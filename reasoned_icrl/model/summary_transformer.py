"""Segment and recurrent-summary memory over AMAGO's Transformer blocks.

The carrier reads one outer task as a sequence of segments of ``C`` admitted
policy-input records. Every segment is one attention problem of ``M + C + M``
slots in the fixed order ``[READ | RECORD | WRITE]``: the READ rows carry the
memory written by the previous segment, the RECORD rows are the segment's
tokens and produce the actor/critic inputs, and the WRITE rows produce the
memory the next segment reads. One mask rule governs content attention in
every block: query slot ``q`` may attend to key slot ``k`` iff ``k <= q`` and
the key is valid, where READ and WRITE slots are always valid and a padded
RECORD slot never is. A segment's WRITE outputs reach the model only as the
next segment's READ inputs.

The relational branch of a dual-attention block follows the summary identity's
``relational_sources``. Under the legacy ``causal_prefix`` route it shares the
content mask. Under the revised ``timestep_records`` route (the 8M study's
``amago-dat-summary-v2``) its sources are the valid RECORD slots the content
mask already admits and its receivers are the valid RECORD and WRITE rows: a
READ row, a padded row and a WRITE row over an all-padding segment have an
exactly zero relational output, projection bias included. The content branch
is unchanged, so a RECORD row still reads the summary through content
attention and a later-layer relation can still see that indirect effect; only
the *direct* relational sources are restricted.

Training runs the complete outer task with gradients through every write and
no detach. Rollouts keep, per actor, the memory ``Z``, one ``M + C + M``-slot
cache per layer and two counters, independent of task length; the cached
path drives every slot one query per row through the same blocks, so dense
and cached execution agree to float32 rounding. The ordinary blocks adopt an
AMAGO ``TransformerLayer``'s own modules, so at step zero they are byte-
identical to the full-prefix carrier's; the summary tables, the write queries
and the memory projection are the only new parameters.

Block execution order and stabilization follow AMAGO v3.4.0; the dual-attention
block, its cache slabs and the cached content attention are those of
:mod:`reasoned_icrl.model.dat_transformer`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol, cast

import torch
from amago.nets.transformer import FixedPosEmb
from einops import rearrange
from torch import nn

from reasoned_icrl.experiments.contracts import (
    ContractError,
    DATSpec,
    SummarySpec,
)
from reasoned_icrl.model.dat_transformer import (
    _CACHE_DTYPES,
    DATBlock,
    DATLayerCache,
    DualAttention,
    _dense_content_attention,
    _masked_softmax,
    _write_cache,
    require_right_padded,
)

WRITE_GATE_INIT = -3.0
"""The gated rewrite's initial gate logit: sigmoid(-3) is about 0.047, a
per-slot gate near closed before training (T3)."""


SUMMARY_HIDDEN_STATE_SCHEMA = "amago-summary-hidden-state.v1"

ROLE_READ, ROLE_RECORD, ROLE_WRITE = 0, 1, 2
"""Rows of the learned role table, in slot order."""


class AttentionProbe(Protocol):
    """An evaluation-only observer of the cached content attention.

    Attached through :meth:`SummaryTransformer.probed`, it sees every ordinary
    block's single cached query per row. ``query_slots`` is the ``[B]`` slot of
    each row's query (its role follows from the slot table), the key axis is
    the retained window in slot order, and ``layer`` indexes the block.
    :meth:`logit_bias` may return an additive ``[B, 1, 1, window]`` score bias
    (``-inf`` blocks a key); :meth:`observe` receives the resulting
    ``[B, H, 1, window]`` weights. With no bias the probed attention equals the
    unprobed one exactly. The study protocol's attention
    declaration defines the probes; no trained identity depends on them.
    """

    def logit_bias(
        self, layer: int, query_slots: torch.Tensor, window: int
    ) -> torch.Tensor | None: ...

    def observe(
        self, layer: int, query_slots: torch.Tensor, weights: torch.Tensor
    ) -> None: ...


class MemoryTransform(Protocol):
    """An evaluation-only intervention on the summary a boundary writes.

    Attached through :meth:`SummaryTransformer.transformed`,
    :meth:`after_write` runs inside :meth:`SummaryTransformer.boundary` after
    the write and the segment counter's increment and before the new
    segment's read pass, on the rows crossing the boundary together; it may
    replace rows of ``hidden.memory`` in place. With nothing attached the
    carrier is unchanged. The representation
    declaration defines the transplant; no trained identity depends on it.
    """

    def after_write(self, hidden: SummaryHiddenState) -> None: ...


def _rows(idxs: Any, device: torch.device) -> torch.Tensor:
    """Normalize a boolean mask or an index sequence to a long index tensor."""
    rows = torch.as_tensor(idxs, device=device)
    if rows.dtype == torch.bool:
        rows = torch.where(rows.reshape(-1))[0]
    return rows.reshape(-1).long()


class SummaryHiddenState:
    """Rollout state of the summary carrier for one actor batch.

    ``memory`` is the summary the current segment reads; ``layers`` hold one
    ``M + C + M``-slot cache per block; ``lengths`` counts the filled slots of
    the current segment (``0`` means the read pass is pending); ``segment``
    counts the boundaries crossed in the current outer task. ``initial_memory``
    is a detached copy of the carrier's learned initial memory, refreshed by
    ``init_hidden_state`` and ``rebuild`` after every learner update, so a
    reset never runs the model. ``summary_cleared`` is the evaluator's
    intervention flag: when set, every boundary writes the initial memory
    instead of the carried summary.
    """

    def __init__(
        self,
        memory: torch.Tensor,
        layers: Sequence[DATLayerCache],
        lengths: torch.Tensor,
        segment: torch.Tensor,
        *,
        spec_sha256: str,
        initial_memory: torch.Tensor,
        capacity: int,
        summary_cleared: bool = False,
    ) -> None:
        if not layers:
            raise ContractError("A summary hidden state needs at least one layer.")
        if memory.ndim != 3 or memory.dtype != torch.float32:
            raise ContractError("Summary memory must be a float32 [B, M, d] tensor.")
        batch, tokens, width = memory.shape
        if lengths.dtype != torch.int32 or lengths.shape != (batch,):
            raise ContractError("Summary lengths must be int32 with one entry per row.")
        if segment.dtype != torch.int64 or segment.shape != (batch,):
            raise ContractError("Summary segment counters must be int64 per row.")
        if initial_memory.shape != (tokens, width):
            raise ContractError("Initial memory must be [M, d].")
        for cache in layers:
            for name, tensor in cache.tensors().items():
                if tensor.shape[:2] != (batch, capacity):
                    raise ContractError(
                        f"Summary cache tensor {name!r} disagrees with batch/capacity."
                    )
        self.memory = memory
        self.layers = tuple(layers)
        self.lengths = lengths
        self.segment = segment
        self.spec_sha256 = spec_sha256
        self.initial_memory = initial_memory
        self.capacity = capacity
        self.summary_cleared = summary_cleared
        self.batch_size = batch
        self.memory_tokens = tokens
        self.device = memory.device

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> DATLayerCache:
        if not 0 <= layer_idx < self.n_layers:
            raise ContractError("Summary layer index outside the cached backbone.")
        return self.layers[layer_idx]

    def clear(self, rows: torch.Tensor) -> None:
        """Poison the caches and restart the slot counter; memory is untouched."""
        if rows.numel() == 0:
            return
        self.lengths[rows] = 0
        for cache in self.layers:
            cache.clear_rows(rows)

    def reset(self, idxs: Any) -> None:
        """Start a new outer task on the given rows; other rows are untouched."""
        rows = _rows(idxs, self.device)
        if rows.numel() == 0:
            return
        self.memory[rows] = self.initial_memory.to(self.memory.dtype)
        self.segment[rows] = 0
        self.clear(rows)

    def select(self, rows: torch.Tensor) -> SummaryHiddenState:
        """A detached copy of the given rows, to be written back by ``assign``."""
        return SummaryHiddenState(
            self.memory[rows].clone(),
            [
                DATLayerCache(
                    cache.variant,
                    cache.content_keys[rows].clone(),
                    cache.content_values[rows].clone(),
                    None
                    if cache.selection_keys is None
                    else cache.selection_keys[rows].clone(),
                    None
                    if cache.selection_values is None
                    else cache.selection_values[rows].clone(),
                    None
                    if cache.relation_keys is None
                    else cache.relation_keys[rows].clone(),
                )
                for cache in self.layers
            ],
            self.lengths[rows].clone(),
            self.segment[rows].clone(),
            spec_sha256=self.spec_sha256,
            initial_memory=self.initial_memory,
            capacity=self.capacity,
            summary_cleared=self.summary_cleared,
        )

    def assign(self, rows: torch.Tensor, sub: SummaryHiddenState) -> None:
        """Scatter a processed row subset back into this state."""
        if sub.batch_size != rows.numel() or sub.n_layers != self.n_layers:
            raise ContractError("Summary row subset does not match its rows.")
        self.memory[rows] = sub.memory
        self.lengths[rows] = sub.lengths
        self.segment[rows] = sub.segment
        for target, source in zip(self.layers, sub.layers, strict=True):
            stored = source.tensors()
            for name, tensor in target.tensors().items():
                tensor[rows] = stored[name]


class MaskedOrdinaryBlock(nn.Module):
    """An AMAGO pre-norm block run under an explicit attention mask.

    Adopts, as the same module objects, the donor ``TransformerLayer``'s
    attention projections, head scaler, normalizations, feed-forward layers,
    activation and dropout, and adds no parameter, so the block's state is
    byte-identical to the donor's. What changes is only how sources are
    selected: an arbitrary ``[B, 1, Q, K]`` boolean mask instead of AMAGO's
    plain causal rule, and a project-owned cache slab for the rollout path.
    Positions are never applied here; the summary carrier adds them at the
    embedding, so RoPE donors are refused.
    """

    def __init__(self, donor: Any) -> None:
        super().__init__()
        attention = donor.attention_layer
        if getattr(attention, "rope", None) is not None:
            raise ContractError(
                "The summary carrier uses fixed slot positions, not RoPE."
            )
        # Registered in the donor's own order so that the block's state_dict
        # iterates exactly as a TransformerLayer's does, and a per-block hash of
        # the two is comparable byte for byte.
        self.attention_layer: Any = attention
        self.ff1: nn.Module = donor.ff1
        self.ff2: nn.Module = donor.ff2
        self.norm1: nn.Module = donor.norm1
        self.norm2: nn.Module = donor.norm2
        self.norm3: nn.Module = donor.norm3
        self.norm4: nn.Module = donor.norm4
        self.dropout_ff: nn.Module = donor.dropout_ff
        self.activation = donor.activation
        self.d_model: int = int(donor.d_model)
        self.n_heads: int = int(attention.n_heads)
        self.head_dim: int = int(attention.qkv_projection.out_features) // (
            3 * self.n_heads
        )
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # Evaluation-only diagnostics (SummaryTransformer.probed); not a module,
        # a parameter or a buffer, so the state dict is unchanged.
        self.probe: AttentionProbe | None = None
        self.probe_layer = -1

    def project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The donor's fused Q/K/V of ``x``, each ``[B, T, heads, head_dim]``."""
        attention = self.attention_layer
        qkv = attention.dropout_qkv(attention.qkv_projection(x))
        qkv = rearrange(
            qkv,
            "batch len (three d_qkv heads) -> batch len three heads d_qkv",
            heads=self.n_heads,
            three=3,
        )
        queries, keys, values = torch.unbind(qkv, dim=2)
        return queries, keys, values

    def attend(
        self,
        x: torch.Tensor,
        allowed: torch.Tensor,
        cache: DATLayerCache | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Content attention over the permitted sources, dense or cached."""
        attention = self.attention_layer
        queries, keys, values = self.project(x)
        if cache is not None:
            if lengths is None:
                raise ContractError("Cached attention needs the retained lengths.")
            sources = _write_cache(
                cache, lengths, content_keys=keys, content_values=values
            )
            keys, values = sources["content_keys"], sources["content_values"]
        if self.probe is None:
            out = _dense_content_attention(
                queries, keys, values, allowed, scale=self.scale
            )
        else:
            out = self._probed_attention(queries, keys, values, allowed, lengths)
        out = rearrange(attention.head_scaler * out, "b q h d -> b q (h d)")
        return cast(torch.Tensor, attention.out_projection(out))

    def _probed_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        allowed: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """:func:`_dense_content_attention` with the probe's bias and observer.

        The same operations in the same order, so an unbiased probe changes no
        value. Probes observe the cached rollout path only, where each row
        holds one query at slot ``lengths[b]``.
        """
        probe = self.probe
        assert probe is not None
        if lengths is None or queries.shape[1] != 1:
            raise ContractError("Attention probes observe the cached path only.")
        scores = torch.einsum("bqhd,bkhd->bhqk", queries, keys) * self.scale
        bias = probe.logit_bias(self.probe_layer, lengths, scores.shape[-1])
        if bias is not None:
            scores = scores + bias
        weights = _masked_softmax(scores, allowed)
        probe.observe(self.probe_layer, lengths, weights)
        return torch.einsum("bhqk,bkhd->bqhd", weights, values)

    def forward(
        self,
        x: torch.Tensor,
        allowed: torch.Tensor,
        cache: DATLayerCache | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """AMAGO's block order: pre-norm attention, normformer norms, feed-forward.

        Args:
            x: ``[B, Q, d_model]`` block input.
            allowed: ``[B, 1, Q, K]`` boolean mask of permitted sources.
            cache: when given, the current query's sources are written at slot
                ``lengths[b]`` and the retained slots supply the keys.
            lengths: ``[B]`` filled slots, required with ``cache``.
        """
        q1 = self.norm1(x)
        q1 = self.attend(q1, allowed, cache=cache, lengths=lengths)
        q1 = self.norm2(q1)
        x = x + q1
        q1 = self.norm3(x)
        q1 = self.norm4(self.activation(self.ff1(q1)))
        q1 = self.dropout_ff(self.ff2(q1))
        return cast(torch.Tensor, x + q1)


class SummaryTransformer(nn.Module):
    """An AMAGO Transformer run segment by segment with a written summary.

    Built by adopting a complete ordinary backbone, exactly as
    :class:`~reasoned_icrl.model.dat_transformer.DATTransformer` does: the
    input projection, sinusoidal position table, embedding dropout, final
    normalization and every unselected block are the donor's own modules, and
    the selected blocks (when ``dat`` is given) run dual attention over the
    segment's slots. New parameters are the initial memory ``Z_init``, the
    write queries ``Q_write``, the role and slot tables and the memory
    projection ``P_mem``.
    """

    def __init__(
        self,
        donor: Any,
        spec: SummarySpec,
        dat: DATSpec | None = None,
        *,
        dropout_qkv: float = 0.0,
        head_scaling: bool = True,
        sigma_reparam: bool = True,
    ) -> None:
        super().__init__()
        layers = list(donor.layers)
        if getattr(donor, "use_rope", False):
            raise ContractError(
                "The summary carrier uses fixed slot positions, not RoPE."
            )
        if not isinstance(donor.position_embedding, FixedPosEmb):
            raise ContractError("The summary carrier needs the donor's FixedPosEmb.")
        d_model = int(donor.d_model)
        if spec.d_model != d_model:
            raise ContractError("Summary width disagrees with the donor backbone.")
        if dat is not None:
            dat.validate_layers(len(layers))
            if dat.d_model != d_model:
                raise ContractError("DAT attention width disagrees with the donor.")
            if dat.max_relative_distance != spec.capacity:
                raise ContractError(
                    "A summary cell's DAT clipping distance must equal the segment "
                    "capacity M + C + M."
                )
            if dat.position_method != "fixed":
                raise ContractError("Summary DAT blocks use fixed slot positions.")
            if spec.writer == "relational_off" and dat.mode != "dat":
                raise ContractError(
                    "The relational-write-off writer needs a relational branch."
                )
            if spec.relational_sources == "timestep_records" and (
                dat.mode == "dual_content"
            ):
                raise ContractError(
                    "The timestep-record route restricts a relational branch; the "
                    "dual-content control has none."
                )
        elif spec.writer == "relational_off":
            raise ContractError(
                "The relational-write-off writer needs a dual-attention block."
            )
        elif spec.relational_sources == "timestep_records":
            raise ContractError(
                "The timestep-record route restricts a relational branch; ordinary "
                "blocks have none, so the route would compute nothing new."
            )
        self.spec = spec
        self.dat = dat
        self.inp: nn.Module = donor.inp
        self.position_embedding: nn.Module = donor.position_embedding
        self.dropout: nn.Module = donor.dropout
        self.norm: nn.Module = donor.norm
        self.d_model = d_model
        self.n_layers = len(layers)
        self.selected = frozenset(() if dat is None else dat.layer_indices)
        built: list[nn.Module] = []
        for index, layer in enumerate(layers):
            if index in self.selected:
                assert dat is not None
                built.append(
                    DATBlock(
                        layer,
                        DualAttention(
                            dat,
                            dropout_qkv=dropout_qkv,
                            head_scaling=head_scaling,
                            sigma_reparam=sigma_reparam,
                        ),
                    )
                )
            else:
                built.append(MaskedOrdinaryBlock(layer))
        self.layers = nn.ModuleList(built)
        self.backbone_heads = int(layers[0].attention_layer.n_heads)
        self.backbone_head_dim = d_model // self.backbone_heads
        tokens = spec.memory_tokens
        self.memory_init = nn.Parameter(torch.empty(tokens, d_model))
        self.write_queries = nn.Parameter(torch.empty(tokens, d_model))
        self.role_embedding = nn.Parameter(torch.empty(3, d_model))
        self.slot_embedding = nn.Parameter(torch.empty(tokens, d_model))
        for table in (
            self.memory_init,
            self.write_queries,
            self.role_embedding,
            self.slot_embedding,
        ):
            nn.init.normal_(table, std=0.02)
        self.memory_projection = nn.Linear(d_model, d_model)
        # T3, the gated residual write:
        # memory <- memory + sigmoid(g) * projection(write), with one gate logit
        # per WRITE slot computed from the normed WRITE outputs. The gate's
        # weights start at zero and its bias at WRITE_GATE_INIT, so at
        # initialisation the gate is a per-slot scalar near closed; it is built
        # for the gated rule only, so every other carrier's parameters are
        # unchanged.
        self.write_gate: nn.Linear | None = None
        if spec.rewrite == "gated":
            self.write_gate = nn.Linear(d_model, 1)
            nn.init.zeros_(self.write_gate.weight)
            nn.init.constant_(self.write_gate.bias, WRITE_GATE_INIT)
        # Evaluation-only intervention (``transformed``); not a module, a
        # parameter or a buffer, so the state dict is unchanged.
        self.memory_transform: MemoryTransform | None = None

    def apply_write(
        self, memory: torch.Tensor, write_outputs: torch.Tensor
    ) -> torch.Tensor:
        """The boundary rule on the normed ``[B, M, d]`` WRITE outputs: the
        projected write replaces the memory (``replace``), is added to it
        (``residual``) or is added scaled by the sigmoid gate (``gated``)."""
        written: torch.Tensor = self.memory_projection(write_outputs)
        rewrite = self.spec.rewrite
        if rewrite == "replace":
            return written
        if rewrite == "gated":
            assert self.write_gate is not None
            written = torch.sigmoid(self.write_gate(write_outputs)) * written
        return memory.to(written.dtype) + written

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def emb_dim(self) -> int:
        return self.d_model

    @property
    def capacity(self) -> int:
        return self.spec.capacity

    def layer_variant(self, index: int) -> str:
        if index in self.selected:
            assert self.dat is not None
            return self.dat.mode
        return "ordinary"

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _positions(self, slots: torch.Tensor) -> torch.Tensor:
        """The donor's sinusoidal table evaluated at ``[B, n]`` slot indices."""
        return cast(torch.Tensor, self.position_embedding(slots.long()))

    def embed_reads(self, memory: torch.Tensor) -> torch.Tensor:
        """READ rows: the carried memory at slots ``0 .. M-1``."""
        batch = memory.shape[0]
        slots = torch.arange(self.spec.memory_tokens, device=memory.device)
        slots = slots.unsqueeze(0).expand(batch, -1)
        x = (
            memory
            + self._positions(slots)
            + self.role_embedding[ROLE_READ]
            + self.slot_embedding
        )
        return cast(torch.Tensor, self.dropout(x))

    def embed_writes(self, batch: int, device: torch.device) -> torch.Tensor:
        """WRITE rows: the learned write queries at slots ``M+C .. M+C+M-1``."""
        spec = self.spec
        slots = torch.arange(spec.write_slots.start, spec.capacity, device=device)
        slots = slots.unsqueeze(0).expand(batch, -1)
        x = (
            self.write_queries
            + self._positions(slots)
            + self.role_embedding[ROLE_WRITE]
            + self.slot_embedding
        )
        return cast(torch.Tensor, self.dropout(x))

    def embed_records(self, records: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """RECORD rows: ``[B, n, token]`` records at the given ``[B, n]`` slots."""
        x = (
            self.inp(records)
            + self._positions(slots)
            + self.role_embedding[ROLE_RECORD]
        )
        return cast(torch.Tensor, self.dropout(x))

    def embed_segment(
        self, records: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """One complete segment ``[B, M + C + M, d]`` in slot order."""
        batch, count, _ = records.shape
        spec = self.spec
        if count != spec.segment_length:
            raise ContractError("A segment embeds exactly C records.")
        slots = torch.arange(
            spec.record_slots.start, spec.record_slots.stop, device=records.device
        )
        slots = slots.unsqueeze(0).expand(batch, -1)
        return torch.cat(
            (
                self.embed_reads(memory),
                self.embed_records(records, slots),
                self.embed_writes(batch, records.device),
            ),
            dim=1,
        )

    # ------------------------------------------------------------------
    # Dense (training) path
    # ------------------------------------------------------------------

    def _row_mask(self, batch: int, device: torch.device) -> torch.Tensor | None:
        """Zero the relational branch on WRITE slots under ``relational_off``.

        The multiplier removes the projected relational half (pairwise
        relations, source symbols and the projection bias alike) from the
        WRITE queries only; content heads and every other row are unchanged.
        """
        if self.spec.writer != "relational_off":
            return None
        mask = torch.ones((batch, self.capacity), device=device)
        mask[:, self.spec.write_slots.start :] = 0.0
        return mask

    def allowed_mask(self, key_valid: torch.Tensor) -> torch.Tensor:
        """The content mask, ``[B, cap, cap]``: ``k <= q`` and ``key_valid[b, k]``.

        In slot order this is exactly the SPEC content column: a READ row reads
        earlier/self READ rows; a RECORD row reads READ and valid causal/self
        RECORD rows; a WRITE row reads READ, every valid RECORD and earlier/self
        WRITE rows.
        """
        capacity = key_valid.shape[1]
        if key_valid.ndim != 2 or capacity != self.capacity:
            raise ContractError("Key validity disagrees with the slot capacity.")
        causal = torch.tril(
            torch.ones((capacity, capacity), dtype=torch.bool, device=key_valid.device)
        )
        return causal.unsqueeze(0) & key_valid.bool().unsqueeze(1)

    def slot_roles(self, device: torch.device) -> torch.Tensor:
        """``[cap]`` role of every slot: READ, RECORD or WRITE, in slot order."""
        spec = self.spec
        roles = torch.full(
            (spec.capacity,), ROLE_RECORD, dtype=torch.long, device=device
        )
        roles[: spec.record_slots.start] = ROLE_READ
        roles[spec.write_slots.start :] = ROLE_WRITE
        return roles

    @contextmanager
    def probed(self, probe: AttentionProbe) -> Iterator[None]:
        """Route every block's cached content attention through ``probe``.

        Evaluation only: the dense training path refuses a probe. Carriers with
        a dual-attention block are refused, since their relational branch
        would bypass the probe.
        """
        if self.selected:
            raise ContractError("Attention probes cover ordinary blocks only.")
        blocks = [cast(MaskedOrdinaryBlock, layer) for layer in self.layers]
        if any(block.probe is not None for block in blocks):
            raise ContractError("An attention probe is already attached.")
        for index, block in enumerate(blocks):
            block.probe = probe
            block.probe_layer = index
        try:
            yield
        finally:
            for block in blocks:
                block.probe = None
                block.probe_layer = -1

    @contextmanager
    def transformed(self, transform: MemoryTransform) -> Iterator[None]:
        """Pass every boundary's written summary through ``transform``.

        Evaluation only (the cached path's :meth:`boundary`; the dense
        training path never calls it). Refused for a carrier that writes no
        summary, and when a transform is already attached.
        """
        if self.spec.regime != "summary":
            raise ContractError(
                "Memory transforms apply to a carrier that writes a summary."
            )
        if self.memory_transform is not None:
            raise ContractError("A memory transform is already attached.")
        self.memory_transform = transform
        try:
            yield
        finally:
            self.memory_transform = None

    @property
    def routes_relational_sources(self) -> bool:
        """Whether the relational branch runs the timestep-record route."""
        return self.spec.relational_sources == "timestep_records"

    def relational_allowed_mask(self, key_valid: torch.Tensor) -> torch.Tensor | None:
        """The relational mask, ``[B, cap, cap]``, or ``None`` on the legacy route.

        A subset of :meth:`allowed_mask`: the key must additionally be a
        RECORD slot, and the query a valid RECORD or WRITE slot. READ rows,
        padded RECORD rows and any row whose segment holds no valid record get
        an all-``False`` row, which the block turns into an exactly zero
        relational output. READ and WRITE slots are never relational sources
        even where the content branch reads them, and a RECORD key is still
        subject to the same causal order and validity as a content key.
        """
        if not self.routes_relational_sources:
            return None
        allowed = self.allowed_mask(key_valid)
        roles = self.slot_roles(key_valid.device)
        record_key = (roles == ROLE_RECORD).view(1, 1, -1)
        routed_query = (roles != ROLE_READ).view(1, -1, 1)
        valid_query = key_valid.bool().unsqueeze(2)
        return allowed & record_key & routed_query & valid_query

    def segment_forward(self, x: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        """Run every block over one embedded segment under the one mask rule.

        Args:
            x: ``[B, cap, d]`` embedded slots.
            key_valid: ``[B, cap]`` boolean; READ and WRITE slots are always
                valid, a padded RECORD slot never is.
        """
        batch, capacity, _ = x.shape
        if capacity != self.capacity or key_valid.shape != (batch, capacity):
            raise ContractError("Segment tensors disagree with the slot capacity.")
        allowed = self.allowed_mask(key_valid)
        relational = self.relational_allowed_mask(key_valid)
        times = torch.arange(capacity, device=x.device).unsqueeze(0).expand(batch, -1)
        row_mask = self._row_mask(batch, x.device)
        for index, layer in enumerate(self.layers):
            if index in self.selected:
                x = layer(
                    x,
                    times,
                    times,
                    allowed,
                    relational_row_mask=row_mask,
                    relational_allowed=relational,
                )
            else:
                x = layer(x, allowed.unsqueeze(1))
        return cast(torch.Tensor, self.norm(x))

    def training_forward(
        self, tokens: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one complete outer task segment by segment.

        Args:
            tokens: ``[B, T, token]`` right-padded policy-input records.
            valid: ``[B, T]`` boolean validity, a contiguous prefix per row.

        Returns the ``[B, T, d]`` RECORD outputs, zeroed where invalid, and the
        ``[B, M, d]`` memory after the last segment's write. Gradients flow
        through every write; nothing is detached, unless the spec's ``detach``
        is ``boundary``, in which case the carried memory is detached at every
        boundary (truncated backpropagation; every forward value is unchanged).
        Under the spec's ``residual`` rewrite every boundary adds the projected
        write to the memory the segment read instead of replacing it, and under
        the ``gated`` rewrite the added write is scaled by the learned per-slot
        gate; the cached rollout and the rebuild apply the same rule
        (:meth:`apply_write`).
        """
        if valid.dtype != torch.bool:
            valid = valid != 0
        require_right_padded(valid)
        batch, length, _ = tokens.shape
        if valid.shape != (batch, length) or length < 1:
            raise ContractError("Summary training needs [B, T] validity for T >= 1.")
        spec = self.spec
        count = spec.segment_length
        segments = -(-length // count)
        padding = segments * count - length
        if padding:
            tokens = torch.nn.functional.pad(tokens, (0, 0, 0, padding))
            valid = torch.nn.functional.pad(valid, (0, padding), value=False)
        memory = self.memory_init.unsqueeze(0).expand(batch, -1, -1)
        always = torch.ones(
            (batch, spec.memory_tokens), dtype=torch.bool, device=tokens.device
        )
        outputs: list[torch.Tensor] = []
        for index in range(segments):
            start, stop = index * count, (index + 1) * count
            x = self.embed_segment(tokens[:, start:stop], memory)
            key_valid = torch.cat((always, valid[:, start:stop], always), dim=1)
            hidden = self.segment_forward(x, key_valid)
            outputs.append(hidden[:, spec.record_slots.start : spec.record_slots.stop])
            if spec.regime == "summary":
                memory = self.apply_write(memory, hidden[:, spec.write_slots.start :])
                if spec.detach == "boundary":
                    memory = memory.detach()
            else:
                memory = self.memory_init.unsqueeze(0).expand(batch, -1, -1)
        out = torch.cat(outputs, dim=1)[:, :length]
        return out * valid[:, :length].unsqueeze(-1).to(out.dtype), memory

    # ------------------------------------------------------------------
    # Rollout state
    # ------------------------------------------------------------------

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> SummaryHiddenState:
        """Allocate poisoned caches and the initial memory; the read pass is pending."""
        spec = self.spec
        dat = self.dat
        capacity = spec.capacity
        dtype = _CACHE_DTYPES[spec.cache_dtype]

        def slab(heads: int, width: int) -> torch.Tensor:
            data = torch.zeros(
                (batch_size, capacity, heads, width), dtype=dtype, device=device
            )
            data[:] = torch.nan
            return data

        caches: list[DATLayerCache] = []
        for index in range(self.n_layers):
            if index not in self.selected:
                caches.append(
                    DATLayerCache(
                        "ordinary",
                        slab(self.backbone_heads, self.backbone_head_dim),
                        slab(self.backbone_heads, self.backbone_head_dim),
                    )
                )
                continue
            assert dat is not None
            content = (dat.content_heads, dat.content_head_dim)
            cache = DATLayerCache(dat.mode, slab(*content), slab(*content))
            if dat.mode == "dual_content":
                cache.selection_keys = slab(dat.relational_heads, dat.second_head_dim)
                cache.selection_values = slab(dat.relational_heads, dat.second_head_dim)
            else:
                cache.selection_keys = slab(dat.relational_heads, dat.head_dim)
                if dat.uses_relations:
                    cache.relation_keys = slab(
                        dat.relation_channels, dat.relation_projection_dim
                    )
            caches.append(cache)
        initial = self.memory_init.detach().to(device=device, dtype=torch.float32)
        return SummaryHiddenState(
            initial.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            caches,
            torch.zeros((batch_size,), dtype=torch.int32, device=device),
            torch.zeros((batch_size,), dtype=torch.int64, device=device),
            spec_sha256=spec.sha256,
            initial_memory=initial.clone(),
            capacity=capacity,
        )

    # ------------------------------------------------------------------
    # Cached (rollout) path
    # ------------------------------------------------------------------

    def _cached_layers(
        self,
        x: torch.Tensor,
        hidden: SummaryHiddenState,
        *,
        relational_row_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One query per row at slot ``lengths[b]`` over keys ``<= lengths[b]``."""
        lengths = hidden.lengths
        batch = hidden.batch_size
        window = int(lengths.max().item()) + 1
        if window > self.capacity:
            raise ContractError("Summary cache slot outside the segment capacity.")
        positions = torch.arange(window, device=hidden.device)
        allowed = (positions.unsqueeze(0) <= lengths.long().unsqueeze(1)).unsqueeze(1)
        relational = self._cached_relational_mask(allowed, lengths)
        key_times = positions.unsqueeze(0).expand(batch, -1)
        query_times = lengths.long().unsqueeze(1)
        for index, layer in enumerate(self.layers):
            cache = hidden[index]
            if index in self.selected:
                x = layer(
                    x,
                    query_times,
                    key_times,
                    allowed,
                    cache=cache,
                    lengths=lengths,
                    relational_row_mask=relational_row_mask,
                    relational_allowed=relational,
                )
            else:
                x = layer(x, allowed.unsqueeze(1), cache=cache, lengths=lengths)
        return x

    def _cached_relational_mask(
        self, allowed: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor | None:
        """The cached path's relational mask, ``[B, 1, window]``, or ``None``.

        The single query of row ``b`` sits at slot ``lengths[b]``, so its role
        is read off the slot table, and every retained slot is a valid source
        (the rollout never processes a padded record), so the key rule reduces
        to the RECORD slots the content mask admits. This is the dense
        :meth:`relational_allowed_mask` evaluated at one query row.
        """
        if not self.routes_relational_sources:
            return None
        roles = self.slot_roles(allowed.device)
        window = allowed.shape[-1]
        record_key = (roles[:window] == ROLE_RECORD).view(1, 1, window)
        routed_query = (roles[lengths.long()] != ROLE_READ).view(-1, 1, 1)
        return allowed & record_key & routed_query

    def read_pass(self, hidden: SummaryHiddenState) -> None:
        """Enter the current memory at slots ``0 .. M-1``; outputs are discarded."""
        if bool((hidden.lengths != 0).any()):
            raise ContractError("The read pass starts from an empty segment.")
        reads = self.embed_reads(hidden.memory)
        for row in range(self.spec.memory_tokens):
            self._cached_layers(reads[:, row : row + 1], hidden)
            hidden.lengths += 1

    def boundary(self, hidden: SummaryHiddenState) -> None:
        """Write the summary of a full segment, start the next one, read it."""
        spec = self.spec
        if bool((hidden.lengths != spec.write_slots.start).any()):
            raise ContractError("A boundary needs a full segment on every row.")
        batch = hidden.batch_size
        writes = self.embed_writes(batch, hidden.device)
        mask = (
            torch.zeros((batch, 1), device=hidden.device)
            if spec.writer == "relational_off"
            else None
        )
        outputs: list[torch.Tensor] = []
        for row in range(spec.memory_tokens):
            outputs.append(
                self._cached_layers(
                    writes[:, row : row + 1], hidden, relational_row_mask=mask
                )
            )
            hidden.lengths += 1
        if spec.regime == "summary" and not hidden.summary_cleared:
            written = self.apply_write(
                hidden.memory, self.norm(torch.cat(outputs, dim=1))
            )
            hidden.memory.copy_(written.to(hidden.memory.dtype))
        else:
            hidden.memory.copy_(
                hidden.initial_memory.unsqueeze(0).expand(batch, -1, -1)
            )
        hidden.clear(torch.arange(batch, device=hidden.device))
        hidden.segment += 1
        if self.memory_transform is not None:
            self.memory_transform.after_write(hidden)
        self.read_pass(hidden)

    def _on_rows(
        self,
        hidden: SummaryHiddenState,
        rows: torch.Tensor,
        step: Any,
    ) -> None:
        """Run ``step`` on the given rows, in place when they are the whole batch."""
        if rows.numel() == 0:
            return
        if rows.numel() == hidden.batch_size:
            step(hidden)
            return
        sub = hidden.select(rows)
        step(sub)
        hidden.assign(rows, sub)

    def cached_forward(
        self, records: torch.Tensor, hidden: SummaryHiddenState
    ) -> torch.Tensor:
        """Encode one record per row, crossing a boundary first where one is due.

        Args:
            records: ``[B, 1, token]`` the current policy-input record per row.
            hidden: the rollout state, advanced in place.
        """
        if self.training:
            raise ContractError("Cached summary execution is evaluation-only.")
        if records.ndim != 3 or records.shape[1] != 1:
            raise ContractError("Cached summary accepts one record per row.")
        if records.shape[0] != hidden.batch_size:
            raise ContractError("Cached summary batch disagrees with the state.")
        if hidden.spec_sha256 != self.spec.sha256:
            raise ContractError("Summary state was built for a different carrier.")
        spec = self.spec
        self._on_rows(hidden, torch.where(hidden.lengths == 0)[0], self.read_pass)
        self._on_rows(
            hidden,
            torch.where(hidden.lengths == spec.write_slots.start)[0],
            self.boundary,
        )
        x = self.embed_records(records, hidden.lengths.long().unsqueeze(1))
        out = self._cached_layers(x, hidden)
        hidden.lengths += 1
        return cast(torch.Tensor, self.norm(out))

    def rebuild(
        self, tokens: torch.Tensor, lengths: Sequence[int]
    ) -> SummaryHiddenState:
        """Rebuild rollout state from complete processed prefixes.

        Rows are grouped by prefix length ``L``. The convention is the cached
        rollout's own, which crosses a boundary lazily (at the start of the
        call that follows a full segment): ``q = (L - 1) // C`` segments are
        completed and written through the dense path and their memory taken,
        then the remaining ``r = L - qC`` records, ``1 <= r <= C``, are encoded
        one at a time through the cached path. A prefix ending exactly at a
        boundary therefore holds its full last segment with the write pending
        (``lengths == M + C``), so the result equals an uninterrupted rollout
        of the same prefix under the current weights, state for state. No
        padded record is ever processed.
        """
        if self.training:
            raise ContractError("Summary rebuild is evaluation-only.")
        batch, columns, _ = tokens.shape
        if len(lengths) != batch:
            raise ContractError("Summary rebuild needs one true length per row.")
        true_lengths = [int(value) for value in lengths]
        if any(value < 0 for value in true_lengths):
            raise ContractError("Summary rebuild prefix lengths must be nonnegative.")
        if max(true_lengths, default=0) > columns:
            raise ContractError("Summary rebuild prefix exceeds the provided packets.")
        hidden = self.init_hidden_state(batch, tokens.device)
        count = self.spec.segment_length
        groups: dict[int, list[int]] = {}
        for row, length in enumerate(true_lengths):
            if length > 0:
                groups.setdefault(length, []).append(row)
        with torch.no_grad():
            for length, rows in groups.items():
                index = torch.as_tensor(rows, device=tokens.device)
                full = (length - 1) // count  # the lazy boundary of cached_forward
                sub = self.init_hidden_state(len(rows), tokens.device)
                if full > 0:
                    processed = tokens[index, : full * count]
                    valid = torch.ones(
                        processed.shape[:2], dtype=torch.bool, device=tokens.device
                    )
                    _, memory = self.training_forward(processed, valid)
                    sub.memory.copy_(memory.to(sub.memory.dtype))
                sub.segment.fill_(full)
                for step in range(full * count, length):
                    self.cached_forward(tokens[index, step : step + 1], sub)
                hidden.assign(index, sub)
        return hidden


__all__ = [
    "ROLE_READ",
    "ROLE_RECORD",
    "ROLE_WRITE",
    "SUMMARY_HIDDEN_STATE_SCHEMA",
    "MaskedOrdinaryBlock",
    "SummaryHiddenState",
    "SummaryTransformer",
]
