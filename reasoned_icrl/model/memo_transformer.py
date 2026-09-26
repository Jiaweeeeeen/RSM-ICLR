"""Memo: accumulated summary tokens over AMAGO's Transformer blocks.

The comparator carrier of the ME0 audit (Gupta et al., 2025, *Memo*; source
``Memory-icrl/memo`` at commit ``9e7044f``, ``AutoCompressorTransformer``),
ported onto the study's donor backbone. One outer task is read as segments
of ``L`` admitted records. Segment ``i`` is one causal attention problem over
the concatenated block

``[ SUMMARY(0 .. i*S-1) | RECORD(0 .. len-1) | QUERY(0 .. S-1) ]``

whose positions are the block's slot indices (the paper's scheme: summaries at
``0 .. nS-1``, the segment's records from ``nS``). The SUMMARY rows are the
final-layer outputs of every earlier segment's QUERY rows, re-entered as
inputs with their positions and the embedding dropout and without the input
projection; the RECORD rows produce the actor/critic inputs; the QUERY rows
are ``S`` learned embeddings whose outputs are *appended* to the carried
summaries, so the readable history grows by ``S`` tokens at every boundary.
One mask rule governs every block: query slot ``q`` reads key slot ``k`` iff
``k <= q`` and the key is valid, where SUMMARY and QUERY slots are always valid
and a padded RECORD slot never is. A boundary's QUERY outputs reach the model
only through the following segments' SUMMARY rows.

Training runs the complete outer task with gradients through every summary
and no detach, drawing one dense segment length per forward from the spec's
jitter range (the paper's +/-20 %); rollout and reconstruction use the fixed
``L``. Rollouts keep, per actor, one cache per layer whose filled length is the
concatenated block's length, plus two counters; the cached path drives every
slot one query per row through the same blocks, so dense and cached execution
agree to float32 rounding. The cache is allocated for the longest task the
carrier is built for and its *live* length grows with the boundaries crossed
(``contracts.memo_live_slots``), which is what the comparator measures. Every
block is the donor's own :class:`MaskedOrdinaryBlock`; the ``S`` summary
embeddings are the only new parameters.

Adaptations from the source (per-actor lazy boundaries, the dense positions
of the QUERY rows on both paths, the sinusoidal donor table, validity masks,
FP32 caches, the uniform jitter) are recorded in the ME0 audit report.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from amago.nets.transformer import FixedPosEmb
from torch import nn

from reasoned_icrl.experiments.contracts import ContractError, MemoSpec
from reasoned_icrl.model.dat_transformer import (
    _CACHE_DTYPES,
    DATLayerCache,
    require_right_padded,
)
from reasoned_icrl.model.summary_transformer import MaskedOrdinaryBlock, _rows

MEMO_HIDDEN_STATE_SCHEMA = "amago-memo-hidden-state.v1"


class MemoHiddenState:
    """Rollout state of the Memo carrier for one actor batch.

    ``layers`` hold one cache per block, ``lengths`` counts the filled slots
    (``held * S`` summary slots plus the open segment's records), ``segment``
    counts the boundaries crossed in the current outer task. ``summary_cleared``
    is the evaluator's intervention flag: when set, every boundary discards
    the summaries instead of appending them, so the next segment starts from
    an empty cache and the carrier reads its current segment only.
    """

    def __init__(
        self,
        layers: Sequence[DATLayerCache],
        lengths: torch.Tensor,
        segment: torch.Tensor,
        *,
        spec_sha256: str,
        capacity: int,
        summary_tokens: int,
        summary_cleared: bool = False,
    ) -> None:
        if not layers:
            raise ContractError("A Memo hidden state needs at least one layer.")
        batch = int(lengths.shape[0]) if lengths.ndim == 1 else -1
        if lengths.dtype != torch.int32 or lengths.ndim != 1:
            raise ContractError("Memo lengths must be int32 with one entry per row.")
        if segment.dtype != torch.int64 or segment.shape != (batch,):
            raise ContractError("Memo segment counters must be int64 per row.")
        for cache in layers:
            if cache.variant != "ordinary":
                raise ContractError("The Memo carrier caches ordinary blocks only.")
            for name, tensor in cache.tensors().items():
                if tensor.shape[:2] != (batch, capacity):
                    raise ContractError(
                        f"Memo cache tensor {name!r} disagrees with batch/capacity."
                    )
        self.layers = tuple(layers)
        self.lengths = lengths
        self.segment = segment
        self.spec_sha256 = spec_sha256
        self.capacity = capacity
        self.summary_tokens = summary_tokens
        self.summary_cleared = summary_cleared
        self.batch_size = batch
        self.device = lengths.device

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> DATLayerCache:
        if not 0 <= layer_idx < self.n_layers:
            raise ContractError("Memo layer index outside the cached backbone.")
        return self.layers[layer_idx]

    @property
    def held(self) -> torch.Tensor:
        """Summary slots each row carries: ``segment * S``, zero when cleared."""
        if self.summary_cleared:
            return torch.zeros_like(self.segment)
        return self.segment * self.summary_tokens

    def truncate(self, keep: torch.Tensor) -> None:
        """Keep the first ``keep[b]`` slots of every row; poison the rest."""
        if keep.shape != (self.batch_size,):
            raise ContractError("Memo truncation needs one slot count per row.")
        positions = torch.arange(self.capacity, device=self.device)
        drop = positions.unsqueeze(0) >= keep.long().unsqueeze(1)
        for cache in self.layers:
            for tensor in cache.tensors().values():
                tensor[drop] = torch.nan
        self.lengths.copy_(keep.to(torch.int32))

    def reset(self, idxs: Any) -> None:
        """Start a new outer task on the given rows; other rows are untouched."""
        rows = _rows(idxs, self.device)
        if rows.numel() == 0:
            return
        self.lengths[rows] = 0
        self.segment[rows] = 0
        for cache in self.layers:
            cache.clear_rows(rows)

    def select(self, rows: torch.Tensor) -> MemoHiddenState:
        """A detached copy of the given rows, to be written back by ``assign``."""
        return MemoHiddenState(
            [
                DATLayerCache(
                    cache.variant,
                    cache.content_keys[rows].clone(),
                    cache.content_values[rows].clone(),
                )
                for cache in self.layers
            ],
            self.lengths[rows].clone(),
            self.segment[rows].clone(),
            spec_sha256=self.spec_sha256,
            capacity=self.capacity,
            summary_tokens=self.summary_tokens,
            summary_cleared=self.summary_cleared,
        )

    def assign(self, rows: torch.Tensor, sub: MemoHiddenState) -> None:
        """Scatter a processed row subset back into this state."""
        if sub.batch_size != rows.numel() or sub.n_layers != self.n_layers:
            raise ContractError("Memo row subset does not match its rows.")
        self.lengths[rows] = sub.lengths
        self.segment[rows] = sub.segment
        for target, source in zip(self.layers, sub.layers, strict=True):
            stored = source.tensors()
            for name, tensor in target.tensors().items():
                tensor[rows] = stored[name]


class MemoTransformer(nn.Module):
    """An AMAGO Transformer run segment by segment with accumulated summaries.

    Built by adopting a complete ordinary backbone: the input projection,
    sinusoidal position table, embedding dropout, final normalization and
    every block are the donor's own modules, wrapped as
    :class:`MaskedOrdinaryBlock` so an explicit mask and a project-owned cache
    replace AMAGO's plain causal rule. ``summary_embedding`` (the ``S`` learned
    query tokens) is the only new parameter. ``max_index`` is the largest
    record index of a task the carrier may run, which fixes the cache
    allocation; the live state stays proportional to the boundaries crossed.
    """

    def __init__(self, donor: Any, spec: MemoSpec, *, max_index: int) -> None:
        super().__init__()
        layers = list(donor.layers)
        if getattr(donor, "use_rope", False):
            raise ContractError("The Memo carrier uses fixed slot positions, not RoPE.")
        if not isinstance(donor.position_embedding, FixedPosEmb):
            raise ContractError("The Memo carrier needs the donor's FixedPosEmb.")
        d_model = int(donor.d_model)
        if spec.d_model != d_model:
            raise ContractError("Memo width disagrees with the donor backbone.")
        if max_index < 0:
            raise ContractError("The longest record index is non-negative.")
        # A task shorter than one segment (a smoke profile) never crosses a
        # boundary; the carrier then holds one open segment and no summary,
        # which the capacity formula covers (L + S slots).
        self.spec = spec
        self.max_index = int(max_index)
        self.capacity = spec.capacity(max_index)
        self.inp: nn.Module = donor.inp
        self.position_embedding: nn.Module = donor.position_embedding
        self.dropout: nn.Module = donor.dropout
        self.norm: nn.Module = donor.norm
        self.d_model = d_model
        self.n_layers = len(layers)
        self.layers = nn.ModuleList([MaskedOrdinaryBlock(layer) for layer in layers])
        self.backbone_heads = int(layers[0].attention_layer.n_heads)
        self.backbone_head_dim = d_model // self.backbone_heads
        self.summary_embedding = nn.Parameter(torch.empty(spec.summary_tokens, d_model))
        nn.init.normal_(self.summary_embedding, std=0.02)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def emb_dim(self) -> int:
        return self.d_model

    @property
    def segment_length(self) -> int:
        return self.spec.segment_length

    @property
    def summary_tokens(self) -> int:
        return self.spec.summary_tokens

    def layer_variant(self, index: int) -> str:
        del index
        return "ordinary"

    def dense_segment_length(self) -> int:
        """The dense path's segment length for this forward: ``L`` outside
        training, one uniform draw from the jitter range inside it."""
        low, high = self.spec.jitter_range
        if not self.training or low == high:
            return self.segment_length
        return int(torch.randint(low, high + 1, (1,)).item())

    # ------------------------------------------------------------------
    # Embedding: every row is placed at its block slot
    # ------------------------------------------------------------------

    def _positions(self, slots: torch.Tensor) -> torch.Tensor:
        """The donor's sinusoidal table evaluated at ``[B, n]`` slot indices."""
        return cast(torch.Tensor, self.position_embedding(slots.long()))

    def embed_records(self, records: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """RECORD rows: ``[B, n, token]`` records at the given ``[B, n]`` slots."""
        return cast(
            torch.Tensor, self.dropout(self.inp(records) + self._positions(slots))
        )

    def embed_queries(self, slots: torch.Tensor) -> torch.Tensor:
        """QUERY rows: the ``S`` learned summary embeddings at ``[B, S]`` slots."""
        if slots.shape[1] != self.summary_tokens:
            raise ContractError("A boundary embeds exactly S summary queries.")
        return cast(
            torch.Tensor, self.dropout(self.summary_embedding + self._positions(slots))
        )

    def embed_query(self, row: int, slots: torch.Tensor) -> torch.Tensor:
        """One QUERY row, the ``row``-th summary embedding at ``[B, 1]`` slots."""
        return cast(
            torch.Tensor,
            self.dropout(self.summary_embedding[row] + self._positions(slots)),
        )

    def embed_summaries(
        self, summaries: torch.Tensor, slots: torch.Tensor
    ) -> torch.Tensor:
        """SUMMARY rows: earlier QUERY outputs ``[B, n, d]`` re-entered at ``[B, n]``
        slots, with positions and dropout and without the input projection."""
        return cast(torch.Tensor, self.dropout(summaries + self._positions(slots)))

    # ------------------------------------------------------------------
    # Dense (training) path
    # ------------------------------------------------------------------

    def allowed_mask(self, key_valid: torch.Tensor) -> torch.Tensor:
        """The one mask rule, ``[B, Q, Q]``: ``k <= q`` and ``key_valid[b, k]``."""
        if key_valid.ndim != 2:
            raise ContractError("Key validity is a [B, Q] boolean tensor.")
        length = key_valid.shape[1]
        causal = torch.tril(
            torch.ones((length, length), dtype=torch.bool, device=key_valid.device)
        )
        return causal.unsqueeze(0) & key_valid.bool().unsqueeze(1)

    def block_forward(self, x: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        """Run every block over one embedded block under the one mask rule."""
        batch, length, _ = x.shape
        if key_valid.shape != (batch, length):
            raise ContractError("Block tensors disagree on their slot count.")
        allowed = self.allowed_mask(key_valid).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, allowed)
        return cast(torch.Tensor, self.norm(x))

    def _dense(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        *,
        segment_length: int,
        summarize_last: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Segment by segment over ``[B, T]`` right-padded records.

        Returns the ``[B, T, d]`` RECORD outputs (zeroed where invalid) and
        the ``[B, n*S, d]`` accumulated summaries, where ``n`` counts the
        segments that were summarized: every segment followed by another, and
        the last one too when ``summarize_last`` (the reconstruction needs the
        summary of a completed segment whose successor is still open).
        """
        batch, length, _ = tokens.shape
        count = int(segment_length)
        summary_count = self.summary_tokens
        device = tokens.device
        summaries = tokens.new_zeros((batch, 0, self.d_model))
        outputs: list[torch.Tensor] = []
        starts = range(0, length, count)
        for start in starts:
            stop = min(start + count, length)
            records = tokens[:, start:stop]
            held = summaries.shape[1]
            last = stop >= length
            queries = 0 if last and not summarize_last else summary_count
            total = held + (stop - start) + queries
            slots = torch.arange(total, device=device).unsqueeze(0).expand(batch, -1)
            rows = [
                self.embed_summaries(summaries, slots[:, :held]),
                self.embed_records(records, slots[:, held : held + stop - start]),
            ]
            if queries:
                rows.append(self.embed_queries(slots[:, held + stop - start :]))
            x = torch.cat(rows, dim=1)
            key_valid = torch.cat(
                (
                    torch.ones((batch, held), dtype=torch.bool, device=device),
                    valid[:, start:stop],
                    torch.ones((batch, queries), dtype=torch.bool, device=device),
                ),
                dim=1,
            )
            hidden = self.block_forward(x, key_valid)
            outputs.append(hidden[:, held : held + stop - start])
            if queries:
                summaries = torch.cat((summaries, hidden[:, held + stop - start :]), 1)
        out = torch.cat(outputs, dim=1)
        return out * valid.unsqueeze(-1).to(out.dtype), summaries

    def training_forward(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        *,
        segment_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one complete outer task segment by segment.

        Args:
            tokens: ``[B, T, token]`` right-padded policy-input records.
            valid: ``[B, T]`` boolean validity, a contiguous prefix per row.
            segment_length: the dense segment length; ``None`` draws it from
                the jitter range in training and uses ``L`` otherwise.

        Returns the ``[B, T, d]`` RECORD outputs, zeroed where invalid, and the
        accumulated summaries of every segment that a later record read.
        Gradients flow through every summary; nothing is detached.
        """
        if valid.dtype != torch.bool:
            valid = valid != 0
        require_right_padded(valid)
        batch, length, _ = tokens.shape
        if valid.shape != (batch, length) or length < 1:
            raise ContractError("Memo training needs [B, T] validity for T >= 1.")
        count = (
            self.dense_segment_length() if segment_length is None else segment_length
        )
        if count <= self.summary_tokens:
            raise ContractError("A Memo segment holds more records than summaries.")
        return self._dense(tokens, valid, segment_length=count, summarize_last=False)

    # ------------------------------------------------------------------
    # Rollout state
    # ------------------------------------------------------------------

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> MemoHiddenState:
        """Allocate poisoned caches for the longest task and empty counters."""
        dtype = _CACHE_DTYPES[self.spec.cache_dtype]

        def slab() -> torch.Tensor:
            data = torch.zeros(
                (
                    batch_size,
                    self.capacity,
                    self.backbone_heads,
                    self.backbone_head_dim,
                ),
                dtype=dtype,
                device=device,
            )
            data[:] = torch.nan
            return data

        return MemoHiddenState(
            [DATLayerCache("ordinary", slab(), slab()) for _ in range(self.n_layers)],
            torch.zeros((batch_size,), dtype=torch.int32, device=device),
            torch.zeros((batch_size,), dtype=torch.int64, device=device),
            spec_sha256=self.spec.sha256,
            capacity=self.capacity,
            summary_tokens=self.summary_tokens,
        )

    # ------------------------------------------------------------------
    # Cached (rollout) path
    # ------------------------------------------------------------------

    def _cached_layers(self, x: torch.Tensor, hidden: MemoHiddenState) -> torch.Tensor:
        """One query per row at slot ``lengths[b]`` over keys ``<= lengths[b]``."""
        lengths = hidden.lengths
        window = int(lengths.max().item()) + 1
        if window > self.capacity:
            raise ContractError(
                "Memo cache slot outside the allocated capacity: the task is "
                "longer than the carrier was built for."
            )
        positions = torch.arange(window, device=hidden.device)
        allowed = (positions.unsqueeze(0) <= lengths.long().unsqueeze(1)).unsqueeze(1)
        for index, layer in enumerate(self.layers):
            x = layer(x, allowed.unsqueeze(1), cache=hidden[index], lengths=lengths)
        return x

    def boundary(self, hidden: MemoHiddenState) -> None:
        """Summarize a full segment, drop its records, append the summary."""
        held = hidden.held
        if bool((hidden.lengths != (held + self.segment_length).to(torch.int32)).any()):
            raise ContractError("A boundary needs a full open segment on every row.")
        outputs: list[torch.Tensor] = []
        for row in range(self.summary_tokens):
            slots = hidden.lengths.long().unsqueeze(1)
            query = self.embed_query(row, slots)
            outputs.append(self.norm(self._cached_layers(query, hidden)))
            hidden.lengths += 1
        written = torch.cat(outputs, dim=1)
        hidden.truncate(held)
        hidden.segment += 1
        if hidden.summary_cleared:
            return
        self._enter_summaries(hidden, written)

    def _enter_summaries(
        self, hidden: MemoHiddenState, summaries: torch.Tensor
    ) -> None:
        """Append ``[B, n, d]`` raw summaries to every row's cache, one slot each."""
        for row in range(summaries.shape[1]):
            slots = hidden.lengths.long().unsqueeze(1)
            self._cached_layers(
                self.embed_summaries(summaries[:, row : row + 1], slots), hidden
            )
            hidden.lengths += 1

    def _on_rows(self, hidden: MemoHiddenState, rows: torch.Tensor, step: Any) -> None:
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
        self, records: torch.Tensor, hidden: MemoHiddenState
    ) -> torch.Tensor:
        """Encode one record per row, crossing a boundary first where one is due.

        Args:
            records: ``[B, 1, token]`` the current policy-input record per row.
            hidden: the rollout state, advanced in place.
        """
        if self.training:
            raise ContractError("Cached Memo execution is evaluation-only.")
        if records.ndim != 3 or records.shape[1] != 1:
            raise ContractError("Cached Memo accepts one record per row.")
        if records.shape[0] != hidden.batch_size:
            raise ContractError("Cached Memo batch disagrees with the state.")
        if hidden.spec_sha256 != self.spec.sha256:
            raise ContractError("Memo state was built for a different carrier.")
        due = hidden.lengths == (hidden.held + self.segment_length).to(torch.int32)
        self._on_rows(hidden, torch.where(due)[0], self.boundary)
        x = self.embed_records(records, hidden.lengths.long().unsqueeze(1))
        out = self._cached_layers(x, hidden)
        hidden.lengths += 1
        return cast(torch.Tensor, self.norm(out))

    def rebuild(self, tokens: torch.Tensor, lengths: Sequence[int]) -> MemoHiddenState:
        """Rebuild rollout state from complete processed prefixes.

        Rows are grouped by prefix length ``P``. The convention is the cached
        rollout's own, which crosses a boundary lazily: ``q = (P - 1) // L``
        segments are completed, summarized through the dense path at the fixed
        ``L`` and their summaries entered into the caches, then the remaining
        ``r = P - qL`` records, ``1 <= r <= L``, are encoded through the cached
        path. A prefix ending exactly at a boundary therefore holds its full
        last segment with the summary pending, so the result equals an
        uninterrupted rollout of the same prefix under the current weights,
        state for state. No padded record is ever processed.
        """
        if self.training:
            raise ContractError("Memo rebuild is evaluation-only.")
        batch, columns, _ = tokens.shape
        if len(lengths) != batch:
            raise ContractError("Memo rebuild needs one true length per row.")
        true_lengths = [int(value) for value in lengths]
        if any(value < 0 for value in true_lengths):
            raise ContractError("Memo rebuild prefix lengths must be nonnegative.")
        if max(true_lengths, default=0) > columns:
            raise ContractError("Memo rebuild prefix exceeds the provided packets.")
        hidden = self.init_hidden_state(batch, tokens.device)
        count = self.segment_length
        groups: dict[int, list[int]] = {}
        for row, length in enumerate(true_lengths):
            if length > 0:
                groups.setdefault(length, []).append(row)
        with torch.no_grad():
            for length, rows in groups.items():
                index = torch.as_tensor(rows, device=tokens.device)
                full = (length - 1) // count
                sub = self.init_hidden_state(len(rows), tokens.device)
                if full > 0:
                    processed = tokens[index, : full * count]
                    valid = torch.ones(
                        processed.shape[:2], dtype=torch.bool, device=tokens.device
                    )
                    _, summaries = self._dense(
                        processed, valid, segment_length=count, summarize_last=True
                    )
                    for slot in range(summaries.shape[1]):
                        slots = sub.lengths.long().unsqueeze(1)
                        self._cached_layers(
                            self.embed_summaries(summaries[:, slot : slot + 1], slots),
                            sub,
                        )
                        sub.lengths += 1
                    sub.segment.fill_(full)
                for step in range(full * count, length):
                    self.cached_forward(tokens[index, step : step + 1], sub)
                hidden.assign(index, sub)
        return hidden


__all__ = [
    "MEMO_HIDDEN_STATE_SCHEMA",
    "MemoHiddenState",
    "MemoTransformer",
]
