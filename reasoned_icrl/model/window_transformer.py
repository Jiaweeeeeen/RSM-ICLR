"""The sliding-window carrier's model: a per-layer band of ``W`` slots (SPEC §5).

Every donor block runs under the band ``A(b, q, k) = (0 <= q - k < W) and
valid[b, k]``: an unselected block as a :class:`MaskedOrdinaryBlock` over the
donor's own parameters, a selected block (when a :class:`DATSpec` is given) as
a :class:`DATBlock` whose content *and* relational branches read the same band.
Every window slot is a RECORD, so the revised timestep-record relational route
of the summary carrier reduces here to the content band itself: there are no
READ or WRITE rows to exclude. Symbols are computed from the sources' actual
trajectory times, so after an eviction a retained source keeps its true
relative position rather than its rank.

Training runs the whole task densely under the band. Rollout keeps a
``W``-slot rolling cache per layer (a :class:`DATHiddenState`), evicted on
AMAGO's post-step schedule, so a query sees itself and the ``W - 1`` most
recent retained sources; the reconstruction after a learner update is one
dense banded forward from the true prefix, from which every layer's sources at
the retained positions are copied. Rebuilding only the last ``W`` raw records
would be wrong: a layer's keys and values are contextualised by the layers
below, which is also why the raw receptive field of a ``d``-layer band is
``1 + d (W - 1)`` records, not ``W``.

The ordinary window (``amago-window-v1``) and the dual-attention window
(``amago-dat-window-v1``) are the same class; their rollout states carry
different schemas and identities so that neither restores onto the other.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from amago.nets.transformer import FixedPosEmb
from torch import nn

from reasoned_icrl.experiments.contracts import ContractError, DATSpec, WindowSpec
from reasoned_icrl.model.dat_transformer import (
    _CACHE_DTYPES,
    EMPTY_TIME,
    DATBlock,
    DATHiddenState,
    DualAttention,
    allocate_layer_caches,
    require_right_padded,
)
from reasoned_icrl.model.summary_transformer import MaskedOrdinaryBlock

WINDOW_HIDDEN_STATE_SCHEMA = "amago-window-hidden-state.v1"
"""The ordinary window's rollout state; its identity is the window spec."""

DAT_WINDOW_HIDDEN_STATE_SCHEMA = "amago-dat-window-hidden-state.v1"
"""The dual-attention window's rollout state; it carries the window identity
and the attention identity, and restores only where both agree."""


def receptive_field(layers: int, window_length: int) -> int:
    """Raw records a ``layers``-deep band of ``window_length`` slots can reach.

    A query reads itself and ``W - 1`` earlier retained sources at every
    layer, and each source is contextualised over its own band by the layer
    below, so influence composes to ``1 + layers * (W - 1)`` records. Records
    older than that cannot affect the top-layer output at all.
    """
    if layers < 1 or window_length < 1:
        raise ContractError("The receptive field needs positive depth and width.")
    return 1 + layers * (window_length - 1)


class WindowTransformer(nn.Module):
    """The donor backbone under a per-layer band of ``W`` slots.

    Args:
        donor: a fully built AMAGO ``Transformer``; its input projection,
            fixed position table, dropout, final normalization and every
            block are adopted as the same module objects.
        spec: the window identity (``W`` and the cache dtype).
        dat: when given, the selected blocks run dual attention over the band,
            with the symbol table clipped at ``W`` (the window's own capacity),
            exactly as a summary cell clips at its segment capacity. ``None``
            is the ordinary window, which adds no parameter.
    """

    def __init__(
        self,
        donor: Any,
        spec: WindowSpec,
        dat: DATSpec | None = None,
        *,
        dropout_qkv: float = 0.0,
        head_scaling: bool = True,
        sigma_reparam: bool = True,
    ) -> None:
        super().__init__()
        if getattr(donor, "use_rope", False):
            raise ContractError("The window carrier uses fixed positions, not RoPE.")
        if not isinstance(donor.position_embedding, FixedPosEmb):
            raise ContractError("The window carrier needs the donor's FixedPosEmb.")
        layers = list(donor.layers)
        d_model = int(donor.d_model)
        if dat is not None:
            dat.validate_layers(len(layers))
            if dat.d_model != d_model:
                raise ContractError("DAT attention width disagrees with the donor.")
            if dat.max_relative_distance != spec.capacity:
                raise ContractError(
                    "A window cell's DAT clipping distance must equal its window "
                    "length W."
                )
            if dat.position_method != "fixed":
                raise ContractError("Window DAT blocks use fixed trajectory positions.")
            if dat.mode == "dual_content":
                raise ContractError(
                    "The revised band runs dual relational attention; the "
                    "dual-content control is not a window cell."
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

    # ------------------------------------------------------------------
    # Geometry and identity
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Retained slots per layer, the query's own slot included: ``W``."""
        return self.spec.capacity

    @property
    def hidden_schema(self) -> str:
        return (
            WINDOW_HIDDEN_STATE_SCHEMA
            if self.dat is None
            else DAT_WINDOW_HIDDEN_STATE_SCHEMA
        )

    @property
    def raw_receptive_field(self) -> int:
        """``1 + n_layers (W - 1)``: the bound the tests probe, not ``W``."""
        return receptive_field(self.n_layers, self.capacity)

    def layer_variant(self, index: int) -> str:
        if index in self.selected:
            assert self.dat is not None
            return self.dat.mode
        return "ordinary"

    def _check_state(self, hidden: DATHiddenState) -> None:
        """Refuse a state built for another schema, window or attention."""
        if hidden.schema != self.hidden_schema:
            raise ContractError("Window state was built for a different carrier.")
        if self.dat is None:
            expected = {"attention_sha256": self.spec.sha256, "window_sha256": None}
        else:
            expected = {
                "attention_sha256": self.dat.sha256,
                "window_sha256": self.spec.sha256,
            }
        actual = {
            "attention_sha256": hidden.attention_sha256,
            "window_sha256": hidden.window_sha256,
        }
        if actual != expected:
            raise ContractError("Window state was built for a different carrier.")

    # ------------------------------------------------------------------
    # Dense (training) path
    # ------------------------------------------------------------------

    def preprocess(self, tokens: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Token embedding plus the donor's fixed positions at ``times``."""
        return cast(
            torch.Tensor,
            self.dropout(self.inp(tokens) + self.position_embedding(times.long())),
        )

    def banded_mask(self, valid: torch.Tensor) -> torch.Tensor:
        """``[B, T, T]``: ``k <= q``, ``q - k < W`` and a valid key.

        One band serves both branches of a dual-attention block: every window
        slot is a RECORD, so the relational branch has no READ or WRITE source
        to exclude and reads exactly the content sources.
        """
        length = valid.shape[1]
        position = torch.arange(length, device=valid.device)
        distance = position.unsqueeze(1) - position.unsqueeze(0)
        band = (distance >= 0) & (distance < self.spec.segment_length)
        return band.unsqueeze(0) & valid.unsqueeze(1)

    def _dense_block(
        self,
        index: int,
        x: torch.Tensor,
        times: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        """One block over a whole sequence: ``[B, T, d]`` under ``[B, T, T]``."""
        block = self.layers[index]
        if index in self.selected:
            return cast(torch.Tensor, block(x, times, times, allowed))
        return cast(torch.Tensor, block(x, allowed.unsqueeze(1)))

    def training_forward(
        self, tokens: torch.Tensor, times: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Dense banded execution over whole right-padded tasks."""
        require_right_padded(valid)
        x = self.preprocess(tokens, times)
        allowed = self.banded_mask(valid)
        for index in range(self.n_layers):
            x = self._dense_block(index, x, times.long(), allowed)
        return cast(torch.Tensor, self.norm(x))

    # ------------------------------------------------------------------
    # Rollout state
    # ------------------------------------------------------------------

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> DATHiddenState:
        """Allocate ``W`` NaN-poisoned slots per layer plus the shared times.

        The slabs follow each layer's variant: keys and values for an ordinary
        block; content keys and values, selection keys and relation keys for a
        selected block. The state's identities are the window spec and, for
        the dual-attention window, the attention spec as well.
        """
        capacity = self.capacity
        caches = allocate_layer_caches(
            batch_size=batch_size,
            capacity=capacity,
            n_layers=self.n_layers,
            selected=self.selected,
            dat=self.dat,
            backbone_heads=self.backbone_heads,
            backbone_head_dim=self.backbone_head_dim,
            dtype=_CACHE_DTYPES[self.spec.cache_dtype],
            device=device,
        )
        return DATHiddenState(
            caches,
            torch.full(
                (batch_size, capacity), EMPTY_TIME, dtype=torch.int64, device=device
            ),
            torch.zeros((batch_size,), dtype=torch.int32, device=device),
            attention_sha256=self.spec.sha256 if self.dat is None else self.dat.sha256,
            schema=self.hidden_schema,
            window_sha256=None if self.dat is None else self.spec.sha256,
        )

    # ------------------------------------------------------------------
    # Cached (rollout) path
    # ------------------------------------------------------------------

    def cached_forward(
        self, record: torch.Tensor, time: torch.Tensor, hidden: DATHiddenState
    ) -> torch.Tensor:
        """Encode one record per row over its retained sources.

        The caller advances the state afterwards (``hidden.update()``), which
        evicts the oldest source of a full row exactly as AMAGO's cache does.
        Retained sources are always within the band: a full row holds exactly
        the ``W - 1`` most recent records, and their stored times (not their
        ranks) supply the relational symbols.
        """
        if self.training:
            raise ContractError("Cached window execution is evaluation-only.")
        if record.ndim != 3 or record.shape[1] != 1:
            raise ContractError("Cached window accepts one record per row.")
        if record.shape[0] != hidden.batch_size:
            raise ContractError("Cached window batch disagrees with the state.")
        self._check_state(hidden)
        lengths = hidden.lengths
        rows = torch.arange(hidden.batch_size, device=hidden.device)
        current = time[:, 0].long()
        stale = hidden.times[rows, lengths.long()]
        if bool(((stale != EMPTY_TIME) & (stale != current)).any()):
            raise ContractError("Window cache slot already holds a different source.")
        hidden.times[rows, lengths.long()] = current
        window = int(lengths.max().item()) + 1
        allowed = (
            torch.arange(window, device=hidden.device).unsqueeze(0)
            <= lengths.long().unsqueeze(1)
        ).view(hidden.batch_size, 1, window)
        key_times = hidden.times[:, :window]
        query_times = current.unsqueeze(1)
        x = self.preprocess(record, query_times)
        for index, block in enumerate(self.layers):
            if index in self.selected:
                x = block(
                    x,
                    query_times,
                    key_times,
                    allowed,
                    cache=hidden[index],
                    lengths=lengths,
                )
            else:
                x = block(x, allowed.unsqueeze(1), cache=hidden[index], lengths=lengths)
        return cast(torch.Tensor, self.norm(x))

    def rebuild(
        self, tokens: torch.Tensor, times: torch.Tensor, lengths: Sequence[int]
    ) -> DATHiddenState:
        """Rebuild rollout state from complete processed prefixes, in one pass.

        One dense banded forward produces, at every position, exactly the layer
        inputs the cached path saw when that position was current (parity), so
        each layer's retained sources at the last positions are copied into the
        slabs in rank order: keys and values for an ordinary block, and every
        cached projection of a dual-attention block (content keys and values,
        selection keys, relation keys). A replay of only the last ``W`` records
        would not be exact: the retained sources at a layer depend on older
        records through the layers below. AMAGO's post-step convention keeps
        ``min(L, W - 1)`` sources for a prefix of ``L`` records, at their true
        trajectory times.
        """
        if self.training:
            raise ContractError("Window rebuild is evaluation-only.")
        batch, columns, _ = tokens.shape
        if len(lengths) != batch:
            raise ContractError("Window rebuild needs one true length per row.")
        true = torch.as_tensor([int(v) for v in lengths], device=tokens.device)
        if bool((true < 0).any()):
            raise ContractError("Window rebuild prefix lengths must be nonnegative.")
        if int(true.max().item() if batch else 0) > columns:
            raise ContractError("Window rebuild prefix exceeds the provided packets.")
        hidden = self.init_hidden_state(batch, tokens.device)
        if columns == 0 or int(true.max().item()) == 0:
            return hidden
        capacity = self.capacity
        with torch.no_grad():
            valid = torch.arange(columns, device=tokens.device).unsqueeze(0) < (
                true.unsqueeze(1)
            )
            long_times = times.long()
            x = self.preprocess(tokens, long_times)
            allowed = self.banded_mask(valid)
            retained = torch.clamp(true, max=capacity - 1)
            slot = torch.arange(capacity, device=tokens.device).unsqueeze(0)
            keep = slot < retained.unsqueeze(1)
            source = (true - retained).unsqueeze(1) + slot
            index = source.clamp(min=0, max=columns - 1)
            keep_slots = keep.view(batch, capacity, 1, 1)
            for layer, block in enumerate(self.layers):
                normalized = cast(Any, block).norm1(x)
                if isinstance(block, DATBlock):
                    sources = block.attention.cache_sources(normalized)
                else:
                    assert isinstance(block, MaskedOrdinaryBlock)
                    _, keys, values = block.project(normalized)
                    sources = {"content_keys": keys, "content_values": values}
                cache = hidden[layer]
                for name, slab in cache.tensors().items():
                    if name not in sources:
                        raise ContractError(
                            f"Window rebuild has no source for {name!r}."
                        )
                    tensor = sources[name]
                    gather = index.view(batch, capacity, 1, 1).expand(
                        -1, -1, tensor.shape[2], tensor.shape[3]
                    )
                    picked = tensor.gather(1, gather).to(slab.dtype)
                    slab.copy_(torch.where(keep_slots, picked, torch.nan))
                x = self._dense_block(layer, x, long_times, allowed)
            hidden.times.copy_(
                torch.where(keep, long_times.gather(1, index), EMPTY_TIME)
            )
            hidden.lengths.copy_(retained.to(torch.int32))
        return hidden


__all__ = [
    "DAT_WINDOW_HIDDEN_STATE_SCHEMA",
    "WINDOW_HIDDEN_STATE_SCHEMA",
    "WindowTransformer",
    "receptive_field",
]
