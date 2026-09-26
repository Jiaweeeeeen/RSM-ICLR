"""Dual attention inside AMAGO's Transformer, with explicit per-layer caches.

The attention operator follows Dual Attention Transformers (Altabaa et al.,
ICML 2025) and its reference implementation at commit
``dce218cbf5ec9aa7f90687c1323050a1fba17966`` (MIT licensed). The surrounding
block execution order, stabilization and cache lifecycle follow AMAGO v3.4.0 at
commit ``0974781a9096ff43df1b708312256f96fc2ab127`` (MIT licensed). Neither
project is a runtime dependency here; this module reimplements the operator
against AMAGO's parameterization, and the tests check it against the pinned
reference numerically.

Two branches share one normalized block input:

* the **content** branch is ordinary self-attention over source features;
* the **relational** branch selects sources with its own query/key projections,
  then retrieves *relations between* the receiver and each source, tagged with a
  symbol identifying the source's position relative to the receiver.

Each branch projects to its own width and the two are concatenated back to
``d_model``. That follows the reference implementation's executable behaviour;
the paper's Appendix B.1 prints full-model output dimensions instead. The
choice is recorded in :class:`~reasoned_icrl.experiments.contracts.DATSpec` and
verified against the reference in ``tests/test_dat_transformer.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from amago.nets.transformer import (
    SigmaReparam,
)
from einops import rearrange
from torch import nn

from reasoned_icrl.experiments.contracts import ContractError, DATSpec

DAT_HIDDEN_STATE_SCHEMA = "amago-dat-hidden-state.v1"

_CACHE_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

#: Sentinel for a cache slot that holds no source. Times are int64, so NaN is
#: unavailable; -1 is never a legal AMAGO trajectory position.
EMPTY_TIME = -1


def _linear(d_in: int, d_out: int, *, bias: bool, sigma_reparam: bool) -> nn.Module:
    """Build AMAGO's stabilized linear layer, or a plain one for parity tests."""
    if sigma_reparam:
        return cast(nn.Module, SigmaReparam(d_in, d_out, bias=bias))
    return nn.Linear(d_in, d_out, bias=bias)


def _masked_softmax(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Softmax over allowed keys only, finite and zero where nothing is allowed.

    Args:
        scores: ``[B, H, Q, K]`` raw attention scores.
        allowed: ``[B, H, Q, K]`` or broadcastable boolean mask.

    A query row with no permitted source occurs at padded rows and at the first
    decision of a freshly reset environment. Softmax over an all -inf row is
    NaN, so those rows are neutralized before the softmax and zeroed after it,
    which keeps both the value and its gradient finite.
    """
    row_ok = allowed.any(-1, keepdim=True)
    scores = scores.masked_fill(~allowed, float("-inf"))
    scores = torch.where(row_ok, scores, torch.zeros_like(scores))
    weights = torch.softmax(scores, dim=-1)
    return torch.where(row_ok, weights, torch.zeros_like(weights))


def _dense_content_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    allowed: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Content attention that materializes the score matrix, for CPU and parity.

    Args:
        queries: ``[B, Q, H, D]`` receiver projections.
        keys: ``[B, K, H, D]`` source projections.
        values: ``[B, K, H, D]`` source projections.
        allowed: ``[B, 1, Q, K]`` boolean mask of permitted sources.
        scale: the ``1/sqrt(head_dim)`` factor for this branch.
    """
    scores = torch.einsum("bqhd,bkhd->bhqk", queries, keys) * scale
    weights = _masked_softmax(scores, allowed)
    return torch.einsum("bhqk,bkhd->bqhd", weights, values)


def _sdpa_layout(x: torch.Tensor) -> torch.Tensor:
    """``[B, T, H, D]`` as ``[B, H, T, D]``, laid out so the fused kernel runs.

    A dimension of extent one addresses nothing, so PyTorch lets it carry any
    stride, and unbinding the fused QKV projection leaves such dimensions
    holding the donor's. The CUDA kernel validates strides before it dispatches
    and refuses the result with "no kernel found to launch": a single cached
    decision arrives with the query stride of the whole projection, and a
    one-head backbone arrives with a head stride of one beside an equal
    innermost stride. ``Tensor.contiguous`` cannot repair either, because by
    PyTorch's definition there is nothing wrong.

    So give every extent-one dimension the stride a standard layout would give
    it. Addressing is unchanged, no element moves and no memory is allocated,
    and dimensions that actually carry data keep the strides they had, which
    leaves the dense path's view exactly as the kernel already accepts it.
    """
    out = x.transpose(1, 2)
    sizes = list(out.shape)
    if 1 not in sizes:
        return out
    strides = list(out.stride())
    step = 1
    for index in range(len(sizes) - 1, -1, -1):
        if sizes[index] == 1:
            strides[index] = step
        step = strides[index] * sizes[index]
    return out.as_strided(sizes, strides)


def _fused_content_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    allowed: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """The same content attention through PyTorch's fused attention kernels.

    Identical mathematics to :func:`_dense_content_attention`, reduced in a
    different order, so the two agree to float32 rounding rather than exactly.
    What it buys is the ``[B, H, Q, K]`` score matrix never being materialized,
    which is the dominant activation at the pilot's 500-decision context.

    The mask is an arbitrary boolean rather than plain causality, because the
    cached path admits a per-environment retained length, so this dispatches
    through ``scaled_dot_product_attention`` rather than a causal-only kernel.
    Fully masked rows are neutralized before the softmax exactly as
    :func:`_masked_softmax` neutralizes them, so no NaN enters the graph, and
    they are zeroed afterwards.
    """
    row_ok = allowed.any(-1, keepdim=True)
    out = torch.nn.functional.scaled_dot_product_attention(
        _sdpa_layout(queries),
        _sdpa_layout(keys),
        _sdpa_layout(values),
        attn_mask=allowed | ~row_ok,
        scale=scale,
    )
    return torch.where(row_ok, out, torch.zeros_like(out)).transpose(1, 2)


def content_backend(device: torch.device | str) -> str:
    """The content-branch kernel this device runs, for the recorded backend map.

    A kernel choice, not a scientific one: it is deliberately outside
    :class:`DATSpec` so it does not enter the attention identity hash and a
    checkpoint stays comparable across the machines it was measured on.
    """
    return "fused" if torch.device(device).type == "cuda" else "vanilla"


@dataclass(slots=True)
class DATLayerCache:
    """Source tensors retained for one layer.

    Which fields exist depends on the layer's variant. Ordinary AMAGO layers use
    ``content_keys``/``content_values`` alone, at the backbone's head width, and
    hand them straight to AMAGO's own cached attention. Dual-attention layers add
    the branch tensors their mode needs and allocate nothing for the branches it
    does not use.
    """

    variant: str
    content_keys: torch.Tensor
    content_values: torch.Tensor
    selection_keys: torch.Tensor | None = None
    selection_values: torch.Tensor | None = None
    relation_keys: torch.Tensor | None = None

    def tensors(self) -> dict[str, torch.Tensor]:
        """Every allocated tensor, keyed by field name."""
        named = {
            "content_keys": self.content_keys,
            "content_values": self.content_values,
            "selection_keys": self.selection_keys,
            "selection_values": self.selection_values,
            "relation_keys": self.relation_keys,
        }
        return {name: value for name, value in named.items() if value is not None}

    def clear_rows(self, rows: torch.Tensor) -> None:
        """Poison every retained slot of the given batch rows."""
        for tensor in self.tensors().values():
            tensor[rows] = torch.nan

    def roll(self, rows: torch.Tensor) -> None:
        """Evict the oldest retained slot of the given batch rows."""
        for tensor in self.tensors().values():
            tensor[rows, :-1] = tensor[rows, 1:].clone()
            tensor[rows, -1] = torch.nan


class DATHiddenState:
    """Rollout state for a mixed ordinary/dual-attention backbone.

    Holds one :class:`DATLayerCache` per layer plus the source times and retained
    lengths those layers share. Times are shared because every layer stores the
    same tokens in the same order; only their contents differ.

    Two index concepts are kept apart on purpose. ``lengths`` is a retained-token
    *rank*, and drives capacity and eviction. ``times`` holds the real trajectory
    position of each retained source, and drives symbol identity. After an
    eviction those disagree, and collapsing them would silently relabel
    non-consecutive sources as consecutive.
    """

    def __init__(
        self,
        layers: Sequence[DATLayerCache],
        times: torch.Tensor,
        lengths: torch.Tensor,
        *,
        attention_sha256: str,
        schema: str = DAT_HIDDEN_STATE_SCHEMA,
        window_sha256: str | None = None,
    ) -> None:
        """``attention_sha256`` is the identity the schema records (the DAT spec
        for the DAT carrier and the dual-attention window, the window spec for
        the ordinary window); ``window_sha256`` is the dual-attention window's
        second identity, ``None`` on every other schema."""
        if not layers:
            raise ContractError("A DAT hidden state needs at least one layer cache.")
        if lengths.dtype != torch.int32:
            raise ContractError("DAT retained lengths must be int32.")
        if times.dtype != torch.int64:
            raise ContractError("DAT source times must be int64.")
        batch, capacity = times.shape
        if lengths.shape != (batch,):
            raise ContractError("DAT lengths must have one entry per environment.")
        for cache in layers:
            for name, tensor in cache.tensors().items():
                if tensor.shape[:2] != (batch, capacity):
                    raise ContractError(
                        f"DAT cache tensor {name!r} disagrees with batch/capacity."
                    )
        self.layers = tuple(layers)
        self.times = times
        self.lengths = lengths
        self.capacity = capacity
        self.batch_size = batch
        self.attention_sha256 = attention_sha256
        self.window_sha256 = window_sha256
        self.schema = schema
        self.device = times.device

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> DATLayerCache:
        if not 0 <= layer_idx < self.n_layers:
            raise ContractError("DAT layer index outside the cached backbone.")
        return self.layers[layer_idx]

    def reset(self, idxs: Any) -> None:
        """Clear whole rows at an outer task boundary; other rows are untouched."""
        rows = torch.as_tensor(idxs, device=self.device)
        if rows.dtype == torch.bool:
            rows = torch.where(rows)[0]
        rows = rows.long()
        if rows.numel() == 0:
            return
        self.lengths[rows] = 0
        self.times[rows] = EMPTY_TIME
        for cache in self.layers:
            cache.clear_rows(rows)

    def update(self) -> None:
        """Advance one decision, evicting the oldest source when full.

        Mirrors AMAGO's post-step convention: the length is incremented first and
        a full row is then rolled back by one, so there is always room for the
        next token. Every layer and the shared time buffer evict the same slot.
        """
        self.lengths += 1
        rows = torch.where(self.lengths == self.capacity)[0]
        if rows.numel():
            self.times[rows, :-1] = self.times[rows, 1:].clone()
            self.times[rows, -1] = EMPTY_TIME
            for cache in self.layers:
                cache.roll(rows)
            self.lengths[rows] -= 1


class DualAttention(nn.Module):
    """One dual-attention computation, replacing a block's ordinary attention.

    Args:
        spec: the resolved attention identity. Determines the mode, branch
            widths, relation channels and symbol clipping distance.
        dropout_qkv: applied to the fused content projection, as AMAGO does.
        head_scaling: whether the per-head scalers are trainable.
        sigma_reparam: use AMAGO's stabilized linear layers. Disabled by the
            parity tests, which compare against the reference's plain linears.
    """

    def __init__(
        self,
        spec: DATSpec,
        *,
        dropout_qkv: float = 0.0,
        head_scaling: bool = True,
        sigma_reparam: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec
        d_model = spec.d_model
        content_inner = spec.content_heads * spec.content_head_dim
        second_inner = spec.relational_heads * spec.second_head_dim

        # Content branch: AMAGO's fused QKV, packed (three, head_dim, heads).
        self.content_qkv = _linear(
            d_model, 3 * content_inner, bias=False, sigma_reparam=sigma_reparam
        )
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.content_out = _linear(
            content_inner, spec.content_width, bias=True, sigma_reparam=sigma_reparam
        )
        self.content_head_scaler = nn.Parameter(
            torch.ones(1, 1, spec.content_heads, 1), requires_grad=head_scaling
        )
        self.second_head_scaler = nn.Parameter(
            torch.ones(1, 1, spec.relational_heads, 1), requires_grad=head_scaling
        )
        self.second_out = _linear(
            second_inner, spec.relational_width, bias=True, sigma_reparam=sigma_reparam
        )

        if spec.mode == "dual_content":
            # The capacity control keeps the 6+2 branch organization and the
            # separate 192/64 output projections, but both branches retrieve
            # source features. Only the internal Q/K/V widths are widened.
            self.second_qkv: nn.Module | None = _linear(
                d_model, 3 * second_inner, bias=False, sigma_reparam=sigma_reparam
            )
            self.selection_q: nn.Module | None = None
            self.selection_k: nn.Module | None = None
        else:
            self.second_qkv = None
            self.selection_q = _linear(
                d_model,
                spec.relational_heads * spec.head_dim,
                bias=False,
                sigma_reparam=sigma_reparam,
            )
            self.selection_k = _linear(
                d_model,
                spec.relational_heads * spec.head_dim,
                bias=False,
                sigma_reparam=sigma_reparam,
            )

        relation_inner = spec.relation_channels * spec.relation_projection_dim
        if spec.uses_relations and spec.mode != "dual_content":
            self.relation_q: nn.Module | None = _linear(
                d_model, relation_inner, bias=False, sigma_reparam=sigma_reparam
            )
            self.relation_k: nn.Module | None = (
                self.relation_q
                if spec.symmetric_relations
                else _linear(
                    d_model, relation_inner, bias=False, sigma_reparam=sigma_reparam
                )
            )
            # W_r maps aggregated relations to head width, per head.
            self.relation_out = nn.Parameter(
                torch.empty(
                    spec.relational_heads, spec.head_dim, spec.relation_channels
                )
            )
            nn.init.kaiming_uniform_(self.relation_out, a=math.sqrt(5))
        else:
            self.relation_q = None
            self.relation_k = None
            self.register_parameter("relation_out", None)

        if spec.uses_symbols:
            # Causal offsets only: sources never follow their receiver, so the
            # table needs [-D, 0] rather than the reference's [-D, D].
            self.symbol_table: nn.Parameter | None = nn.Parameter(
                torch.empty(spec.max_relative_distance + 1, spec.symbol_dim)
            )
            nn.init.xavier_uniform_(self.symbol_table)
            self.symbol_projection: nn.Module | None = _linear(
                spec.symbol_dim,
                spec.relational_heads * spec.head_dim,
                bias=False,
                sigma_reparam=sigma_reparam,
            )
        else:
            self.register_parameter("symbol_table", None)
            self.symbol_projection = None

        self.content_scale = 1.0 / math.sqrt(spec.content_head_dim)
        self.second_scale = 1.0 / math.sqrt(spec.second_head_dim)
        self.relation_scale = 1.0 / math.sqrt(spec.relation_projection_dim)
        # DATSpec admits only the identity relation activation today; the
        # reference's other choices are not qualified here.
        self.activation: nn.Module = nn.Identity()
        # Evaluation-only instrument; absent from state_dict and inert in fits.
        self.evaluation_relation_transform: (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None

    def project_content(self, x: torch.Tensor) -> torch.Tensor:
        """Return fused content Q/K/V as ``[B, T, 3, heads, head_dim]``."""
        qkv = self.dropout_qkv(self.content_qkv(x))
        return cast(
            torch.Tensor,
            rearrange(
                qkv,
                "batch len (three d_qkv heads) -> batch len three heads d_qkv",
                heads=self.spec.content_heads,
                three=3,
            ),
        )

    def project_second(self, x: torch.Tensor) -> torch.Tensor:
        """Return the dual-content control's second-branch Q/K/V."""
        assert self.second_qkv is not None
        qkv = self.dropout_qkv(self.second_qkv(x))
        return cast(
            torch.Tensor,
            rearrange(
                qkv,
                "batch len (three d_qkv heads) -> batch len three heads d_qkv",
                heads=self.spec.relational_heads,
                three=3,
            ),
        )

    def project_selection(self, x: torch.Tensor, *, query: bool) -> torch.Tensor:
        """Return selection queries or keys as ``[B, T, heads, head_dim]``."""
        projection = self.selection_q if query else self.selection_k
        assert projection is not None
        return cast(
            torch.Tensor,
            rearrange(
                projection(x), "b l (h d) -> b l h d", h=self.spec.relational_heads
            ),
        )

    def project_relation(self, x: torch.Tensor, *, query: bool) -> torch.Tensor:
        """Return relation queries or keys as ``[B, T, channels, width]``."""
        projection = self.relation_q if query else self.relation_k
        assert projection is not None
        return cast(
            torch.Tensor,
            rearrange(
                projection(x), "b l (r d) -> b l r d", r=self.spec.relation_channels
            ),
        )

    def cache_sources(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """The source projections a layer cache retains for ``x``, by slab name.

        ``x`` is the normalized block input ``[B, T, d_model]``; every value is
        ``[B, T, heads, width]`` and comes from exactly the projections
        :meth:`forward` writes through ``_write_cache``, so a rebuild that
        gathers these at the retained positions reproduces the live cache to
        float32 rounding. Keys follow :class:`DATLayerCache`'s field names.
        """
        spec = self.spec
        _, content_k, content_v = torch.unbind(self.project_content(x), dim=2)
        sources = {"content_keys": content_k, "content_values": content_v}
        if spec.mode == "dual_content":
            _, second_k, second_v = torch.unbind(self.project_second(x), dim=2)
            sources["selection_keys"] = second_k
            sources["selection_values"] = second_v
        else:
            sources["selection_keys"] = self.project_selection(x, query=False)
            if spec.uses_relations:
                sources["relation_keys"] = self.project_relation(x, query=False)
        return sources

    def content_branch(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        allowed: torch.Tensor,
        *,
        scale: float,
        head_scaler: torch.Tensor,
    ) -> torch.Tensor:
        """Ordinary attention over source features; used by both content branches.

        CUDA takes the fused kernel and CPU the dense one. The two are the same
        operator, so the choice follows the device rather than the run's
        configuration and never reaches the attention identity hash.
        """
        attention = (
            _fused_content_attention if queries.is_cuda else _dense_content_attention
        )
        out = attention(queries, keys, values, allowed, scale=scale)
        return rearrange(head_scaler * out, "b q h d -> b q (h d)")

    def relational_weights(
        self,
        selection_q: torch.Tensor,
        selection_k: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        """The relational branch's source weights ``[B, H, Q, K]``.

        This is the branch's score normalization: a source outside ``allowed``
        receives exactly zero weight, so it contributes nothing to the relation
        aggregation or to the symbol buckets that both consume these weights,
        and a receiver with no permitted source gets an all-zero, finite row.
        Exposed on its own so a routing test can inspect the normalization
        rather than only the projected output.
        """
        scores = torch.einsum("bqhd,bkhd->bhqk", selection_q, selection_k)
        return _masked_softmax(scores * (1.0 / math.sqrt(self.spec.head_dim)), allowed)

    def symbol_buckets(self, beta: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Attention weight summed per clipped offset, ``[B, H, Q, D + 1]``.

        This is the symbol aggregation: the weight a receiver puts on every
        source at a given relative position. A source with zero weight in
        ``beta`` adds nothing to its bucket, which is how a source excluded by
        the relational mask stays out of the symbol term as well. Exposed on
        its own so a routing test can inspect the aggregation directly.
        """
        buckets = beta.new_zeros(
            (
                beta.shape[0],
                self.spec.relational_heads,
                beta.shape[2],
                self.spec.max_relative_distance + 1,
            )
        )
        index = offsets.unsqueeze(1).expand_as(beta)
        return buckets.scatter_add_(3, index, beta)

    def relational_branch(
        self,
        selection_q: torch.Tensor,
        selection_k: torch.Tensor,
        relation_q: torch.Tensor | None,
        relation_k: torch.Tensor | None,
        allowed: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Relations between receiver and sources, tagged by relative position.

        Args:
            selection_q: ``[B, Q, H, head_dim]`` receiver selection queries.
            selection_k: ``[B, K, H, head_dim]`` source selection keys.
            relation_q: ``[B, Q, R, width]`` or None in symbol-only mode.
            relation_k: ``[B, K, R, width]`` or None in symbol-only mode.
            allowed: ``[B, 1, Q, K]`` boolean mask of permitted sources.
            offsets: ``[B, Q, K]`` symbol table indices, already clipped.

        Relations are aggregated in channel space *before* being projected to
        head width, so the largest tensor is ``[B, Q, K, R]`` rather than
        ``[B, Q, K, H, head_dim]``. Symbols are handled the same way: attention
        weight is summed per offset bucket, then multiplied by the projected
        table, so ``[B, Q, K, symbol_dim]`` is never materialized.
        """
        spec = self.spec
        beta = self.relational_weights(selection_q, selection_k, allowed)

        out = beta.new_zeros(
            (beta.shape[0], beta.shape[2], spec.relational_heads, spec.head_dim)
        )
        if relation_q is not None and relation_k is not None:
            relations = (
                torch.einsum("bqrd,bkrd->bqkr", relation_q, relation_k)
                * self.relation_scale
            )
            relations = self.activation(relations)
            if self.evaluation_relation_transform is not None:
                if self.training:
                    raise ContractError(
                        "Relation interventions require a frozen evaluation model."
                    )
                relations = self.evaluation_relation_transform(relations, beta, allowed)
            aggregated = torch.einsum("bhqk,bqkr->bqhr", beta, relations)
            assert self.relation_out is not None
            out = out + torch.einsum("bqhr,hdr->bqhd", aggregated, self.relation_out)
        if self.symbol_table is not None:
            assert self.symbol_projection is not None
            buckets = self.symbol_buckets(beta, offsets)
            symbols = rearrange(
                self.symbol_projection(self.symbol_table),
                "n (h d) -> n h d",
                h=spec.relational_heads,
            )
            out = out + torch.einsum("bhqn,nhd->bqhd", buckets, symbols)
        return rearrange(self.second_head_scaler * out, "b q h d -> b q (h d)")

    def forward(
        self,
        x: torch.Tensor,
        query_times: torch.Tensor,
        key_times: torch.Tensor,
        allowed: torch.Tensor,
        cache: DATLayerCache | None = None,
        lengths: torch.Tensor | None = None,
        relational_row_mask: torch.Tensor | None = None,
        relational_allowed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run both branches and concatenate them back to ``d_model``.

        Args:
            x: ``[B, Q, d_model]`` normalized block input.
            query_times: ``[B, Q]`` receiver trajectory positions.
            key_times: ``[B, K]`` source trajectory positions.
            allowed: ``[B, Q, K]`` boolean mask of permitted content sources.
            cache: when given, the current step's sources are written into it
                first and its retained slots supply the keys.
            lengths: ``[B]`` retained lengths, required with ``cache``.
            relational_row_mask: optional ``[B, Q]`` multiplier applied to the
                second branch's projected output (bias included) before
                concatenation, so a receiver row keeps its content heads and
                loses the second branch. The summary carrier's legacy
                ``relational_off`` writer zeroes it on WRITE rows; ``None``
                leaves every row unchanged.
            relational_allowed: optional ``[B, Q, K]`` boolean mask of the
                sources the *relational* branch may read, when they are a
                subset of ``allowed``. It governs the branch's score
                normalization, relation values and symbol aggregation alike,
                and a receiver with no permitted relational source has its
                projected relational output, bias included, set to exactly
                zero. ``None`` is the legacy route: both branches share
                ``allowed``. The dual-content control has no relational branch
                and refuses the argument.
        """
        spec = self.spec
        if relational_allowed is not None:
            if spec.mode == "dual_content":
                raise ContractError(
                    "A relational source route restricts a relational branch; the "
                    "dual-content control has none."
                )
            if relational_allowed.shape != allowed.shape:
                raise ContractError(
                    "The relational source mask must match the content mask shape."
                )
            if bool((relational_allowed & ~allowed).any()):
                raise ContractError(
                    "The relational branch may not read a source the content "
                    "branch may not."
                )
        content = self.project_content(x)
        content_q, content_k, content_v = torch.unbind(content, dim=2)
        second_q: torch.Tensor | None = None
        second_k: torch.Tensor | None = None
        second_v: torch.Tensor | None = None
        selection_q: torch.Tensor | None = None
        selection_k: torch.Tensor | None = None
        relation_q: torch.Tensor | None = None
        relation_k: torch.Tensor | None = None
        if spec.mode == "dual_content":
            second = self.project_second(x)
            second_q, second_k, second_v = torch.unbind(second, dim=2)
        else:
            selection_q = self.project_selection(x, query=True)
            selection_k = self.project_selection(x, query=False)
            relation_q = (
                self.project_relation(x, query=True) if spec.uses_relations else None
            )
            relation_k = (
                self.project_relation(x, query=False) if spec.uses_relations else None
            )

        if cache is not None:
            assert lengths is not None
            sources = _write_cache(
                cache,
                lengths,
                content_keys=content_k,
                content_values=content_v,
                selection_keys=second_k if spec.mode == "dual_content" else selection_k,
                selection_values=second_v if spec.mode == "dual_content" else None,
                relation_keys=relation_k,
            )
            content_k = sources["content_keys"]
            content_v = sources["content_values"]
            if spec.mode == "dual_content":
                second_k = sources["selection_keys"]
                second_v = sources["selection_values"]
            else:
                selection_k = sources["selection_keys"]
                relation_k = sources.get("relation_keys")

        mask = allowed.unsqueeze(1)
        content_out = self.content_branch(
            content_q,
            content_k,
            content_v,
            mask,
            scale=self.content_scale,
            head_scaler=self.content_head_scaler,
        )
        if spec.mode == "dual_content":
            assert second_q is not None and second_k is not None
            assert second_v is not None
            second_out = self.content_branch(
                second_q,
                second_k,
                second_v,
                mask,
                scale=self.second_scale,
                head_scaler=self.second_head_scaler,
            )
        else:
            assert selection_q is not None and selection_k is not None
            offsets = symbol_offsets(query_times, key_times, spec.max_relative_distance)
            relational_mask = (
                mask if relational_allowed is None else relational_allowed.unsqueeze(1)
            )
            second_out = self.relational_branch(
                selection_q,
                selection_k,
                relation_q,
                relation_k,
                relational_mask,
                offsets,
            )
        second_projected = self.second_out(second_out)
        if relational_allowed is not None:
            # A receiver with no permitted relational source (a READ row, a
            # padded row, a WRITE row over an all-padding segment) is zero after
            # the projection, bias included; the softmax above already gave it
            # a finite all-zero row, so no NaN reaches this multiplier.
            reachable = relational_allowed.any(-1, keepdim=True)
            second_projected = second_projected * reachable.to(second_projected.dtype)
        if relational_row_mask is not None:
            second_projected = second_projected * relational_row_mask.unsqueeze(-1)
        return torch.cat((self.content_out(content_out), second_projected), dim=-1)


def symbol_offsets(
    query_times: torch.Tensor, key_times: torch.Tensor, max_distance: int
) -> torch.Tensor:
    """Return clipped symbol table indices for every receiver/source pair.

    The symbol identifies ``p_key - p_query``, clipped into ``[-D, 0]`` and
    shifted by ``D`` so the causal-only table is indexed from zero. Sources are
    never after their receiver under a causal mask; any such pair is masked out
    anyway and clamps harmlessly to offset zero.
    """
    offsets = key_times.unsqueeze(1) - query_times.unsqueeze(2)
    return offsets.clamp(-max_distance, 0) + max_distance


def _write_cache(
    cache: DATLayerCache,
    lengths: torch.Tensor,
    **current: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Store this step's sources, then return every retained source.

    One valid decision per environment is written at that row's retained length,
    exactly as AMAGO's cached attention does, and the readable window spans the
    longest row so short rows are simply masked out by the caller.
    """
    rows = torch.arange(lengths.shape[0], device=lengths.device)
    window = int(lengths.max().item()) + 1
    out: dict[str, torch.Tensor] = {}
    for name, value in current.items():
        stored = getattr(cache, name)
        if stored is None:
            if value is not None:
                raise ContractError(f"DAT cache has no slot for {name!r}.")
            continue
        if value is None:
            raise ContractError(f"DAT cache expected a value for {name!r}.")
        stored[rows, lengths.long()] = value[:, 0].to(stored.dtype)
        out[name] = torch.nan_to_num(stored[:, :window]).to(value.dtype)
    return out


def allocate_layer_caches(
    *,
    batch_size: int,
    capacity: int,
    n_layers: int,
    selected: frozenset[int],
    dat: DATSpec | None,
    backbone_heads: int,
    backbone_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[DATLayerCache]:
    """One NaN-poisoned cache per layer, laid out by the layer's variant.

    An unselected layer keeps keys and values at the backbone's head geometry.
    A selected layer keeps content keys and values at the content head width
    plus its second branch's sources: selection keys and, for the dual-content
    control, selection values; relation keys when relations are used. This is
    the layout ``contracts.cache_slot_floats`` counts, so allocated bytes and
    the closed-form accounting agree tensor for tensor.
    """
    if selected and dat is None:
        raise ContractError("A selected layer needs a dual-attention spec.")

    def slab(heads: int, width: int) -> torch.Tensor:
        data = torch.empty(
            (batch_size, capacity, heads, width), dtype=dtype, device=device
        )
        data.fill_(torch.nan)
        return data

    caches: list[DATLayerCache] = []
    for index in range(n_layers):
        if index not in selected:
            caches.append(
                DATLayerCache(
                    "ordinary",
                    slab(backbone_heads, backbone_head_dim),
                    slab(backbone_heads, backbone_head_dim),
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
    return caches


class DATBlock(nn.Module):
    """AMAGO's pre-norm block with its attention replaced by dual attention.

    Construction takes a fully built AMAGO ``TransformerLayer`` and adopts its
    normalizations, feed-forward layers, activation and dropout. The donor's
    attention modules are dropped rather than kept as unused parameters, so the
    optimizer never sees a projection this block does not use.
    """

    def __init__(self, donor: Any, attention: DualAttention) -> None:
        super().__init__()
        self.attention = attention
        self.norm1: nn.Module = donor.norm1
        self.norm2: nn.Module = donor.norm2
        self.norm3: nn.Module = donor.norm3
        self.norm4: nn.Module = donor.norm4
        self.ff1: nn.Module = donor.ff1
        self.ff2: nn.Module = donor.ff2
        self.dropout_ff: nn.Module = donor.dropout_ff
        self.activation = donor.activation
        self.d_model: int = int(donor.d_model)

    def forward(
        self,
        self_seq: torch.Tensor,
        query_times: torch.Tensor,
        key_times: torch.Tensor,
        allowed: torch.Tensor,
        cache: DATLayerCache | None = None,
        lengths: torch.Tensor | None = None,
        relational_row_mask: torch.Tensor | None = None,
        relational_allowed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q1 = self.norm1(self_seq)  # pre-norm
        q1 = self.attention(
            q1,
            query_times,
            key_times,
            allowed,
            cache=cache,
            lengths=lengths,
            relational_row_mask=relational_row_mask,
            relational_allowed=relational_allowed,
        )
        q1 = self.norm2(q1)  # normformer extra norm 1
        self_seq = self_seq + q1
        q1 = self.norm3(self_seq)  # regular norm
        q1 = self.norm4(self.activation(self.ff1(q1)))  # normformer extra norm 2
        q1 = self.dropout_ff(self.ff2(q1))
        return cast(torch.Tensor, self_seq + q1)


class DATTransformer(nn.Module):
    """An AMAGO Transformer whose selected blocks run dual attention.

    Built by adopting a complete ordinary backbone and replacing only the
    attention computation of the selected blocks. Everything else -- the input
    projection, positional encoding, embedding dropout, unselected blocks and the
    final normalization -- is the same module object AMAGO constructed, so the
    ordinary parts of two matched conditions are identical at step zero.
    """

    def __init__(
        self,
        donor: Any,
        spec: DATSpec,
        *,
        dropout_qkv: float = 0.0,
        head_scaling: bool = True,
        sigma_reparam: bool = True,
    ) -> None:
        super().__init__()
        layers = list(donor.layers)
        spec.validate_layers(len(layers))
        if getattr(donor, "use_rope", False):
            raise ContractError("DAT carries real times and does not use RoPE.")
        self.spec = spec
        self.inp: nn.Module = donor.inp
        self.position_embedding: nn.Module | None = donor.position_embedding
        self.dropout: nn.Module = donor.dropout
        self.norm: nn.Module = donor.norm
        self.d_model: int = int(donor.d_model)
        self.n_layers = len(layers)
        self.selected = frozenset(spec.layer_indices)
        built: list[nn.Module] = []
        for index, layer in enumerate(layers):
            if index in self.selected:
                built.append(
                    DATBlock(
                        layer,
                        DualAttention(
                            spec,
                            dropout_qkv=dropout_qkv,
                            head_scaling=head_scaling,
                            sigma_reparam=sigma_reparam,
                        ),
                    )
                )
            else:
                built.append(layer)
        self.layers = nn.ModuleList(built)
        # Unselected blocks keep the backbone's own head geometry.
        self.backbone_heads = int(layers[0].attention_layer.n_heads)
        self.backbone_head_dim = self.d_model // self.backbone_heads

    @property
    def emb_dim(self) -> int:
        return self.d_model

    def layer_variant(self, index: int) -> str:
        return self.spec.mode if index in self.selected else "ordinary"

    def preprocess_seq(self, seq: torch.Tensor, pos_idxs: torch.Tensor) -> torch.Tensor:
        traj_emb = self.inp(seq)
        if self.position_embedding is not None:
            traj_emb = traj_emb + self.position_embedding(pos_idxs.squeeze(-1))
        return cast(torch.Tensor, self.dropout(traj_emb))

    def init_hidden_state(
        self, batch_size: int, device: torch.device, capacity: int
    ) -> DATHiddenState:
        """Allocate one cache per layer, poisoned so stale reads are loud."""
        spec = self.spec
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
            content = (spec.content_heads, spec.content_head_dim)
            cache = DATLayerCache("", slab(*content), slab(*content))
            cache.variant = spec.mode
            if spec.mode == "dual_content":
                cache.selection_keys = slab(spec.relational_heads, spec.second_head_dim)
                cache.selection_values = slab(
                    spec.relational_heads, spec.second_head_dim
                )
            else:
                cache.selection_keys = slab(spec.relational_heads, spec.head_dim)
                if spec.uses_relations:
                    cache.relation_keys = slab(
                        spec.relation_channels, spec.relation_projection_dim
                    )
            caches.append(cache)
        return DATHiddenState(
            caches,
            torch.full(
                (batch_size, capacity), EMPTY_TIME, dtype=torch.int64, device=device
            ),
            torch.zeros((batch_size,), dtype=torch.int32, device=device),
            attention_sha256=spec.sha256,
        )

    def forward(
        self,
        seq: torch.Tensor,
        pos_idxs: torch.Tensor,
        hidden_state: DATHiddenState | None = None,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DATHiddenState | None]:
        """Run the backbone over a full sequence, or one cached decision.

        Args:
            seq: ``[B, T, tstep_dim]`` timestep tokens.
            pos_idxs: ``[B, T, 1]`` real trajectory positions.
            hidden_state: rollout cache, or None for full-sequence training.
            valid: ``[B, T]`` source validity. Required for the full-sequence
                path, where it must be a contiguous right-padded prefix.
        """
        times = pos_idxs.squeeze(-1).long()
        traj_emb = self.preprocess_seq(seq, pos_idxs)
        if hidden_state is None:
            return self._full_forward(traj_emb, times, valid), None
        return self._cached_forward(traj_emb, times, hidden_state), hidden_state

    def _full_forward(
        self, traj_emb: torch.Tensor, times: torch.Tensor, valid: torch.Tensor | None
    ) -> torch.Tensor:
        rows, length = times.shape
        if valid is None:
            valid = torch.ones((rows, length), dtype=torch.bool, device=times.device)
        require_right_padded(valid)
        causal = torch.tril(
            torch.ones((length, length), dtype=torch.bool, device=times.device)
        )
        allowed = causal.unsqueeze(0) & valid.unsqueeze(1)
        for index, layer in enumerate(self.layers):
            if index in self.selected:
                traj_emb = layer(traj_emb, times, times, allowed)
            else:
                traj_emb = layer(traj_emb)
        return cast(torch.Tensor, self.norm(traj_emb))

    def _cached_forward(
        self, traj_emb: torch.Tensor, times: torch.Tensor, hidden: DATHiddenState
    ) -> torch.Tensor:
        if self.training:
            raise ContractError("Cached DAT execution is evaluation-only.")
        if traj_emb.shape[1] != 1:
            raise ContractError("Cached DAT accepts one decision per environment.")
        if hidden.attention_sha256 != self.spec.sha256:
            raise ContractError("Cached DAT state was built for a different attention.")
        lengths = hidden.lengths
        window = int(lengths.max().item()) + 1
        rows = torch.arange(hidden.batch_size, device=hidden.device)
        current = times[:, 0]
        stale = hidden.times[rows, lengths.long()]
        if bool(((stale != EMPTY_TIME) & (stale != current)).any()):
            raise ContractError("DAT cache slot already holds a different source.")
        hidden.times[rows, lengths.long()] = current
        key_times = hidden.times[:, :window]
        allowed = (
            torch.arange(window, device=hidden.device).unsqueeze(0)
            <= lengths.long().unsqueeze(1)
        ).unsqueeze(1)
        for index, layer in enumerate(self.layers):
            cache = hidden[index]
            if index in self.selected:
                traj_emb = layer(
                    traj_emb,
                    current.unsqueeze(1),
                    key_times,
                    allowed,
                    cache=cache,
                    lengths=lengths,
                )
            else:
                traj_emb = layer(
                    traj_emb,
                    cache.content_keys,
                    cache.content_values,
                    lengths,
                )
        return cast(torch.Tensor, self.norm(traj_emb))


def require_right_padded(valid: torch.Tensor) -> None:
    """Reject validity masks with holes or left padding.

    The replay contract supplies complete outer-task prefixes, so a valid mask is
    always a contiguous run from position zero. Arbitrary holes would be silently
    accepted by AMAGO's ordinary blocks, which mask by causal order alone, so the
    contract is checked rather than assumed.
    """
    lengths = valid.sum(-1)
    expected = torch.arange(valid.shape[1], device=valid.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    if not torch.equal(valid, expected):
        raise ContractError(
            "DAT requires contiguous right-padded validity; got holes or left padding."
        )


def selected_parameter_count(model: DATTransformer) -> int:
    """Active parameters in the replaced attention modules only."""
    total = 0
    for index, layer in enumerate(model.layers):
        if index in model.selected:
            block = cast(DATBlock, layer)
            total += sum(int(p.numel()) for p in block.attention.parameters())
    return total


def dual_content_head_dims(
    spec: DATSpec, target: int, *, second_bounds: Iterable[int] = range(16, 129, 8)
) -> DATSpec:
    """Choose control branch widths whose attention parameters match ``target``.

    Used to freeze the active-capacity control before any reward is observed, as
    the plan requires. Returns the spec with both branch widths filled in.
    """
    best: tuple[int, DATSpec] | None = None
    for second in second_bounds:
        for content in range(16, 129, 8):
            candidate = replace(
                spec,
                mode="dual_content",
                control_content_head_dim=content,
                control_second_head_dim=second,
            )
            module = DualAttention(candidate)
            count = sum(p.numel() for p in module.parameters())
            distance = abs(count - target)
            if best is None or distance < best[0]:
                best = (distance, candidate)
    assert best is not None
    return best[1]


__all__ = [
    "DAT_HIDDEN_STATE_SCHEMA",
    "EMPTY_TIME",
    "DATBlock",
    "DATHiddenState",
    "DATLayerCache",
    "DATTransformer",
    "DualAttention",
    "allocate_layer_caches",
    "content_backend",
    "dual_content_head_dims",
    "require_right_padded",
    "selected_parameter_count",
    "symbol_offsets",
]
