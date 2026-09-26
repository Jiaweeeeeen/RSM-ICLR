"""The condition tables, architecture identities and shared contract errors.

This module is the single source of truth for what every condition is: the five
Stage-1 cells in ``CONDITIONS``, the attention variants in ``DAT_CONDITIONS``
and the memory regimes in ``SUMMARY_CONDITIONS``. Everything downstream --
environment wrapping, encoder selection, replay requirements, checkpoint
identity -- is derived from the ``ConditionSpec`` rows here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Final, Literal

import numpy as np


class ContractError(ValueError):
    """An architecture or research contract was violated."""


class ResultValidationError(ContractError):
    """Result rows cannot reconstruct the declared evaluation."""


def arrays_equal(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> bool:
    """Compare observation dictionaries directly and without derived identities."""
    if tuple(sorted(left)) != tuple(sorted(right)):
        return False
    return all(np.array_equal(left[key], right[key]) for key in sorted(left))


Evidence = Literal["raw", "transition"]
"""What one history token carries: the current timestep, or the causal transition."""

Condition = Literal[
    "feedforward",
    "raw",
    "raw_bypass",
    "transition",
    "transition_bypass",
]

DATCondition = Literal[
    "transition_dat",
    "transition_dat_symbol_only",
    "transition_dual_content",
    "raw_dat",
    "raw_dual_content",
]

SummaryCondition = Literal[
    "raw_gru",
    "raw_segment",
    "raw_summary",
    "raw_summary_detach",
    "raw_summary_residual",
    "raw_summary_gated",
    "raw_dat_segment",
    "raw_dat_summary",
    "raw_dat_summary_relational_write_off",
    "raw_dual_content_summary",
    "raw_window",
]

RevisedCondition = Literal[
    "full_context",
    "full_dual_relational",
    "full_dual_content",
    "full_gru",
    "fixed_summary",
    "fixed_segment",
    "fixed_window",
]
"""The 8M study's six required cells plus the optional GRU.

These are study identities, not renames. ``full_context``,
``full_dual_relational``, ``full_dual_content`` and ``full_gru`` run the same
operators as ``raw``, ``raw_dat``, ``raw_dual_content`` and ``raw_gru`` and keep
their architecture identities, so a legacy checkpoint still loads. The three
``fixed_*`` cells change routing or window semantics and carry new architecture
identities; ``raw_window`` in particular is ordinary attention and is not the
revised band. Never present a measurement made under a legacy name as one of
these cells."""

AnyCondition = Condition | DATCondition | SummaryCondition | RevisedCondition

AttentionVariant = Literal["ordinary", "dat", "dat_symbol_only", "dual_content"]
"""Which attention computation the selected trajectory blocks run."""

MemoryRegime = Literal["full", "segment", "summary", "window", "accumulated"]
"""How much history the carrier may read at a decision.

``full`` is the complete outer-task prefix; ``segment`` is the current segment
of ``C`` records only; ``summary`` is the segment plus ``M`` carried summary
tokens written at every boundary; ``window`` is a sliding band of the most
recent records (``model.window.segment_length`` of them); ``accumulated`` is
the Memo comparator's regime: the current segment plus *every* summary token
written so far, ``S`` per boundary, so the readable state grows with the
number of boundaries crossed rather than staying fixed.
"""

Writer = Literal["same", "relational_off"]
"""What the summary WRITE rows compute: the block's own attention, or the same
block with its relational branch zeroed at those rows (a branch ablation, not an
ordinary-attention writer: the content heads are unchanged and nothing replaces
the removed branch)."""

RelationalSources = Literal["causal_prefix", "timestep_records"]
"""Which keys the relational branch may read inside a bounded block.

``causal_prefix`` is the legacy route: content and relational branches share one
causal key mask, so a relational query also relates summary tokens. The 8M
study's ``timestep_records`` route restricts relational sources to valid
timestep RECORDs and zeroes the relational output of READ rows. The two compute
different functions on identical weights, so they never share an architecture
identity. A full-prefix block has no READ or summary rows, which makes the two
routes identical there; those conditions keep the default and the legacy
identity rather than minting a distinction that computes nothing."""

TrainingSegmentation = Literal["jittered", "fixed"]
"""How the accumulated-summary (Memo) regime segments replayed tasks in the
learner: ``jittered`` draws one segment length per forward from the spec's
+/-20 % range (the published recipe); ``fixed`` trains on the rollout's own
fixed ``L`` (the paper's ablated variant, the study's ``memo_fixed``). Every
other regime segments training exactly as it rolls out and keeps the default."""

DATMode = Literal["dat", "symbol_only", "dual_content"]
RelationActivation = Literal["identity"]
RelationalBackend = Literal["dense"]
PositionMethod = Literal["fixed"]

_VARIANT_MODES: Final[dict[AttentionVariant, DATMode | None]] = {
    "ordinary": None,
    "dat": "dat",
    "dat_symbol_only": "symbol_only",
    "dual_content": "dual_content",
}

Device = Literal["auto", "cpu", "mps", "cuda"]

AttentionBackend = Literal["vanilla", "flash"]

ArchitectureID = Literal[
    "amago-ff-tstep-v3.4.0",
    "amago-history-v1",
    "amago-transition-dat-v1",
    "amago-transition-dual-content-v1",
    "amago-gru-history-v1",
    "amago-summary-v1",
    "amago-dat-summary-v1",
    "amago-dual-content-summary-v1",
    "amago-window-v1",
    "amago-dat-summary-v2",
    "amago-dat-window-v1",
    "amago-memo-v1",
]

HISTORY_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-history-v1"
FEEDFORWARD_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-ff-tstep-v3.4.0"
DAT_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-transition-dat-v1"
DUAL_CONTENT_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-transition-dual-content-v1"
GRU_HISTORY_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-gru-history-v1"
SUMMARY_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-summary-v1"
DAT_SUMMARY_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-dat-summary-v1"
DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID: Final[ArchitectureID] = (
    "amago-dual-content-summary-v1"
)
WINDOW_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-window-v1"
DAT_SUMMARY_V2_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-dat-summary-v2"
DAT_WINDOW_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-dat-window-v1"
MEMO_ARCHITECTURE_ID: Final[ArchitectureID] = "amago-memo-v1"
"""The Memo comparator (ME0/ME1): ordinary blocks over
``[accumulated summaries | segment | summary queries]`` with full gradients
through every accumulated summary, an AMAGO adaptation of
https://github.com/Memory-icrl/memo at commit ``9e7044f`` under the audited
recipe of :class:`MemoSpec`. It is neither a summary identity (no READ/WRITE
slots, no memory projection, no overwrite) nor a window; it hashes its own
spec and restores only its own hidden-state schema."""
"""The 8M study's two revised carriers. ``amago-dat-summary-v2`` is the
timestep-record relational route over the segment and summary regimes;
``amago-dat-window-v1`` is the dual relational per-layer band, which the legacy
``amago-window-v1`` is not. Both were registered here before their carriers
existed and left unbound in :mod:`reasoned_icrl.runtime.amago` until R2 and
R3 implemented them; both are bound now, and a fit still cannot run legacy
semantics under a revised name because each identity hashes its own spec."""

# The identity names the carrier; the packet identity carries the evidence, so
# `amago-history-v1` serves `raw` and `transition` alike, and the two DAT
# carriers serve `raw_dat`/`raw_dual_content` as they serve the transition rows.
_HISTORY_PACKET_ARCHITECTURES: Final = frozenset(
    {
        HISTORY_ARCHITECTURE_ID,
        DAT_ARCHITECTURE_ID,
        DUAL_CONTENT_ARCHITECTURE_ID,
        GRU_HISTORY_ARCHITECTURE_ID,
        SUMMARY_ARCHITECTURE_ID,
        DAT_SUMMARY_ARCHITECTURE_ID,
        DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID,
        WINDOW_ARCHITECTURE_ID,
        DAT_SUMMARY_V2_ARCHITECTURE_ID,
        DAT_WINDOW_ARCHITECTURE_ID,
        MEMO_ARCHITECTURE_ID,
    }
)

_DAT_ARCHITECTURES: Final = frozenset(
    {
        DAT_ARCHITECTURE_ID,
        DUAL_CONTENT_ARCHITECTURE_ID,
        DAT_SUMMARY_ARCHITECTURE_ID,
        DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID,
        DAT_SUMMARY_V2_ARCHITECTURE_ID,
        DAT_WINDOW_ARCHITECTURE_ID,
    }
)

_SUMMARY_ARCHITECTURES: Final = frozenset(
    {
        SUMMARY_ARCHITECTURE_ID,
        DAT_SUMMARY_ARCHITECTURE_ID,
        DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID,
        DAT_SUMMARY_V2_ARCHITECTURE_ID,
    }
)


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """The switches that define a condition.

    ``trajectory_encoder`` picks memory or no memory (``"gru"`` is AMAGO's
    recurrent carrier), ``evidence`` picks what a history token carries, and
    ``bypass`` decides whether the current state also reaches the actor and
    critic without passing through memory. ``attention`` is ``"ordinary"`` for
    every Stage-1 baseline and names a dual-attention variant for the opt-in
    comparison conditions. ``memory`` bounds what a decision may read and
    ``writer`` selects the attention of the summary WRITE rows; every Stage-1
    and DAT-study row keeps ``"full"`` and ``"same"``. ``detach`` and
    ``rewrite`` are the recurrent summary's gradient and boundary-write
    rules; every cell but the two named ablations keeps ``"none"`` and
    ``"replace"``.
    """

    trajectory_encoder: Literal["feedforward", "transformer", "gru"]
    evidence: Evidence | None = None
    bypass: bool = False
    attention: AttentionVariant = "ordinary"
    memory: MemoryRegime = "full"
    writer: Writer = "same"
    relational_sources: RelationalSources = "causal_prefix"
    segmentation: TrainingSegmentation = "jittered"
    detach: SummaryDetach = "none"
    rewrite: SummaryRewrite = "replace"

    # Running without the state bypass is the default, so condition names leave
    # it unmarked: `transition` is transition evidence with no bypass, and only
    # `transition_bypass` adds one.

    def __post_init__(self) -> None:
        if self.trajectory_encoder == "feedforward" and self.evidence is not None:
            raise ContractError("The feed-forward reference reads no history token.")
        if self.trajectory_encoder != "feedforward" and self.evidence is None:
            raise ContractError("A history carrier needs an evidence selection.")
        if self.memory != "full" and self.trajectory_encoder != "transformer":
            raise ContractError(
                "Bounded memory regimes exist for the Transformer carrier only."
            )
        if self.trajectory_encoder == "gru" and (
            self.attention != "ordinary" or self.bypass
        ):
            raise ContractError(
                "The GRU carrier runs no attention variant and no state bypass."
            )
        if self.writer == "relational_off" and not (
            self.attention == "dat" and self.memory == "summary"
        ):
            raise ContractError(
                "The relational-write-off writer applies to dual-attention summary "
                "cells only."
            )
        if self.memory == "window" and self.attention not in ("ordinary", "dat"):
            raise ContractError(
                "The sliding window runs ordinary or dual attention only."
            )
        if self.memory == "accumulated" and (
            self.attention != "ordinary" or self.writer != "same"
        ):
            raise ContractError(
                "The accumulated-summary (Memo) regime is the published ordinary-"
                "attention method; it has no dual-attention or writer variant."
            )
        if self.segmentation != "jittered" and self.memory != "accumulated":
            raise ContractError(
                "Training-segment jitter is a setting of the accumulated-summary "
                "(Memo) regime; every other carrier trains on its rollout segments."
            )
        if self.detach != "none" and self.memory != "summary":
            raise ContractError(
                "Truncated carry gradients apply to the recurrent summary regime, "
                "which alone carries a written memory across boundaries."
            )
        if self.rewrite != "replace" and self.memory != "summary":
            raise ContractError(
                "A residual or gated rewrite applies to the recurrent summary "
                "regime, which alone carries a written memory across boundaries."
            )
        if (
            self.memory == "window"
            and self.attention == "dat"
            and self.relational_sources != "timestep_records"
        ):
            raise ContractError(
                "A dual-attention window is the revised band and requires the "
                "timestep-record relational route."
            )
        if self.relational_sources == "timestep_records":
            if self.attention != "dat":
                raise ContractError(
                    "The timestep-record route restricts a relational branch; only "
                    "dual attention has one."
                )
            if self.memory == "full":
                raise ContractError(
                    "A full-prefix block has no summary or READ rows, so the "
                    "timestep-record route computes the causal route; keep the "
                    "legacy identity rather than minting an equal one."
                )
            if self.writer != "same":
                raise ContractError(
                    "The revised route defines its own WRITE-row behavior; it does "
                    "not compose with the legacy writer ablation."
                )

    @property
    def uses_history_packet(self) -> bool:
        """History conditions need the public outcome adapter and full prefixes."""
        return self.evidence is not None

    @property
    def dat_mode(self) -> DATMode | None:
        """The dual-attention mode this condition runs, or None if ordinary."""
        return _VARIANT_MODES[self.attention]

    @property
    def bounded(self) -> bool:
        """Whether a decision reads less than the complete outer-task prefix."""
        return self.memory != "full"


@dataclass(frozen=True, slots=True)
class DATSpec:
    """Every scientific choice in the dual-attention block, hashed as one identity.

    Constructed once at configuration time and serialized with the run. Its
    ``sha256`` is written into the model as a ``protocol_identity`` buffer, so a
    checkpoint whose tensor shapes happen to agree but whose attention semantics
    differ is still rejected.

    Widths follow the executable DAT branch convention: each branch projects to
    its own width and the two are concatenated back to ``d_model``. The paper's
    Appendix B.1 prints full-model output dimensions instead; the code is
    authoritative here and this choice is verified numerically in the tests.
    """

    layer_indices: tuple[int, ...]
    mode: DATMode = "dat"
    d_model: int = 256
    total_heads: int = 8
    relational_heads: int = 2
    relation_channels: int = 4
    relation_projection_dim: int = 16
    relation_activation: RelationActivation = "identity"
    symmetric_relations: bool = False
    symbol_dim: int = 256
    max_relative_distance: int = 320
    relational_backend: RelationalBackend = "dense"
    position_method: PositionMethod = "fixed"
    cache_dtype: str = "float32"
    # dual_content control only: widened internal Q/K/V dimensions per branch.
    control_content_head_dim: int | None = None
    control_second_head_dim: int | None = None
    schema: str = "dat-attention.v1"

    def __post_init__(self) -> None:
        if not self.layer_indices:
            raise ContractError("DAT requires at least one selected layer index.")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ContractError("DAT selected layer indices must be unique.")
        if any(index < 0 for index in self.layer_indices):
            raise ContractError("DAT selected layer indices must be non-negative.")
        if tuple(sorted(self.layer_indices)) != self.layer_indices:
            raise ContractError("DAT selected layer indices must be ordered.")
        if self.total_heads <= 0 or self.d_model <= 0:
            raise ContractError("DAT residual width and head count must be positive.")
        if self.d_model % self.total_heads:
            raise ContractError("DAT residual width must divide by the head count.")
        if not 0 < self.relational_heads < self.total_heads:
            raise ContractError(
                "DAT needs at least one content head and one relational head."
            )
        if self.relation_channels <= 0 or self.relation_projection_dim <= 0:
            raise ContractError("DAT relation channels and width must be positive.")
        if (
            self.relation_channels * self.relation_projection_dim
            != self.relational_heads * self.head_dim
        ):
            raise ContractError(
                "DAT requires relation_channels * relation_projection_dim == "
                "relational_heads * head_dim."
            )
        if self.symbol_dim <= 0 or self.max_relative_distance <= 0:
            raise ContractError(
                "DAT symbol width and clipping distance must be positive."
            )
        if self.relation_activation != "identity":
            raise ContractError("Only the identity relation activation is qualified.")
        if self.relational_backend != "dense":
            raise ContractError("Only the dense relational backend is qualified.")
        if self.position_method != "fixed":
            raise ContractError("Only fixed positional encoding is qualified.")
        if self.cache_dtype not in ("float32", "bfloat16", "float16"):
            raise ContractError(f"Unsupported DAT cache dtype: {self.cache_dtype!r}.")
        control = (self.control_content_head_dim, self.control_second_head_dim)
        if self.mode == "dual_content":
            if any(value is None or value <= 0 for value in control):
                raise ContractError(
                    "The dual-content control requires positive branch head widths."
                )
        elif any(value is not None for value in control):
            raise ContractError(
                "Control branch head widths belong to the dual-content mode only."
            )
        if self.mode == "symbol_only" and self.symmetric_relations:
            raise ContractError(
                "Symbol-only mode has no relation projections to make symmetric."
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.total_heads

    @property
    def content_heads(self) -> int:
        return self.total_heads - self.relational_heads

    @property
    def content_width(self) -> int:
        """Width of the content branch output projection."""
        return self.content_heads * self.head_dim

    @property
    def relational_width(self) -> int:
        """Width of the relational branch output projection."""
        return self.relational_heads * self.head_dim

    @property
    def uses_relations(self) -> bool:
        """Symbol-only mode drops relation Q/K and the relation output map."""
        return self.mode in ("dat", "dual_content")

    @property
    def uses_symbols(self) -> bool:
        """The dual-content control retrieves source features, never symbols."""
        return self.mode in ("dat", "symbol_only")

    @property
    def content_head_dim(self) -> int:
        """Internal Q/K/V width of each content head."""
        if self.mode == "dual_content":
            assert self.control_content_head_dim is not None
            return self.control_content_head_dim
        return self.head_dim

    @property
    def second_head_dim(self) -> int:
        """Internal Q/K/V width of each head in the second branch."""
        if self.mode == "dual_content":
            assert self.control_second_head_dim is not None
            return self.control_second_head_dim
        return self.head_dim

    @property
    def sha256(self) -> str:
        payload = {**asdict(self), "layer_indices": list(self.layer_indices)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serializable form, with the identity attached."""
        return {
            **asdict(self),
            "layer_indices": list(self.layer_indices),
            "sha256": self.sha256,
        }

    def validate_layers(self, n_layers: int) -> None:
        """Reject selected indices outside a concrete backbone."""
        if any(index >= n_layers for index in self.layer_indices):
            raise ContractError(
                f"DAT selected layer index outside a {n_layers}-layer backbone."
            )


SummaryRegime = Literal["segment", "summary"]
SummaryDetach = Literal["none", "boundary"]
"""``none``: complete-task gradients through every carried write (the study's
recipe); ``boundary``: the carried memory is detached at every segment boundary
of the dense training path, truncated backpropagation at boundaries (the
``raw_summary_detach`` ablation; rollout, rebuild and every forward value are
unchanged)."""
SummaryRewrite = Literal["replace", "residual", "gated"]
"""How a boundary write updates the carried memory. ``replace``: the memory
becomes the projected WRITE outputs (the study's recipe); ``residual``: the
projected WRITE outputs are added to the memory the segment read, so the
identity map is the default rewrite (the ``raw_summary_residual`` ablation declared
after the recurrent summary's collapse on the passive T-Maze, the paper's method);
``gated``: the added
write is scaled by a sigmoid gate, one logit per WRITE slot computed from the
normed WRITE outputs, whose weights start at zero and whose bias starts near
closed, so at initialisation the gate is a per-slot scalar near zero and the
carrier can learn to close the write on segments that carry nothing new (T3, the gated
residual write; T-Maze v3 only). Dense
training, the cached rollout and the rebuild apply the same rule; the
replacing write is unmarked in the identity hash so that every summary
identity recorded before the field existed is unchanged."""
SummaryPosition = Literal["segment-local"]


CONTROL_HEAD_WIDTHS: Final[tuple[int, ...]] = tuple(range(16, 129, 8))
"""The per-head Q/K/V widths the dual-content control is chosen from."""


def _linear_parameters(
    d_in: int, d_out: int, *, bias: bool, sigma_reparam: bool
) -> int:
    """Parameters of one linear layer: weight, bias and the sigma-reparam gain."""
    return d_in * d_out + (d_out if bias else 0) + (1 if sigma_reparam else 0)


def attention_parameter_count(spec: DATSpec, *, sigma_reparam: bool = True) -> int:
    """Parameters of one ``DualAttention`` module built from ``spec``, in closed form.

    Mirrors the module's construction (content branch, second branch, relation
    projections, symbol table, head scalers) so a capacity control can be
    matched without building modules; ``tests/model/test_dat_diagnostics.py``
    pins it against the module's actual parameter count.
    """
    d_model = spec.d_model
    content_inner = spec.content_heads * spec.content_head_dim
    second_inner = spec.relational_heads * spec.second_head_dim
    selection = spec.relational_heads * spec.head_dim
    total = _linear_parameters(
        d_model, 3 * content_inner, bias=False, sigma_reparam=sigma_reparam
    )
    total += _linear_parameters(
        content_inner, spec.content_width, bias=True, sigma_reparam=sigma_reparam
    )
    total += spec.content_heads + spec.relational_heads  # the head scalers
    total += _linear_parameters(
        second_inner, spec.relational_width, bias=True, sigma_reparam=sigma_reparam
    )
    if spec.mode == "dual_content":
        total += _linear_parameters(
            d_model, 3 * second_inner, bias=False, sigma_reparam=sigma_reparam
        )
    else:
        total += 2 * _linear_parameters(
            d_model, selection, bias=False, sigma_reparam=sigma_reparam
        )
    if spec.uses_relations and spec.mode != "dual_content":
        relation_inner = spec.relation_channels * spec.relation_projection_dim
        projections = 1 if spec.symmetric_relations else 2
        total += projections * _linear_parameters(
            d_model, relation_inner, bias=False, sigma_reparam=sigma_reparam
        )
        total += spec.relational_heads * spec.head_dim * spec.relation_channels
    if spec.uses_symbols:
        total += (spec.max_relative_distance + 1) * spec.symbol_dim
        total += _linear_parameters(
            spec.symbol_dim, selection, bias=False, sigma_reparam=sigma_reparam
        )
    return total


def capacity_matched_control_dims(spec: DATSpec) -> tuple[int, int]:
    """Control branch widths matched to the DAT block of the same geometry.

    The reference is ``spec`` in ``dat`` mode (same heads, relation channels,
    symbol width and clipping distance); the result is the first pair on the
    ``CONTROL_HEAD_WIDTHS`` grid, second width outer, whose dual-content
    parameter count is nearest the reference's — the search order of
    ``dual_content_head_dims`` in the model package. A bounded cell derives
    its widths at its own clipping distance: the symbol table shrinks with
    it, so widths matched at the full-prefix distance over-match.
    """
    reference = replace(
        spec, mode="dat", control_content_head_dim=None, control_second_head_dim=None
    )
    target = attention_parameter_count(reference)
    best: tuple[int, int, int] | None = None
    for second in CONTROL_HEAD_WIDTHS:
        for content in CONTROL_HEAD_WIDTHS:
            candidate = replace(
                reference,
                mode="dual_content",
                control_content_head_dim=content,
                control_second_head_dim=second,
            )
            distance = abs(attention_parameter_count(candidate) - target)
            if best is None or distance < best[0]:
                best = (distance, content, second)
    assert best is not None
    return best[1], best[2]


@dataclass(frozen=True, slots=True)
class SummarySpec:
    """Every scientific choice in the segment/summary carrier, hashed as one identity.

    ``segment_length`` (``C``) records fill one segment; ``memory_tokens``
    (``M``) summary tokens are read at its start and written at its end, so a
    segment occupies ``M + C + M`` slots. ``regime`` says whether the written
    memory is carried into the next segment (``summary``) or discarded for the
    task-independent initial memory (``segment``); ``writer`` selects the
    attention the WRITE rows run in a dual-attention block;
    ``relational_sources`` selects which slots that block's relational branch
    may read (the legacy ``causal_prefix`` route shares the content mask, the
    revised ``timestep_records`` route reads valid RECORD slots only and zeroes
    READ rows). ``detach`` and ``position`` name the fixed decisions of the
    study (full-task gradients, segment-local slot positions) so that a
    different choice would be a different identity rather than a silent change;
    ``detach="boundary"`` is that different identity for the truncated-gradient
    ablation, hashed apart from the study's carrier, and
    ``rewrite="residual"`` is the identity of the carrier
    whose boundary write is added to the memory it read instead of replacing
    it, and ``rewrite="gated"`` the identity of the carrier
    whose added write is scaled by a learned per-slot gate.

    The identity hash omits ``relational_sources`` on the legacy route and
    ``rewrite`` under the replacing write so that every summary identity
    recorded before either field existed still hashes to the same value:
    legacy resolved configs, checkpoints and hidden states keep loading against
    their own carrier. The revised route and the residual rewrite are hashed,
    so no two settings share an identity.
    """

    segment_length: int
    memory_tokens: int
    regime: SummaryRegime
    writer: Writer = "same"
    relational_sources: RelationalSources = "causal_prefix"
    detach: SummaryDetach = "none"
    rewrite: SummaryRewrite = "replace"
    position: SummaryPosition = "segment-local"
    d_model: int = 256
    cache_dtype: str = "float32"
    schema: str = "summary-memory.v1"

    def __post_init__(self) -> None:
        if type(self.segment_length) is not int or self.segment_length < 4:
            raise ContractError("Summary segment_length must be an integer >= 4.")
        if type(self.memory_tokens) is not int or self.memory_tokens < 1:
            raise ContractError("Summary memory_tokens must be an integer >= 1.")
        if self.regime not in ("segment", "summary"):
            raise ContractError(f"Unknown summary regime: {self.regime!r}.")
        if self.writer not in ("same", "relational_off"):
            raise ContractError(f"Unknown summary writer: {self.writer!r}.")
        if self.writer == "relational_off" and self.regime != "summary":
            raise ContractError(
                "The relational-write-off writer applies to the summary regime."
            )
        if self.relational_sources not in ("causal_prefix", "timestep_records"):
            raise ContractError(
                f"Unknown summary relational_sources: {self.relational_sources!r}."
            )
        if self.relational_sources == "timestep_records" and self.writer != "same":
            raise ContractError(
                "The timestep-record route defines its own WRITE-row behavior; it "
                "does not compose with the legacy writer ablation."
            )
        if self.detach not in ("none", "boundary"):
            raise ContractError(f"Unknown summary detach: {self.detach!r}.")
        if self.detach == "boundary" and self.regime != "summary":
            raise ContractError(
                "Boundary detach applies to the summary regime, which alone carries "
                "a written memory across boundaries."
            )
        if self.rewrite not in ("replace", "residual", "gated"):
            raise ContractError(f"Unknown summary rewrite: {self.rewrite!r}.")
        if self.rewrite != "replace" and self.regime != "summary":
            raise ContractError(
                "A residual or gated rewrite applies to the summary regime, which "
                "alone carries a written memory across boundaries."
            )
        if self.position != "segment-local":
            raise ContractError("Only segment-local positions are qualified.")
        if type(self.d_model) is not int or self.d_model <= 0:
            raise ContractError("Summary residual width must be positive.")
        if self.cache_dtype not in ("float32", "bfloat16", "float16"):
            raise ContractError(
                f"Unsupported summary cache dtype: {self.cache_dtype!r}."
            )
        if self.schema != "summary-memory.v1":
            raise ContractError(f"Unknown summary schema: {self.schema!r}.")

    @property
    def capacity(self) -> int:
        """Slots per segment: ``M + C + M``."""
        return self.memory_tokens + self.segment_length + self.memory_tokens

    @property
    def read_slots(self) -> range:
        return range(0, self.memory_tokens)

    @property
    def record_slots(self) -> range:
        return range(self.memory_tokens, self.memory_tokens + self.segment_length)

    @property
    def write_slots(self) -> range:
        return range(self.memory_tokens + self.segment_length, self.capacity)

    def _hashed(self) -> dict[str, Any]:
        """The fields that enter the identity: the legacy route and the
        replacing write are unmarked."""
        fields = asdict(self)
        if fields["relational_sources"] == "causal_prefix":
            del fields["relational_sources"]
        if fields["rewrite"] == "replace":
            del fields["rewrite"]
        return fields

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self._hashed(), sort_keys=True).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serializable form, with the identity attached."""
        return {**asdict(self), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """The sliding-window reference's identity: a per-layer band of ``W`` slots.

    The window carrier is the ordinary backbone whose every block may attend
    to at most the last ``segment_length`` slots, the query included: a banded
    mask in training and a ``segment_length``-slot rolling cache per layer at
    rollout. It holds no memory tokens and no regime, so it is not a
    ``SummarySpec``. The study runs ``segment_length = 40``: the same slot
    count per layer as the summary carrier's cache, for equal allocated bytes.
    """

    segment_length: int
    cache_dtype: str = "float32"
    schema: str = "window-memory.v1"

    def __post_init__(self) -> None:
        if type(self.segment_length) is not int or self.segment_length < 4:
            raise ContractError("Window segment_length must be an integer >= 4.")
        if self.cache_dtype not in ("float32", "bfloat16", "float16"):
            raise ContractError(
                f"Unsupported window cache dtype: {self.cache_dtype!r}."
            )
        if self.schema != "window-memory.v1":
            raise ContractError(f"Unknown window schema: {self.schema!r}.")

    @property
    def capacity(self) -> int:
        """Retained slots per layer, the query's own slot included."""
        return self.segment_length

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serializable form, with the identity attached."""
        return {**asdict(self), "sha256": self.sha256}


MemoCarry = Literal["accumulate"]
MemoPosition = Literal["concatenated"]


@dataclass(frozen=True, slots=True)
class MemoSpec:
    """Every scientific choice of the Memo comparator, hashed as one identity.

    The audited AMAGO adaptation of Memo (Gupta et al., 2025; source commit
    ``9e7044f`` of ``Memory-icrl/memo``): ``segment_length`` (``L``) admitted
    records fill one segment; at every boundary ``summary_tokens`` (``S``)
    learned summary queries read the segment and every earlier summary, and
    their final-layer outputs are *appended* to the carried summaries
    (``carry="accumulate"``: the paper's method; keeping only the newest
    summary is the RMT variant and is not an option here). Positions are the
    slot index of the concatenated block ``[summaries | records | queries]``
    (``position="concatenated"``: summaries at ``0 .. nS-1``, records from
    ``nS``, the paper's scheme). Training runs the whole task with gradients
    through every summary (``detach="none"``) and one uniform draw of the
    dense segment length from ``[ceil((1-j)L), floor((1+j)L)]`` per learner
    forward, ``j = training_segment_jitter`` (the paper's +/-20 %); rollout
    and reconstruction use the fixed ``L``.

    The rollout cache is allocated for the longest task the carrier is built
    for, but the *live* state grows with the boundaries crossed:
    :func:`memo_live_slots` gives the filled slots per layer at a prefix
    length, which is what the comparator measures instead of asserting a
    byte match with the fixed-size carriers.
    """

    segment_length: int
    summary_tokens: int
    carry: MemoCarry = "accumulate"
    training_segment_jitter: float = 0.2
    detach: SummaryDetach = "none"
    position: MemoPosition = "concatenated"
    d_model: int = 256
    cache_dtype: str = "float32"
    schema: str = "memo-summary.v1"

    def __post_init__(self) -> None:
        if type(self.segment_length) is not int or self.segment_length < 4:
            raise ContractError("Memo segment_length must be an integer >= 4.")
        if type(self.summary_tokens) is not int or self.summary_tokens < 1:
            raise ContractError("Memo summary_tokens must be an integer >= 1.")
        if self.summary_tokens >= self.segment_length:
            raise ContractError(
                "Memo writes fewer summary tokens than records per segment."
            )
        if self.carry != "accumulate":
            raise ContractError(
                "Memo accumulates summaries; an overwriting carry is the RMT "
                "variant, a different method."
            )
        if type(self.training_segment_jitter) not in (int, float) or not (
            0.0 <= float(self.training_segment_jitter) < 1.0
        ):
            raise ContractError("Memo training_segment_jitter must be in [0, 1).")
        if self.detach != "none":
            raise ContractError("Only full-task gradients (detach none) are qualified.")
        if self.position != "concatenated":
            raise ContractError("Only concatenated block positions are qualified.")
        if type(self.d_model) is not int or self.d_model <= 0:
            raise ContractError("Memo residual width must be positive.")
        if self.cache_dtype not in ("float32", "bfloat16", "float16"):
            raise ContractError(f"Unsupported Memo cache dtype: {self.cache_dtype!r}.")
        if self.schema != "memo-summary.v1":
            raise ContractError(f"Unknown Memo schema: {self.schema!r}.")

    @property
    def jitter_range(self) -> tuple[int, int]:
        """The inclusive dense segment lengths training draws from."""
        jitter = float(self.training_segment_jitter)
        low = math.ceil((1.0 - jitter) * self.segment_length)
        high = math.floor((1.0 + jitter) * self.segment_length)
        return max(low, self.summary_tokens + 1), max(high, self.segment_length)

    def summaries_before(self, index: int) -> int:
        """Boundaries crossed before the record at 0-based ``index``."""
        if index < 0:
            raise ContractError("A record index is non-negative.")
        return index // self.segment_length

    def capacity(self, max_index: int) -> int:
        """Cache slots per layer for tasks of at most ``max_index + 1`` records:
        every summary the longest task accumulates, one open segment and the
        ``S`` queries of a boundary."""
        if max_index < 0:
            raise ContractError("The longest record index is non-negative.")
        return (
            self.summaries_before(max_index) * self.summary_tokens
            + self.segment_length
            + self.summary_tokens
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serializable form, with the identity attached."""
        return {**asdict(self), "sha256": self.sha256}


def memo_live_slots(spec: MemoSpec, prefix_length: int) -> int:
    """Filled cache slots per layer after ``prefix_length`` records of one task.

    The lazy boundary convention of the carrier: after ``P`` records the
    ``(P - 1) // L`` completed boundaries hold ``S`` summary slots each and the
    open segment holds the remaining ``P - qL`` records (``1 <= r <= L``); a
    prefix ending exactly at a boundary keeps its full segment with the write
    pending. Zero for an empty prefix.
    """
    if prefix_length < 0:
        raise ContractError("A prefix length is non-negative.")
    if prefix_length == 0:
        return 0
    written = (prefix_length - 1) // spec.segment_length
    return written * spec.summary_tokens + prefix_length - written * spec.segment_length


# ----------------------------------------------------------------------
# Bounded-state accounting: the window is matched to the summary by bytes
# ----------------------------------------------------------------------

CACHE_ELEMENT_BYTES: Final[dict[str, int]] = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
}
"""Bytes per retained cache value, by the carriers' ``cache_dtype`` names."""


def cache_slot_floats(
    dat: DATSpec | None, *, layers: int, heads: int, width: int
) -> tuple[int, ...]:
    """Floating-point values one retained slot occupies in each layer's cache.

    An unselected block keeps a key and a value at the backbone head geometry,
    ``2 * width`` values. A selected dual-attention block keeps its content keys
    and values at the content head width plus what its second branch retains:
    selection keys (and, for the dual-content control, selection values) and,
    when relations are used, relation keys. This is the layout
    ``model.dat_transformer.allocate_layer_caches`` builds; the tests pin the
    two to each other tensor for tensor, which is what makes the closed form a
    measured count rather than a nominal slot count.
    """
    if layers < 1 or heads < 1 or width < 1 or width % heads:
        raise ContractError("Cache accounting needs a valid layers/heads/width.")
    selected: frozenset[int] = frozenset()
    if dat is not None:
        dat.validate_layers(layers)
        if dat.d_model != width or dat.total_heads != heads:
            raise ContractError("DAT attention geometry disagrees with the model.")
        selected = frozenset(dat.layer_indices)
    per_layer: list[int] = []
    for index in range(layers):
        if index not in selected:
            per_layer.append(2 * width)
            continue
        assert dat is not None
        floats = 2 * dat.content_heads * dat.content_head_dim
        if dat.mode == "dual_content":
            floats += 2 * dat.relational_heads * dat.second_head_dim
        else:
            floats += dat.relational_heads * dat.head_dim
            if dat.uses_relations:
                floats += dat.relation_channels * dat.relation_projection_dim
        per_layer.append(floats)
    return tuple(per_layer)


def summary_state_bytes(
    summary: SummarySpec, dat: DATSpec | None, *, layers: int, heads: int, width: int
) -> int:
    """Allocated per-actor rollout state of the summary carrier, in bytes.

    Every cache slab at ``M + C + M`` slots in the summary's cache dtype, the
    float32 memory ``[M, d]``, the int32 filled-slot counter and the int64
    boundary counter. The carrier's detached copy of the learned initial
    memory is shared by every actor and is a parameter's shadow, not per-actor
    state, so it is deliberately absent here and reported separately.
    """
    if summary.d_model != width:
        raise ContractError("Summary width disagrees with the model.")
    element = CACHE_ELEMENT_BYTES[summary.cache_dtype]
    slabs = sum(cache_slot_floats(dat, layers=layers, heads=heads, width=width))
    return (
        slabs * summary.capacity * element
        + summary.memory_tokens * summary.d_model * 4
        + 4  # int32 filled slots
        + 8  # int64 boundary counter
    )


def window_state_bytes(
    window_length: int,
    dat: DATSpec | None,
    *,
    layers: int,
    heads: int,
    width: int,
    cache_dtype: str = "float32",
) -> int:
    """Allocated per-actor rollout state of the window carrier, in bytes.

    Every cache slab at ``W`` slots, the int64 source time of every slot (the
    window carries real trajectory times, which the summary's slot-indexed
    cache does not need) and the int32 retained length.
    """
    if window_length < 1:
        raise ContractError("Window accounting needs a positive window length.")
    element = CACHE_ELEMENT_BYTES[cache_dtype]
    slabs = sum(cache_slot_floats(dat, layers=layers, heads=heads, width=width))
    return slabs * window_length * element + window_length * 8 + 4


def memo_state_bytes(
    memo: MemoSpec, *, max_index: int, layers: int, heads: int, width: int
) -> int:
    """Allocated per-actor rollout state of the Memo carrier, in bytes.

    Every ordinary cache slab at :meth:`MemoSpec.capacity` slots (the
    carrier's blocks are all ordinary), the int32 filled-slot counter and the
    int64 boundary counter. The allocation is fixed by the longest task; the
    live figure at a prefix is ``memo_live_slots`` slots of the same slabs.
    """
    if memo.d_model != width:
        raise ContractError("Memo width disagrees with the model.")
    element = CACHE_ELEMENT_BYTES[memo.cache_dtype]
    slabs = sum(cache_slot_floats(None, layers=layers, heads=heads, width=width))
    return slabs * memo.capacity(max_index) * element + 4 + 8


def memo_live_state_bytes(
    memo: MemoSpec, *, prefix_length: int, layers: int, heads: int, width: int
) -> int:
    """Bytes of Memo state actually filled after ``prefix_length`` records:
    the live slots of every slab plus the two counters."""
    if memo.d_model != width:
        raise ContractError("Memo width disagrees with the model.")
    element = CACHE_ELEMENT_BYTES[memo.cache_dtype]
    slabs = sum(cache_slot_floats(None, layers=layers, heads=heads, width=width))
    return slabs * memo_live_slots(memo, prefix_length) * element + 4 + 8


@dataclass(frozen=True, slots=True)
class StateMatch:
    """The window length chosen from measured bytes, with its residual.

    ``window_length`` is the largest ``W`` whose allocated per-actor window
    state does not exceed the summary carrier's allocated per-actor state at
    the reference ``(C, M)``; ``residual_bytes`` is what the summary still
    holds beyond it. ``slot_parity_length`` is the provisional ``C + 2M`` the
    contracts started from, kept so the two rules can be compared.
    """

    reference_segment_length: int
    reference_memory_tokens: int
    reference_cache_dtype: str
    summary_state_bytes: int
    window_length: int
    window_state_bytes: int
    residual_bytes: int
    slot_parity_length: int
    method: str = (
        "largest W whose allocated per-actor window state (cache slabs, int64 "
        "source times, int32 length) does not exceed the summary carrier's "
        "allocated per-actor state (cache slabs at M+C+M, float32 memory, "
        "counters); both counted tensor by tensor, shared initial memory excluded"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def match_window_to_summary(
    summary: SummarySpec,
    dat: DATSpec | None,
    *,
    layers: int,
    heads: int,
    width: int,
    cache_dtype: str = "float32",
) -> StateMatch:
    """Choose ``W`` from bytes: the largest window within the summary's state.

    Deterministic in the specs and geometry alone, so it is fixed before any
    policy result exists. The DAT spec's clipping distance does not enter (the
    symbol table is a parameter, not state), so the same spec serves the
    summary reference and the window it is matched to.
    """
    budget = summary_state_bytes(summary, dat, layers=layers, heads=heads, width=width)
    element = CACHE_ELEMENT_BYTES[cache_dtype]
    per_slot = (
        sum(cache_slot_floats(dat, layers=layers, heads=heads, width=width)) * element
        + 8
    )
    window_length = (budget - 4) // per_slot
    if window_length < 4:
        raise ContractError(
            "The summary state is too small to match a window of at least 4 slots."
        )
    allocated = window_state_bytes(
        window_length,
        dat,
        layers=layers,
        heads=heads,
        width=width,
        cache_dtype=cache_dtype,
    )
    assert allocated <= budget < allocated + per_slot
    return StateMatch(
        reference_segment_length=summary.segment_length,
        reference_memory_tokens=summary.memory_tokens,
        reference_cache_dtype=summary.cache_dtype,
        summary_state_bytes=budget,
        window_length=window_length,
        window_state_bytes=allocated,
        residual_bytes=budget - allocated,
        slot_parity_length=summary.capacity,
    )


def control_parameter_match(spec: DATSpec) -> dict[str, int | float | bool]:
    """One dual-content block's attention parameters against its DAT reference.

    The reference is the same geometry in ``dat`` mode (heads, channels, symbol
    width and clipping distance); the relative difference is reported with the
    two tolerances the condition table names (aim 2%, require 5%). Recorded per
    resolved run so the match is a measurement on that environment's clipping
    distance rather than a figure carried over from another study.
    """
    if spec.mode != "dual_content":
        raise ContractError("Only the dual-content control has a parameter match.")
    reference = replace(
        spec, mode="dat", control_content_head_dim=None, control_second_head_dim=None
    )
    reference_count = attention_parameter_count(reference)
    control_count = attention_parameter_count(spec)
    relative = (control_count - reference_count) / reference_count
    return {
        "reference_attention_parameters": reference_count,
        "control_attention_parameters": control_count,
        "relative_difference": relative,
        "within_two_percent": abs(relative) <= 0.02,
        "within_five_percent": abs(relative) <= 0.05,
    }


def dat_architecture_id(mode: DATMode) -> ArchitectureID:
    """Map an attention mode to its architecture identity."""
    if mode in ("dat", "symbol_only"):
        return DAT_ARCHITECTURE_ID
    if mode == "dual_content":
        return DUAL_CONTENT_ARCHITECTURE_ID
    raise ContractError(f"Unknown DAT mode: {mode!r}.")


CONDITIONS: dict[Condition, ConditionSpec] = {
    "feedforward": ConditionSpec("feedforward"),
    "raw": ConditionSpec("transformer", "raw", False),
    "raw_bypass": ConditionSpec("transformer", "raw", True),
    "transition": ConditionSpec("transformer", "transition", False),
    "transition_bypass": ConditionSpec("transformer", "transition", True),
}


DAT_CONDITIONS: dict[DATCondition, ConditionSpec] = {
    "transition_dat": ConditionSpec("transformer", "transition", False, "dat"),
    "transition_dat_symbol_only": ConditionSpec(
        "transformer", "transition", False, "dat_symbol_only"
    ),
    "transition_dual_content": ConditionSpec(
        "transformer", "transition", False, "dual_content"
    ),
    "raw_dat": ConditionSpec("transformer", "raw", False, "dat"),
    "raw_dual_content": ConditionSpec("transformer", "raw", False, "dual_content"),
}
"""Opt-in dual-attention comparison conditions over the full prefix.

Deliberately kept out of ``CONDITIONS``: that table is the frozen Stage-1
factorial, and several analysis and lifecycle paths require exactly those five
cells. The ``transition_*`` rows are the DAT study's; the ``raw_*`` rows are
the summary-memory study's full-prefix attention variants (no raw symbol-only
row: that mode is not used there). Use ``ALL_CONDITIONS`` where any condition
name must resolve.
"""

SUMMARY_CONDITIONS: dict[SummaryCondition, ConditionSpec] = {
    "raw_gru": ConditionSpec("gru", "raw", False),
    "raw_segment": ConditionSpec("transformer", "raw", False, "ordinary", "segment"),
    "raw_summary": ConditionSpec("transformer", "raw", False, "ordinary", "summary"),
    "raw_summary_detach": ConditionSpec(
        "transformer", "raw", False, "ordinary", "summary", detach="boundary"
    ),
    "raw_summary_residual": ConditionSpec(
        "transformer", "raw", False, "ordinary", "summary", rewrite="residual"
    ),
    "raw_summary_gated": ConditionSpec(
        "transformer", "raw", False, "ordinary", "summary", rewrite="gated"
    ),
    "raw_dat_segment": ConditionSpec("transformer", "raw", False, "dat", "segment"),
    "raw_dat_summary": ConditionSpec("transformer", "raw", False, "dat", "summary"),
    "raw_dat_summary_relational_write_off": ConditionSpec(
        "transformer", "raw", False, "dat", "summary", "relational_off"
    ),
    "raw_dual_content_summary": ConditionSpec(
        "transformer", "raw", False, "dual_content", "summary"
    ),
    "raw_window": ConditionSpec("transformer", "raw", False, "ordinary", "window"),
}
"""The summary-memory study's grounding baseline and bounded memory regimes.

``raw_gru`` is AMAGO's own recurrent carrier over the full prefix; every other
row reads a bounded history. All share the ``raw`` packet with no bypass.
"""

REVISED_CONDITIONS: dict[RevisedCondition, ConditionSpec] = {
    "full_context": ConditionSpec("transformer", "raw", False),
    "full_dual_relational": ConditionSpec("transformer", "raw", False, "dat"),
    "full_dual_content": ConditionSpec("transformer", "raw", False, "dual_content"),
    "full_gru": ConditionSpec("gru", "raw", False),
    "fixed_summary": ConditionSpec(
        "transformer",
        "raw",
        False,
        "dat",
        "summary",
        relational_sources="timestep_records",
    ),
    "fixed_segment": ConditionSpec(
        "transformer",
        "raw",
        False,
        "dat",
        "segment",
        relational_sources="timestep_records",
    ),
    "fixed_window": ConditionSpec(
        "transformer",
        "raw",
        False,
        "dat",
        "window",
        relational_sources="timestep_records",
    ),
}
"""The 8M study's roster. The four full-prefix rows are switch-identical to
their legacy analogues by design: the study identity changes, the operator does
not. ``fixed_summary`` and ``fixed_segment`` share one model and differ only in
cross-segment carry, exactly as ``raw_dat_summary`` and ``raw_dat_segment`` do.
"""


MemoCondition = Literal["memo", "memo_fixed"]

MEMO_CONDITIONS: dict[MemoCondition, ConditionSpec] = {
    "memo": ConditionSpec("transformer", "raw", False, "ordinary", "accumulated"),
    "memo_fixed": ConditionSpec(
        "transformer", "raw", False, "ordinary", "accumulated", segmentation="fixed"
    ),
}
"""The Memo comparator: ordinary attention over the raw
packet without bypass, the accumulated-summary regime. ``memo`` trains with the
published +/-20 % segment jitter; ``memo_fixed`` (ME4, added with the figure
redesign) is the same recipe with jitter 0, so training and rollout both use
the fixed ``L`` segments, the paper's ablated variant and the training-
procedure-matched reading against ``fixed_summary``. Both are published-method
comparators run at matched experience beside the 8M study, under their own
study roots, sharing the ``amago-memo-v1`` carrier with distinct spec hashes;
neither belongs to a tier of the 8M roster or renames an existing cell
(``raw_summary`` is the ordinary fixed-size writer, not Memo)."""


ALL_CONDITIONS: dict[str, ConditionSpec] = {
    **{str(name): spec for name, spec in CONDITIONS.items()},
    **{str(name): spec for name, spec in DAT_CONDITIONS.items()},
    **{str(name): spec for name, spec in SUMMARY_CONDITIONS.items()},
    **{str(name): spec for name, spec in REVISED_CONDITIONS.items()},
    **{str(name): spec for name, spec in MEMO_CONDITIONS.items()},
}


CONDITION_LABELS: dict[Condition, str] = {
    "feedforward": "Feed-forward, no history",
    "raw": "Raw history",
    "raw_bypass": "Raw history, with state bypass",
    "transition": "Transition history",
    "transition_bypass": "Transition history, with state bypass",
}

DAT_CONDITION_LABELS: dict[DATCondition, str] = {
    "transition_dat": "Transition history, dual attention",
    "transition_dat_symbol_only": ("Transition history, dual attention, symbols only"),
    "transition_dual_content": (
        "Transition history, two content branches (capacity control)"
    ),
    "raw_dat": "Raw history, dual attention",
    "raw_dual_content": "Raw history, two content branches (capacity control)",
}

SUMMARY_CONDITION_LABELS: dict[SummaryCondition, str] = {
    "raw_gru": "Raw history, GRU (AMAGO recurrent baseline)",
    "raw_segment": "Raw history, segment memory (no carry)",
    "raw_summary": "Raw history, recurrent summary",
    "raw_summary_detach": (
        "Raw history, recurrent summary, gradients truncated at segment boundaries"
    ),
    "raw_summary_residual": (
        "Raw history, recurrent summary, residual rewrite (memory + write)"
    ),
    "raw_summary_gated": (
        "Raw history, recurrent summary, gated residual rewrite "
        "(memory + sigmoid(gate) * write)"
    ),
    "raw_dat_segment": "Raw history, dual attention, segment memory",
    "raw_dat_summary": "Raw history, dual attention, recurrent summary",
    "raw_dat_summary_relational_write_off": (
        "Raw history, dual attention, recurrent summary, relational branch off "
        "at WRITE rows"
    ),
    "raw_dual_content_summary": (
        "Raw history, two content branches, recurrent summary (capacity control)"
    ),
    "raw_window": "Raw history, sliding window",
}

REVISED_CONDITION_LABELS: dict[RevisedCondition, str] = {
    "full_context": "Full context (ordinary causal attention, entire prefix)",
    "full_dual_relational": "Full dual relational (DAT content and relational)",
    "full_dual_content": "Full dual content (two content branches, capacity control)",
    "full_gru": "Full GRU (AMAGO recurrent carrier, optional reference)",
    "fixed_summary": "Fixed summary (DAT, C records and M carried summary tokens)",
    "fixed_segment": "Fixed segment (same DAT model, summary reset at each boundary)",
    "fixed_window": "Fixed window (DAT, W retained slots per layer, no summary)",
}


MEMO_CONDITION_LABELS: dict[MemoCondition, str] = {
    "memo": (
        "Memo (AMAGO adaptation, matched 8M): ordinary attention, accumulated "
        "summary tokens with full gradients, jittered training segments"
    ),
    "memo_fixed": (
        "Memo, fixed segments (AMAGO adaptation, matched 8M): the same recipe "
        "with training-segment jitter 0"
    ),
}


ALL_CONDITION_LABELS: dict[str, str] = {
    **{str(name): label for name, label in CONDITION_LABELS.items()},
    **{str(name): label for name, label in DAT_CONDITION_LABELS.items()},
    **{str(name): label for name, label in SUMMARY_CONDITION_LABELS.items()},
    **{str(name): label for name, label in REVISED_CONDITION_LABELS.items()},
    **{str(name): label for name, label in MEMO_CONDITION_LABELS.items()},
}


ARCHITECTURE_LABELS: dict[str, str] = {
    HISTORY_ARCHITECTURE_ID: "AMAGO public history and observation carrier (v1)",
    FEEDFORWARD_ARCHITECTURE_ID: "AMAGO feed-forward timestep encoder (v3.4)",
    DAT_ARCHITECTURE_ID: "Transition history with dual attention (v1)",
    DUAL_CONTENT_ARCHITECTURE_ID: ("Transition history with two content branches (v1)"),
    GRU_HISTORY_ARCHITECTURE_ID: (
        "AMAGO GRU trajectory encoder over the history packet (v1)"
    ),
    SUMMARY_ARCHITECTURE_ID: "Ordinary blocks, segment or summary memory regime (v1)",
    DAT_SUMMARY_ARCHITECTURE_ID: (
        "One dual-attention block, segment or summary memory regime (v1)"
    ),
    DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID: (
        "One block with two content branches, summary memory regime (v1)"
    ),
    WINDOW_ARCHITECTURE_ID: "Ordinary blocks over a sliding window (v1)",
    DAT_SUMMARY_V2_ARCHITECTURE_ID: (
        "One dual-attention block, segment or summary regime, timestep-record "
        "relational route (v2)"
    ),
    DAT_WINDOW_ARCHITECTURE_ID: (
        "Dual-attention blocks over a per-layer relational band (v1)"
    ),
    MEMO_ARCHITECTURE_ID: (
        "Memo: ordinary blocks over accumulated summary tokens, full gradients "
        "(AMAGO adaptation, v1)"
    ),
}


def condition_label(condition: str) -> str:
    if condition not in ALL_CONDITION_LABELS:
        raise ContractError(f"Unknown condition: {condition!r}.")
    return ALL_CONDITION_LABELS[condition]


def architecture_label(architecture_id: str) -> str:
    if architecture_id not in ARCHITECTURE_LABELS:
        raise ContractError(f"Unknown architecture ID: {architecture_id!r}.")
    return ARCHITECTURE_LABELS[architecture_id]


def architecture_uses_history_packet(architecture_id: str) -> bool:
    """Return whether a resolved architecture consumes history packets."""
    if architecture_id not in ARCHITECTURE_LABELS:
        raise ContractError(f"Unknown architecture ID: {architecture_id!r}.")
    return architecture_id in _HISTORY_PACKET_ARCHITECTURES


HISTORY_CONDITIONS: tuple[Condition, ...] = tuple(
    key for key, value in CONDITIONS.items() if value.uses_history_packet
)


def condition_spec(condition: str) -> ConditionSpec:
    """Return the typed switch record for an active condition."""
    if condition not in ALL_CONDITIONS:
        raise ContractError(f"Unknown condition: {condition!r}.")
    return ALL_CONDITIONS[condition]


def architecture_uses_dat(architecture_id: str) -> bool:
    """Return whether a resolved architecture runs a dual-attention block."""
    if architecture_id not in ARCHITECTURE_LABELS:
        raise ContractError(f"Unknown architecture ID: {architecture_id!r}.")
    return architecture_id in _DAT_ARCHITECTURES


def architecture_uses_memo(architecture_id: str) -> bool:
    """Return whether a resolved architecture runs the Memo carrier."""
    if architecture_id not in ARCHITECTURE_LABELS:
        raise ContractError(f"Unknown architecture ID: {architecture_id!r}.")
    return architecture_id == MEMO_ARCHITECTURE_ID


def architecture_uses_summary(architecture_id: str) -> bool:
    """Return whether a resolved architecture runs the segment/summary carrier."""
    if architecture_id not in ARCHITECTURE_LABELS:
        raise ContractError(f"Unknown architecture ID: {architecture_id!r}.")
    return architecture_id in _SUMMARY_ARCHITECTURES


def summary_architecture_id(
    attention: AttentionVariant,
    memory: MemoryRegime,
    relational_sources: RelationalSources = "causal_prefix",
) -> ArchitectureID:
    """Map a bounded memory regime and its attention to the carrier identity."""
    if memory not in ("segment", "summary"):
        raise ContractError(f"{memory!r} is not a segment or summary regime.")
    if relational_sources == "timestep_records":
        if attention != "dat":
            raise ContractError(
                "The timestep-record route belongs to dual attention only."
            )
        return DAT_SUMMARY_V2_ARCHITECTURE_ID
    if attention == "ordinary":
        return SUMMARY_ARCHITECTURE_ID
    if attention == "dat":
        return DAT_SUMMARY_ARCHITECTURE_ID
    if attention == "dual_content":
        return DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID
    raise ContractError(f"No summary carrier runs {attention!r} attention.")


__all__ = [
    "ALL_CONDITIONS",
    "ALL_CONDITION_LABELS",
    "ARCHITECTURE_LABELS",
    "CACHE_ELEMENT_BYTES",
    "CONDITIONS",
    "CONDITION_LABELS",
    "CONTROL_HEAD_WIDTHS",
    "DAT_ARCHITECTURE_ID",
    "DAT_CONDITIONS",
    "DAT_CONDITION_LABELS",
    "DAT_SUMMARY_ARCHITECTURE_ID",
    "DUAL_CONTENT_ARCHITECTURE_ID",
    "DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID",
    "FEEDFORWARD_ARCHITECTURE_ID",
    "GRU_HISTORY_ARCHITECTURE_ID",
    "HISTORY_ARCHITECTURE_ID",
    "HISTORY_CONDITIONS",
    "MEMO_ARCHITECTURE_ID",
    "MEMO_CONDITIONS",
    "MEMO_CONDITION_LABELS",
    "SUMMARY_ARCHITECTURE_ID",
    "SUMMARY_CONDITIONS",
    "SUMMARY_CONDITION_LABELS",
    "WINDOW_ARCHITECTURE_ID",
    "AnyCondition",
    "ArchitectureID",
    "AttentionBackend",
    "AttentionVariant",
    "Condition",
    "ConditionSpec",
    "ContractError",
    "DATCondition",
    "DATMode",
    "DATSpec",
    "Device",
    "Evidence",
    "MemoCarry",
    "MemoCondition",
    "MemoPosition",
    "MemoSpec",
    "MemoryRegime",
    "PositionMethod",
    "RelationActivation",
    "RelationalBackend",
    "RelationalSources",
    "ResultValidationError",
    "StateMatch",
    "SummaryCondition",
    "SummaryDetach",
    "SummaryPosition",
    "SummaryRegime",
    "SummaryRewrite",
    "SummarySpec",
    "TrainingSegmentation",
    "WindowSpec",
    "Writer",
    "architecture_label",
    "architecture_uses_dat",
    "architecture_uses_history_packet",
    "architecture_uses_memo",
    "architecture_uses_summary",
    "arrays_equal",
    "attention_parameter_count",
    "cache_slot_floats",
    "capacity_matched_control_dims",
    "condition_label",
    "condition_spec",
    "control_parameter_match",
    "dat_architecture_id",
    "match_window_to_summary",
    "memo_live_slots",
    "memo_live_state_bytes",
    "memo_state_bytes",
    "summary_architecture_id",
    "summary_state_bytes",
    "window_state_bytes",
]
