"""Environment contracts, study rosters, and the resolver that joins them.

A *contract* (``configs/environments/<name>.yaml``) declares one environment
protocol: its recipe, training schedule, evaluation rosters and qualification
thresholds. A *study* (``configs/<study>.yaml``) names the contracts it runs,
the conditions it compares, its seeds and the shared model backbone.
:func:`experiment_config` joins one contract, one study, one condition and one
seed into the resolved :class:`ExperimentConfig` the runtime consumes, so there
is never a second authored copy of a recipe that could drift.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, cast

from reasoned_icrl.environments import ENVIRONMENT_NAMES
from reasoned_icrl.environments.xland_minigrid import XLAND_HORIZON
from reasoned_icrl.experiments.config import (
    CONFIG_VERSION,
    EXPERIMENT_ID_PATTERN,
    EnvironmentConfig,
    ExperimentConfig,
    TrackingConfig,
    TrainingConfig,
    environment_config,
    mapping,
    model_config,
    read_yaml,
    smoke_profile,
    training_config,
)
from reasoned_icrl.experiments.contracts import ALL_CONDITIONS, ContractError
from reasoned_icrl.experiments.environments import roster as environment_roster
from reasoned_icrl.experiments.records import (
    COUNT_METRICS,
    PRIMARY_METRICS,
    RETENTIONS,
    Retention,
)
from reasoned_icrl.experiments.xland_one_rule import (
    LAYOUT_FIXTURES,
    XLAND_ONE_RULE_HORIZON,
)

RETENTION_KINDS = frozenset({"attempt", "decision"})
"""Event kinds whose contracts may declare ``complete`` retention (R4): every
attempt of the task is exported, the partial one included. Query and episode
records already hold every scored unit."""

CONTRACT_CHECKPOINT_RULES = frozenset(
    {
        "best-development-primary-ties-earliest",
        "collection-endpoint-primary-development-selected-secondary",
    }
)
"""How a contract chooses the checkpoint its panels report.

The legacy rule takes the best development-primary checkpoint, ties earliest.
The 8M study makes the weights at the common collection endpoint primary and
keeps the development-selected checkpoint as a separate companion panel, so a
cell cannot be represented by whichever of the two happens to look better. Both
panels stay separate even when the two checkpoints coincide."""

EVENT_KINDS = {
    "darkroom": "attempt",
    "dark_key_to_door": "attempt",
    "count_recall": "query",
    "mazerunner": "episode",
    "concentration": "episode",
    "xland_minigrid": "attempt",
    "xland_one_rule": "attempt",
    "match_pattern": "decision",
    "tmaze": "episode",
}
"""The evaluation unit each environment scores."""


def _text(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{key} must be nonempty text.")
    return value


def _integer(raw: Mapping[str, object], key: str, minimum: int = 1) -> int:
    value = raw.get(key)
    if type(value) is not int or value < minimum:
        raise ContractError(f"{key} must be an integer >= {minimum}.")
    return value


def _real(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if type(value) not in (int, float):
        raise ContractError(f"{key} must be numeric.")
    result = float(cast(float, value))
    if not math.isfinite(result):
        raise ContractError(f"{key} must be finite.")
    return result


def _strings(raw: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ContractError(f"{key} must be a nonempty list.")
    if any(not isinstance(v, str) or not v for v in value):
        raise ContractError(f"{key} must contain nonempty text.")
    return tuple(value)


def _seeds(raw: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ContractError(f"{key} must be a nonempty seed roster.")
    if any(type(v) is not int or v < 0 for v in value):
        raise ContractError(f"{key} contains invalid seeds.")
    if len(set(value)) != len(value):
        raise ContractError(f"{key} contains duplicate seeds.")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class EvaluationSplit:
    """One evaluation roster: ``count`` tasks of the environment split ``source``."""

    source: str
    count: int
    offset: int = 0


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    splits: Mapping[str, EvaluationSplit]
    rollout_seeds: tuple[int, ...]
    primary_metric: str
    sampling_unit: str
    checkpoint_rule: str
    retention: Retention = "scored-band"
    """How the evaluator exports events (R4): the legacy scored band, or every
    event of the task with the partial one marked."""

    @property
    def panel_rules(self) -> tuple[str, ...]:
        """The checkpoint panels the contract reports, primary first."""
        if self.checkpoint_rule == (
            "collection-endpoint-primary-development-selected-secondary"
        ):
            return ("endpoint", "selected")
        return ("selected", "final-epoch")


@dataclass(frozen=True, slots=True)
class QualificationPlan:
    reference: str
    minimum_improvement: float
    maximum_primary: float
    sustained_checkpoints: int
    maximum_fits: int
    maximum_device_hours: float
    fallback: str
    evidence_diagnostic: str


@dataclass(frozen=True, slots=True)
class BenchmarkContract:
    """One environment protocol with everything a study needs to run it."""

    protocol: str
    status: str
    environment: EnvironmentConfig
    training: TrainingConfig
    evaluation: EvaluationPlan
    qualification: QualificationPlan
    documentation: Mapping[str, object]
    memory: Mapping[str, Mapping[str, object]] | None = None
    """Per-protocol overrides of the study's ``model.summary``/``model.window``
    blocks. Merged over the study block before the condition's identity is
    resolved, so the resolved config and its sha carry the effective values.
    The bounded capacity is fixed here per environment (R4 retired the former
    ``capacity`` development-selection grids); there is no sweep."""

    @property
    def name(self) -> str:
        return self.environment.name

    @property
    def event_kind(self) -> str:
        """The evaluation unit: the environment's, except that a MazeRunner
        task replayed in laps (``meta_horizon``, evaluation only) scores one
        attempt record per lap, as Key-to-Door does."""
        if (
            self.environment.name == "mazerunner"
            and self.environment.meta_horizon is not None
        ):
            return "attempt"
        return EVENT_KINDS[self.environment.name]

    def roster(self, split: str) -> tuple[int, ...]:
        """The concrete task identities of one declared evaluation split."""
        try:
            spec = self.evaluation.splits[split]
        except KeyError as error:
            raise ContractError(
                f"{self.protocol} declares no {split!r} split."
            ) from error
        selected = environment_roster(
            {"environment": asdict(self.environment)},
            spec.source,
            task_count=spec.count,
            offset=spec.offset,
        )
        return tuple(int(value) for value in selected)


MEMORY_BLOCKS = ("summary", "window", "memo", "critic")
"""The study model blocks a contract may override per protocol: the memory
settings (segment length, memory tokens, window, the Memo comparator's segment
and summary sizes) and, since decision 15, the two-hot critic support an
environment's reward scale needs."""


def _memory(value: object) -> Mapping[str, Mapping[str, object]] | None:
    """The contract's optional ``memory`` section: a model block per name."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ContractError("Contract memory must be a mapping of blocks.")
    unknown = set(value) - set(MEMORY_BLOCKS)
    if unknown:
        raise ContractError(f"Unknown contract memory blocks: {sorted(unknown)}.")
    blocks: dict[str, Mapping[str, object]] = {}
    for name, block in value.items():
        if not isinstance(block, Mapping) or not block:
            raise ContractError(f"Contract memory.{name} must be a non-empty mapping.")
        blocks[str(name)] = {str(key): item for key, item in block.items()}
    return blocks


def study_model(contract: BenchmarkContract, study: Study) -> dict[str, object]:
    """The study's model block with the contract's memory overrides applied."""
    model = dict(study.model)
    for name, override in (contract.memory or {}).items():
        base = model.get(name)
        if base is not None and not isinstance(base, Mapping):
            raise ContractError(f"Study model.{name} must be a mapping.")
        model[name] = {**(dict(base) if base is not None else {}), **override}
    return model


def load_contract(path: str | Path) -> BenchmarkContract:
    """Load and validate one environment contract."""
    raw = read_yaml(path, name="environment contract")
    expected = {
        "protocol",
        "status",
        "environment",
        "training",
        "evaluation",
        "qualification",
        "documentation",
    }
    if "capacity" in raw:
        raise ContractError(
            "Contract capacity grids were retired in R4: the bounded capacity is "
            "fixed in memory.summary/memory.window with no development selection."
        )
    if set(raw) - {"memory"} != expected:
        raise ContractError(
            f"Unexpected/missing contract sections: {set(raw) ^ expected}"
        )
    memory = _memory(raw.get("memory"))
    protocol = _text(raw, "protocol")
    if not EXPERIMENT_ID_PATTERN.fullmatch(protocol):
        raise ContractError("Protocol must be a lowercase hyphenated name.")
    status = _text(raw, "status")
    if status not in ("C0", "C1"):
        # C1 is earned by the environment's own execution tests. C2--C4 depend
        # on measured learning and reporting, which no manifest can assert.
        raise ContractError("Contracts cannot assert unverified C2--C4 status.")
    environment = environment_config(mapping(raw["environment"], "environment"))
    if environment.benchmark != protocol:
        raise ContractError("Contract protocol disagrees with its environment.")
    training_raw = mapping(raw["training"], "training")
    unknown = set(training_raw) - {f.name for f in fields(TrainingConfig)}
    if unknown:
        raise ContractError(f"Unknown training fields: {sorted(unknown)}")
    training = training_config(training_raw)
    if training.epsilon_anneal_steps > training.epochs * training.timesteps_per_epoch:
        raise ContractError("epsilon_anneal_steps exceeds the actor-local budget.")
    if training.max_sequence_length < environment.outer_length:
        raise ContractError("Context would drop the task prefix.")
    if training.trajectory_length < environment.outer_length:
        raise ContractError("Replay files would split a task.")
    if not 0 <= training.start_learning_epoch < training.epochs:
        raise ContractError("Training must contain learner updates.")
    for value in (
        training.learning_rate,
        training.reward_multiplier,
        training.gradient_clip,
        training.weight_decay,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ContractError("Numerical settings must be finite and positive.")
    ev = mapping(raw["evaluation"], "evaluation")
    splits = {
        str(name): EvaluationSplit(
            _text(spec, "source"),
            _integer(spec, "count"),
            _integer({"offset": spec.get("offset", 0)}, "offset", 0),
        )
        for name, spec in mapping(ev.get("splits"), "evaluation.splits").items()
    }
    if "development" not in splits:
        raise ContractError("Every contract declares a development split.")
    retention = cast(Retention, str(ev.get("retention", "scored-band")))
    if retention not in RETENTIONS:
        raise ContractError(f"Unknown event retention: {retention!r}.")
    if (
        retention == "complete"
        and environment.name not in ("concentration", "count_recall")
        and EVENT_KINDS[environment.name] not in RETENTION_KINDS
    ):
        raise ContractError(
            f"{environment.name} records already hold every scored unit; "
            "complete retention applies to attempts, flips or CountRecall queries."
        )
    evaluation = EvaluationPlan(
        splits,
        _seeds(ev, "rollout_seeds"),
        _text(ev, "primary_metric"),
        _text(ev, "sampling_unit"),
        _text(ev, "checkpoint_rule"),
        retention,
    )
    if evaluation.primary_metric not in PRIMARY_METRICS[environment.name]:
        raise ContractError("Primary metric disagrees with the environment.")
    if evaluation.primary_metric in COUNT_METRICS and retention != "complete":
        raise ContractError(
            f"{evaluation.primary_metric} counts every attempt, so the contract "
            "must retain complete events."
        )
    if environment.name == "dark_key_to_door" and environment.attempts != 8:
        raise ContractError("first8 endpoint requires eight scored attempts.")
    if environment.name == "xland_minigrid":
        if environment.attempts != 5 or environment.scored_from != 4:
            raise ContractError("success_last2 scores attempts 4-5 of five.")
        if environment.horizon != XLAND_HORIZON:
            raise ContractError("The XLand contract keeps the native episode limit.")
    if environment.name == "xland_one_rule":
        if environment.attempts != 5 or environment.scored_from != 4:
            raise ContractError("success_last2 scores attempts 4-5 of five.")
        if environment.horizon != XLAND_ONE_RULE_HORIZON:
            raise ContractError("The one-rule contract fixes 128 actions per attempt.")
        if len(evaluation.rollout_seeds) > len(LAYOUT_FIXTURES) or any(
            seed not in LAYOUT_FIXTURES for seed in evaluation.rollout_seeds
        ):
            raise ContractError("One-rule rollout seeds are the declared layout roots.")
    if evaluation.checkpoint_rule not in CONTRACT_CHECKPOINT_RULES:
        raise ContractError("Unknown checkpoint-selection rule.")
    q = mapping(raw["qualification"], "qualification")
    qualification = QualificationPlan(
        _text(q, "reference"),
        _real(q, "minimum_improvement"),
        _real(q, "maximum_primary"),
        _integer(q, "sustained_checkpoints"),
        _integer(q, "maximum_fits"),
        _real(q, "maximum_device_hours"),
        _text(q, "fallback"),
        _text(q, "evidence_diagnostic"),
    )
    if not (
        0 < qualification.minimum_improvement < 1
        and 0 < qualification.maximum_primary < 1
        and qualification.maximum_device_hours > 0
    ):
        raise ContractError("Invalid bounded qualification criteria.")
    documentation = dict(mapping(raw["documentation"], "documentation"))
    for key in ("source", "information", "lifecycle"):
        mapping(documentation.get(key), f"documentation.{key}")
    information = mapping(documentation["information"], "documentation.information")
    policy = _strings(information, "policy_fields")
    excluded = _strings(information, "excluded_fields")
    if set(policy) & set(excluded):
        raise ContractError("Policy and excluded fields overlap.")
    contract = BenchmarkContract(
        protocol,
        status,
        environment,
        training,
        evaluation,
        qualification,
        documentation,
        memory=memory,
    )
    rosters = [set(contract.roster(name)) for name in splits]
    if any(a & b for index, a in enumerate(rosters) for b in rosters[index + 1 :]):
        raise ContractError("Evaluation task rosters overlap.")
    return contract


@dataclass(frozen=True, slots=True)
class TierContrast:
    """One predeclared contrast of a tier, ``left - right`` in the units of the
    contract's primary metric."""

    name: str
    left: str
    right: str

    def __post_init__(self) -> None:
        if not self.name.strip() or self.left == self.right:
            raise ContractError("A tier contrast names two different cells.")


TierGroup = Literal["primary", "supplementary"]


@dataclass(frozen=True, slots=True)
class TierPlan:
    """One environment's tier-specific groups, reference and contrasts (R4).

    ``primary`` is the group whose three-seed completion delivers the tier's
    report; ``supplementary`` is a separately tracked lower-priority panel that
    never blocks the primary readout and is never marked complete while unrun;
    ``qualification_reference`` is the primary cell whose C2/C3 qualify the
    environment; ``practical_effect`` is the tier's delta in the units of its
    contract's primary metric; ``contract_pending`` marks a tier declared before
    its contract joins the study (CountRecallMedium until R8). Cells outside
    ``cells`` are *outside scope* on this environment, not missing runs.
    """

    environment: str
    tier: int
    purpose: str
    primary: tuple[str, ...]
    supplementary: tuple[str, ...]
    qualification_reference: str
    practical_effect: float
    primary_contrasts: tuple[TierContrast, ...]
    companion_contrasts: tuple[TierContrast, ...] = ()
    contract_pending: bool = False
    reference_cells: tuple[str, ...] = ()
    reference_root: str | None = None
    """Frozen reference cells of a comparator study (the Memo comparator, ME0):
    cells whose endpoint policies were fitted under another study root
    (``reference_root``) and are read from there, paired with this tier's
    fits on the same rosters. They are never fits of this root, never planned
    or launched here, and may carry the qualification the other study
    established; a contrast may name them."""

    def __post_init__(self) -> None:
        if self.environment not in ENVIRONMENT_NAMES:
            raise ContractError(f"Unknown tier environment: {self.environment!r}.")
        if self.tier < 0:
            raise ContractError("A tier index is a non-negative integer.")
        if not self.primary:
            raise ContractError(f"Tier {self.environment} declares no primary cell.")
        if len(set(self.compared_cells)) != len(self.compared_cells):
            raise ContractError(
                f"Tier {self.environment} lists a cell twice across its groups."
            )
        for name in self.compared_cells:
            if name not in ALL_CONDITIONS:
                raise ContractError(f"Unknown tier condition: {name!r}.")
        if bool(self.reference_cells) != (self.reference_root is not None):
            raise ContractError(
                f"Tier {self.environment}: reference cells and their root are "
                "declared together."
            )
        if self.qualification_reference not in self.primary + self.reference_cells:
            raise ContractError(
                f"Tier {self.environment}: the qualification reference "
                f"{self.qualification_reference!r} must be a primary cell (or a "
                "frozen reference cell of a comparator study)."
            )
        if not math.isfinite(self.practical_effect) or self.practical_effect <= 0:
            raise ContractError("A tier's practical effect is a positive number.")
        if not self.primary_contrasts:
            raise ContractError(f"Tier {self.environment} declares no contrast.")
        for contrast in self.contrasts:
            for name in (contrast.left, contrast.right):
                if name not in self.compared_cells:
                    raise ContractError(
                        f"Tier {self.environment}: contrast {contrast.name!r} "
                        f"names {name!r}, which is outside its groups."
                    )
        for contrast in self.primary_contrasts:
            if any(
                name in self.supplementary for name in (contrast.left, contrast.right)
            ):
                raise ContractError(
                    f"Tier {self.environment}: primary contrast {contrast.name!r} "
                    "may not depend on a supplementary cell."
                )

    @property
    def cells(self) -> tuple[str, ...]:
        """Every cell that may run on this environment, primary first."""
        return self.primary + self.supplementary

    @property
    def compared_cells(self) -> tuple[str, ...]:
        """The fits of this root plus the frozen reference cells read elsewhere."""
        return self.cells + self.reference_cells

    @property
    def contrasts(self) -> tuple[TierContrast, ...]:
        return self.primary_contrasts + self.companion_contrasts

    def role_of(self, condition: str) -> str:
        """``primary``, ``supplementary`` or ``reference`` (a frozen cell read
        from ``reference_root``); a cell outside the tier is refused."""
        if condition in self.reference_cells:
            return "reference"
        return self.group_of(condition)

    def group_of(self, condition: str) -> TierGroup:
        if condition in self.primary:
            return "primary"
        if condition in self.supplementary:
            return "supplementary"
        if condition in self.reference_cells:
            raise ContractError(
                f"{condition!r} is a frozen reference cell of the "
                f"{self.environment} tier, read from {self.reference_root}; it "
                "is not fitted under this study root."
            )
        raise ContractError(
            f"{condition!r} is outside the {self.environment} tier's groups "
            f"({', '.join(self.cells)}); it is not a missing run there."
        )

    def fits(self, seeds: Sequence[int], group: str = "primary") -> int:
        """How many fits the named group needs at the given seeds."""
        cells = {
            "primary": self.primary,
            "supplementary": self.supplementary,
            "all": self.cells,
        }[group]
        return len(cells) * len(seeds)


@dataclass(frozen=True, slots=True)
class Study:
    """One experiment set: which contracts, conditions and seeds it runs.

    ``conditions`` is the union roster every cell name must resolve through;
    a study with ``tiers`` additionally says, per environment, which of those
    cells run there and in which group (R4). An untiered study runs every
    condition on every contract, as the retired 4M roster did.
    """

    name: str
    contracts: tuple[BenchmarkContract, ...]
    conditions: tuple[str, ...]
    training_seeds: tuple[int, ...]
    model: Mapping[str, object]
    pilot_seeds: tuple[int, ...] = ()
    control_conditions: tuple[str, ...] = ()
    control_protocol: str | None = None
    output_root: str = "outputs"
    tiers: Mapping[str, TierPlan] = field(default_factory=dict)

    def contract(self, name: str) -> BenchmarkContract:
        """Resolve one environment by name or protocol."""
        for contract in self.contracts:
            if name in (contract.environment.name, contract.protocol):
                return contract
        raise ContractError(f"Unknown environment for {self.name}: {name!r}.")

    @property
    def all_conditions(self) -> tuple[str, ...]:
        return self.conditions + self.control_conditions

    @property
    def tiered(self) -> bool:
        return bool(self.tiers)

    def tier(self, name: str) -> TierPlan:
        """The tier plan of one environment, by environment name or protocol."""
        if name in self.tiers:
            return self.tiers[name]
        for contract in self.contracts:
            if contract.protocol == name and contract.environment.name in self.tiers:
                return self.tiers[contract.environment.name]
        raise ContractError(f"{self.name} declares no tier for {name!r}.")

    def cells(self, contract: BenchmarkContract) -> tuple[str, ...]:
        """The cells that may run on ``contract``: its tier's groups, or the
        whole roster for an untiered study."""
        if self.tiered:
            return self.tier(contract.environment.name).cells
        return self.all_conditions

    def cell_root(
        self, contract: BenchmarkContract, condition: str, root: str | Path
    ) -> Path:
        """Where ``condition``'s runs on ``contract`` live: the study root, or
        the tier's ``reference_root`` for a frozen reference cell (a relative
        reference root is resolved against the repository, as the study's own
        ``output_root`` is)."""
        if self.tiered:
            plan = self.tier(contract.environment.name)
            if condition in plan.reference_cells:
                assert plan.reference_root is not None
                reference = Path(plan.reference_root)
                if not reference.is_absolute():
                    from reasoned_icrl.utils import repository_root

                    reference = repository_root() / reference
                return reference
        return Path(root)

    def compared_cells(self, contract: BenchmarkContract) -> tuple[str, ...]:
        """``cells`` plus the frozen reference cells of a comparator tier."""
        if self.tiered:
            return self.tier(contract.environment.name).compared_cells
        return self.all_conditions

    def primary_cells(self, contract: BenchmarkContract) -> tuple[str, ...]:
        """The cells whose completion the environment's report requires."""
        if self.tiered:
            return self.tier(contract.environment.name).primary
        return self.all_conditions


def _string_list(raw: Mapping[str, object], key: str) -> tuple[str, ...]:
    """A possibly empty list of nonempty strings."""
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise ContractError(f"{key} must be a list.")
    if any(not isinstance(v, str) or not v for v in value):
        raise ContractError(f"{key} must contain nonempty text.")
    return tuple(value)


def _contrasts(raw: Mapping[str, object], key: str) -> tuple[TierContrast, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise ContractError(f"{key} must be a list of contrasts.")
    output: list[TierContrast] = []
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "left", "right"}:
            raise ContractError(f"{key} entries are {{name, left, right}} mappings.")
        mapping_entry = cast(Mapping[str, object], entry)
        output.append(
            TierContrast(
                _text(mapping_entry, "name"),
                _text(mapping_entry, "left"),
                _text(mapping_entry, "right"),
            )
        )
    return tuple(output)


TIER_KEYS = frozenset(
    {
        "tier",
        "purpose",
        "contract",
        "primary",
        "supplementary",
        "qualification_reference",
        "practical_effect",
        "primary_contrasts",
        "companion_contrasts",
        "reference_cells",
        "reference_root",
    }
)


def _tiers(
    value: object,
    *,
    conditions: Sequence[str],
    contracts: Sequence[BenchmarkContract],
) -> dict[str, TierPlan]:
    """The study's per-environment tier plans, checked against its roster."""
    if value is None:
        return {}
    raw = mapping(value, "tiers")
    present = {contract.environment.name for contract in contracts}
    plans: dict[str, TierPlan] = {}
    for environment, entry in raw.items():
        block = mapping(entry, f"tiers.{environment}")
        unknown = set(block) - TIER_KEYS
        if unknown:
            raise ContractError(f"Unknown tiers.{environment} keys: {sorted(unknown)}.")
        contract_state = block.get("contract", "present")
        if contract_state not in ("present", "pending"):
            raise ContractError(
                f"tiers.{environment}.contract is 'pending' or omitted."
            )
        pending = contract_state == "pending"
        if pending == (str(environment) in present):
            raise ContractError(
                f"tiers.{environment}: a tier is 'pending' exactly when the study "
                "has no contract for its environment."
            )
        plan = TierPlan(
            environment=str(environment),
            tier=_integer(block, "tier", 0),
            purpose=_text(block, "purpose"),
            primary=_strings(block, "primary"),
            supplementary=_string_list(block, "supplementary"),
            qualification_reference=_text(block, "qualification_reference"),
            practical_effect=_real(block, "practical_effect"),
            primary_contrasts=_contrasts(block, "primary_contrasts"),
            companion_contrasts=_contrasts(block, "companion_contrasts"),
            contract_pending=pending,
            reference_cells=_string_list(block, "reference_cells"),
            reference_root=(
                None
                if block.get("reference_root") is None
                else _text(block, "reference_root")
            ),
        )
        outside = [name for name in plan.compared_cells if name not in conditions]
        if outside:
            raise ContractError(
                f"tiers.{environment} names cells outside the study roster: {outside}."
            )
        plans[str(environment)] = plan
    missing = sorted(present - set(plans))
    if missing:
        raise ContractError(
            f"A tiered study declares a tier for every contract; missing {missing}."
        )
    return plans


def load_study(path: str | Path) -> Study:
    """Load one study roster and every contract it names."""
    source = Path(path)
    raw = read_yaml(source, name="study")
    contracts = tuple(
        load_contract(source.parent / entry) for entry in _strings(raw, "contracts")
    )
    if len({c.protocol for c in contracts}) != len(contracts):
        raise ContractError("Duplicate environment protocol in study.")
    conditions = _strings(raw, "conditions")
    controls = (
        _strings(raw, "control_conditions") if raw.get("control_conditions") else ()
    )
    for name in conditions + controls:
        if name not in ALL_CONDITIONS:
            raise ContractError(f"Unknown study condition: {name!r}.")
    model = mapping(raw.get("model"), "model")
    for condition in conditions + controls:
        model_config(model, condition=condition)
    pilots = _seeds(raw, "pilot_seeds") if raw.get("pilot_seeds") else ()
    training = _seeds(raw, "training_seeds")
    if set(pilots) & set(training):
        raise ContractError("Pilot and final training seeds overlap.")
    # The control protocol is a preselection; its contract may not exist yet.
    control_protocol = raw.get("control_protocol")
    return Study(
        name=_text(raw, "name"),
        contracts=contracts,
        conditions=conditions,
        training_seeds=training,
        model=model,
        pilot_seeds=pilots,
        control_conditions=controls,
        control_protocol=None if control_protocol is None else str(control_protocol),
        output_root=str(raw.get("output_root", "outputs")),
        tiers=_tiers(
            raw.get("tiers"), conditions=conditions + controls, contracts=contracts
        ),
    )


def experiment_config(
    contract: BenchmarkContract,
    study: Study,
    *,
    condition: str,
    seed: int,
    repository: str | Path,
    device: str = "auto",
    output_root: str | Path | None = None,
    smoke: bool = False,
    wandb: bool = False,
) -> ExperimentConfig:
    """Resolve one contract and condition into a runnable experiment.

    The contract owns the task, budget and evaluation roster (and may override
    the study's memory blocks); the study owns the shared backbone and the
    condition roster. Nothing here invents a training setting that the two
    files do not already declare. ``wandb`` mirrors the local telemetry to
    Weights & Biases (project ``REASONED_ICRL_WANDB_PROJECT`` or
    ``reasoned-icrl``, group ``<study>/<protocol>``); a smoke profile never
    opens a run.
    """
    if condition not in study.all_conditions and not (
        contract.name == "count_recall" and condition == "feedforward"
    ):
        raise ContractError(f"Condition {condition!r} is outside the study roster.")
    if study.tiered and condition != "feedforward":
        # R4: a cell runs on an environment only where its tier declares it;
        # an undeclared cell is outside scope there, not a missing run.
        study.tier(contract.name).group_of(condition)
    root = Path(repository).resolve()
    environment = contract.environment
    model = model_config(study_model(contract, study), condition=condition)
    training = contract.training
    tracking = TrackingConfig(
        wandb=wandb,
        project=os.environ.get("REASONED_ICRL_WANDB_PROJECT", "reasoned-icrl"),
        group=f"{study.name}/{contract.protocol}",
        log_interval=16,
    )
    if smoke:
        environment, model, training, tracking = smoke_profile(
            environment, model, training, tracking, device=device
        )
        if wandb:
            # Explicit tracking opt-in survives the otherwise untracked smoke
            # profile, in a distinct group from the twelve primary fits.
            tracking = replace(tracking, wandb=True, group=f"{tracking.group}/smoke")
    if condition == "feedforward":
        # Checkpoint selection needs the scheduled epochs; the ordinary
        # feedforward cleanup would otherwise remove the entire ckpts tree.
        training = replace(training, retain_training_state=True)
    return ExperimentConfig(
        version=CONFIG_VERSION,
        experiment=contract.protocol,
        condition=cast(Any, condition),
        seed=seed,
        device=cast(Any, device),
        output_root=(
            root / study.output_root
            if output_root is None
            else Path(output_root).resolve()
        ),
        repository=root,
        environment=environment,
        model=model,
        training=training,
        tracking=tracking,
        smoke=smoke,
    )


def saved_config(config: ExperimentConfig) -> ExperimentConfig:
    """Evaluate the actual saved recipe, never retroactively apply a new one."""
    from reasoned_icrl.experiments.config import load_resolved_config

    saved = load_resolved_config(
        config.run_directory / "config.yaml", repository=config.repository
    )
    saved_identity = (
        saved.condition,
        saved.seed,
        saved.experiment,
        saved.environment.name,
    )
    requested = (
        config.condition,
        config.seed,
        config.experiment,
        config.environment.name,
    )
    if saved_identity != requested:
        raise ContractError(
            "Saved checkpoint identity/environment does not match the requested run."
        )
    return replace(saved, device=config.device, output_root=config.output_root)


__all__ = [
    "COUNT_METRICS",
    "EVENT_KINDS",
    "MEMORY_BLOCKS",
    "PRIMARY_METRICS",
    "RETENTION_KINDS",
    "BenchmarkContract",
    "EvaluationPlan",
    "EvaluationSplit",
    "QualificationPlan",
    "Study",
    "TierContrast",
    "TierGroup",
    "TierPlan",
    "experiment_config",
    "load_contract",
    "load_study",
    "saved_config",
    "study_model",
]
