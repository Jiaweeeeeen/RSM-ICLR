"""The fully resolved configuration saved with every run.

A run is (environment recipe, model recipe, training recipe, condition, seed).
Studies build an :class:`ExperimentConfig` from their contract and study YAML
files; the runtime consumes only :meth:`ExperimentConfig.as_runtime_mapping`.
``config.yaml`` in a run directory is the resolved dump and is reloaded by
:func:`load_resolved_config` for evaluation and resume.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from reasoned_icrl.environments import ENVIRONMENT_NAMES
from reasoned_icrl.environments.concentration import (
    CONCENTRATION_CARDS,
    CONCENTRATION_HORIZONS,
    concentration_variant,
)
from reasoned_icrl.environments.count_recall import (
    COUNT_RECALL_CATEGORIES,
    COUNT_RECALL_HORIZONS,
    count_recall_variant,
)
from reasoned_icrl.environments.dark_key_to_door import KEY_TO_DOOR_PROTOCOL
from reasoned_icrl.environments.darkroom import (
    DARKROOM_GOAL_PARTITION,
    DARKROOM_PROTOCOL,
)
from reasoned_icrl.environments.mazerunner import mazerunner_variant
from reasoned_icrl.environments.tmaze import (
    TMAZE_CORRIDOR,
    TMAZE_V3_PROTOCOL,
    tmaze_horizon,
)
from reasoned_icrl.environments.xland_minigrid import (
    XLAND_ATTEMPTS,
    XLAND_GRID,
    XLAND_HORIZON,
    XLAND_PROTOCOL,
    XLAND_SCORED_FROM,
)
from reasoned_icrl.experiments.contracts import (
    ALL_CONDITIONS,
    DAT_WINDOW_ARCHITECTURE_ID,
    FEEDFORWARD_ARCHITECTURE_ID,
    GRU_HISTORY_ARCHITECTURE_ID,
    HISTORY_ARCHITECTURE_ID,
    MEMO_ARCHITECTURE_ID,
    REVISED_CONDITIONS,
    WINDOW_ARCHITECTURE_ID,
    AnyCondition,
    ArchitectureID,
    AttentionBackend,
    ConditionSpec,
    ContractError,
    DATSpec,
    Device,
    MemoSpec,
    StateMatch,
    SummarySpec,
    WindowSpec,
    architecture_label,
    architecture_uses_history_packet,
    capacity_matched_control_dims,
    condition_label,
    dat_architecture_id,
    match_window_to_summary,
    summary_architecture_id,
)
from reasoned_icrl.experiments.xland_one_rule import (
    XLAND_ONE_RULE_ATTEMPTS,
    XLAND_ONE_RULE_HORIZON,
    XLAND_ONE_RULE_PROTOCOL,
    XLAND_ONE_RULE_SCORED_FROM,
    CurriculumSchedule,
)

KEY_TO_DOOR_PROTOCOL_1000 = "native-keydoor-fixed1000-first8"
"""The same Key-to-Door task family at a 1,000-call outer budget: the
in-distribution long-horizon control declared (split
record). The environment class keeps its 500-call protocol attribute as
self-consistency metadata; the contract names the budget and the run records
carry it."""
KEY_TO_DOOR_PROTOCOLS: tuple[str, ...] = (
    KEY_TO_DOOR_PROTOCOL,
    KEY_TO_DOOR_PROTOCOL_1000,
)

CONFIG_VERSION = "0.8.0"
EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DEVICES = ("auto", "cpu", "mps", "cuda")
_LEGACY_ENVIRONMENT_NAMES = {"dark_key_to_door_native": "dark_key_to_door"}
"""Environment identities recorded by runs trained before the layout refactor."""

EnvironmentName = Literal[
    "darkroom",
    "dark_key_to_door",
    "count_recall",
    "mazerunner",
    "concentration",
    "xland_minigrid",
    "xland_one_rule",
    "match_pattern",
    "tmaze",
]
XLAND_NAMES = ("xland_minigrid", "xland_one_rule")
"""The environments that play episodes of the pinned XLand simulator with a
reset-only decision between them and read packets with the xland trunk."""
TimestepEncoder = Literal["ff", "xland"]
"""The timestep trunk an environment's packets are read with: the shared
feed-forward trunk, or the XLand grid trunk that reproduces AMAGO's example."""


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """One environment recipe. ``benchmark`` is the environment's protocol."""

    name: EnvironmentName
    benchmark: str
    size: int
    attempts: int
    horizon: int
    parallel_envs: int
    goal_partition: Literal["quadrant-v1"] | None = None
    meta_horizon: int | None = None
    """Dark Key-to-Door's native meta budget in charged calls; on MazeRunner the
    evaluation-only repeated-laps budget (the trained task replayed on the same map
    until this many charged calls, derived by the horizon adapter, never declared by a
    study); ``None`` elsewhere."""
    goals: int | None = None
    randomized_actions: bool = False
    scored_from: int = 1
    encoder: TimestepEncoder = "ff"
    curriculum: Mapping[str, float] | None = None
    """The one-rule XLand training curriculum (global charged-call boundaries
    and the mixture probability); ``None`` for every other environment."""
    movement_penalty: float | None = None
    """The passive T-Maze's movement penalty (AMAGO's ``-1 / corridor_length``, paid by
    every non-forward action before the final decision; v2 of the protocol); ``None``
    for every other environment."""
    training_corridors: tuple[int, ...] | None = None
    """The passive T-Maze v3 training draw: every training
    task's corridor is drawn from these lengths by its identity; evaluation
    keeps ``size``. ``None`` under v2 and for every other environment."""
    protocol_size: int | None = None
    """MazeRunner's larger-maze axis: the trained maze
    dimension behind ``benchmark`` when ``size`` is a larger evaluation-only
    maze (derived by the horizon adapter, never declared by a study);
    ``None`` when the maze played is the protocol's."""

    @property
    def longest_training_length(self) -> int:
        """Charged calls the learner may see in one training task: the outer
        length, under the v3 corridor draw the longest corridor plus its
        turn."""
        if self.training_corridors:
            return max(self.training_corridors) + 1
        return self.outer_length

    @property
    def outer_length(self) -> int:
        """Decisions in one outer task, whatever supplies its boundary.

        DarkRoom runs exactly ``attempts`` physical attempts of ``horizon``
        steps. The native Key-to-Door task instead runs a fixed meta budget in
        which its scored attempts are guaranteed to fit. CountRecall and
        MazeRunner run one native episode of ``horizon`` decisions.
        XLand-MiniGrid runs ``attempts`` episodes of at most ``horizon`` steps
        with one reset-only decision between consecutive episodes, and so does
        a CountRecall task continued over ``attempts`` deck pairs (the
        continued-stream axis, evaluation only; one pair in training).
        """
        if self.meta_horizon is not None:
            return self.meta_horizon
        if self.name in XLAND_NAMES or self.name == "count_recall":
            return self.attempts * (self.horizon + 1) - 1
        return self.attempts * self.horizon

    @property
    def scored_attempts(self) -> int:
        """How many attempts of the ``attempts`` budget the primary endpoint scores."""
        return self.attempts - self.scored_from + 1

    @property
    def has_fixed_native_horizon(self) -> bool:
        """A CountRecall stream is exactly as long as its deck; a Concentration
        board's flip budget is fixed by its deck as well."""
        return self.name in ("count_recall", "concentration", "match_pattern")


@dataclass(frozen=True, slots=True)
class CriticConfig:
    """Optional AMAGO two-hot return support."""

    min_return: float
    max_return: float
    output_bins: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_return) or not math.isfinite(self.max_return):
            raise ContractError("Critic return bounds must be finite.")
        if not self.min_return < self.max_return:
            raise ContractError("Critic min_return must be smaller than max_return.")
        if self.output_bins < 2:
            raise ContractError("Critic output_bins must be at least two.")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture_id: ArchitectureID
    width: int
    layers: int
    heads: int
    feedforward_multiplier: int
    attention_backend: AttentionBackend = "vanilla"
    critic: CriticConfig | None = None
    dat: DATSpec | None = None
    summary: SummarySpec | None = None
    window: WindowSpec | None = None
    window_match: StateMatch | None = None
    """The revised dual-attention window's measured state match (R3): the
    summary reference its ``W`` was chosen against and the residual bytes.
    ``None`` for every other cell, including the legacy ordinary window."""
    memo: MemoSpec | None = None
    """The Memo comparator's audited recipe (ME0); ``None`` on every other cell."""


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Local telemetry with optional W&B mirroring."""

    wandb: bool = False
    project: str = "reasoned-icrl"
    group: str | None = None
    log_interval: int = 300


@dataclass(frozen=True, slots=True)
class NativeAgentConfig:
    """Native AMAGO objective coefficients, validated once at construction."""

    critic_loss_weight: float = 10.0
    online_coeff: float = 1.0
    offline_coeff: float = 0.1
    popart: bool = True
    tau: float = 0.003
    n_step: int = 1

    def __post_init__(self) -> None:
        weights = (self.critic_loss_weight, self.online_coeff, self.offline_coeff)
        if any(not math.isfinite(v) or v < 0 for v in weights):
            raise ContractError("Native objective weights must be finite/nonnegative.")
        if type(self.popart) is not bool:
            raise ContractError("Native PopArt selection must be boolean.")
        if type(self.n_step) is not int or self.n_step < 1:
            raise ContractError("Native n-step horizon must be a positive integer.")
        if not 0 < self.tau <= 1:
            raise ContractError("Native target tau must be in (0, 1].")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    start_learning_epoch: int
    timesteps_per_epoch: int
    batches_per_epoch: int
    validation_timesteps: int
    validation_interval: int
    checkpoint_interval: int
    batch_size: int
    max_sequence_length: int
    trajectory_length: int
    replay_capacity: int
    reward_multiplier: float
    epsilon_anneal_steps: int
    learning_rate: float
    warmup_steps: int
    gradient_clip: float
    weight_decay: float
    mixed_precision: str
    dataloader_workers: int
    torch_compile: bool
    exploration: Literal[
        "epsilon_greedy", "bilevel_epsilon_greedy", "tmaze_epsilon_greedy"
    ]
    exploration_rollout_horizon: int
    verbose: bool
    retain_training_state: bool = False
    native_agent: NativeAgentConfig | None = None


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Fully resolved configuration saved with every run."""

    version: str
    experiment: str
    condition: AnyCondition
    seed: int
    device: Device
    output_root: Path
    repository: Path
    environment: EnvironmentConfig
    model: ModelConfig
    training: TrainingConfig
    tracking: TrackingConfig
    smoke: bool = False

    def __post_init__(self) -> None:
        collection = self.training.epochs * self.training.timesteps_per_epoch
        if self.training.epsilon_anneal_steps > collection:
            raise ContractError(
                "epsilon_anneal_steps counts vector decisions in each actor and "
                f"cannot exceed the configured per-actor collection budget "
                f"({collection})."
            )
        if architecture_uses_history_packet(self.model.architecture_id):
            outer = self.environment.longest_training_length
            if (
                self.training.max_sequence_length < outer
                or self.training.trajectory_length < outer
            ):
                raise ContractError(
                    "History conditions require complete outer-task replay prefixes."
                )

    @property
    def condition_spec(self) -> ConditionSpec:
        return ALL_CONDITIONS[self.condition]

    @property
    def run_directory(self) -> Path:
        return self.output_root / self.experiment / self.condition / f"seed-{self.seed}"

    def as_runtime_mapping(self) -> dict[str, Any]:
        """Return the compact mapping consumed at the AMAGO boundary."""
        environment = asdict(self.environment)
        environment["physical_horizon"] = environment["horizon"]
        if self.environment.goal_partition is not None:
            environment["goal_protocol"] = DARKROOM_PROTOCOL
        spec = self.condition_spec
        runtime: dict[str, Any] = {
            "version": self.version,
            "experiment": self.experiment,
            "condition": self.condition,
            "condition_label": condition_label(self.condition),
            "seed": self.seed,
            "device": self.device,
            "smoke": self.smoke,
            "environment": environment,
            "model": {
                **asdict(self.model),
                "architecture_label": architecture_label(self.model.architecture_id),
                "trajectory_encoder": spec.trajectory_encoder,
                "initialization_seed": self.seed,
            },
            "training": {**asdict(self.training), "smoke": self.smoke},
            "tracking": asdict(self.tracking),
        }
        if self.model.dat is None:
            runtime["model"].pop("dat")
        else:
            # Carry the hashed attention identity into provenance and gin.
            runtime["model"]["dat"] = self.model.dat.to_dict()
        if self.model.summary is None:
            runtime["model"].pop("summary")
        else:
            runtime["model"]["summary"] = self.model.summary.to_dict()
        if self.model.window is None:
            runtime["model"].pop("window")
        else:
            runtime["model"]["window"] = self.model.window.to_dict()
        if self.model.window_match is None:
            runtime["model"].pop("window_match")
        else:
            runtime["model"]["window_match"] = self.model.window_match.to_dict()
        if self.model.memo is None:
            runtime["model"].pop("memo")
        else:
            runtime["model"]["memo"] = self.model.memo.to_dict()
        if self.training.native_agent is None:
            runtime["training"].pop("native_agent")
        if not self.training.retain_training_state:
            runtime["training"].pop("retain_training_state")
        if architecture_uses_history_packet(self.model.architecture_id):
            runtime["model"].update({"evidence": spec.evidence, "bypass": spec.bypass})
        if self.tracking.group is not None:
            # One group per study and protocol; the condition is the run name
            # and a tag (experiment.create_experiment), never part of the group.
            runtime["tracking"]["group"] = self.tracking.group
        return runtime


# ----------------------------------------------------------------------
# Section parsers, shared by the study loaders and the resolved reloader
# ----------------------------------------------------------------------


def mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"Configuration section {name!r} must be a mapping.")
    return cast(Mapping[str, Any], value)


def positive_int(raw: Mapping[str, Any], name: str) -> int:
    value = raw.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{name} must be a positive integer.")
    return value


def environment_config(raw: Mapping[str, Any]) -> EnvironmentConfig:
    """Validate one environment recipe against its environment's protocol."""
    name = str(raw.get("name", ""))
    name = _LEGACY_ENVIRONMENT_NAMES.get(name, name)
    if name not in ENVIRONMENT_NAMES:
        raise ContractError(f"Unknown environment name: {name!r}.")
    size = positive_int(raw, "size")
    attempts = positive_int(raw, "attempts")
    horizon = positive_int(raw, "horizon")
    benchmark = str(raw.get("benchmark", ""))
    partition_value = raw.get("goal_partition")
    goal_partition = None if partition_value in (None, "") else str(partition_value)
    meta_value = raw.get("meta_horizon")
    goals_value = raw.get("goals")
    protocol_size_value = raw.get("protocol_size")
    if name != "mazerunner" and protocol_size_value is not None:
        raise ContractError(f"{name} does not accept a MazeRunner protocol size.")
    if name != "darkroom" and goal_partition is not None:
        raise ContractError(f"{name} does not accept a DarkRoom goal partition.")
    if name not in ("dark_key_to_door", "mazerunner") and meta_value is not None:
        raise ContractError(f"{name} does not accept a native meta budget.")
    if name != "mazerunner" and goals_value is not None:
        raise ContractError(f"{name} does not accept a goal sequence length.")
    scored_from = raw.get("scored_from", 1)
    if type(scored_from) is not int or not 1 <= scored_from <= attempts:
        raise ContractError("scored_from must name an attempt of the budget.")
    encoder = str(raw.get("encoder", "ff"))
    if encoder not in ("ff", "xland"):
        raise ContractError(f"Unknown timestep encoder: {encoder!r}.")
    if name not in XLAND_NAMES and (scored_from != 1 or encoder != "ff"):
        raise ContractError(f"{name} scores every attempt with the shared trunk.")
    curriculum_value = raw.get("curriculum")
    if name != "xland_one_rule" and curriculum_value is not None:
        raise ContractError(f"{name} does not accept a training curriculum.")
    penalty_value = raw.get("movement_penalty")
    if name != "tmaze" and penalty_value is not None:
        raise ContractError(f"{name} does not accept a movement penalty.")
    movement_penalty: float | None = None
    if name == "tmaze":
        if isinstance(penalty_value, bool) or not isinstance(
            penalty_value, int | float
        ):
            raise ContractError(
                "The passive T-Maze declares its movement_penalty (AMAGO's "
                "-1 / corridor_length at the paper's settings)."
            )
        movement_penalty = float(penalty_value)
        if not math.isfinite(movement_penalty) or movement_penalty > 0.0:
            raise ContractError("The T-Maze movement_penalty is a finite value <= 0.")
    training_corridors: tuple[int, ...] | None = None
    trained_size: int | None = None
    if name == "darkroom":
        if goal_partition not in (None, DARKROOM_GOAL_PARTITION):
            raise ContractError(f"Unknown DarkRoom goal partition: {goal_partition!r}.")
        if goal_partition is not None and (size != 5 or benchmark != DARKROOM_PROTOCOL):
            raise ContractError(f"{DARKROOM_PROTOCOL} requires a 5x5 grid.")
    elif name == "dark_key_to_door":
        if benchmark not in KEY_TO_DOOR_PROTOCOLS:
            raise ContractError(
                f"Dark Key-to-Door requires one of {sorted(KEY_TO_DOOR_PROTOCOLS)}."
            )
        if positive_int(raw, "meta_horizon") < attempts * (horizon + 1):
            raise ContractError(
                "The native meta budget cannot guarantee its scored attempts."
            )
    elif name == "mazerunner":
        variant, protocol_size = mazerunner_variant(benchmark)
        if protocol_size_value is not None:
            # The larger-maze axis: the protocol names the trained size, the
            # maze played is at least as large (evaluation-only contracts).
            if positive_int(raw, "protocol_size") != protocol_size:
                raise ContractError(
                    "MazeRunner protocol_size disagrees with its protocol."
                )
            if size < protocol_size:
                raise ContractError(
                    "A larger-maze MazeRunner task is at least the protocol's size."
                )
            trained_size = protocol_size
        elif size != protocol_size:
            raise ContractError("MazeRunner size disagrees with its protocol.")
        if meta_value is not None:
            # The repeated-laps axis (evaluation only): one map replayed until
            # the budget, at least one native episode long, on the trained maze.
            if positive_int(raw, "meta_horizon") < horizon:
                raise ContractError(
                    "The MazeRunner repeated-laps budget covers at least one "
                    "native episode."
                )
            if protocol_size_value is not None:
                raise ContractError(
                    "MazeRunner repeated laps and the larger maze are separate axes."
                )
        if bool(raw.get("randomized_actions", False)) != (
            variant == "randomized-actions"
        ):
            raise ContractError(
                "MazeRunner randomized_actions disagrees with its protocol."
            )
        if size < 7 or size % 2 != 1:
            raise ContractError("MazeRunner requires an odd maze of at least 7.")
        if attempts != 1:
            raise ContractError("MazeRunner has no meta-attempt wrapper.")
        positive_int(raw, "goals")
    elif name == "concentration":
        variant = concentration_variant(benchmark)
        if horizon != CONCENTRATION_HORIZONS[variant]:
            raise ContractError("The Concentration flip budget is fixed by its deck.")
        if size != CONCENTRATION_CARDS[variant]:
            raise ContractError("Concentration size must be its card count.")
        if attempts != 1:
            raise ContractError("Concentration has no meta-attempt wrapper.")
    elif name == "match_pattern":
        from reasoned_icrl.experiments.match_pattern import (
            HORIZON,
            PROTOCOL,
            VOCABULARY,
        )

        if (benchmark, size, horizon, attempts) != (PROTOCOL, VOCABULARY, HORIZON, 1):
            raise ContractError(
                "Match-pattern requires its frozen vocabulary and seven-call lifecycle."
            )
    elif name == "xland_minigrid":
        if benchmark != XLAND_PROTOCOL:
            raise ContractError(f"XLand-MiniGrid requires {XLAND_PROTOCOL}.")
        if size != XLAND_GRID or not 1 <= horizon <= XLAND_HORIZON:
            # The contract pins the native limit; a smoke profile shortens it.
            raise ContractError(
                "The XLand grid is native; the limit is at most native."
            )
        if attempts != XLAND_ATTEMPTS or scored_from != XLAND_SCORED_FROM:
            raise ContractError("XLand plays five episodes and scores the last two.")
        if encoder != "xland":
            raise ContractError("XLand-MiniGrid packets are read by the xland trunk.")
    elif name == "xland_one_rule":
        if benchmark != XLAND_ONE_RULE_PROTOCOL:
            raise ContractError(f"XLand one-rule requires {XLAND_ONE_RULE_PROTOCOL}.")
        if size != XLAND_GRID or not 1 <= horizon <= XLAND_ONE_RULE_HORIZON:
            raise ContractError("The one-rule attempt is at most 128 actions.")
        if (
            attempts != XLAND_ONE_RULE_ATTEMPTS
            or scored_from != XLAND_ONE_RULE_SCORED_FROM
        ):
            raise ContractError(
                "XLand one-rule plays five attempts and scores the last two."
            )
        if encoder != "xland":
            raise ContractError("XLand one-rule packets are read by the xland trunk.")
        if not isinstance(curriculum_value, Mapping):
            raise ContractError("XLand one-rule declares its training curriculum.")
        unknown = set(curriculum_value) - {
            "warmup_calls",
            "mixed_calls",
            "mixed_probability",
        }
        if unknown:
            raise ContractError(f"Unknown curriculum fields: {sorted(unknown)}.")
        curriculum_value = {
            "warmup_calls": positive_int(curriculum_value, "warmup_calls"),
            "mixed_calls": positive_int(curriculum_value, "mixed_calls"),
            "mixed_probability": float(
                cast(float, curriculum_value.get("mixed_probability", 0.5))
            ),
        }
        CurriculumSchedule(
            **curriculum_value, actors=positive_int(raw, "parallel_envs")
        )
    elif name == "tmaze":
        if benchmark != TMAZE_V3_PROTOCOL:
            raise ContractError(f"The passive T-Maze requires {TMAZE_V3_PROTOCOL}.")
        if size != TMAZE_CORRIDOR:
            raise ContractError(
                f"{benchmark} evaluates at corridor length {TMAZE_CORRIDOR}."
            )
        if horizon != tmaze_horizon(size):
            raise ContractError(
                "The T-Maze budget is the corridor plus one turn, with no slack."
            )
        if attempts != 1:
            raise ContractError("The passive T-Maze has no meta-attempt wrapper.")
        corridors_value = raw.get("training_corridors")
        if not isinstance(corridors_value, Sequence) or isinstance(
            corridors_value, str
        ):
            raise ContractError(f"{TMAZE_V3_PROTOCOL} declares its training corridors.")
        corridors = tuple(corridors_value)
        if (
            len(corridors) < 2
            or len(set(corridors)) != len(corridors)
            or any(
                isinstance(length, bool) or type(length) is not int or length < 1
                for length in corridors
            )
        ):
            raise ContractError(
                "T-Maze training corridors are two or more distinct positive integers."
            )
        if not min(corridors) <= size <= max(corridors):
            raise ContractError(
                "The T-Maze evaluation corridor lies inside the training draw."
            )
        training_corridors = tuple(int(length) for length in corridors)
    else:
        variant = count_recall_variant(benchmark)
        if horizon != COUNT_RECALL_HORIZONS[variant]:
            raise ContractError(
                "The CountRecall stream length is fixed by its native deck."
            )
        if size != COUNT_RECALL_CATEGORIES[variant]:
            raise ContractError("CountRecall size must be its category count.")
        if attempts != 1:
            raise ContractError("CountRecall has no meta-attempt wrapper.")
    return EnvironmentConfig(
        name=cast(EnvironmentName, name),
        benchmark=benchmark,
        size=size,
        attempts=attempts,
        horizon=horizon,
        parallel_envs=positive_int(raw, "parallel_envs"),
        goal_partition=cast(Any, goal_partition),
        meta_horizon=None if meta_value is None else positive_int(raw, "meta_horizon"),
        movement_penalty=movement_penalty,
        training_corridors=training_corridors,
        protocol_size=trained_size,
        goals=None if goals_value is None else positive_int(raw, "goals"),
        randomized_actions=bool(raw.get("randomized_actions", False)),
        scored_from=scored_from,
        encoder=cast(TimestepEncoder, encoder),
        curriculum=None if curriculum_value is None else dict(curriculum_value),
    )


def architecture_id(condition: str) -> ArchitectureID:
    """One condition resolves to exactly one encoder architecture.

    The identity names the carrier; the evidence travels in the packet identity,
    so ``raw`` and ``transition`` rows share carriers. Every attention variant
    and every bounded regime runs without the state bypass.
    """
    spec = ALL_CONDITIONS[condition]
    if spec.trajectory_encoder == "gru":
        return GRU_HISTORY_ARCHITECTURE_ID
    if spec.memory == "accumulated":
        return MEMO_ARCHITECTURE_ID
    if spec.memory == "window":
        if spec.relational_sources == "timestep_records":
            return DAT_WINDOW_ARCHITECTURE_ID
        return WINDOW_ARCHITECTURE_ID
    if spec.memory in ("segment", "summary"):
        return summary_architecture_id(
            spec.attention, spec.memory, spec.relational_sources
        )
    mode = spec.dat_mode
    if mode is not None:
        if spec.bypass:
            raise ContractError(
                "Dual-attention conditions run without the state bypass."
            )
        return dat_architecture_id(mode)
    return (
        HISTORY_ARCHITECTURE_ID
        if spec.uses_history_packet
        else FEEDFORWARD_ARCHITECTURE_ID
    )


def _summary(
    raw: Mapping[str, Any], *, condition: str, width: int
) -> SummarySpec | None:
    """Build the summary identity, or None for a full-prefix condition.

    A study's ``model.summary`` block belongs to its segment and summary
    conditions; the regime, writer and relational source route come from the
    condition row and a block that names them must agree. A bounded condition
    without the block is refused rather than run at an invented segment length.
    """
    spec = ALL_CONDITIONS[condition]
    value = raw.get("summary")
    if spec.memory not in ("segment", "summary"):
        return None
    if value is None:
        raise ContractError(
            f"Condition {condition!r} requires a model.summary configuration block."
        )
    settings = dict(mapping(value, "model.summary"))
    recorded_sha = settings.pop("sha256", None)
    for name, derived in (
        ("regime", spec.memory),
        ("writer", spec.writer),
        ("relational_sources", spec.relational_sources),
        ("d_model", width),
    ):
        if name in settings and settings.pop(name) != derived:
            raise ContractError(
                f"model.summary.{name} disagrees with the condition and model width."
            )
    unknown = set(settings) - {field.name for field in fields(SummarySpec)}
    if unknown:
        raise ContractError(f"Unknown model.summary settings: {sorted(unknown)}.")
    for name in ("segment_length", "memory_tokens"):
        if name not in settings:
            raise ContractError(f"model.summary requires {name}.")
    # The study block carries the full-gradient, replacing-write recipe every
    # summary cell shares; the truncated-gradient and residual-rewrite cells
    # derive their rules from their condition rows, and a resolved config
    # already records those rules with its own hash. A block that names
    # another rule for a cell is refused.
    recorded_detach = settings.pop("detach", "none")
    if recorded_detach not in ("none", spec.detach):
        raise ContractError(
            f"model.summary.detach {recorded_detach!r} disagrees with the "
            f"{condition!r} condition row."
        )
    settings["detach"] = spec.detach
    recorded_rewrite = settings.pop("rewrite", "replace")
    if recorded_rewrite not in ("replace", spec.rewrite):
        raise ContractError(
            f"model.summary.rewrite {recorded_rewrite!r} disagrees with the "
            f"{condition!r} condition row."
        )
    settings["rewrite"] = spec.rewrite
    summary = SummarySpec(
        regime=spec.memory,
        writer=spec.writer,
        relational_sources=spec.relational_sources,
        d_model=width,
        **settings,
    )
    if recorded_sha is not None and recorded_sha != summary.sha256:
        raise ContractError("Recorded summary identity does not match.")
    return summary


def _memo(raw: Mapping[str, Any], *, condition: str, width: int) -> MemoSpec | None:
    """Build the Memo identity, or None for every other condition.

    A study's ``model.memo`` block belongs to the accumulated-summary
    condition; the block must carry the audited segment and summary sizes
    (ME0), and a Memo condition without it is refused rather than run at an
    invented recipe. Every other condition ignores the block as it ignores
    ``summary`` and ``window``.
    """
    spec = ALL_CONDITIONS[condition]
    if spec.memory != "accumulated":
        return None
    value = raw.get("memo")
    if value is None:
        raise ContractError(
            f"Condition {condition!r} requires a model.memo configuration block."
        )
    settings = dict(mapping(value, "model.memo"))
    recorded_sha = settings.pop("sha256", None)
    if "d_model" in settings and settings.pop("d_model") != width:
        raise ContractError("model.memo.d_model disagrees with the model width.")
    unknown = set(settings) - {field.name for field in fields(MemoSpec)}
    if unknown:
        raise ContractError(f"Unknown model.memo settings: {sorted(unknown)}.")
    for name in ("segment_length", "summary_tokens"):
        if name not in settings:
            raise ContractError(f"model.memo requires {name}.")
    if spec.segmentation == "fixed":
        # The study block carries the jittered recipe the two Memo cells
        # share; the fixed-segment cell derives jitter 0 from its condition
        # row, and a resolved config already records 0 with its own hash.
        settings["training_segment_jitter"] = 0.0
    memo = MemoSpec(d_model=width, **settings)
    if spec.segmentation == "jittered" and memo.training_segment_jitter == 0:
        raise ContractError(
            f"Condition {condition!r} trains with segment jitter; a jitter of 0 "
            "is the fixed-segment cell, which has its own name."
        )
    if recorded_sha is not None and recorded_sha != memo.sha256:
        raise ContractError("Recorded Memo identity does not match.")
    return memo


def _window(raw: Mapping[str, Any], *, condition: str) -> WindowSpec | None:
    """Build the window identity, or None for every other condition.

    A study's ``model.window`` block belongs to its sliding-window condition. A
    window condition without the block is refused rather than run at an
    invented width; every other condition ignores the block as it ignores
    ``summary``.
    """
    if ALL_CONDITIONS[condition].memory != "window":
        return None
    value = raw.get("window")
    if value is None:
        raise ContractError(
            f"Condition {condition!r} requires a model.window configuration block."
        )
    settings = dict(mapping(value, "model.window"))
    recorded_sha = settings.pop("sha256", None)
    unknown = set(settings) - {field.name for field in fields(WindowSpec)}
    if unknown:
        raise ContractError(f"Unknown model.window settings: {sorted(unknown)}.")
    if "segment_length" not in settings:
        raise ContractError("model.window requires segment_length.")
    window = WindowSpec(**settings)
    if recorded_sha is not None and recorded_sha != window.sha256:
        raise ContractError("Recorded window identity does not match.")
    return window


def _dat(
    raw: Mapping[str, Any],
    *,
    condition: str,
    width: int,
    heads: int,
    layers: int,
    capacity: int | None = None,
) -> DATSpec | None:
    """Build the attention identity, or None for an ordinary condition.

    A study's ``model.dat`` block belongs to its dual-attention conditions only.
    An ordinary condition resolves to ``None`` and carries no DAT settings at
    all; a dual-attention condition *requires* the block so that a
    misconfigured arm cannot masquerade as a working baseline. A bounded
    (segment/summary) cell replaces the block's ``max_relative_distance`` by
    its segment ``capacity``: slot indices are the only positions its symbols
    ever see, and the resolved config records that value. A bounded
    dual-content cell likewise derives its control branch widths at that
    capacity (``capacity_matched_control_dims``): the study block's widths
    match the full-prefix block and would over-match here, and the resolved
    config records the derived pair.
    """
    mode = ALL_CONDITIONS[condition].dat_mode
    value = raw.get("dat")
    if mode is None:
        return None
    if value is None:
        raise ContractError(
            f"Condition {condition!r} requires a model.dat configuration block."
        )
    settings = dict(mapping(value, "model.dat"))
    # A resolved config round-trips the whole spec, so the derived fields are
    # accepted when they agree and rejected when they disagree.
    recorded_sha = settings.pop("sha256", None)
    for name, derived in (("mode", mode), ("d_model", width), ("total_heads", heads)):
        if name in settings and settings.pop(name) != derived:
            raise ContractError(
                f"model.dat.{name} disagrees with the condition and model width."
            )
    if capacity is not None:
        if settings.get("max_relative_distance", capacity) != capacity:
            # The study block carries the full-prefix distance; a resolved
            # bounded config already carries the capacity. Both are accepted.
            recorded_sha = None
        settings["max_relative_distance"] = capacity
    indices = settings.pop("layer_indices", None)
    if not isinstance(indices, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in indices
    ):
        raise ContractError("model.dat.layer_indices must be a list of integers.")
    if mode != "dual_content":
        # Only the capacity-control arm has branches to widen; carrying the
        # widths elsewhere would change the other arms' attention identity.
        settings.pop("control_content_head_dim", None)
        settings.pop("control_second_head_dim", None)
    unknown = set(settings) - {field.name for field in fields(DATSpec)}
    if unknown:
        raise ContractError(f"Unknown model.dat settings: {sorted(unknown)}.")
    spec = DATSpec(
        layer_indices=tuple(indices),
        mode=mode,
        d_model=width,
        total_heads=heads,
        **settings,
    )
    spec.validate_layers(layers)
    if mode == "dual_content" and (
        capacity is not None or condition in REVISED_CONDITIONS
    ):
        # Bounded cells derive the control widths at their clipped distance;
        # the 8M study's full dual-content cell (R3) derives them at its own
        # resolved distance too, so the match is recalculated per environment
        # rather than carried over from the legacy study's block.
        matched = capacity_matched_control_dims(spec)
        if (spec.control_content_head_dim, spec.control_second_head_dim) != matched:
            # The study block carries the legacy widths; a resolved config
            # already carries the derived pair.
            recorded_sha = None
            spec = replace(
                spec,
                control_content_head_dim=matched[0],
                control_second_head_dim=matched[1],
            )
    if recorded_sha is not None and recorded_sha != spec.sha256:
        raise ContractError("Recorded DAT attention identity does not match.")
    return spec


def model_config(
    raw: Mapping[str, Any], *, condition: str, require_architecture: bool = False
) -> ModelConfig:
    """Validate one model recipe for one condition."""
    architecture = architecture_id(condition)
    configured = raw.get("architecture_id")
    if require_architecture and configured is None:
        raise ContractError("Resolved model configuration lacks its architecture_id.")
    if configured is not None and str(configured) != architecture:
        raise ContractError(
            "Configured encoder architecture does not match the condition."
        )
    attention_backend = str(raw.get("attention_backend", ""))
    if attention_backend not in ("vanilla", "flash"):
        raise ContractError("Model attention_backend must be vanilla or flash.")
    critic: CriticConfig | None = None
    if raw.get("critic") is not None:
        critic_raw = mapping(raw["critic"], "model.critic")
        try:
            minimum = float(critic_raw["min_return"])
            maximum = float(critic_raw["max_return"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(
                "Critic min_return and max_return must be finite numbers."
            ) from error
        critic = CriticConfig(minimum, maximum, positive_int(critic_raw, "output_bins"))
    width = positive_int(raw, "width")
    heads = positive_int(raw, "heads")
    layers = positive_int(raw, "layers")
    if width % heads:
        raise ContractError("Model width must be divisible by the head count.")
    summary = _summary(raw, condition=condition, width=width)
    window = _window(raw, condition=condition)
    memo = _memo(raw, condition=condition, width=width)
    spec = ALL_CONDITIONS[condition]
    revised_window = (
        spec.memory == "window" and spec.relational_sources == "timestep_records"
    )
    capacity: int | None = None
    if summary is not None:
        capacity = summary.capacity
    elif revised_window and window is not None:
        # The band's symbols see offsets within W only, so the dual-attention
        # window clips at its window length exactly as a summary cell clips at
        # its segment capacity.
        capacity = window.capacity
    dat = _dat(
        raw,
        condition=condition,
        width=width,
        heads=heads,
        layers=layers,
        capacity=capacity,
    )
    window_match: StateMatch | None = None
    if revised_window:
        assert window is not None and dat is not None
        window_match = _window_match(
            raw, window=window, dat=dat, width=width, heads=heads, layers=layers
        )
    return ModelConfig(
        architecture_id=architecture,
        width=width,
        layers=layers,
        heads=heads,
        feedforward_multiplier=positive_int(raw, "feedforward_multiplier"),
        attention_backend=cast(Any, attention_backend),
        critic=critic,
        dat=dat,
        summary=summary,
        window=window,
        window_match=window_match,
        memo=memo,
    )


def _window_match(
    raw: Mapping[str, Any],
    *,
    window: WindowSpec,
    dat: DATSpec,
    width: int,
    heads: int,
    layers: int,
) -> StateMatch:
    """The revised band's ``W`` must be the measured state match (R3).

    The reference is the summary carrier the window is compared with: the
    study's ``model.summary`` block (with the contract's overrides applied) or,
    in a resolved config, the recorded ``model.window_match`` reference, which
    travels with the run so the match can be reproduced without the study
    file. A declared ``segment_length`` that is not the largest ``W`` within
    the summary's allocated bytes is refused, naming the match, so slot parity
    or any other rule cannot stand in for the measurement.
    """
    recorded = raw.get("window_match")
    if recorded is not None:
        reference_raw = mapping(recorded, "model.window_match")
        try:
            settings: dict[str, Any] = {
                "segment_length": reference_raw["reference_segment_length"],
                "memory_tokens": reference_raw["reference_memory_tokens"],
                "cache_dtype": reference_raw["reference_cache_dtype"],
            }
        except KeyError as error:
            raise ContractError(
                "model.window_match lacks its summary reference."
            ) from error
    else:
        block = raw.get("summary")
        if block is None:
            raise ContractError(
                "The revised window is matched to the summary carrier's state; the "
                "study needs a model.summary block beside model.window."
            )
        summary_raw = mapping(block, "model.summary")
        settings = {
            "segment_length": summary_raw.get("segment_length"),
            "memory_tokens": summary_raw.get("memory_tokens"),
            "cache_dtype": summary_raw.get("cache_dtype", "float32"),
        }
    reference = SummarySpec(regime="summary", d_model=width, **settings)
    match = match_window_to_summary(
        reference,
        dat,
        layers=layers,
        heads=heads,
        width=width,
        cache_dtype=window.cache_dtype,
    )
    if window.segment_length != match.window_length:
        raise ContractError(
            f"model.window.segment_length {window.segment_length} is not the "
            f"measured state match: the summary carrier at C="
            f"{reference.segment_length}, M={reference.memory_tokens} allocates "
            f"{match.summary_state_bytes} bytes per actor, and the largest window "
            f"within it is W={match.window_length} ({match.window_state_bytes} "
            f"bytes; slot parity would give {match.slot_parity_length})."
        )
    if recorded is not None and dict(recorded) != match.to_dict():
        raise ContractError("Recorded window state match does not reproduce.")
    return match


def tracking_config(value: object) -> TrackingConfig:
    raw = {} if value is None else mapping(value, "tracking")
    wandb = raw.get("wandb", False)
    if not isinstance(wandb, bool):
        raise ContractError("tracking.wandb must be a boolean.")
    project = str(raw.get("project", "reasoned-icrl")).strip()
    if not project:
        raise ContractError("tracking.project cannot be empty.")
    group_value = raw.get("group")
    interval = raw.get("log_interval", 300)
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise ContractError("tracking.log_interval must be a positive integer.")
    return TrackingConfig(
        wandb=wandb,
        project=project,
        group=None if group_value in (None, "") else str(group_value),
        log_interval=interval,
    )


def training_config(raw: Mapping[str, Any]) -> TrainingConfig:
    exploration = str(raw.get("exploration", ""))
    if exploration not in (
        "epsilon_greedy",
        "bilevel_epsilon_greedy",
        "tmaze_epsilon_greedy",
    ):
        raise ContractError("Unknown exploration schedule.")
    return TrainingConfig(
        epochs=positive_int(raw, "epochs"),
        start_learning_epoch=int(raw.get("start_learning_epoch", 0)),
        timesteps_per_epoch=positive_int(raw, "timesteps_per_epoch"),
        batches_per_epoch=positive_int(raw, "batches_per_epoch"),
        validation_timesteps=positive_int(raw, "validation_timesteps"),
        validation_interval=positive_int(raw, "validation_interval"),
        checkpoint_interval=positive_int(raw, "checkpoint_interval"),
        batch_size=positive_int(raw, "batch_size"),
        max_sequence_length=positive_int(raw, "max_sequence_length"),
        trajectory_length=positive_int(raw, "trajectory_length"),
        replay_capacity=positive_int(raw, "replay_capacity"),
        reward_multiplier=float(raw.get("reward_multiplier", 10.0)),
        epsilon_anneal_steps=positive_int(raw, "epsilon_anneal_steps"),
        learning_rate=float(raw.get("learning_rate", 1e-4)),
        warmup_steps=positive_int(raw, "warmup_steps"),
        gradient_clip=float(raw.get("gradient_clip", 2.0)),
        weight_decay=float(raw.get("weight_decay", 0.001)),
        mixed_precision=str(raw.get("mixed_precision", "no")),
        dataloader_workers=int(raw.get("dataloader_workers", 0)),
        torch_compile=bool(raw.get("torch_compile", False)),
        exploration=cast(Any, exploration),
        exploration_rollout_horizon=positive_int(raw, "exploration_rollout_horizon"),
        verbose=bool(raw.get("verbose", True)),
        retain_training_state=bool(raw.get("retain_training_state", False)),
        native_agent=None
        if raw.get("native_agent") is None
        else NativeAgentConfig(**mapping(raw["native_agent"], "training.native_agent")),
    )


def smoke_profile(
    environment: EnvironmentConfig,
    model: ModelConfig,
    training: TrainingConfig,
    tracking: TrackingConfig,
    *,
    device: str,
) -> tuple[EnvironmentConfig, ModelConfig, TrainingConfig, TrackingConfig]:
    """Shrink one resolved recipe to a bounded execution check.

    The outer task keeps its shape: every domain still runs whole tasks, and a
    native fixed-budget meta-task keeps a budget in which all of its scored
    attempts still fit. Only sizes, counts and precision change.
    """
    tracking = replace(tracking, wandb=False, log_interval=1)
    environment = replace(
        environment,
        parallel_envs=1,
        horizon=environment.horizon
        if environment.has_fixed_native_horizon
        else min(environment.horizon, 8),
    )
    if environment.meta_horizon is not None:
        environment = replace(
            environment,
            meta_horizon=min(
                environment.meta_horizon,
                environment.attempts * (environment.horizon + 1),
            ),
        )
    if environment.name == "tmaze":
        # A short corridor keeps the exact budget (corridor plus one turn); a
        # v3 draw shrinks to three short corridors around it so the smoke
        # lifecycle still trains over variable episode lengths.
        corridor = min(environment.size, 7)
        environment = replace(
            environment, size=corridor, horizon=tmaze_horizon(corridor)
        )
        if environment.training_corridors:
            environment = replace(environment, training_corridors=(3, 5, 7))
    # The rollout, context and replay sizes follow the longest training task
    # (the outer length, or the longest corridor of a v3 draw).
    outer_length = environment.longest_training_length
    if environment.curriculum is not None:
        # One actor, two epochs of one lifetime each: the first lifetime is
        # warmup, the second primary, and the learner samples the primary pool.
        environment = replace(
            environment,
            curriculum={
                **dict(environment.curriculum),
                "warmup_calls": outer_length,
                "mixed_calls": outer_length,
            },
        )
    if model.dat is None:
        model = replace(model, width=32, layers=1, heads=2, feedforward_multiplier=2)
        if model.summary is not None:
            model = replace(model, summary=replace(model.summary, d_model=32))
        if model.memo is not None:
            model = replace(model, memo=replace(model.memo, d_model=32))
    # A dual-attention arm keeps its declared residual width, head split and
    # selected layers: those are the attention identity the check exists to
    # exercise. Only the task and the schedule become small.
    training = replace(
        training,
        epochs=2,
        start_learning_epoch=1,
        timesteps_per_epoch=outer_length,
        batches_per_epoch=1,
        validation_timesteps=outer_length,
        batch_size=1,
        max_sequence_length=outer_length,
        trajectory_length=outer_length,
        replay_capacity=16,
        epsilon_anneal_steps=2 * outer_length,
        validation_interval=1,
        checkpoint_interval=1,
        dataloader_workers=0,
        torch_compile=False,
        mixed_precision=training.mixed_precision
        if model.attention_backend == "flash" and device in ("auto", "cuda")
        else "no",
        exploration_rollout_horizon=outer_length,
        verbose=False,
    )
    return environment, model, training, tracking


def read_yaml(path: str | Path, *, name: str) -> Mapping[str, Any]:
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ContractError(f"Could not read {name} {source}.") from error
    return mapping(loaded, name)


def dump_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Write a human-readable resolved configuration."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    if config.model.dat is not None:
        # asdict drops the derived identity; the run must record it.
        payload["model"]["dat"] = config.model.dat.to_dict()
    if config.model.summary is not None:
        payload["model"]["summary"] = config.model.summary.to_dict()
    if config.model.window is not None:
        payload["model"]["window"] = config.model.window.to_dict()
    if config.model.window_match is not None:
        payload["model"]["window_match"] = config.model.window_match.to_dict()
    if config.model.memo is not None:
        payload["model"]["memo"] = config.model.memo.to_dict()
    try:
        payload["output_root"] = str(config.output_root.relative_to(config.repository))
    except ValueError:
        payload["output_root"] = str(config.output_root)
    payload["repository"] = "."
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output


def load_resolved_config(
    path: str | Path, *, repository: str | Path | None = None
) -> ExperimentConfig:
    """Reload the exact configuration stored in one run directory."""
    raw = read_yaml(path, name="resolved configuration")
    version = str(raw.get("version", ""))
    if version != CONFIG_VERSION:
        raise ContractError("Resolved run has an unsupported configuration version.")
    condition = str(raw.get("condition", ""))
    experiment = str(raw.get("experiment", ""))
    device = str(raw.get("device", "auto"))
    if condition not in ALL_CONDITIONS:
        raise ContractError("Resolved run has an unknown condition.")
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment):
        raise ContractError("Resolved run has an invalid experiment ID.")
    if device not in DEVICES:
        raise ContractError("Resolved run has an unknown device.")
    root = (
        Path(repository).resolve()
        if repository is not None
        else Path(str(raw["repository"])).resolve()
    )
    output_value = Path(str(raw["output_root"]))
    return ExperimentConfig(
        version=version,
        experiment=experiment,
        condition=cast(AnyCondition, condition),
        seed=int(raw["seed"]),
        device=cast(Any, device),
        output_root=output_value.resolve()
        if output_value.is_absolute()
        else root / output_value,
        repository=root,
        environment=environment_config(mapping(raw["environment"], "environment")),
        model=model_config(
            mapping(raw["model"], "model"),
            condition=condition,
            require_architecture=True,
        ),
        training=training_config(mapping(raw["training"], "training")),
        tracking=tracking_config(raw.get("tracking")),
        smoke=bool(raw.get("smoke", False)),
    )


__all__ = [
    "CONFIG_VERSION",
    "DEVICES",
    "EXPERIMENT_ID_PATTERN",
    "CriticConfig",
    "EnvironmentConfig",
    "EnvironmentName",
    "ExperimentConfig",
    "ModelConfig",
    "NativeAgentConfig",
    "TrackingConfig",
    "TrainingConfig",
    "architecture_id",
    "dump_config",
    "environment_config",
    "load_resolved_config",
    "mapping",
    "model_config",
    "positive_int",
    "read_yaml",
    "smoke_profile",
    "tracking_config",
    "training_config",
]
