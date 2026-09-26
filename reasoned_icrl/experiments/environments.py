"""Task rosters and scalar Gymnasium environments from a resolved runtime mapping.

One factory per environment turns the ``environment`` section of
:meth:`ExperimentConfig.as_runtime_mapping` plus a split and an actor seed into
a scalar Gymnasium environment, and :func:`roster` slices the ordered task
roster of one split. Training wraps these in AMAGO's adapter
(:mod:`reasoned_icrl.runtime.environments`); evaluation and the evaluation-only
references use the same factories with explicit rosters, so a training rollout
and an evaluation rollout differ only in seed and task roster.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from reasoned_icrl.environments.base import BaseEnv, benchmark_task_sources
from reasoned_icrl.environments.concentration import (
    ConcentrationEnv,
    concentration_variant,
)
from reasoned_icrl.environments.count_recall import CountRecallEnv, count_recall_variant
from reasoned_icrl.environments.dark_key_to_door import DarkKeyToDoorEnv
from reasoned_icrl.environments.darkroom import DarkRoomEnv, darkroom_source_indices
from reasoned_icrl.environments.match_pattern import MatchPatternEnv
from reasoned_icrl.environments.mazerunner import MazeRunnerEnv, mazerunner_variant
from reasoned_icrl.environments.tmaze import TMazeEnv
from reasoned_icrl.environments.xland_minigrid import (
    XLandMiniGridEnv,
    xland_task_sources,
)
from reasoned_icrl.environments.xland_one_rule import XLandOneRuleEnv
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.match_pattern import (
    task_sources as match_pattern_task_sources,
)
from reasoned_icrl.experiments.xland_one_rule import (
    CurriculumSchedule,
    xland_one_rule_task_sources,
)

VALIDATION_SPLITS = {
    "darkroom": "validation",
    "dark_key_to_door": "development",
    "count_recall": "development",
    "mazerunner": "development",
    "concentration": "development",
    "xland_minigrid": "development",
    "xland_one_rule": "development",
    "match_pattern": "development",
    "tmaze": "development",
}
"""The split AMAGO's validation actors draw from during training."""


def _group(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ContractError(f"Missing configuration section {name!r}.")
    return cast(Mapping[str, Any], value)


def roster(
    config: Mapping[str, Any],
    split: str,
    *,
    task_count: int | None = None,
    offset: int = 0,
) -> Sequence[int] | range:
    """Slice one environment's ordered task roster for ``split``."""
    environment = _group(config, "environment")
    name = str(environment["name"])
    if name == "darkroom":
        partition = environment.get("goal_partition")
        full: Sequence[int] | range = darkroom_source_indices(
            size=int(environment["size"]),
            split=split,
            goal_partition=None if partition in (None, "") else str(partition),
        )
    elif name == "match_pattern":
        full = match_pattern_task_sources(split)
    elif name == "xland_minigrid":
        full = xland_task_sources(split)
    elif name == "xland_one_rule":
        full = xland_one_rule_task_sources(split)
    else:
        full = benchmark_task_sources(split)
    if offset < 0 or (task_count is not None and task_count <= 0):
        raise ContractError("Evaluation task slices must be positive.")
    end = len(full) if task_count is None else offset + task_count
    if offset >= len(full) or end > len(full):
        raise ContractError(f"Requested {name} {split!r} slice exceeds its roster.")
    if isinstance(full, range):
        return range(full[offset], full[offset] + (end - offset))
    return tuple(int(value) for value in full[offset:end])


def build_environment(
    config: Mapping[str, Any],
    *,
    split: str,
    seed: int,
    task_count: int | None = None,
    offset: int = 0,
) -> BaseEnv:
    """Build the configured environment over one ordered task roster."""
    environment = _group(config, "environment")
    name = str(environment["name"])
    selected = roster(config, split, task_count=task_count, offset=offset)
    if name == "match_pattern":
        return MatchPatternEnv(split=split, source_indices=selected, initial_seed=seed)
    if name == "darkroom":
        partition = environment.get("goal_partition")
        return DarkRoomEnv(
            size=int(environment["size"]),
            attempts=int(environment["attempts"]),
            horizon=int(environment["horizon"]),
            split=split,
            goal_partition=None if partition in (None, "") else str(partition),
            source_indices=selected,
            initial_seed=seed,
        )
    if name == "dark_key_to_door":
        return DarkKeyToDoorEnv(
            size=int(environment["size"]),
            physical_horizon=int(environment["horizon"]),
            meta_horizon=int(environment["meta_horizon"]),
            scored_attempts=int(environment["attempts"]),
            randomized_actions=bool(environment.get("randomized_actions", False)),
            split=split,
            source_indices=selected,
            initial_seed=seed,
        )
    if name == "tmaze":
        # The corridor is the contract's ``size``; the budget follows from it.
        # Under v3 the training split draws each task's corridor from the
        # declared set; every evaluation split keeps the contract's corridor.
        corridors = environment.get("training_corridors")
        return TMazeEnv(
            corridor_length=int(environment["size"]),
            movement_penalty=float(environment["movement_penalty"]),
            split=split,
            source_indices=selected,
            initial_seed=seed,
            protocol=str(environment["benchmark"]),
            corridor_lengths=tuple(int(c) for c in corridors)
            if split == "train" and corridors
            else None,
        )
    if name == "count_recall":
        # Every split plays the contract's ``attempts`` pairs (one, or the
        # continued-stream adapter's N).
        return CountRecallEnv(
            variant=count_recall_variant(str(environment["benchmark"])),
            horizon=int(environment["horizon"]),
            split=split,
            source_indices=selected,
            initial_seed=seed,
            streams=int(environment.get("attempts", 1)),
        )
    if name == "concentration":
        return ConcentrationEnv(
            variant=concentration_variant(str(environment["benchmark"])),
            horizon=int(environment["horizon"]),
            split=split,
            source_indices=selected,
            initial_seed=seed,
        )
    if name == "mazerunner":
        # The protocol carries the trained size; environment_config checked they
        # agree, or that a larger evaluation-only maze names it as protocol_size.
        variant, _ = mazerunner_variant(str(environment["benchmark"]))
        trained = environment.get("protocol_size")
        budget = environment.get("meta_horizon")
        return MazeRunnerEnv(
            size=int(environment["size"]),
            goals=int(environment["goals"]),
            horizon=int(environment["horizon"]),
            variant=variant,
            split=split,
            source_indices=selected,
            initial_seed=seed,
            protocol_size=None if trained in (None, "") else int(trained),
            meta_horizon=None if budget in (None, "") else int(budget),
        )
    if name == "xland_minigrid":
        return XLandMiniGridEnv(
            physical_horizon=int(environment["horizon"]),
            attempts=int(environment["attempts"]),
            scored_from=int(environment.get("scored_from", 1)),
            split=split,
            source_indices=selected,
            initial_seed=seed,
        )
    if name == "xland_one_rule":
        curriculum = environment.get("curriculum")
        return XLandOneRuleEnv(
            physical_horizon=int(environment["horizon"]),
            attempts=int(environment["attempts"]),
            scored_from=int(environment.get("scored_from", 1)),
            split=split,
            source_indices=selected,
            initial_seed=seed,
            # The curriculum drives training lifetimes only; evaluation rosters
            # are primary tasks on the declared layout roots.
            curriculum=CurriculumSchedule(
                **dict(curriculum), actors=int(environment["parallel_envs"])
            )
            if split == "train" and isinstance(curriculum, Mapping)
            else None,
        )
    raise ContractError(f"Unknown environment: {name!r}.")


__all__ = [
    "VALIDATION_SPLITS",
    "build_environment",
    "roster",
]
