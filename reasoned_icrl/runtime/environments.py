"""AMAGO's environment adapter and the actor factories training hands it.

:func:`amago_environment` wraps a scalar project environment in AMAGO's
adapter, promoting Gymnasium truncation to the stored task terminal signal;
:func:`environment_builders` produces the training and validation callables
AMAGO expects, one seeded scalar environment per actor; and
:func:`benchmark_relabeler` builds the replay relabeler an environment
declares. The environments themselves come from
:mod:`reasoned_icrl.experiments.environments`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from amago.envs import AMAGOEnv

from reasoned_icrl.environments.mazerunner import (
    HindsightGoalRelabeler,
    rebuild_transition_packet,
)
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.environments import (
    VALIDATION_SPLITS,
    _group,
    build_environment,
)
from reasoned_icrl.runtime.replay import ReconstructingRelabeler


class TaskBoundaryAMAGOEnv(AMAGOEnv):
    """Promote Gymnasium truncation to AMAGO's stored task terminal signal."""

    def step(self, action: np.ndarray) -> Any:
        # Canonicalize after exploration and before AMAGO constructs RL2/replay.
        from reasoned_icrl.environments.match_pattern import MatchPatternEnv

        if isinstance(self.env.unwrapped, MatchPatternEnv):
            action = self.env.unwrapped.canonical_action(action)
        return super().step(action)

    @property
    def env_name(self) -> str:
        """The declared name, suffixed by the live lifetime's replay pool.

        AMAGO names every trajectory file after ``env_name`` at the moment the
        lifetime ends, so an environment that exposes ``replay_pool`` (the
        one-rule XLand curriculum) tags its files ``<name>-warmup`` or
        ``<name>-primary``; the replay sampler reads the tag back. Returns
        and frame counts are logged per pool for the same reason.
        """
        base = str(super().env_name)
        try:
            pool = self.env.get_wrapper_attr("replay_pool")
        except AttributeError:
            return base
        return f"{base}-{pool}"

    def inner_step(
        self, action: np.ndarray
    ) -> tuple[Any, Any, np.ndarray, np.ndarray, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        terminal = np.logical_or(terminated, truncated)
        return observation, reward, terminal, truncated, info

    def state_dict(self) -> dict[str, object]:
        """Snapshot learner counters and the wrapped scientific environment."""
        try:
            reader = self.env.get_wrapper_attr("state_dict")
        except AttributeError:
            reader = None
        if not callable(reader):
            raise ContractError("The wrapped Gymnasium environment is not restorable.")
        step_count = getattr(self, "step_count", None)
        if not isinstance(step_count, np.ndarray):
            raise ContractError(
                "The AMAGO environment must be reset before snapshotting."
            )
        return {
            "schema": "task-boundary-amago-env.v0.3",
            "step_count": step_count.copy(),
            "environment_state": deepcopy(reader()),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a snapshot without changing the task."""
        if state.get("schema") != "task-boundary-amago-env.v0.3":
            raise ContractError("Unsupported task-boundary AMAGO state.")
        try:
            loader = self.env.get_wrapper_attr("load_state_dict")
        except AttributeError:
            loader = None
        environment_state = state.get("environment_state")
        if not callable(loader) or not isinstance(environment_state, Mapping):
            raise ContractError("The wrapped Gymnasium environment is not restorable.")
        step_count = np.asarray(state.get("step_count"), dtype=np.int64)
        if step_count.shape != (self.batched_envs,):
            raise ContractError("The restored AMAGO step counter changed shape.")
        loader(deepcopy(environment_state))
        self.step_count = step_count.copy()


def amago_environment(
    env: gym.Env[Any, Any], *, name: str, batched_envs: int = 1, seed: int = 0
) -> AMAGOEnv:
    """Create the official AMAGO adapter around a project environment."""
    wrapper = TaskBoundaryAMAGOEnv(env=env, env_name=name, batched_envs=batched_envs)
    wrapper.research_seed = seed
    return wrapper


def benchmark_relabeler(config: Mapping[str, Any]) -> ReconstructingRelabeler | None:
    """Build the replay relabeler declared by this environment, if any.

    Only MazeRunner declares hindsight relabeling. The relabeler is seeded from
    the training seed alone, so every condition of one seed applies the
    identical relabeling function to the trajectories it samples.
    """
    environment = _group(config, "environment")
    if str(environment.get("name")) != "mazerunner":
        return None
    return ReconstructingRelabeler(
        HindsightGoalRelabeler(
            size=int(environment["size"]),
            goals=int(environment["goals"]),
            horizon=int(environment["horizon"]),
            strategy="some",
            seed=int(config["seed"]),
        ),
        rebuild_transition_packet,
    )


def environment_builders(
    config: Mapping[str, Any],
) -> tuple[Sequence[Callable[[], AMAGOEnv]], Sequence[Callable[[], AMAGOEnv]]]:
    """Create AMAGO's training and validation environment callables.

    Every actor owns one scalar environment with its own seed, so replay never
    mixes actors and a resumed run reproduces each actor's task sequence.
    """
    environment = _group(config, "environment")
    name = str(environment["name"])
    if name not in VALIDATION_SPLITS:
        raise ContractError(f"Unknown environment: {name!r}.")
    seed = int(config["seed"])
    actors = int(environment["parallel_envs"])

    def builder(split: str, actor_seed: int) -> Callable[[], AMAGOEnv]:
        def create() -> AMAGOEnv:
            base = build_environment(config, split=split, seed=actor_seed)
            return amago_environment(
                base, name=f"{type(base).__name__}-{split}", seed=actor_seed
            )

        return create

    return (
        tuple(builder("train", seed * 10_000 + slot) for slot in range(actors)),
        tuple(
            builder(VALIDATION_SPLITS[name], seed * 10_000 + 5_000 + slot)
            for slot in range(actors)
        ),
    )


__all__ = [
    "TaskBoundaryAMAGOEnv",
    "amago_environment",
    "benchmark_relabeler",
    "environment_builders",
]
