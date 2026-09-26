"""Native AMAGO v3.4.0 MazeRunner as a public-packet policy environment.

Wraps the pinned ``amago.envs.builtin.mazerunner`` task through its own
``RelabelInfoWrapper``. Upstream source is not patched.

The native lifecycle is kept as it is and has no meta-attempt structure: one
reset draws a fresh maze and an ordered goal sequence, each step moves the agent
(an invalid move leaves it in place), reaching the active goal pays 1 and
advances the sequence, and the episode ends at full-sequence completion or at
the native finite-horizon timer. Both endings set ``terminated``. Under the
randomized protocols the five action identities are permuted once per outer
reset by the native task, from a generator reseeded with the map identity, so
the permutation is a hidden property of the map and never of the actor.

A protocol identity names the action protocol *and* the maze size: the 11x11
protocols belong to the DAT study, ``mazerunner-15-randomized-actions`` (15x15,
three goals, 500 steps) to the summary-memory study. No dynamics differ between
sizes; the public projection divides every coordinate by the maze dimension.

A larger maze at evaluation (the paper's MazeRunner beyond-horizon axis) keeps the
trained protocol identity through ``protocol_size``:
the maze played is ``size`` (odd, never below the trained size), every
coordinate is divided by it, the step budget is the caller's (the area-scaled
timer of :mod:`reasoned_icrl.experiments.horizon`) and the packet width is
unchanged, so the frozen weights read the same fields.

The repeated-laps axis (the paper's second MazeRunner beyond-horizon axis) keeps the
trained protocol and the trained maze and plays one
map for a fixed budget of charged calls (``meta_horizon``): a lap is one native
episode of the trained task on the map's own seed (same maze, same ordered
goals, same hidden action permutation, the agent at the start cell, the native
timer per lap); when a lap ends the next call is a reset-only step, the
K-attempt boundary of the shared lifecycle (event new-attempt, no physical
action, no reward, charged), and the carried memory continues across it; the
outer task ends when the budget is spent, the last lap cut and marked partial.
The packet, its width and every field are the trained ones, so the frozen
weights see only what training showed them, for longer. Evaluation only: a
training split never runs laps.

Only the pinned public projection reaches the policy: seven observation values
(position, four wall distances, timer) and the flattened goal sequence, each
divided by the maze dimension, with ``-1`` marking a completed goal. The
``achieved`` helper field is never an actor input and is never stored: it is
exactly the agent's public position, so hindsight relabeling recovers it from
the packet rather than carrying privileged metadata alongside it.

Hindsight relabeling lives here because it is task semantics, not replay
plumbing. :class:`HindsightGoalRelabeler` rewrites the instruction and replays
the pinned reward and termination rules against it;
:func:`rebuild_transition_packet` then rebuilds every derived field from the
relabeled observations, so no precomputed transition field survives relabeling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np
from amago.hindsight import FrozenTraj, Relabeler, Trajectory
from numpy.typing import NDArray

from reasoned_icrl.environments.base import (
    Attempt,
    BaseEnv,
    PublicDecision,
    PublicField,
    benchmark_task_sources,
    unit_fields,
)
from reasoned_icrl.environments.utils import NativeSource
from reasoned_icrl.experiments.contracts import ContractError

MAZERUNNER_STATE_SCHEMA = "mazerunner-state.v1"
MAZERUNNER_VARIANTS = ("fixed-actions", "randomized-actions")
MAZERUNNER_PROTOCOLS: dict[tuple[str, int], str] = {
    ("fixed-actions", 11): "mazerunner-fixed-actions",
    ("randomized-actions", 11): "mazerunner-randomized-actions",
    ("randomized-actions", 15): "mazerunner-15-randomized-actions",
}
"""The action protocol and the maze size are one identity; results are never
pooled. A (variant, size) pair outside this table has no protocol and is refused."""

MAZERUNNER_ACTIONS = 5
MAZERUNNER_OBSERVATION_WIDTH = 7
"""Native `obs`: position (2), wall distances (4), timer (1)."""

RELAYOUT_TASKS = range(6_000_000, 7_000_000)
"""The seed band of the maps a map-change continuation draws: disjoint from the training
band and from every evaluation roster, so a
new map was never trained on or scored as a roster map."""

TIMER_INDEX = 6
GOAL_SENTINEL = -1
RELABEL_STRATEGIES = ("none", "some", "all")


def mazerunner_protocol(variant: str, size: int) -> str:
    """Return the protocol identity of one action protocol at one maze size."""
    if variant not in MAZERUNNER_VARIANTS:
        raise ContractError(f"Unknown MazeRunner variant: {variant!r}.")
    try:
        return MAZERUNNER_PROTOCOLS[(variant, size)]
    except KeyError as error:
        raise ContractError(
            f"No MazeRunner protocol declares {variant!r} at size {size}."
        ) from error


def mazerunner_variant(protocol: str) -> tuple[str, int]:
    """Return the action protocol and the maze size behind one protocol identity."""
    for (variant, size), name in MAZERUNNER_PROTOCOLS.items():
        if name == protocol:
            return variant, size
    raise ContractError(f"Unknown MazeRunner protocol: {protocol!r}.")


def public_fields(goals: int) -> tuple[PublicField, ...]:
    """Name the pinned public projection for provenance and leakage review."""
    names = [
        "position_i",
        "position_j",
        "space_west",
        "space_north",
        "space_east",
        "space_south",
        "timer",
    ]
    fields = list(unit_fields(names))
    for index in range(goals):
        for axis in ("i", "j"):
            # A completed goal is published as -1 before normalization.
            fields.append(
                PublicField(f"goal_{index}_{axis}", (1,), "float32", (-1.0,), (1.0,))
            )
    return tuple(fields)


def public_width(goals: int) -> int:
    return MAZERUNNER_OBSERVATION_WIDTH + 2 * goals


def decode_grid(values: NDArray[np.float32], maze_dim: int) -> NDArray[np.int64]:
    """Recover integer maze coordinates from the pinned public projection."""
    scaled = np.asarray(values, dtype=np.float64) * maze_dim
    grid = np.rint(scaled).astype(np.int64)
    if np.any(np.abs(scaled - grid) > 1e-3):
        raise ContractError("MazeRunner public value is not a maze coordinate.")
    if np.any(grid < GOAL_SENTINEL) or np.any(grid >= maze_dim):
        raise ContractError("MazeRunner coordinate lies outside the maze.")
    return grid


def encode_grid(
    grid: NDArray[np.int64] | Sequence[int], maze_dim: int
) -> NDArray[np.float32]:
    """Publish integer maze coordinates exactly as the pinned wrapper does."""
    return (np.asarray(grid, dtype=np.float64) / maze_dim).astype(np.float32)


@dataclass(frozen=True, slots=True)
class GoalCompletion:
    """One goal reached, reconstructed from public observations only."""

    index: int
    step: int

    def __post_init__(self) -> None:
        if self.index < 0 or self.step < 1:
            raise ContractError("MazeRunner goal completion counters must be valid.")


@dataclass(frozen=True, slots=True)
class MazeLap(Attempt):
    """One lap of the repeated-laps axis: one native episode of the trained task
    on the same map, from public observations only.

    ``goals`` counts the goals reached in the lap and ``goal_steps`` their
    completion steps as one-based charged calls of the outer task, in order;
    ``success`` is the full ordered sequence, so a partial lap never succeeds
    but may hold goals. ``native_return`` equals ``goals``.
    """

    goals: int
    goal_steps: tuple[int, ...]
    layout: int = 0
    """Maps the task has had when the lap ran: 0 for the roster map, ``k``
    after the ``k``-th change of the evaluation-only map-change continuation."""

    def __post_init__(self) -> None:
        Attempt.__post_init__(self)
        steps = tuple(int(step) for step in self.goal_steps)
        if self.goals != len(steps) or self.native_return != float(self.goals):
            raise ContractError("A lap's goals, their steps and its return disagree.")
        if any(not self.first_step <= step <= self.last_step for step in steps) or any(
            later <= earlier for earlier, later in pairwise(steps)
        ):
            raise ContractError("A lap's goal steps lie inside it, in order.")


class MazeRunnerEnv(BaseEnv):
    """Public five-key packet over the pinned native MazeRunner task."""

    label: ClassVar[str] = "MazeRunner"
    state_schema: ClassVar[str] = MAZERUNNER_STATE_SCHEMA

    def __init__(
        self,
        *,
        size: int = 11,
        goals: int = 3,
        horizon: int = 250,
        variant: str = "fixed-actions",
        split: str = "train",
        source_indices: Sequence[int] | range | None = None,
        fixed_task_index: int | None = None,
        initial_seed: int = 0,
        protocol_size: int | None = None,
        meta_horizon: int | None = None,
    ) -> None:
        trained = size if protocol_size is None else int(protocol_size)
        self.protocol = mazerunner_protocol(variant, trained)
        if meta_horizon is not None:
            if isinstance(meta_horizon, bool) or int(meta_horizon) < horizon:
                raise ContractError(
                    "The repeated-laps budget covers at least one native episode."
                )
            if protocol_size is not None:
                raise ContractError(
                    "Repeated laps and the larger maze are separate axes; a laps "
                    "task plays the trained maze."
                )
        self.meta_horizon = None if meta_horizon is None else int(meta_horizon)
        """The repeated-laps budget in charged calls, or ``None`` for the
        trained one-episode task."""
        if size < 7 or size % 2 != 1:
            # The native task silently rounds an even dimension up.
            raise ContractError("MazeRunner requires an odd maze of at least 7.")
        if size < trained:
            raise ContractError(
                f"A {size}x{size} maze is smaller than the trained {trained}x{trained} "
                "protocol; the larger-maze axis only enlarges the maze."
            )
        self.protocol_size = trained
        """The maze dimension of the protocol identity (the trained size)."""
        super().__init__(
            split=split,
            roster=benchmark_task_sources(split)
            if source_indices is None
            else source_indices,
            fixed_task_index=fixed_task_index,
            initial_seed=initial_seed,
        )
        self.variant = variant
        self.size = size
        self.goals = goals
        self.horizon = horizon
        self.randomized_actions = variant == "randomized-actions"
        self._native = NativeSource(
            self._make_native, seed=initial_seed, reseed=self._reseed
        )
        self._install_contract(public_fields(goals), MAZERUNNER_ACTIONS)
        self._step = 0
        self._episode_return = 0.0
        self._completions: list[GoalCompletion] = []
        self._source = 0
        self._lap = 0
        self._lap_step = 0
        self._lap_first_step = 0
        self._lap_open = False
        self._pending_reset = False
        self._laps: list[MazeLap] = []
        self._task_return = 0.0
        self._layout_period: int | None = None
        self._layout = 0
        self._permutation: NDArray[np.int64] = np.zeros(0, dtype=np.int64)

    def set_layout_period(self, period: int | None) -> None:
        """Evaluation-only map-change continuation of the repeated-laps axis
        (a different map layout and goal position at the trained size). With a period
        ``P``, the lap that
        starts at the first lap boundary at or after every ``k * P``-th
        charged call, ``k >= 1``, plays a new 15x15 maze with new ordered goals,
        drawn from its own seed in :data:`RELAYOUT_TASKS`; the hidden action
        permutation of the task is kept, the carried memory is never reset and
        the packet carries no signal of the change. The first ``P`` calls are
        the roster map's laps exactly."""
        if period is not None:
            if isinstance(period, bool) or type(period) is not int or period < 1:
                raise ContractError("The layout period is a positive number of calls.")
            if self.meta_horizon is None:
                raise ContractError(
                    "A MazeRunner map change needs the repeated-laps budget."
                )
            if period < self.horizon:
                raise ContractError(
                    "The layout period must not fall inside the trained episode."
                )
        self._layout_period = period

    @property
    def layout_period(self) -> int | None:
        return self._layout_period

    @property
    def layout_index(self) -> int:
        """Maps so far: 0 for the roster map, ``k`` after ``k`` changes."""
        return self._layout

    def relayout_seed(self, layout: int) -> int:
        """The native seed of map ``layout`` (``>= 1``) of this task, inside
        :data:`RELAYOUT_TASKS` and distinct for every (roster, map, change)."""
        band, offset = divmod(int(self._source), 1_000_000)
        seed = RELAYOUT_TASKS.start + ((band * 4096 + offset) * 16 + int(layout))
        if layout < 1 or layout >= 16 or offset >= 4096 or seed not in RELAYOUT_TASKS:
            raise ContractError("No map-change seed for this task and change.")
        return seed

    def _make_native(self) -> gym.Env[Any, Any]:
        from amago.envs.builtin.mazerunner import MazeRunnerGymEnv, RelabelInfoWrapper

        return cast(
            gym.Env[Any, Any],
            RelabelInfoWrapper(
                MazeRunnerGymEnv(
                    maze_dim=self.size,
                    min_num_goals=self.goals,
                    max_num_goals=self.goals,
                    time_limit=self.horizon,
                    goal_in_obs=False,
                    randomized_action_space=self.randomized_actions,
                )
            ),
        )

    @staticmethod
    def _reseed(native: Any, seed: int) -> None:
        native.rng = np.random.default_rng(seed)

    @property
    def native(self) -> gym.Env[Any, Any]:
        """The wrapped upstream task, for inspection and audits only."""
        return self._native

    # ------------------------------------------------------------------
    # Evaluator-only accessors
    # ------------------------------------------------------------------

    @property
    def completed_goals(self) -> tuple[GoalCompletion, ...]:
        """Every goal reached in the live episode (the current lap under the
        repeated-laps axis), in completion order; ``step`` is the one-based
        charged call of the outer task."""
        return tuple(self._completions)

    @property
    def episode_return(self) -> float:
        """Unscaled native return of the live episode (the current lap under
        the repeated-laps axis); equals the number of goals completed."""
        return self._episode_return

    @property
    def decisions(self) -> int:
        return self._step

    # The repeated-laps axis exposes the K-attempt interface the evaluator
    # scores Key-to-Door with: every finished lap, the lap the budget cut, and
    # the return over the whole outer task.

    @property
    def completed_attempts(self) -> tuple[MazeLap, ...]:
        """Every lap that reached its native end (third goal or timer)."""
        return tuple(self._laps)

    @property
    def partial_attempt(self) -> MazeLap | None:
        """The lap the budget cut short, when one exists."""
        if (
            self.meta_horizon is None
            or not self._done
            or self._pending_reset
            or not self._lap_open
        ):
            return None
        return self._lap_record(complete=False)

    @property
    def task_return(self) -> float:
        """Unscaled native return over the whole outer task (goals reached)."""
        return self._task_return

    def decode(self, packet: Mapping[str, np.ndarray]) -> dict[str, Any]:
        """Decode one packet's ``current`` token to named native values."""
        current = np.asarray(packet["current"], dtype=np.float32)
        if current.shape != (public_width(self.goals),):
            raise ContractError("Packet width disagrees with the public contract.")
        position = decode_grid(current[:2], self.size)
        goals = decode_grid(current[MAZERUNNER_OBSERVATION_WIDTH:], self.size)
        return {
            "position": tuple(int(value) for value in position),
            "goals": goals.reshape(self.goals, 2),
            "timer": round(float(current[TIMER_INDEX]) * self.horizon),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _project(self, observation: object) -> np.ndarray:
        """Apply the pinned public projection; the native scale is preserved.

        The ``-1`` completed-goal sentinel is valid public data despite the
        native Dict space's incorrectly nonnegative declared lower bound.
        """
        if not isinstance(observation, Mapping) or set(observation) not in (
            {"obs", "goals", "achieved"},
            {"obs", "goals"},
        ):
            raise ContractError("MazeRunner requires only declared native fields.")
        state = np.asarray(observation["obs"])
        goals = np.asarray(observation["goals"])
        if state.shape != (MAZERUNNER_OBSERVATION_WIDTH,) or goals.shape != (
            self.goals,
            2,
        ):
            raise ContractError("MazeRunner public observation shape mismatch.")
        if not np.issubdtype(goals.dtype, np.integer):
            raise ContractError("MazeRunner goal coordinates must be integers.")
        if np.any(goals < GOAL_SENTINEL) or np.any(goals >= self.size):
            raise ContractError("MazeRunner goal coordinates outside native bounds.")
        if np.any(
            (goals == GOAL_SENTINEL).any(axis=1) != (goals == GOAL_SENTINEL).all(axis=1)
        ):
            raise ContractError("Completed MazeRunner goals require paired -1 values.")
        result = np.concatenate((state, goals.reshape(-1) / self.size))
        if result.shape != (public_width(self.goals),) or not np.isfinite(result).all():
            raise ContractError(
                "Native public observation has invalid shape or values."
            )
        if np.any(result < GOAL_SENTINEL / self.size) or np.any(result > 1):
            raise ContractError("Native public observation outside declared bounds.")
        return np.array(result, dtype=np.float32, copy=True)

    def _begin_task(self, source: int) -> np.ndarray:
        observation, _ = self._native.reset(seed=source)
        self._source = int(source)
        self._permutation = np.array(
            cast(Any, self._native.unwrapped).action_dirs, copy=True
        )
        self._step = 0
        self._episode_return = 0.0
        self._completions = []
        self._lap = 0
        self._lap_step = 0
        self._lap_first_step = 0
        self._lap_open = False
        self._pending_reset = False
        self._laps = []
        self._task_return = 0.0
        self._layout = 0
        return self._project(observation)

    def _reset_info(self) -> dict[str, Any]:
        return self._info(reward=0.0)

    def _info(
        self, *, reward: float, attempt_done: bool = False, reset_only: bool = False
    ) -> dict[str, Any]:
        """Evaluator-only counters; ``attempt_done`` marks the physical step that
        ended a lap (the ``attempt-cleared`` boundary), ``reset_only`` the call
        that started the next one."""
        return {
            "evaluator_task_index": self.evaluator_task_index,
            "step_in_episode": self._lap_step,
            "goal_reached": bool(reward),
            "goals_completed": len(self._completions),
            "episode_return": self._episode_return,
            "attempt_index": self._lap,
            "attempt_done": attempt_done,
            "reset_only": reset_only,
            "step_in_task": self._step,
            "task_return": self._task_return,
        }

    def _lap_record(self, *, complete: bool) -> MazeLap:
        goals = len(self._completions)
        return MazeLap(
            index=self._lap,
            first_step=self._lap_first_step,
            last_step=self._step,
            steps=self._lap_step,
            success=complete and goals == self.goals,
            native_return=float(goals),
            complete=complete,
            goals=goals,
            goal_steps=tuple(record.step for record in self._completions),
            layout=self._layout,
        )

    def _lap_reset(
        self,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """The reset-only call between laps: the native task is reset on the
        map's own seed (same maze, goals and permutation, the agent at the
        start), no action executes, nothing is paid, the call is charged."""
        assert self.meta_horizon is not None
        self._count_reset_only_step()
        period = self._layout_period
        if period is not None and self._step + 1 >= (self._layout + 1) * period:
            # A new maze and goal sequence at the trained size; the memory
            # carries on and nothing in the packet marks the change.
            self._layout += 1
        if self._layout == 0:
            observation, _ = self._native.reset(seed=self._source)
        else:
            observation, _ = self._native.reset(seed=self.relayout_seed(self._layout))
            # The task's hidden action permutation is kept across the change.
            cast(Any, self._native.unwrapped).action_dirs = np.array(
                self._permutation, copy=True
            )
        self._step += 1
        self._lap += 1
        self._lap_step = 0
        self._lap_first_step = 0
        self._lap_open = False
        self._episode_return = 0.0
        self._completions = []
        self._pending_reset = False
        self._current = self._project(observation)
        outer_terminated = self._step >= self.meta_horizon
        self._done = outer_terminated
        decision = PublicDecision(
            current=self._current, new_attempt=True, outer_terminated=outer_terminated
        )
        return (
            self._packet(decision),
            0.0,
            outer_terminated,
            False,
            self._info(reward=0.0, reset_only=True),
        )

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one native decision and build its causal public record."""
        selected = self._validate_action(action)
        if self._pending_reset:
            return self._lap_reset()
        observation, reward, terminated, truncated, _ = self._native.step(selected)
        self._step += 1
        self._lap_step += 1
        if self._lap_step == 1:
            self._lap_first_step = self._step
            self._lap_open = True
        if reward not in (0.0, 1.0):
            raise ContractError("Native MazeRunner pays one unit per goal.")
        self._episode_return += float(reward)
        self._task_return += float(reward)
        if reward:
            self._completions.append(
                GoalCompletion(index=len(self._completions), step=self._step)
            )
        previous = self._current
        self._current = self._project(observation)
        lap_done = bool(terminated or truncated)
        if self.meta_horizon is None:
            outer_terminated, outer_truncated = bool(terminated), bool(truncated)
        else:
            # A lap ends by the native rule; the outer task ends by its budget.
            # The lap's last record is an ordinary physical record (the trained
            # episode never flagged its end either); the boundary the policy
            # sees is the reset-only record that follows.
            if lap_done:
                self._laps.append(self._lap_record(complete=True))
                self._lap_open = False
            outer_terminated = self._step >= self.meta_horizon
            outer_truncated = False
            self._pending_reset = lap_done and not outer_terminated
        self._done = bool(outer_terminated or outer_truncated)
        decision = PublicDecision(
            current=self._current,
            previous=previous,
            outcome=self._current,
            executed_action=selected,
            reward=float(reward),
            outer_terminated=outer_terminated,
            outer_truncated=outer_truncated,
        )
        return (
            self._packet(decision),
            float(reward),
            outer_terminated,
            outer_truncated,
            self._info(reward=float(reward), attempt_done=lap_done),
        )

    def close(self) -> None:
        cast(Any, self._native).close()

    # ------------------------------------------------------------------
    # Restorable state
    # ------------------------------------------------------------------

    def _identity(self) -> tuple[object, ...]:
        # The trained identity is unchanged; a laps task names its budget, so a
        # one-episode snapshot never restores into a laps task or back.
        identity: tuple[object, ...] = (
            self.variant,
            self.size,
            self.goals,
            self.horizon,
        )
        if self.meta_horizon is not None:
            identity = (*identity, ("laps", self.meta_horizon))
        return identity

    def _state(self) -> dict[str, object]:
        native = cast(Any, self._native.unwrapped)
        laps: dict[str, object] = {}
        if self.meta_horizon is not None:
            laps = {
                "laps": {
                    "source": self._source,
                    "lap": self._lap,
                    "lap_step": self._lap_step,
                    "lap_first_step": self._lap_first_step,
                    "lap_open": self._lap_open,
                    "pending_reset": self._pending_reset,
                    "task_return": self._task_return,
                    "layout": self._layout,
                    "layout_period": self._layout_period,
                    "permutation": self._permutation.tolist(),
                    "records": [asdict(record) for record in self._laps],
                }
            }
        return {
            **laps,
            "step": self._step,
            "episode_return": self._episode_return,
            "completions": [asdict(record) for record in self._completions],
            "native": {
                "maze": np.asarray(native.maze).tolist(),
                "start": [int(value) for value in native.start],
                "goal_positions": [
                    [int(value) for value in goal] for goal in native.goal_positions
                ],
                "action_dirs": np.asarray(native.action_dirs).tolist(),
                "active_goal_idx": int(native.active_goal_idx),
                "pos": [int(value) for value in native.pos],
                "timer": int(native.timer),
                "enforce_reset": bool(native._enforce_reset),
            },
            "native_rng": deepcopy(native.rng.bit_generator.state),
            "task_rng": deepcopy(self._native.generator_state()),
        }

    def _load_state(self, state: Mapping[str, object]) -> None:
        native_state = state.get("native")
        completions = state.get("completions")
        if not isinstance(native_state, Mapping) or not isinstance(
            completions, Sequence
        ):
            raise ContractError("MazeRunner checkpoint is malformed.")
        restored = [GoalCompletion(**cast(Any, row)) for row in completions]
        native = cast(Any, self._native.unwrapped)
        native.maze = np.asarray(native_state["maze"], dtype=np.int64)
        native.start = tuple(int(v) for v in cast(Any, native_state["start"]))
        native.goal_positions = [
            tuple(int(v) for v in goal)
            for goal in cast(Any, native_state["goal_positions"])
        ]
        native.action_dirs = np.asarray(native_state["action_dirs"], dtype=np.int64)
        native.active_goal_idx = int(cast(Any, native_state["active_goal_idx"]))
        native.pos = tuple(int(v) for v in cast(Any, native_state["pos"]))
        native.timer = int(cast(Any, native_state["timer"]))
        native._enforce_reset = bool(native_state["enforce_reset"])
        native.rng.bit_generator.state = deepcopy(cast(Any, state["native_rng"]))
        self._native.load_generator_state(cast(Any, state["task_rng"]))
        self._step = int(cast(Any, state["step"]))
        self._episode_return = float(cast(Any, state["episode_return"]))
        self._completions = restored
        laps = state.get("laps")
        if (laps is None) != (self.meta_horizon is None) or (
            laps is not None and not isinstance(laps, Mapping)
        ):
            raise ContractError("MazeRunner checkpoint's laps block is malformed.")
        if laps is None:
            # A one-episode task: the live episode is lap 0 and its step count
            # is the task's, so a snapshot written before the laps axis (or by
            # the plain task) restores the same evaluator counters.
            self._lap = 0
            self._lap_step = self._step
            self._lap_first_step = 1 if self._step else 0
            self._lap_open = self._step > 0 and not self._done
            self._pending_reset = False
            self._laps = []
            self._task_return = self._episode_return
        else:
            block = cast(Mapping[str, Any], laps)
            self._source = int(block["source"])
            self._lap = int(block["lap"])
            self._lap_step = int(block["lap_step"])
            self._lap_first_step = int(block["lap_first_step"])
            self._lap_open = bool(block["lap_open"])
            self._pending_reset = bool(block["pending_reset"])
            self._task_return = float(block["task_return"])
            self._layout = int(block.get("layout", 0))
            self._layout_period = block.get("layout_period")
            self._permutation = np.asarray(
                block.get("permutation", native_state["action_dirs"]), dtype=np.int64
            )
            self._laps = [
                MazeLap(**{**row, "goal_steps": tuple(row["goal_steps"])})
                for row in block["records"]
            ]


# ----------------------------------------------------------------------
# Hindsight relabeling
# ----------------------------------------------------------------------


def rebuild_transition_packet(traj: FrozenTraj) -> FrozenTraj:
    """Rebuild every derived field from the relabeled observation stream.

    Only ``obs['current']``, ``actions``, ``rews``, ``dones`` and ``time_idxs``
    are read. The stored ``previous``, ``outcome``, ``event``, ``valid`` and
    ``rl2s`` are overwritten unconditionally, so no precomputed transition field
    can survive relabeling. A trained MazeRunner task has no reset-only step:
    every decision executes a physical action whose endpoint is the observation
    it returned (the evaluation-only repeated-laps axis never trains).
    """
    current = np.asarray(traj.obs["current"], dtype=np.float32)
    if current.ndim != 2 or current.shape[0] < 2:
        raise ContractError("MazeRunner reconstruction needs a token sequence.")
    tokens = current.shape[0]
    actions = np.asarray(traj.actions, dtype=np.float32)
    rews = np.asarray(traj.rews, dtype=np.float32).reshape(-1, 1)
    if actions.shape[0] != tokens - 1 or rews.shape[0] != tokens - 1:
        raise ContractError("MazeRunner reconstruction has inconsistent lengths.")
    previous = np.zeros_like(current)
    previous[1:] = current[:-1]
    outcome = np.zeros_like(current)
    outcome[1:] = current[1:]
    event = np.zeros((tokens, 3), dtype=np.float32)
    event[0] = (0.0, 1.0, 0.0)
    event[1:] = (1.0, 0.0, 0.0)
    rl2s = np.zeros((tokens, 1 + actions.shape[1]), dtype=np.float32)
    rl2s[1:, :1] = rews
    rl2s[1:, 1:] = actions
    return FrozenTraj(
        obs={
            "current": current,
            "previous": previous,
            "outcome": outcome,
            "event": event,
            "valid": np.ones((tokens, 1), dtype=np.float32),
        },
        rl2s=rl2s,
        time_idxs=np.arange(tokens, dtype=np.int64).reshape(-1, 1),
        rews=rews,
        dones=np.asarray(traj.dones, dtype=bool).reshape(-1, 1),
        actions=actions,
    )


class HindsightGoalRelabeler(Relabeler):  # type: ignore[misc]
    """Native-inspired hindsight instructions with causal packet reconstruction.

    ``some`` samples a replacement count uniformly from zero through the number
    of uncompleted goals; ``all`` requests all of them. Alternative achieved
    timesteps are sampled and merged chronologically with genuinely achieved
    goals. Remaining original goals stay at the tail, allowing valid timeout
    trajectories. The pinned example is documented in the M5 retry report.

    Project corrections exclude time-zero and repeated goal coordinates, keep
    native reward/termination semantics, and reconstruct every derived packet
    field. This is not an exact upstream implementation.
    """

    def __init__(
        self,
        *,
        size: int,
        goals: int,
        horizon: int,
        strategy: str = "some",
        seed: int = 0,
    ) -> None:
        if strategy not in RELABEL_STRATEGIES:
            raise ContractError(f"Unknown relabeling strategy: {strategy!r}.")
        self.size = size
        self.goals = goals
        self.horizon = horizon
        self.strategy = strategy
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _positions(self, current: NDArray[np.float32]) -> NDArray[np.int64]:
        return decode_grid(current[:, :2], self.size)

    def relabel(self, traj: Trajectory | FrozenTraj) -> FrozenTraj:
        frozen = traj.freeze() if isinstance(traj, Trajectory) else traj
        if self.strategy == "none":
            return frozen
        current = np.asarray(frozen.obs["current"], dtype=np.float32)
        tokens = current.shape[0]
        rews = np.asarray(frozen.rews, dtype=np.float32).reshape(-1)
        if tokens < 2 or rews.shape[0] != tokens - 1:
            raise ContractError("MazeRunner relabeling needs a complete trajectory.")
        positions = self._positions(current)
        start = tuple(int(value) for value in positions[0])
        kept = decode_grid(current[0, MAZERUNNER_OBSERVATION_WIDTH:], self.size)
        kept = kept.reshape(self.goals, 2)
        completed = round(float(rews.sum()))
        if completed >= self.goals:
            return frozen
        needed = self.goals - completed
        replacements = (
            needed if self.strategy == "all" else int(self._rng.integers(needed + 1))
        )
        if replacements == 0:
            return frozen
        reached = np.flatnonzero(rews > 0.0) + 1
        actual = [
            (int(step), (int(kept[index, 0]), int(kept[index, 1])))
            for index, step in enumerate(reached)
        ]
        # Select timestep alternatives randomly, rejecting duplicate coordinates
        # and original goals (which remain available at the uncompleted tail).
        excluded = {tuple(int(v) for v in goal) for goal in kept}
        excluded.add(start)
        alternatives: list[tuple[int, tuple[int, int]]] = []
        for step in self._rng.permutation(np.arange(1, tokens)):
            spot = tuple(int(v) for v in positions[step])
            if spot in excluded:
                continue
            alternatives.append((int(step), (spot[0], spot[1])))
            excluded.add(spot)
            if len(alternatives) == replacements:
                break
        if not alternatives:
            return frozen
        achieved = sorted(actual + alternatives)
        instruction = [spot for _, spot in achieved]
        instruction += [
            (int(goal[0]), int(goal[1])) for goal in kept[len(instruction) :]
        ]
        return self._replay(frozen, current, positions, instruction)

    def _replay(
        self,
        frozen: FrozenTraj,
        current: NDArray[np.float32],
        positions: NDArray[np.int64],
        instruction: Sequence[tuple[int, int]],
    ) -> FrozenTraj:
        """Re-derive rewards, progress and termination under the new goals.

        This mirrors the pinned ``MazeRunnerGymEnv.step``: one goal check per
        step against the active goal only, reward one on success, the sequence
        advancing by one, and termination at the final goal.
        """
        tokens = current.shape[0]
        active = 0
        rewards = np.zeros((tokens - 1, 1), dtype=np.float32)
        progress = np.zeros(tokens, dtype=np.int64)
        end: int | None = None
        for step in range(1, tokens):
            spot = (int(positions[step][0]), int(positions[step][1]))
            if active < len(instruction) and spot == instruction[active]:
                rewards[step - 1, 0] = 1.0
                active += 1
                if active == len(instruction):
                    end = step
            progress[step] = active
            if end is not None:
                break
        if end is None:
            # Incomplete instructions legitimately terminate at the native time
            # limit. A shorter unfinished fragment is not a full-task replay.
            if tokens - 1 != self.horizon or not bool(frozen.dones[-1]):
                raise ContractError(
                    "Uncompleted relabeling needs a native horizon terminal."
                )
            end = tokens - 1
        published = np.array(current[: end + 1], dtype=np.float32, copy=True)
        for token in range(end + 1):
            goals = np.array(instruction, dtype=np.int64)
            goals[: progress[token]] = GOAL_SENTINEL
            published[token, MAZERUNNER_OBSERVATION_WIDTH:] = encode_grid(
                goals.reshape(-1), self.size
            )
        dones = np.zeros((end, 1), dtype=bool)
        dones[end - 1, 0] = True
        return FrozenTraj(
            obs={"current": published},
            rl2s=np.zeros((end + 1, 1 + frozen.actions.shape[1]), dtype=np.float32),
            time_idxs=np.arange(end + 1, dtype=np.int64).reshape(-1, 1),
            rews=rewards[:end],
            dones=dones,
            actions=np.asarray(frozen.actions[:end], dtype=np.float32),
        )


__all__ = [
    "GOAL_SENTINEL",
    "MAZERUNNER_ACTIONS",
    "MAZERUNNER_OBSERVATION_WIDTH",
    "MAZERUNNER_PROTOCOLS",
    "MAZERUNNER_VARIANTS",
    "RELABEL_STRATEGIES",
    "TIMER_INDEX",
    "GoalCompletion",
    "HindsightGoalRelabeler",
    "MazeRunnerEnv",
    "decode_grid",
    "encode_grid",
    "mazerunner_protocol",
    "mazerunner_variant",
    "public_fields",
    "public_width",
    "rebuild_transition_packet",
]
