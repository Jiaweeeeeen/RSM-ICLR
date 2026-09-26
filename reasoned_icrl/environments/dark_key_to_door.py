"""Native AMAGO v3.4.0 Dark Key-to-Door as a public-packet policy environment.

Wraps the pinned ``amago.envs.builtin.toy_gym.RoomKeyDoor`` without editing
upstream source. The native lifecycle is kept exactly as it is: one hidden
start/key/door layout persists for the whole ``meta_rollout_horizon`` budget, a
physical attempt ends at door completion or at the ``max_episode_steps`` limit,
and the *following* native step performs a soft reset that ignores whichever
action was selected for it.

This adapter therefore distinguishes three kinds of timestep:

* the initial observation, which carries no physical evidence;
* a physical step, whose selected action moved the agent and whose public
  endpoint is the observation it returned;
* a reset-only step, which executed no physical action at all.

Only the four public Box values reach the policy. Hidden key/door/start
coordinates and the native action permutation are never read for an
observation; :meth:`DarkKeyToDoorEnv.render` draws them for a human observer
only. Per-attempt outcomes are retained on the environment for the evaluator;
they are not part of any observation and never enter ``info`` as structured
objects.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from reasoned_icrl.environments.base import (
    Attempt,
    BaseEnv,
    PublicDecision,
    benchmark_task_sources,
    unit_fields,
)
from reasoned_icrl.environments.rendering import (
    PALETTE,
    ansi_frame,
    color_grid,
    paint_cells,
)
from reasoned_icrl.environments.utils import NativeSource, project_unit_interval
from reasoned_icrl.experiments.contracts import ContractError

KEY_TO_DOOR_PROTOCOL = "native-keydoor-fixed500-first8"
KEY_TO_DOOR_STATE_SCHEMA = "dark-key-to-door-native-state.v1"
KEY_TO_DOOR_ACTIONS = 5
KEY_TO_DOOR_FIELDS: tuple[str, ...] = (
    "position_x",
    "position_y",
    "has_key",
    "episode_time",
)
"""The four native Box values, named for provenance and leakage review."""
KEY_TO_DOOR_ACTION_NAMES: tuple[str, ...] = ("left", "up", "right", "down", "stay")
"""The native action table under the fixed (non-randomised) permutation:
``dirs = [[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]]`` over ``(row, column)``,
the axes the native ``render`` draws. Observer labels only."""


@dataclass(frozen=True, slots=True)
class KeyToDoorAttempt(Attempt):
    """One attempt with one-based key and door timings, or ``None``.

    ``layout`` counts the hidden layouts the task has had so far: 0 for the
    native task, ``k`` after the ``k``-th change of the evaluation-only
    layout-change continuation (:meth:`DarkKeyToDoorEnv.set_layout_period`).
    """

    key_step: int | None
    door_step: int | None
    layout: int = 0

    def __post_init__(self) -> None:
        Attempt.__post_init__(self)
        if self.success != (self.door_step is not None):
            raise ContractError("Door success and door timing disagree.")
        expected = float(self.key_step is not None) + float(self.door_step is not None)
        if self.native_return != expected:
            raise ContractError("Native attempt return disagrees with its events.")
        if self.door_step is not None and (
            self.key_step is None or self.key_step > self.door_step
        ):
            raise ContractError("A door cannot be opened before the key is acquired.")


class DarkKeyToDoorEnv(BaseEnv):
    """Public five-key packet over the pinned native Key-to-Door task."""

    label: ClassVar[str] = "Dark Key-to-Door"
    state_schema: ClassVar[str] = KEY_TO_DOOR_STATE_SCHEMA
    protocol = KEY_TO_DOOR_PROTOCOL

    def __init__(
        self,
        *,
        size: int = 8,
        physical_horizon: int = 50,
        meta_horizon: int = 500,
        scored_attempts: int = 8,
        randomized_actions: bool = False,
        split: str = "train",
        source_indices: Sequence[int] | range | None = None,
        fixed_task_index: int | None = None,
        initial_seed: int = 0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(
            split=split,
            roster=benchmark_task_sources(split)
            if source_indices is None
            else source_indices,
            fixed_task_index=fixed_task_index,
            initial_seed=initial_seed,
            render_mode=render_mode,
        )
        self.size = size
        self.horizon = physical_horizon
        self.meta_horizon = meta_horizon
        self.attempts = scored_attempts
        self.randomized_actions = randomized_actions
        self._native = NativeSource(self._make_native, seed=initial_seed)
        self._install_contract(unit_fields(KEY_TO_DOOR_FIELDS), KEY_TO_DOOR_ACTIONS)
        self._pending_reset = False
        self._has_key = False
        self._attempt = 0
        self._attempt_step = 0
        self._attempt_first_step = 0
        self._attempt_return = 0.0
        self._key_step: int | None = None
        self._door_step: int | None = None
        self._global_step = 0
        self._task_return = 0.0
        self._records: list[KeyToDoorAttempt] = []
        self._layout_period: int | None = None
        self._layout = 0
        self._source = 0

    def set_layout_period(self, period: int | None) -> None:
        """Evaluation-only layout-change continuation.

        With a period ``P``, a new hidden start, key and door replace the
        task's layout at the first attempt boundary at or after every
        ``k * P``-th charged call, ``k >= 1``; the agent's memory is never
        reset, the public packet carries no signal of the change, and the
        first ``P`` calls are the native task exactly. Layout ``k`` is drawn
        from its own generator seeded by the task source and ``k``, so it is
        deterministic per task and independent of the native RNG stream.
        """
        if period is not None and (
            isinstance(period, bool) or type(period) is not int or period < 1
        ):
            raise ContractError("The layout period is a positive number of calls.")
        self._layout_period = period

    @property
    def layout_period(self) -> int | None:
        return self._layout_period

    @property
    def layout_index(self) -> int:
        """Hidden layouts so far: 0 for the native task, ``k`` after ``k`` changes."""
        return self._layout

    def _change_layout(self) -> None:
        """Draw the next hidden layout for the attempt about to begin."""
        native = cast(Any, self._native.unwrapped)
        self._layout += 1
        draw = random.Random(f"relayout:{self._source}:{self._layout}")
        native.start = np.array(draw.choices(range(self.size), k=2))
        native.key = np.array(draw.choices(range(self.size), k=2))
        native.goal = np.array(draw.choices(range(self.size), k=2))

    def _make_native(self) -> gym.Env[Any, Any]:
        from amago.envs.builtin.toy_gym import RoomKeyDoor

        return cast(
            gym.Env[Any, Any],
            RoomKeyDoor(
                dark=True,
                size=self.size,
                max_episode_steps=self.horizon,
                meta_rollout_horizon=self.meta_horizon,
                randomize_actions=self.randomized_actions,
            ),
        )

    @property
    def native(self) -> gym.Env[Any, Any]:
        """The wrapped upstream task, for inspection and audits only."""
        return self._native

    # ------------------------------------------------------------------
    # Evaluator-only accessors
    # ------------------------------------------------------------------

    @property
    def completed_attempts(self) -> tuple[KeyToDoorAttempt, ...]:
        """Every attempt that reached a physical boundary in the live task."""
        return tuple(self._records)

    @property
    def task_return(self) -> float:
        """Unscaled native return accumulated over the whole meta budget."""
        return self._task_return

    @property
    def partial_attempt(self) -> KeyToDoorAttempt | None:
        """The unfinished attempt at the outer boundary, when one exists."""
        if not self._done or self._pending_reset or self._attempt_step < 1:
            return None
        return self._attempt_record(complete=False)

    def public_fields(self, packet: Mapping[str, np.ndarray]) -> dict[str, float]:
        """Decode one packet's ``current`` token back to named native values."""
        current = np.asarray(packet["current"], dtype=np.float32)
        if current.shape != (len(self.fields),):
            raise ContractError("Packet width disagrees with the public contract.")
        return {
            field.name: float((value + 1.0) / 2.0)
            for field, value in zip(self.fields, current, strict=True)
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _project(self, observation: object) -> np.ndarray:
        return project_unit_interval(observation, len(self.fields), label=self.label)

    def _attempt_record(self, *, complete: bool) -> KeyToDoorAttempt:
        return KeyToDoorAttempt(
            index=self._attempt,
            first_step=self._attempt_first_step,
            last_step=self._global_step,
            steps=self._attempt_step,
            success=complete and self._door_step is not None,
            native_return=self._attempt_return,
            complete=complete,
            key_step=self._key_step,
            door_step=self._door_step if complete else None,
            layout=self._layout,
        )

    def _begin_task(self, source: int) -> np.ndarray:
        observation, _ = self._native.reset(seed=source)
        self._source = int(source)
        self._layout = 0
        self._pending_reset = False
        self._has_key = False
        self._attempt = 0
        self._attempt_step = 0
        self._attempt_first_step = 0
        self._attempt_return = 0.0
        self._key_step = None
        self._door_step = None
        self._global_step = 0
        self._task_return = 0.0
        self._records = []
        return self._project(observation)

    def _reset_info(self) -> dict[str, Any]:
        return {
            "evaluator_task_index": self.evaluator_task_index,
            "attempt_index": 0,
            "attempt_done": False,
            "attempt_success": False,
            "attempt_return": 0.0,
            "attempt_steps": 0,
            "reset_only": False,
            "step_in_task": 0,
            "task_return": 0.0,
        }

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one native decision and build its causal public record."""
        selected = self._validate_action(action)
        if (
            self._pending_reset
            and self._layout_period is not None
            and self._global_step + 1 < self.meta_horizon
            and self._global_step + 1 >= (self._layout + 1) * self._layout_period
        ):
            # The reset-only call about to run starts the next attempt from
            # the new layout's start; the change is invisible to the packet.
            self._change_layout()
        observation, reward, terminated, truncated, _ = self._native.step(selected)
        current = self._project(observation)
        self._global_step += 1
        reset_only = self._pending_reset
        if reset_only:
            # The native task ignored this command and soft-reset the attempt.
            self._count_reset_only_step()
            if reward:
                raise ContractError("A reset-only native step cannot pay a reward.")
            self._attempt += 1
            self._attempt_step = 0
            self._attempt_first_step = 0
            self._attempt_return = 0.0
            self._key_step = None
            self._door_step = None
            self._has_key = False
            self._pending_reset = False
            decision = PublicDecision(
                current=current,
                new_attempt=True,
                outer_terminated=terminated,
                outer_truncated=truncated,
            )
            attempt_done = False
        else:
            self._attempt_step += 1
            if self._attempt_step == 1:
                self._attempt_first_step = self._global_step
            self._attempt_return += float(reward)
            self._task_return += float(reward)
            had_key = self._has_key
            self._has_key = bool(current[2] > 0.0)
            if reward == 1.0 and not had_key:
                self._key_step = self._attempt_step
            elif reward == 1.0:
                self._door_step = self._attempt_step
            elif reward:
                raise ContractError("Native Key-to-Door pays only unit rewards.")
            timed_out = self._attempt_step >= self.horizon
            attempt_done = self._door_step is not None or timed_out
            self._pending_reset = attempt_done
            decision = PublicDecision(
                current=current,
                previous=self._current,
                outcome=current,
                executed_action=selected,
                reward=float(reward),
                physical_done=attempt_done,
                outer_terminated=terminated,
                outer_truncated=truncated,
            )
            if attempt_done:
                self._records.append(self._attempt_record(complete=True))
        self._current = current
        self._done = bool(terminated or truncated)
        info = {
            "evaluator_task_index": self.evaluator_task_index,
            "attempt_index": self._attempt,
            "attempt_done": attempt_done,
            "attempt_success": attempt_done and self._door_step is not None,
            "attempt_return": self._attempt_return if attempt_done else 0.0,
            "attempt_steps": self._attempt_step if attempt_done else 0,
            "reset_only": reset_only,
            "step_in_task": self._global_step,
            "task_return": self._task_return,
        }
        return self._packet(decision), float(reward), terminated, truncated, info

    def close(self) -> None:
        cast(Any, self._native).close()

    # ------------------------------------------------------------------
    # Rendering (observer only; reads the hidden layout, never the packet)
    # ------------------------------------------------------------------

    @property
    def action_names(self) -> tuple[str, ...]:
        """Observer labels for the five actions; opaque under randomisation."""
        if self.randomized_actions:
            return tuple(f"action {index}" for index in range(self.action_count))
        return KEY_TO_DOOR_ACTION_NAMES

    def _layout_cells(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """``(agent, key, door)`` as ``(row, column)`` cells of the hidden layout."""
        native = cast(Any, self._native.unwrapped)
        cells = []
        for value in (native.pos, native.key, native.goal):
            row, column = (int(part) for part in np.asarray(value).reshape(2))
            cells.append((row, column))
        return cells[0], cells[1], cells[2]

    def _render_header(self) -> list[str]:
        attempt = self._attempt + 1
        if self._pending_reset:
            phase = "attempt ended, next call resets"
        elif self._done:
            phase = "task over"
        else:
            phase = f"step {self._attempt_step}/{self.horizon}"
        key = "held" if self._has_key else "not held"
        if self._key_step is not None:
            key += f" (step {self._key_step})"
        door = f"opened at step {self._door_step}" if self._door_step else "closed"
        return [
            f"{self.label}  task {self.evaluator_task_index}  "
            f"call {self._global_step}/{self.meta_horizon}  attempt {attempt}  "
            f"{phase}  return {self._task_return:g}",
            f"key {key}  door {door}  doors so far "
            f"{sum(record.success for record in self._records)}",
        ]

    def _render_ansi(self) -> str:
        agent, key, door = self._layout_cells()
        rows: list[str] = []
        for row in range(self.size):
            cells: list[str] = []
            for column in range(self.size):
                cell = (row, column)
                if cell == agent:
                    glyph = "A"
                elif cell == door:
                    glyph = "D"
                elif cell == key:
                    glyph = "k" if self._has_key else "K"
                else:
                    glyph = "."
                cells.append(glyph)
            rows.append(" ".join(cells))
        return ansi_frame(self._render_header(), rows)

    def _render_rgb(self) -> NDArray[np.uint8]:
        agent, key, door = self._layout_cells()
        board = color_grid(self.size, self.size, PALETTE["floor"])
        board[key] = PALETTE["key_taken" if self._has_key else "key"]
        board[door] = PALETTE["door"]
        board[agent] = PALETTE["agent"]
        return paint_cells(board)

    # ------------------------------------------------------------------
    # Restorable state
    # ------------------------------------------------------------------

    def _identity(self) -> tuple[object, ...]:
        return (
            self.size,
            self.horizon,
            self.meta_horizon,
            self.attempts,
            self.randomized_actions,
        )

    def _state(self) -> dict[str, object]:
        native = cast(Any, self._native.unwrapped)
        return {
            "pending_reset": self._pending_reset,
            "has_key": self._has_key,
            "attempt": self._attempt,
            "attempt_step": self._attempt_step,
            "attempt_first_step": self._attempt_first_step,
            "attempt_return": self._attempt_return,
            "key_step": self._key_step,
            "door_step": self._door_step,
            "global_step": self._global_step,
            "task_return": self._task_return,
            "layout_period": self._layout_period,
            "layout": self._layout,
            "source": self._source,
            "records": [asdict(record) for record in self._records],
            "native": {
                "start": np.asarray(native.start).tolist(),
                "key": np.asarray(native.key).tolist(),
                "goal": np.asarray(native.goal).tolist(),
                "dirs": [list(pair) for pair in native.dirs],
                "pos": np.asarray(native.pos).tolist(),
                "episode_time": int(native.episode_time),
                "global_time": int(native.global_time),
                "native_has_key": bool(native.has_key),
                "reset_next_step": bool(native.reset_next_step),
            },
            "task_rng": deepcopy(self._native.generator_state()),
        }

    def _load_state(self, state: Mapping[str, object]) -> None:
        native_state = state.get("native")
        records = state.get("records")
        if not isinstance(native_state, Mapping) or not isinstance(records, Sequence):
            raise ContractError("Dark Key-to-Door checkpoint is malformed.")
        restored = [KeyToDoorAttempt(**cast(Any, row)) for row in records]
        native = cast(Any, self._native.unwrapped)
        native.start = np.asarray(native_state["start"])
        native.key = np.asarray(native_state["key"])
        native.goal = np.asarray(native_state["goal"])
        native.dirs = [list(pair) for pair in cast(Any, native_state["dirs"])]
        native.pos = np.asarray(native_state["pos"])
        native.episode_time = int(cast(Any, native_state["episode_time"]))
        native.global_time = int(cast(Any, native_state["global_time"]))
        native.has_key = bool(native_state["native_has_key"])
        native.reset_next_step = bool(native_state["reset_next_step"])
        self._native.load_generator_state(cast(Any, state["task_rng"]))
        self._pending_reset = bool(state["pending_reset"])
        self._has_key = bool(state["has_key"])
        self._attempt = int(cast(Any, state["attempt"]))
        self._attempt_step = int(cast(Any, state["attempt_step"]))
        self._attempt_first_step = int(cast(Any, state["attempt_first_step"]))
        self._attempt_return = float(cast(Any, state["attempt_return"]))
        key_step = state.get("key_step")
        door_step = state.get("door_step")
        self._key_step = None if key_step is None else int(cast(Any, key_step))
        self._door_step = None if door_step is None else int(cast(Any, door_step))
        self._global_step = int(cast(Any, state["global_step"]))
        self._task_return = float(cast(Any, state["task_return"]))
        period = state.get("layout_period")
        self._layout_period = None if period is None else int(cast(Any, period))
        self._layout = int(cast(Any, state.get("layout", 0)))
        self._source = int(cast(Any, state.get("source", 0)))
        self._records = restored


__all__ = [
    "KEY_TO_DOOR_ACTIONS",
    "KEY_TO_DOOR_ACTION_NAMES",
    "KEY_TO_DOOR_FIELDS",
    "KEY_TO_DOOR_PROTOCOL",
    "DarkKeyToDoorEnv",
    "KeyToDoorAttempt",
]
