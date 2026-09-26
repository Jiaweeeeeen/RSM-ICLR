"""DarkRoom: a hidden goal cell that persists for K physical attempts.

The agent starts every attempt at the centre of a ``size`` x ``size`` grid and
sees only its own position and the attempt/step counters. Reaching the hidden
goal pays one and ends the attempt; otherwise the attempt ends at ``horizon``
steps. Internal attempt resets keep the task and the interaction history; the
outer task ends after ``attempts`` attempts.

Task identities are goal cells ``row * size + column``. The quadrant partition
splits the 5x5 goal set into train, IID and OOD rosters so that adaptation to
unseen goals can be measured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, ClassVar, cast

import numpy as np

from reasoned_icrl.environments.base import (
    Attempt,
    BaseEnv,
    PublicDecision,
    PublicField,
)
from reasoned_icrl.environments.utils import normalize_fields, validate_roster
from reasoned_icrl.experiments.contracts import ContractError

DARKROOM_PROTOCOL = "darkroom-5x5-quadrant-v1"
DARKROOM_STATE_SCHEMA = "darkroom-state.v2"
DARKROOM_GOAL_PARTITION = "quadrant-v1"
DARKROOM_SPLITS = ("train", "validation", "iid", "ood")
DARKROOM_TRAIN_SOURCE_INDICES = (1, 2, 4, 5, 6, 7, 9, 10, 11, 13, 15, 16, 20, 21, 22)
DARKROOM_IID_SOURCE_INDICES = (0, 3, 8, 14, 17)
DARKROOM_OOD_SOURCE_INDICES = (18, 19, 23, 24)
DARKROOM_ACTIONS = 5
_MOVES = np.asarray(((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)), dtype=np.int16)


def darkroom_source_indices(
    *, size: int, split: str, goal_partition: str | None
) -> tuple[int, ...]:
    """Resolve the ordered task roster for one DarkRoom split."""
    if split not in DARKROOM_SPLITS:
        raise ContractError(f"Unknown DarkRoom split: {split!r}.")
    if goal_partition is None:
        if split == "ood":
            raise ContractError("All-goal DarkRoom has no OOD-goal split.")
        return tuple(range(size * size))
    if goal_partition not in (DARKROOM_GOAL_PARTITION, DARKROOM_PROTOCOL):
        raise ContractError(f"Unknown DarkRoom goal partition: {goal_partition!r}.")
    if size != 5:
        raise ContractError(f"{DARKROOM_PROTOCOL} requires a 5x5 grid.")
    if split == "train":
        return DARKROOM_TRAIN_SOURCE_INDICES
    if split in ("validation", "iid"):
        return DARKROOM_IID_SOURCE_INDICES
    return DARKROOM_OOD_SOURCE_INDICES


def public_fields(size: int, attempts: int, horizon: int) -> tuple[PublicField, ...]:
    """The public observation: position and the attempt/step counters."""
    return (
        PublicField("position", (2,), "int16", (0.0, 0.0), (size - 1.0, size - 1.0)),
        PublicField("attempt", (1,), "int16", (0.0,), (attempts - 1.0,)),
        PublicField("attempt_step", (1,), "int16", (0.0,), (float(horizon),)),
        PublicField("task_step", (1,), "int32", (0.0,), (float(attempts * horizon),)),
        PublicField("attempt_boundary", (1,), "uint8", (0.0,), (1.0,)),
    )


class DarkRoomEnv(BaseEnv):
    """Public five-key packet over the K-attempt hidden-goal grid."""

    label: ClassVar[str] = "DarkRoom"
    state_schema: ClassVar[str] = DARKROOM_STATE_SCHEMA

    def __init__(
        self,
        *,
        size: int = 5,
        attempts: int = 5,
        horizon: int = 32,
        split: str = "train",
        goal_partition: str | None = DARKROOM_GOAL_PARTITION,
        source_indices: Sequence[int] | None = None,
        fixed_task_index: int | None = None,
        initial_seed: int = 0,
    ) -> None:
        if size < 3 or attempts <= 0 or horizon <= 0:
            raise ContractError("DarkRoom dimensions must be positive.")
        self.protocol = (
            DARKROOM_PROTOCOL if goal_partition else f"darkroom-{size}x{size}-all"
        )
        partition = darkroom_source_indices(
            size=size, split=split, goal_partition=goal_partition
        )
        super().__init__(
            split=split,
            roster=partition
            if source_indices is None
            else validate_roster(source_indices, partition, label=self.label),
            fixed_task_index=fixed_task_index,
            initial_seed=initial_seed,
        )
        self.size = size
        self.attempts = attempts
        self.horizon = horizon
        self.goal_partition = goal_partition
        self._install_contract(public_fields(size, attempts, horizon), DARKROOM_ACTIONS)
        self._low = np.concatenate([field.low for field in self.fields])
        self._high = np.concatenate([field.high for field in self.fields])
        self._goal = np.zeros(2, dtype=np.int16)
        self._position = np.zeros(2, dtype=np.int16)
        self._attempt = 0
        self._attempt_step = 0
        self._attempt_first_step = 0
        self._attempt_return = 0.0
        self._task_step = 0
        self._task_return = 0.0
        self._boundary = 1
        self._records: list[Attempt] = []

    # ------------------------------------------------------------------
    # Evaluator-only accessors
    # ------------------------------------------------------------------

    @property
    def completed_attempts(self) -> tuple[Attempt, ...]:
        """Every attempt that reached its boundary in the live task."""
        return tuple(self._records)

    @property
    def task_return(self) -> float:
        return self._task_return

    @property
    def partial_attempt(self) -> Attempt | None:
        """DarkRoom tasks always end on an attempt boundary."""
        return None

    @property
    def goal(self) -> tuple[int, int]:
        """The hidden goal cell. Evaluator-only; never a policy input."""
        return int(self._goal[0]), int(self._goal[1])

    def public_fields(self, packet: Mapping[str, np.ndarray]) -> dict[str, Any]:
        """Decode one packet's ``current`` token back to named native values."""
        current = np.asarray(packet["current"], dtype=np.float64)
        if current.shape != (self.width,):
            raise ContractError("Packet width disagrees with the public contract.")
        raw = (current + 1) / 2 * np.maximum(self._high - self._low, 1) + self._low
        values = [int(value) for value in np.rint(raw)]
        return {
            "position": (values[0], values[1]),
            "attempt": values[2],
            "attempt_step": values[3],
            "task_step": values[4],
            "attempt_boundary": values[5],
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        raw = np.array(
            (
                *self._position,
                self._attempt,
                self._attempt_step,
                self._task_step,
                self._boundary,
            ),
            dtype=np.float64,
        )
        return normalize_fields(raw, self._low, self._high)

    def _start_position(self) -> np.ndarray:
        center = self.size // 2
        return np.asarray((center, center), dtype=np.int16)

    def _begin_task(self, source: int) -> np.ndarray:
        self._goal = np.asarray(divmod(source, self.size), dtype=np.int16)
        self._position = self._start_position()
        self._attempt = 0
        self._attempt_step = 0
        self._attempt_first_step = 0
        self._attempt_return = 0.0
        self._task_step = 0
        self._task_return = 0.0
        self._boundary = 1
        self._records = []
        return self._observation()

    def _info(self, *, attempt_done: bool, success: bool) -> dict[str, Any]:
        return {
            "evaluator_task_index": self.evaluator_task_index,
            "attempt_index": self._records[-1].index if attempt_done else self._attempt,
            "attempt_done": attempt_done,
            "attempt_success": success,
            "attempt_return": self._records[-1].native_return if attempt_done else 0.0,
            "attempt_steps": self._records[-1].steps if attempt_done else 0,
            "reset_only": False,
            "step_in_task": self._task_step,
            "task_return": self._task_return,
        }

    def _reset_info(self) -> dict[str, Any]:
        return self._info(attempt_done=False, success=False)

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Move once; an attempt ends at the goal or at the horizon."""
        selected = self._validate_action(action)
        self._boundary = 0
        candidate = self._position + _MOVES[selected]
        self._position = np.clip(candidate, 0, self.size - 1).astype(np.int16)
        self._attempt_step += 1
        self._task_step += 1
        if self._attempt_step == 1:
            self._attempt_first_step = self._task_step
        success = bool(np.array_equal(self._position, self._goal))
        reward = float(success)
        self._attempt_return += reward
        self._task_return += reward
        attempt_done = success or self._attempt_step >= self.horizon
        outcome = self._observation()
        truncated = False
        if attempt_done:
            self._records.append(
                Attempt(
                    index=self._attempt,
                    first_step=self._attempt_first_step,
                    last_step=self._task_step,
                    steps=self._attempt_step,
                    success=success,
                    native_return=self._attempt_return,
                    complete=True,
                )
            )
            if self._attempt + 1 >= self.attempts:
                truncated = True
            else:
                self._attempt += 1
                self._attempt_step = 0
                self._attempt_return = 0.0
                self._position = self._start_position()
                self._boundary = 1
        current = self._observation()
        decision = PublicDecision(
            current=current,
            previous=self._current,
            outcome=outcome,
            executed_action=selected,
            reward=reward,
            new_attempt=attempt_done and not truncated,
            physical_done=attempt_done,
            outer_truncated=truncated,
        )
        self._current = current
        self._done = truncated
        info = self._info(attempt_done=attempt_done, success=success)
        return self._packet(decision), reward, False, truncated, info

    def render(self, mode: str | None = None) -> Any:
        rows: list[str] = []
        for row in range(self.size):
            cells: list[str] = []
            for column in range(self.size):
                cell = np.asarray((row, column), dtype=np.int16)
                agent_here = np.array_equal(cell, self._position)
                goal_here = np.array_equal(cell, self._goal)
                if agent_here and goal_here:
                    glyph = "*"
                elif agent_here:
                    glyph = "A"
                elif goal_here:
                    glyph = "G"
                else:
                    glyph = "."
                cells.append(glyph)
            rows.append(" ".join(cells))
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Restorable state
    # ------------------------------------------------------------------

    def _identity(self) -> tuple[object, ...]:
        return (self.size, self.attempts, self.horizon, self.goal_partition)

    def _state(self) -> dict[str, object]:
        return {
            "goal": self._goal.tolist(),
            "position": self._position.tolist(),
            "attempt": self._attempt,
            "attempt_step": self._attempt_step,
            "attempt_first_step": self._attempt_first_step,
            "attempt_return": self._attempt_return,
            "task_step": self._task_step,
            "task_return": self._task_return,
            "boundary": self._boundary,
            "records": [asdict(record) for record in self._records],
        }

    def _load_state(self, state: Mapping[str, object]) -> None:
        records = state.get("records")
        if not isinstance(records, Sequence):
            raise ContractError("DarkRoom checkpoint is malformed.")
        goal = np.asarray(state["goal"], dtype=np.int16)
        position = np.asarray(state["position"], dtype=np.int16)
        if goal.shape != (2,) or position.shape != (2,):
            raise ContractError("Restored DarkRoom coordinates changed shape.")
        task = int(cast(Any, state["task"]))
        if not np.array_equal(goal, np.asarray(divmod(task, self.size), np.int16)):
            raise ContractError("Restored DarkRoom goal disagrees with its task.")
        self._goal = goal
        self._position = position
        self._attempt = int(cast(Any, state["attempt"]))
        self._attempt_step = int(cast(Any, state["attempt_step"]))
        self._attempt_first_step = int(cast(Any, state["attempt_first_step"]))
        self._attempt_return = float(cast(Any, state["attempt_return"]))
        self._task_step = int(cast(Any, state["task_step"]))
        self._task_return = float(cast(Any, state["task_return"]))
        self._boundary = int(cast(Any, state["boundary"]))
        self._records = [Attempt(**cast(Any, row)) for row in records]


__all__ = [
    "DARKROOM_ACTIONS",
    "DARKROOM_GOAL_PARTITION",
    "DARKROOM_IID_SOURCE_INDICES",
    "DARKROOM_OOD_SOURCE_INDICES",
    "DARKROOM_PROTOCOL",
    "DARKROOM_SPLITS",
    "DARKROOM_TRAIN_SOURCE_INDICES",
    "DarkRoomEnv",
    "darkroom_source_indices",
    "public_fields",
]
