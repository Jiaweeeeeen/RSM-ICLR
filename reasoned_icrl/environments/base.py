"""The shared public-packet environment contract.

Every benchmark environment publishes the same five-key packet to the policy:

======== ==================================================================
current  the public observation the next action is chosen from, in [-1, 1]
previous the public observation the last executed action was chosen from
outcome  the public endpoint that action reached, before any reset
event    ``[physical tuple available, new attempt, physical done]``
valid    always one; AMAGO pads sequences with zeros
======== ==================================================================

:class:`PublicDecision` builds that packet from one causal decision record and
refuses inconsistent records. :class:`BaseEnv` owns everything the four
environments share: the task roster and its deterministic selection on reset,
the packet observation space and its serialisable contract, action validation,
and the restorable-state envelope. Subclasses supply the task dynamics.

Task identities are integers. The native benchmarks use them as upstream reset
seeds drawn from the study-wide bands below; DarkRoom uses them as hidden goal
cells. A training rollout never draws an evaluation identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from reasoned_icrl.environments.rendering import RENDER_MODES
from reasoned_icrl.experiments.contracts import ContractError

OUTCOME_PROTOCOL = "public-physical-outcome.v1"
PACKET_KEYS = ("current", "previous", "outcome", "event", "valid")

TRAINING_TASKS = range(0, 1_000_000)
DEVELOPMENT_TASKS = range(1_000_000, 1_000_064)
FINAL_TASKS = range(2_000_000, 2_000_256)
FINAL_OOD_TASKS = range(3_000_000, 3_000_256)
"""A fourth disjoint band for a contract's declared out-of-distribution final
split (decision 17): the same identities-as-seeds convention, a shift the
environment applies from the split name alone (Concentration's uneven deck)."""

REVISED_FINAL_TASKS = range(4_000_000, 4_000_256)
"""The 8M study's untouched final panel (R1).

Every task in :data:`FINAL_TASKS` was evaluated by the 4M study, so it can serve
only as a labelled historical-comparison panel. New training seeds do not make a
fresh test set, so the revised study draws its 256 final tasks from this fifth
disjoint band. Identity disjointness is necessary and not sufficient: XLand maps
identities through a pinned ruleset permutation, so a duplicate rule encoding can
still cross the split, which R5 checks before any tier-3 pilot."""

CONFIRMATION_TASKS = range(5_000_000, 5_000_256)
"""The Memo comparator's confirmation panel (ME0).

A sixth disjoint band: the comparator scores its own endpoints and the saved
8M reference endpoints on tasks that no training seed, the development roster,
the 4M study's inspected ``final`` band, the out-of-distribution band or the
8M study's ``final-revised`` band ever contained. It is opened only after the
recipe, the contrasts and every Memo endpoint are fixed."""

_SPLIT_SOURCES: dict[str, range] = {
    "train": TRAINING_TASKS,
    "development": DEVELOPMENT_TASKS,
    "final": FINAL_TASKS,
    "final-ood": FINAL_OOD_TASKS,
    "final-revised": REVISED_FINAL_TASKS,
    "confirmation": CONFIRMATION_TASKS,
}


def benchmark_task_sources(split: str) -> range:
    """Return the deterministic native identity roster for one split."""
    try:
        return _SPLIT_SOURCES[split]
    except KeyError:
        raise ContractError(f"Unknown benchmark split: {split!r}.") from None


@dataclass(frozen=True, slots=True)
class PublicField:
    """One named public value, recorded for provenance and leakage review."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    low: tuple[float, ...]
    high: tuple[float, ...]


def unit_fields(names: Sequence[str]) -> tuple[PublicField, ...]:
    """Scalar float fields whose native range is ``[0, 1]``."""
    return tuple(PublicField(name, (1,), "float32", (0.0,), (1.0,)) for name in names)


@dataclass(frozen=True, slots=True)
class PublicDecision:
    """States [D], executed discrete action, and unscaled public feedback.

    None marks unavailable data; a zero endpoint is valid evidence. An ignored
    command on a reset-only step is not an executed physical action.
    """

    current: NDArray[np.float32]
    previous: NDArray[np.float32] | None = None
    outcome: NDArray[np.float32] | None = None
    executed_action: int | None = None
    reward: float = 0.0
    new_attempt: bool = False
    physical_done: bool = False
    outer_terminated: bool = False
    outer_truncated: bool = False

    def inputs(
        self, action_count: int
    ) -> tuple[dict[str, NDArray[np.float32]], NDArray[np.float32]]:
        """Build the five-key packet and the ``[reward, one-hot action]`` RL2.

        Outer terminals remain learner targets, not extra policy inputs or
        inferred physical resets.
        """
        if type(action_count) is not int or action_count < 1:
            raise ContractError("Action count must be positive.")
        if self.current.ndim != 1 or not np.isfinite(self.current).all():
            raise ContractError("Current public observation must be a finite vector.")
        available = self.outcome is not None
        if available != (self.previous is not None) or available != (
            self.executed_action is not None
        ):
            raise ContractError("Physical evidence requires previous/outcome/action.")
        if not math.isfinite(self.reward) or (not available and self.reward != 0):
            raise ContractError("Unavailable physical evidence cannot hide feedback.")
        if self.physical_done and not available:
            raise ContractError("Physical completion requires a real outcome.")
        for value in (self.previous, self.outcome):
            if value is not None and (
                value.shape != self.current.shape or not np.isfinite(value).all()
            ):
                raise ContractError("Physical endpoint shape/value mismatch.")
        feedback = np.zeros(action_count + 1, dtype=np.float32)
        if self.executed_action is not None:
            if (
                type(self.executed_action) is not int
                or not 0 <= self.executed_action < action_count
            ):
                raise ContractError("Executed action outside native action space.")
            feedback[0] = self.reward
            feedback[1 + self.executed_action] = 1
        packet = {
            "current": self.current.astype(np.float32, copy=True),
            "previous": np.zeros_like(self.current, dtype=np.float32)
            if self.previous is None
            else self.previous.astype(np.float32, copy=True),
            "outcome": np.zeros_like(self.current, dtype=np.float32)
            if self.outcome is None
            else self.outcome.astype(np.float32, copy=True),
            "event": np.array(
                [available, self.new_attempt, self.physical_done], dtype=np.float32
            ),
            "valid": np.ones(1, dtype=np.float32),
        }
        return packet, feedback

    @property
    def learner_terminal(self) -> bool:
        return self.outer_terminated or self.outer_truncated


@dataclass(frozen=True, slots=True)
class Attempt:
    """One physical attempt of a K-attempt task, from public observations only.

    ``steps`` counts executed physical actions, never a reset-only step that
    follows a boundary. ``complete`` is False for a final partial attempt cut
    short by the outer budget.
    """

    index: int
    first_step: int
    last_step: int
    steps: int
    success: bool
    native_return: float
    complete: bool

    def __post_init__(self) -> None:
        if self.index < 0 or self.steps < 1 or self.first_step < 1:
            raise ContractError("Attempt counters must be positive.")
        if self.last_step - self.first_step + 1 != self.steps:
            raise ContractError("Attempt step span disagrees with its length.")
        if self.success and not self.complete:
            raise ContractError("A partial attempt cannot report success.")


class BaseEnv(gym.Env[dict[str, np.ndarray], int]):
    """Roster, task selection, packet contract and state envelope for every task.

    Subclasses set ``label``, ``protocol`` and ``state_schema``; call
    :meth:`_install_contract` with their public fields and action count; and
    implement :meth:`_begin_task`, :meth:`step`, :meth:`_reset_info`,
    :meth:`_identity`, :meth:`_state` and :meth:`_load_state`. An environment
    that can be watched implements :meth:`_render_ansi` and
    :meth:`_render_rgb`; :meth:`render` dispatches on ``render_mode`` the way
    any Gymnasium environment does.
    """

    label: ClassVar[str] = "Environment"
    state_schema: ClassVar[str] = "environment-state.v1"
    protocol: str
    # Gymnasium declares ``metadata`` as an instance variable, so ClassVar is
    # refused by the strict checker; the dict is never mutated.
    metadata: dict[str, Any] = {"render_modes": list(RENDER_MODES)}  # noqa: RUF012

    def __init__(
        self,
        *,
        split: str,
        roster: Sequence[int] | range,
        fixed_task_index: int | None,
        initial_seed: int,
        render_mode: str | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in RENDER_MODES:
            raise ContractError(
                f"{self.label} render modes: {', '.join(RENDER_MODES)}."
            )
        self.render_mode = render_mode
        self.split = split
        if isinstance(roster, range):
            if roster.step != 1 or len(roster) < 1:
                raise ContractError(f"{self.label} task rosters must be contiguous.")
            self.source_indices: range | tuple[int, ...] = roster
        else:
            values = tuple(int(value) for value in roster)
            if not values or len(set(values)) != len(values):
                raise ContractError(f"{self.label} roster must be unique and nonempty.")
            self.source_indices = values
        if fixed_task_index is not None and fixed_task_index not in self.source_indices:
            raise ContractError(
                f"{self.label} fixed_task_index lies outside the roster."
            )
        self.fixed_task_index = fixed_task_index
        self.initial_seed = initial_seed
        self._task_index: int | None = None
        self._current: np.ndarray | None = None
        # `_has_reset` gates the task-completion counter in the `_done` setter,
        # so it must exist before the first assignment to `_done`.
        self._has_reset = False
        self._done = True
        self._charged_calls = 0
        self._reset_only_steps = 0
        self._tasks_started = 0
        self._tasks_completed = 0

    def collection_counters(self) -> dict[str, int]:
        """Measured interaction counts since this actor was built (R1).

        These are observed totals, never a nominal product of the recipe.
        ``charged_calls`` counts every decision the collector paid for,
        ``physical_actions`` counts only the ones the environment executed, and
        the difference is ``reset_only_steps``. ``tasks_completed`` counts outer
        tasks that reached a terminal, so it excludes a task cut short when
        collection stopped mid-episode."""
        return {
            "charged_calls": self._charged_calls,
            "physical_actions": self._charged_calls - self._reset_only_steps,
            "reset_only_steps": self._reset_only_steps,
            "tasks_started": self._tasks_started,
            "tasks_completed": self._tasks_completed,
        }

    def _count_reset_only_step(self) -> None:
        """Record that the decision just validated executed no physical action."""
        self._reset_only_steps += 1

    # Every environment ends a task by assigning ``self._done``; counting the
    # transition here keeps one definition of "completed" instead of five.
    @property
    def _done(self) -> bool:
        return self.__done

    @_done.setter
    def _done(self, value: bool) -> None:
        became_terminal = bool(value) and not getattr(self, "_BaseEnv__done", False)
        self.__done = bool(value)
        if became_terminal and self._has_reset:
            self._tasks_completed += 1

    def _install_contract(
        self, fields: Sequence[PublicField], action_count: int
    ) -> None:
        """Declare the public fields and the discrete action space."""
        self.fields = tuple(fields)
        self.action_count = action_count
        width = sum(len(field.low) for field in self.fields)
        self._width = width
        self.action_space: gym.spaces.Space[Any] = gym.spaces.Discrete(action_count)
        self.observation_space = gym.spaces.Dict(
            {
                "current": gym.spaces.Box(-1, 1, (width,), np.float32),
                "previous": gym.spaces.Box(-1, 1, (width,), np.float32),
                "outcome": gym.spaces.Box(-1, 1, (width,), np.float32),
                "event": gym.spaces.Box(0, 1, (3,), np.float32),
                "valid": gym.spaces.Box(0, 1, (1,), np.float32),
            }
        )
        self.contract: dict[str, Any] = {
            "schema": OUTCOME_PROTOCOL,
            "fields": [asdict(field) for field in self.fields],
            "environment_protocol": self.protocol,
            "goal_schedule_sha256": None,
        }

    @property
    def width(self) -> int:
        return self._width

    # ------------------------------------------------------------------
    # Task identity
    # ------------------------------------------------------------------

    @property
    def evaluator_task_index(self) -> int:
        """The identity of the live task. Evaluator-only; never a policy input."""
        if self._task_index is None:
            raise ContractError(f"{self.label} must be reset before use.")
        return self._task_index

    def set_task(self, task_index: int) -> None:
        """Pin the task the next reset selects, for a declared roster order."""
        if isinstance(task_index, bool) or not isinstance(task_index, int):
            raise ContractError(f"{self.label} task_index must be an integer.")
        if task_index not in self.source_indices:
            raise ContractError(f"{self.label} task_index lies outside the roster.")
        self.fixed_task_index = task_index

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _packet(self, decision: PublicDecision) -> dict[str, np.ndarray]:
        packet, _ = decision.inputs(self.action_count)
        return packet

    def _begin_task(self, source: int) -> np.ndarray:
        """Start the task ``source`` and return its first public observation."""
        raise NotImplementedError

    def _reset_info(self) -> dict[str, Any]:
        raise NotImplementedError

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Select the next task from the roster and clear all history."""
        if seed is None and not self._has_reset:
            seed = self.initial_seed
        super().reset(seed=seed)
        requested = None if options is None else options.get("task_index")
        if requested is None:
            requested = self.fixed_task_index
        if requested is None:
            offset = int(self.np_random.integers(len(self.source_indices)))
            source = int(self.source_indices[offset])
        elif isinstance(requested, bool) or not isinstance(requested, int):
            raise ContractError(f"{self.label} task_index must be an integer.")
        else:
            source = int(requested)
        if source not in self.source_indices:
            raise ContractError(f"{self.label} task_index lies outside the roster.")
        self._current = self._begin_task(source)
        self._task_index = source
        self._done = False
        self._has_reset = True
        self._tasks_started += 1
        decision = PublicDecision(current=self._current, new_attempt=True)
        return self._packet(decision), self._reset_info()

    def close(self) -> None:
        """Release upstream resources; typed once for every environment."""
        return None

    def _validate_action(self, action: Any) -> int:
        if self._current is None or self._done:
            raise ContractError(f"{self.label} requires an outer reset.")
        selected = int(np.asarray(action).reshape(-1)[0])
        if not 0 <= selected < self.action_count:
            raise ContractError("Selected action lies outside the action space.")
        self._charged_calls += 1
        return selected

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_ansi(self) -> str:
        """The observer's text frame of the live task (hidden state included)."""
        raise NotImplementedError

    def _render_rgb(self) -> NDArray[np.uint8]:
        """The observer's ``(H, W, 3)`` uint8 frame of the live task."""
        raise NotImplementedError

    def render(self, mode: str | None = None) -> Any:
        """Render the live task in ``mode``, else the constructor's ``render_mode``.

        ``ansi`` (and ``None``, so a bare ``render()`` always has a value)
        returns a text frame, ``rgb_array`` an ``(H, W, 3)`` uint8 array and
        ``human`` prints the text frame. The frame is the observer's view:
        hidden layouts and cues are drawn; the policy's packet is unchanged.
        """
        if mode is None:
            mode = self.render_mode
        elif mode not in RENDER_MODES:
            raise ContractError(
                f"{self.label} render modes: {', '.join(RENDER_MODES)}."
            )
        if self._task_index is None:
            raise ContractError(f"{self.label} must be reset before rendering.")
        try:
            if mode == "rgb_array":
                return self._render_rgb()
            frame = self._render_ansi()
        except NotImplementedError:
            raise ContractError(f"{self.label} does not render.") from None
        if mode == "human":
            print(frame)
            return None
        return frame

    # ------------------------------------------------------------------
    # Restorable state
    # ------------------------------------------------------------------

    def _identity(self) -> tuple[object, ...]:
        """Task-specific contract values that a checkpoint must reproduce."""
        return ()

    def _contract_identity(self) -> tuple[object, ...]:
        return (
            self.protocol,
            self.split,
            len(self.source_indices),
            int(self.source_indices[0]),
            *self._identity(),
        )

    def _state(self) -> dict[str, object]:
        """Task-specific restorable state; the envelope adds the shared keys."""
        raise NotImplementedError

    def _load_state(self, state: Mapping[str, object]) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict[str, object]:
        """Snapshot the task, the current public vector and the roster RNG."""
        if self._task_index is None:
            raise ContractError(f"{self.label} must be reset before snapshotting.")
        return {
            "schema": self.state_schema,
            "contract": self._contract_identity(),
            "task": self._task_index,
            "current": None if self._current is None else self._current.tolist(),
            "done": self._done,
            "rng": deepcopy(self.np_random.bit_generator.state),
            # R6: the measured interaction counters travel with the snapshot so
            # a resumed run keeps counting from where the state was saved.
            "counters": self.collection_counters(),
            **self._state(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a snapshot without changing the declared task contract."""
        if (
            state.get("schema") != self.state_schema
            or tuple(cast(Sequence[object], state.get("contract", ())))
            != self._contract_identity()
        ):
            raise ContractError(f"{self.label} checkpoint contract does not match.")
        current = state.get("current")
        if not isinstance(current, Sequence):
            raise ContractError(f"{self.label} checkpoint is malformed.")
        vector = np.asarray(current, dtype=np.float32)
        if vector.shape != (self.width,):
            raise ContractError(f"{self.label} checkpoint changed the packet width.")
        self._load_state(state)
        self._task_index = int(cast(Any, state["task"]))
        self._current = vector
        self._done = bool(state["done"])
        self._has_reset = True
        self.np_random.bit_generator.state = deepcopy(cast(Any, state["rng"]))
        counters = state.get("counters")
        if isinstance(counters, Mapping):  # snapshots before R6 carry none
            self._charged_calls = int(cast(int, counters["charged_calls"]))
            self._reset_only_steps = int(cast(int, counters["reset_only_steps"]))
            self._tasks_started = int(cast(int, counters["tasks_started"]))
            self._tasks_completed = int(cast(int, counters["tasks_completed"]))


__all__ = [
    "CONFIRMATION_TASKS",
    "DEVELOPMENT_TASKS",
    "FINAL_TASKS",
    "OUTCOME_PROTOCOL",
    "PACKET_KEYS",
    "TRAINING_TASKS",
    "Attempt",
    "BaseEnv",
    "PublicDecision",
    "PublicField",
    "benchmark_task_sources",
    "unit_fields",
]
