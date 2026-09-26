"""Native AMAGO v3.4.0 passive T-Maze as a public-packet policy environment.

Wraps the pinned ``amago.envs.builtin.tmaze.TMazeAltPassive`` (Ni et al.,
2023, "When Do Transformers Shine in RL? Decoupling Memory from Credit
Assignment", as modified by AMAGO) without editing upstream source. It is the
memory-length unit test the bounded-summary paper uses as its third
benchmark: the goal side is shown once, in the first observation, and the
agent must carry it down a corridor of ``corridor_length`` steps to the
junction, where it turns.

The native lifecycle is kept exactly as it is and has no meta-attempt
structure: one reset draws the goal side (``+1`` up, ``-1`` down) from the
task identity, the agent starts at the oracle cell, every step moves it by
one of four directions (a move into a wall leaves it in place and still
costs a step), and the episode ends after exactly ``corridor_length + 1``
steps, paying ``1`` if the agent then stands on the goal side of the
junction and ``0`` otherwise. Every non-forward action before the last step
pays the ``TMazeAlt`` movement penalty (``movement_penalty``, ``-1 / L`` at
the paper's settings, AMAGO's own T-Maze recipe): the corridor is walked
under a dense signal, and only the turn is paid by the cue.

**There is no spare step.** Reaching the junction takes ``corridor_length``
forward moves and the turn takes one more, which is the whole budget; a
single wasted or lateral move in the corridor makes the episode fail. Every
successful corridor action is therefore the forward move, so the previous
action the learner reads through RL2 carries no information about the cue,
and success depends on the memory carried from the first observation alone.
Pinned by the environment tests; the contract keeps the budget exact.

Only the two native Box values reach the policy: ``at_junction`` (``1`` at
the junction or on a goal cell, ``0`` on the corridor) and
``cue_or_lateral`` (the goal side at the very first observation, then ``0``
along the corridor, then the lateral position at the junction). The native
coordinates, the goal side and the oracle flag are never read for an
observation; :meth:`TMazeEnv.render` draws them for a human observer only.
The evaluator reconstructs the cue and the outcome from the public
observations alone.
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
    BaseEnv,
    PublicDecision,
    PublicField,
    benchmark_task_sources,
)
from reasoned_icrl.environments.rendering import (
    CELL_PIXELS,
    PALETTE,
    ansi_frame,
    color_grid,
    paint_cells,
)
from reasoned_icrl.environments.utils import NativeSource, normalize_fields
from reasoned_icrl.experiments.contracts import ContractError

TMAZE_V3_PROTOCOL = "tmaze-passive-l32-256-v3"
"""The paper's protocol identity: the
passive variant with AMAGO's movement penalty, where every training task draws
its corridor length from ``TMAZE_V3_TRAINING_CORRIDORS`` (one to eight rewrites
at C = 32) so the writer is trained under a variable number of repetitions.
Evaluation contracts stay at corridor 128."""
TMAZE_V3_TRAINING_CORRIDORS: tuple[int, ...] = (32, 64, 96, 128, 160, 192, 224, 256)
"""The v3 training draw: the multiples of the C32 segment from one to eight
rewrites; the task identity fixes the draw (``corridor_for_task``)."""
TMAZE_STATE_SCHEMA = "tmaze-passive-state.v2"
TMAZE_CORRIDOR = 128
TMAZE_MOVEMENT_PENALTY = -1.0 / TMAZE_CORRIDOR
"""The paper's movement penalty, AMAGO's ``-1 / corridor_length``: paid by
every non-forward action before the final decision (``TMazeAlt.reward_fn``),
never by the turn, so it shapes the walk and leaves the cue to the memory."""
TMAZE_ACTIONS = 4
FORWARD, UP, BACK, DOWN = 0, 1, 2, 3
"""The native action table: +x, +y, -x, -y."""
TMAZE_ACTION_NAMES: tuple[str, ...] = ("forward", "up", "back", "down")
"""Observer labels for the native action table."""
RENDER_COLUMNS = 64
"""At most this many text columns for the corridor; longer corridors are drawn
at one column per ``ceil((L + 1) / 64)`` cells, the header keeping the exact
position."""
TMAZE_FIELDS: tuple[PublicField, ...] = (
    PublicField("at_junction", (1,), "float32", (0.0,), (1.0,)),
    PublicField("cue_or_lateral", (1,), "float32", (-1.0,), (1.0,)),
)
"""The two native Box values, named for provenance and leakage review."""
_LOW = np.array([field.low[0] for field in TMAZE_FIELDS], dtype=np.float64)
_HIGH = np.array([field.high[0] for field in TMAZE_FIELDS], dtype=np.float64)


def corridor_for_task(source: int, corridor_lengths: Sequence[int]) -> int:
    """The corridor length of training task ``source`` under a per-task draw:
    a uniform choice from ``corridor_lengths`` by a generator seeded from the
    task identity alone (a string seed, so the draw is stable across
    processes and independent of the native cue draw, which the upstream task
    makes from its own seed)."""
    if type(source) is not int or source < 0:
        raise ContractError("A T-Maze task identity is a nonnegative integer.")
    return int(random.Random(f"tmaze-corridor:{source}").choice(list(corridor_lengths)))


def tmaze_horizon(corridor_length: int) -> int:
    """Decisions in one episode: the corridor plus the turn, no slack."""
    if isinstance(corridor_length, bool) or int(corridor_length) < 1:
        raise ContractError("The T-Maze corridor length is a positive integer.")
    return int(corridor_length) + 1


@dataclass(frozen=True, slots=True)
class TMazeEpisode:
    """One finished episode, reconstructed from public observations only."""

    cue: int
    """The goal side shown in the first observation: ``+1`` up, ``-1`` down."""
    lateral: int
    """Where the agent stood at the end: ``+1``, ``-1`` or ``0`` (never turned)."""
    success: bool
    steps: int
    junction_step: int | None
    """The first decision index (one-based) whose observation showed the
    junction, or ``None`` if the agent never reached it."""
    forward_moves: int
    """Executed forward actions; the budget is met only when every corridor
    action was forward."""
    native_return: float
    penalised_moves: int = 0
    """Non-forward actions before the final decision, each paid the penalty."""
    movement_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.cue not in (-1, 1) or self.lateral not in (-1, 0, 1):
            raise ContractError("T-Maze cue and lateral position are -1/0/+1.")
        if self.steps < 1 or self.forward_moves < 0 or self.forward_moves > self.steps:
            raise ContractError("T-Maze step counters must be valid.")
        if self.success != (self.lateral == self.cue):
            raise ContractError("T-Maze success disagrees with the public outcome.")
        if self.penalised_moves < 0 or self.penalised_moves > self.steps:
            raise ContractError("T-Maze penalised moves must be valid.")
        expected = float(self.success) + self.penalised_moves * self.movement_penalty
        if not np.isclose(self.native_return, expected, rtol=0, atol=1e-9):
            raise ContractError("T-Maze native return disagrees with its outcome.")
        if self.junction_step is not None and not 1 <= self.junction_step <= self.steps:
            raise ContractError("T-Maze junction step lies outside the episode.")
        if self.success and self.junction_step is None:
            raise ContractError("A T-Maze success requires the junction.")


class TMazeEnv(BaseEnv):
    """Public five-key packet over the pinned native passive T-Maze."""

    label: ClassVar[str] = "Passive T-Maze"
    state_schema: ClassVar[str] = TMAZE_STATE_SCHEMA
    protocol = TMAZE_V3_PROTOCOL

    def __init__(
        self,
        *,
        corridor_length: int = TMAZE_CORRIDOR,
        movement_penalty: float = TMAZE_MOVEMENT_PENALTY,
        split: str = "train",
        source_indices: Sequence[int] | range | None = None,
        fixed_task_index: int | None = None,
        initial_seed: int = 0,
        render_mode: str | None = None,
        protocol: str = TMAZE_V3_PROTOCOL,
        corridor_lengths: Sequence[int] | None = None,
    ) -> None:
        if protocol != TMAZE_V3_PROTOCOL:
            raise ContractError(f"Unknown T-Maze protocol {protocol!r}.")
        self.protocol = protocol
        if corridor_lengths is not None:
            lengths = tuple(int(length) for length in corridor_lengths)
            if (
                not lengths
                or len(set(lengths)) != len(lengths)
                or any(
                    isinstance(length, bool) or length < 1
                    for length in corridor_lengths
                )
            ):
                raise ContractError(
                    "T-Maze corridor lengths are distinct positive integers."
                )
            corridor_lengths = lengths
        self.corridor_lengths: tuple[int, ...] | None = corridor_lengths
        """The per-task training draw (v3), or ``None`` for a fixed corridor."""
        super().__init__(
            split=split,
            roster=benchmark_task_sources(split)
            if source_indices is None
            else source_indices,
            fixed_task_index=fixed_task_index,
            initial_seed=initial_seed,
            render_mode=render_mode,
        )
        self.fixed_corridor = int(corridor_length)
        """The declared corridor: the evaluation contracts' length; training
        tasks draw theirs from ``corridor_lengths``."""
        self.corridor_length = self.fixed_corridor
        self.horizon = tmaze_horizon(self.corridor_length)
        penalty = float(movement_penalty)
        if not np.isfinite(penalty) or penalty > 0.0:
            raise ContractError("The T-Maze movement penalty is a finite value <= 0.")
        self.movement_penalty = penalty
        self._penalised_moves = 0
        self._natives: dict[int, NativeSource] = {}
        self._native = self._native_for(self.corridor_length)
        self._install_contract(TMAZE_FIELDS, TMAZE_ACTIONS)
        self._cue = 0
        self._step = 0
        self._junction_step: int | None = None
        self._forward_moves = 0
        self._episode_return = 0.0
        self._episode: TMazeEpisode | None = None

    def _make_native(self) -> gym.Env[Any, Any]:
        from amago.envs.builtin.tmaze import TMazeAltPassive

        return cast(
            gym.Env[Any, Any],
            TMazeAltPassive(
                corridor_length=self.corridor_length,
                goal_reward=1.0,
                penalty=self.movement_penalty,
                distract_reward=0.0,
            ),
        )

    def _native_for(self, corridor_length: int) -> NativeSource:
        """The upstream task at ``corridor_length``, built once per length and
        reseeded from the task identity at every reset."""
        source = self._natives.get(corridor_length)
        if source is None:
            self.corridor_length = corridor_length
            source = NativeSource(self._make_native, seed=self.initial_seed)
            self._natives[corridor_length] = source
        return source

    def _set_corridor(self, corridor_length: int) -> None:
        """Switch the live corridor (and its budget) before a reset or restore."""
        self._native = self._native_for(corridor_length)
        self.corridor_length = int(corridor_length)
        self.horizon = tmaze_horizon(self.corridor_length)

    def corridor_for_task(self, source: int) -> int:
        """The corridor the task ``source`` runs: the declared one, or under a
        per-task draw the identity's choice."""
        if self.corridor_lengths is None:
            return self.fixed_corridor
        return corridor_for_task(source, self.corridor_lengths)

    @property
    def native(self) -> gym.Env[Any, Any]:
        """The wrapped upstream task, for inspection and audits only."""
        return self._native

    # ------------------------------------------------------------------
    # Evaluator-only accessors
    # ------------------------------------------------------------------

    @property
    def cue(self) -> int:
        """The goal side read from the public reset observation."""
        return self._cue

    @property
    def episode(self) -> TMazeEpisode | None:
        """The finished episode's public record, once the task is done."""
        return self._episode

    @property
    def episode_return(self) -> float:
        return self._episode_return

    def public_fields(self, packet: Mapping[str, np.ndarray]) -> dict[str, int]:
        """Decode one packet's ``current`` token back to the native values."""
        current = np.asarray(packet["current"], dtype=np.float32)
        if current.shape != (len(self.fields),):
            raise ContractError("Packet width disagrees with the public contract.")
        raw = (current.astype(np.float64) + 1.0) / 2.0 * (_HIGH - _LOW) + _LOW
        values = np.rint(raw)
        if not np.allclose(raw, values, atol=1e-6):
            raise ContractError("T-Maze public values are integers.")
        return {
            field.name: int(value)
            for field, value in zip(self.fields, values, strict=True)
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _project(self, observation: object) -> NDArray[np.float32]:
        raw = np.asarray(observation, dtype=np.float64)
        if raw.shape != (len(self.fields),):
            raise ContractError(f"{self.label} observation has an unexpected shape.")
        return normalize_fields(raw, _LOW, _HIGH)

    def _begin_task(self, source: int) -> np.ndarray:
        corridor = self.corridor_for_task(source)
        if corridor != self.corridor_length:
            self._set_corridor(corridor)
        observation, _ = self._native.reset(seed=source)
        current = self._project(observation)
        fields = self.public_fields({"current": current})
        if fields["at_junction"] != 0 or fields["cue_or_lateral"] not in (-1, 1):
            raise ContractError("The first T-Maze observation must show the cue.")
        self._cue = fields["cue_or_lateral"]
        self._step = 0
        self._junction_step = None
        self._forward_moves = 0
        self._penalised_moves = 0
        self._episode_return = 0.0
        self._episode = None
        return current

    def _reset_info(self) -> dict[str, Any]:
        return {
            "evaluator_task_index": self.evaluator_task_index,
            "step_in_task": 0,
            "at_junction": False,
            "episode_done": False,
            "episode_success": False,
            "episode_return": 0.0,
        }

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one native decision and build its causal public record."""
        selected = self._validate_action(action)
        observation, reward, terminated, truncated, _ = self._native.step(selected)
        current = self._project(observation)
        self._step += 1
        self._forward_moves += int(selected == FORWARD)
        fields = self.public_fields({"current": current})
        at_junction = fields["at_junction"] == 1
        if at_junction and self._junction_step is None:
            self._junction_step = self._step
        if self._step > 1 and not at_junction and fields["cue_or_lateral"] != 0:
            raise ContractError("The corridor shows no cue after the first step.")
        done = bool(terminated or truncated)
        if done != (self._step >= self.horizon):
            raise ContractError("T-Maze terminated away from its fixed budget.")
        lateral = fields["cue_or_lateral"] if at_junction else 0
        success = bool(done and lateral == self._cue)
        penalised = not done and selected != FORWARD
        # TMazeAlt.reward_fn: the goal reward on the final decision, the
        # movement penalty for every other non-forward action, nothing else.
        expected = float(success) if done else self.movement_penalty * float(penalised)
        if not np.isclose(reward, expected, rtol=0, atol=1e-9):
            raise ContractError(
                "Public T-Maze outcome disagrees with the native reward."
            )
        self._penalised_moves += int(penalised)
        self._episode_return += float(reward)
        previous = self._current
        self._current = current
        self._done = done
        if done:
            self._episode = TMazeEpisode(
                cue=self._cue,
                lateral=lateral,
                success=success,
                steps=self._step,
                junction_step=self._junction_step,
                forward_moves=self._forward_moves,
                native_return=self._episode_return,
                penalised_moves=self._penalised_moves,
                movement_penalty=self.movement_penalty,
            )
        decision = PublicDecision(
            current=current,
            previous=previous,
            outcome=current,
            executed_action=selected,
            reward=float(reward),
            physical_done=done,
            outer_terminated=terminated,
            outer_truncated=truncated,
        )
        info = {
            "evaluator_task_index": self.evaluator_task_index,
            "step_in_task": self._step,
            "at_junction": at_junction,
            "episode_done": done,
            "episode_success": success,
            "episode_return": self._episode_return,
        }
        return self._packet(decision), float(reward), terminated, truncated, info

    def close(self) -> None:
        cast(Any, self._native).close()

    # ------------------------------------------------------------------
    # Rendering (observer only; reads the native position, never the packet)
    # ------------------------------------------------------------------

    @property
    def action_names(self) -> tuple[str, ...]:
        return TMAZE_ACTION_NAMES

    def _agent(self) -> tuple[int, int]:
        """The native ``(x, y)``: ``x`` along the corridor, ``y`` the lateral."""
        native = cast(Any, self._native.unwrapped)
        return int(native.x), int(native.y)

    def _render_header(self) -> list[str]:
        x, _ = self._agent()
        side = "up" if self._cue == 1 else "down"
        line = (
            f"{self.label}  task {self.evaluator_task_index}  "
            f"step {self._step}/{self.horizon}  cue {side}  "
            f"x {x}/{self.corridor_length}  return {self._episode_return:+.4f}  "
            f"forward {self._forward_moves}  penalised {self._penalised_moves}"
        )
        if self._episode is not None:
            line += "  " + ("success" if self._episode.success else "failure")
        return [line]

    def _render_ansi(self) -> str:
        x, y = self._agent()
        cells = self.corridor_length + 1
        scale = -(-cells // RENDER_COLUMNS)
        columns = -(-cells // scale)
        junction = columns - 1
        agent_column = x // scale
        rows: list[str] = []
        for lateral in (1, 0, -1):
            if lateral == 0:
                row = ["."] * columns
                row[junction] = "+"
            else:
                row = [" "] * columns
                row[junction] = "G" if lateral == self._cue else "."
            if y == lateral:
                row[agent_column] = "A"
            rows.append("".join(row))
        return ansi_frame(self._render_header(), rows)

    def _render_rgb(self) -> NDArray[np.uint8]:
        x, y = self._agent()
        cells = self.corridor_length + 1
        board = color_grid(3, cells, PALETTE["wall"])
        board[1, :] = PALETTE["floor"]
        board[0, cells - 1] = PALETTE["goal" if self._cue == 1 else "goal_other"]
        board[2, cells - 1] = PALETTE["goal" if self._cue == -1 else "goal_other"]
        board[1 - y, x] = PALETTE["agent"]
        pixels = max(6, min(CELL_PIXELS, 1024 // cells))
        return paint_cells(
            board, cell_pixels=pixels, grid=None if pixels < 8 else PALETTE["grid"]
        )

    # ------------------------------------------------------------------
    # Restorable state
    # ------------------------------------------------------------------

    def _identity(self) -> tuple[object, ...]:
        # The declared geometry, never the live task's: under the v3 draw the
        # live corridor changes with the task and travels in the state.
        if self.corridor_lengths is None:
            return (
                self.fixed_corridor,
                tmaze_horizon(self.fixed_corridor),
                self.movement_penalty,
            )
        return (
            self.fixed_corridor,
            tmaze_horizon(self.fixed_corridor),
            self.movement_penalty,
            self.corridor_lengths,
        )

    def _state(self) -> dict[str, object]:
        native = cast(Any, self._native.unwrapped)
        return {
            "corridor_length": self.corridor_length,
            "cue": self._cue,
            "step": self._step,
            "junction_step": self._junction_step,
            "forward_moves": self._forward_moves,
            "penalised_moves": self._penalised_moves,
            "episode_return": self._episode_return,
            "episode": None if self._episode is None else asdict(self._episode),
            "native": {
                "x": int(native.x),
                "y": int(native.y),
                "goal_y": int(native.goal_y),
                "oracle_visited": bool(native.oracle_visited),
                "time_step": int(native.time_step),
            },
            "task_rng": deepcopy(self._native.generator_state()),
        }

    def _load_state(self, state: Mapping[str, object]) -> None:
        native_state = state.get("native")
        if not isinstance(native_state, Mapping):
            raise ContractError("T-Maze checkpoint is malformed.")
        corridor = int(cast(Any, state.get("corridor_length", self.corridor_length)))
        if self.corridor_lengths is None and corridor != self.fixed_corridor:
            raise ContractError("T-Maze checkpoint ran a different corridor.")
        if self.corridor_lengths is not None and corridor not in self.corridor_lengths:
            raise ContractError("T-Maze checkpoint ran an undeclared corridor.")
        if corridor != self.corridor_length:
            self._set_corridor(corridor)
        native = cast(Any, self._native.unwrapped)
        native.x = int(cast(Any, native_state["x"]))
        native.y = int(cast(Any, native_state["y"]))
        native.goal_y = int(cast(Any, native_state["goal_y"]))
        native.oracle_visited = bool(native_state["oracle_visited"])
        native.time_step = int(cast(Any, native_state["time_step"]))
        self._native.load_generator_state(cast(Any, state["task_rng"]))
        self._cue = int(cast(Any, state["cue"]))
        self._step = int(cast(Any, state["step"]))
        junction = state.get("junction_step")
        self._junction_step = None if junction is None else int(cast(Any, junction))
        self._forward_moves = int(cast(Any, state["forward_moves"]))
        self._penalised_moves = int(cast(Any, state["penalised_moves"]))
        self._episode_return = float(cast(Any, state["episode_return"]))
        episode = state.get("episode")
        self._episode = None if episode is None else TMazeEpisode(**cast(Any, episode))


__all__ = [
    "BACK",
    "DOWN",
    "FORWARD",
    "TMAZE_ACTIONS",
    "TMAZE_ACTION_NAMES",
    "TMAZE_CORRIDOR",
    "TMAZE_FIELDS",
    "TMAZE_MOVEMENT_PENALTY",
    "TMAZE_V3_PROTOCOL",
    "TMAZE_V3_TRAINING_CORRIDORS",
    "UP",
    "TMazeEnv",
    "TMazeEpisode",
    "corridor_for_task",
    "tmaze_horizon",
]
