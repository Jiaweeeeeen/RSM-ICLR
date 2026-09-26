"""XLand-MiniGrid (goal visible, rules hidden) as a public-packet policy environment.

Drives the pinned ``xminigrid`` 0.9.3 simulator directly (decision 15): one
``XLand-MiniGrid-R1-9x9`` task per outer reset, whose ruleset is one entry of
the released ``small-1m`` benchmark, played for ``attempts`` episodes under the
same goal and the same hidden rules. Every episode ends at the goal or at the
native step limit (``3 * 9 * 9 = 243``); the *following* step is a reset-only
step that ignores its action and starts the next episode from a fresh physical
layout (objects and the agent are repositioned by the episode's reset key).
The outer task ends with its last episode.

AMAGO's own ``XLandMinigridVectorizedGym`` is not used: it samples rulesets at
random, batches environments and has no restorable state, so it cannot honour
the roster, checkpoint and evaluator contracts of :class:`BaseEnv`. Its policy
observation is reproduced exactly, in the packet's ``current`` vector: the
partial ``5 x 5`` view as tile and colour ids, the direction one-hot and the
episode-done flag (AMAGO's ``direction_done``), the goal encoding (AMAGO's
``goal``), plus the normalised step within the episode. The ruleset's
``rule_encoding``, the ruleset id and the simulator state never reach a packet.

Task identities are the study-wide bands of :mod:`reasoned_icrl.environments.base`;
:func:`ruleset_index` maps them through one pinned permutation of the
benchmark's rulesets so that training, development and final identities select
disjoint rulesets. A physical layout is a deterministic function of the ruleset,
the episode index and the actor seed, so every condition meets the same layout
schedule at a task id.

JAX runs on the CPU beside the PyTorch learner; the module pins the platform
before JAX is imported and keeps the simulator single-threaded, so sixteen actor
processes do not each open a thread pool.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, ClassVar, cast

import numpy as np

from reasoned_icrl.environments.base import (
    Attempt,
    BaseEnv,
    PublicDecision,
    PublicField,
    benchmark_task_sources,
    unit_fields,
)
from reasoned_icrl.environments.utils import normalize_fields
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.utils import repository_root

XLAND_PROTOCOL = "xland-r1-9x9-small1m-goal-visible-5attempts"
XLAND_STATE_SCHEMA = "xland-minigrid-state.v1"
XLAND_ENVIRONMENT_ID = "XLand-MiniGrid-R1-9x9"
XLAND_BENCHMARK = "small-1m"
XLAND_GRID = 9
XLAND_VIEW = 5
"""The native partial view: ``view_size`` of the pinned environment."""
XLAND_ACTIONS = 6
XLAND_HORIZON = 3 * XLAND_GRID * XLAND_GRID
"""The pinned environment's native episode limit: ``3 * height * width``."""
XLAND_ATTEMPTS = 5
XLAND_SCORED_FROM = 4
"""The first scored attempt: the primary endpoint is success on attempts 4-5."""
XLAND_TILE_HIGH = 12
"""Tile ids ``0 .. 12`` (thirteen tiles) in the view's first layer."""
XLAND_COLOR_HIGH = 11
"""Colour ids ``0 .. 11`` (twelve colours) in the view's second layer."""
XLAND_GOAL_HIGH = 14
"""Goal-encoding ids: AMAGO's wrapper declares ``Box(0, 14)`` for the goal."""
XLAND_GOAL_LENGTH = 5
"""The released benchmarks' goal encoding: ``[goal id, tile, colour, tile, colour]``."""
XLAND_DIRECTIONS = 4
XLAND_SPLIT_KEY = 20260912
"""Seed of the pinned ruleset permutation behind :func:`ruleset_index`."""
XLAND_RULESETS = 1_000_000
XLAND_HELD_OUT = 320
"""Rulesets reserved at the end of the permutation: 64 development + 256 final."""
XLAND_TRAINING_TASKS = range(0, XLAND_RULESETS - XLAND_HELD_OUT)
"""The training roster: the study band, shortened to the rulesets not held out."""
XLAND_FIELDS: tuple[str, ...] = (
    "grid_tile",
    "grid_color",
    "direction",
    "attempt_done",
    "goal",
    "attempt_time",
)
XLAND_DATA_VARIABLE = "XLAND_MINIGRID_DATA"

_JAX_DEFAULTS = {
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}


def xland_task_sources(split: str) -> range:
    """The task roster of one split: the study bands, training shortened."""
    if split == "train":
        return XLAND_TRAINING_TASKS
    return benchmark_task_sources(split)


@lru_cache(maxsize=1)
def _permutation() -> np.ndarray:
    return np.random.default_rng(XLAND_SPLIT_KEY).permutation(XLAND_RULESETS)


def ruleset_index(source: int) -> int:
    """Map one task identity to its ``small-1m`` ruleset.

    Training identities take the first ``1,000,000 - 320`` entries of the pinned
    permutation; the development band the next 64; the final band the last 256.
    The three images are disjoint by construction.
    """
    development = benchmark_task_sources("development")
    final = benchmark_task_sources("final")
    permutation = _permutation()
    if source in XLAND_TRAINING_TASKS:
        return int(permutation[source])
    if source in development:
        return int(permutation[len(XLAND_TRAINING_TASKS) + (source - development[0])])
    if source in final:
        offset = len(XLAND_TRAINING_TASKS) + len(development)
        return int(permutation[offset + (source - final[0])])
    raise ContractError(f"Task identity {source} lies outside every XLand band.")


def public_fields(goal_length: int) -> tuple[PublicField, ...]:
    """The declared packet layout, in order; bounds drive the id decoding."""
    cells = XLAND_VIEW * XLAND_VIEW
    return (
        PublicField(
            "grid_tile",
            (cells,),
            "float32",
            (0.0,) * cells,
            (float(XLAND_TILE_HIGH),) * cells,
        ),
        PublicField(
            "grid_color",
            (cells,),
            "float32",
            (0.0,) * cells,
            (float(XLAND_COLOR_HIGH),) * cells,
        ),
        PublicField(
            "direction",
            (XLAND_DIRECTIONS,),
            "float32",
            (0.0,) * XLAND_DIRECTIONS,
            (1.0,) * XLAND_DIRECTIONS,
        ),
        *unit_fields(("attempt_done",)),
        PublicField(
            "goal",
            (goal_length,),
            "float32",
            (0.0,) * goal_length,
            (float(XLAND_GOAL_HIGH),) * goal_length,
        ),
        *unit_fields(("attempt_time",)),
    )


def _configure_jax_environment() -> None:
    for name, value in _JAX_DEFAULTS.items():
        os.environ.setdefault(name, value)
    # xminigrid reads its data directory when it is imported: the pinned
    # benchmark lives under the checkout unless the launcher says otherwise.
    os.environ.setdefault(
        XLAND_DATA_VARIABLE, str(repository_root() / "outputs" / "xland-data")
    )


def _import_simulator() -> tuple[Any, Any, Any]:
    """Import JAX and xminigrid lazily; the rest of the package never needs them."""
    _configure_jax_environment()
    try:
        import jax
        import jax.numpy as jnp
        import xminigrid
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise ContractError(
            "XLand-MiniGrid needs the 'xland' extra: uv sync --extra xland."
        ) from error
    return jax, jnp, xminigrid


@lru_cache(maxsize=1)
def _benchmark() -> Any:
    """The pinned ruleset benchmark, loaded once per process."""
    _, _, xminigrid = _import_simulator()
    benchmark = xminigrid.load_benchmark(name=XLAND_BENCHMARK)
    if int(benchmark.num_rulesets()) != XLAND_RULESETS:
        raise ContractError(
            f"{XLAND_BENCHMARK} has {benchmark.num_rulesets()} rulesets; "
            f"the protocol pins {XLAND_RULESETS}."
        )
    return benchmark


def benchmark_arrays() -> tuple[np.ndarray, np.ndarray]:
    """The pinned benchmark's goal and rule encodings as NumPy arrays.

    Evaluator-only: the rule encodings never reach a packet. Used by the task
    diagnostic to measure how many distinct rule sets the public goal admits.
    """
    benchmark = _benchmark()
    goals = np.asarray(benchmark.goals)
    rules = np.asarray(benchmark.rules)
    return goals.reshape(len(goals), -1), rules.reshape(len(rules), -1)


@dataclass(frozen=True, slots=True)
class BenchmarkCorpus:
    """The pinned benchmark as immutable NumPy arrays, with the file's digest.

    ``goals`` is ``(N, 5)``, ``rules`` ``(N, 4, 7)``, ``init_tiles`` ``(N, 10, 2)``
    and ``num_rules`` ``(N,)``; ``sha256`` digests the released file the
    arrays were unpickled from. Evaluator-only, like :func:`benchmark_arrays`:
    the one-rule manifest builder reads every row, the policy reads none.
    """

    goals: np.ndarray
    rules: np.ndarray
    init_tiles: np.ndarray
    num_rules: np.ndarray
    sha256: str
    file: str


@lru_cache(maxsize=1)
def benchmark_corpus() -> BenchmarkCorpus:
    """Load the pinned benchmark once and expose every encoding array."""
    import hashlib

    _configure_jax_environment()
    benchmark = _benchmark()
    from xminigrid.benchmarks import DATA_PATH, NAME2HFFILENAME

    file = NAME2HFFILENAME[XLAND_BENCHMARK]
    digest = hashlib.sha256()
    with open(os.path.join(DATA_PATH, file), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    goals = np.asarray(benchmark.goals, dtype=np.int64)
    rules = np.asarray(benchmark.rules, dtype=np.int64)
    init_tiles = np.asarray(benchmark.init_tiles, dtype=np.int64)
    num_rules = np.asarray(benchmark.num_rules, dtype=np.int64).reshape(-1)
    for name, array in (("goals", goals), ("rules", rules), ("init_tiles", init_tiles)):
        array.setflags(write=False)
        if len(array) != XLAND_RULESETS:
            raise ContractError(f"{XLAND_BENCHMARK} {name} changed length.")
    num_rules.setflags(write=False)
    return BenchmarkCorpus(
        goals=goals,
        rules=rules,
        init_tiles=init_tiles,
        num_rules=num_rules,
        sha256=digest.hexdigest(),
        file=file,
    )


@dataclass(frozen=True, slots=True)
class XLandAttempt(Attempt):
    """One episode under the task's ruleset; ``success_step`` is one-based."""

    success_step: int | None

    def __post_init__(self) -> None:
        Attempt.__post_init__(self)
        if self.success != (self.success_step is not None):
            raise ContractError("Episode success and its timing disagree.")
        if self.success and not 0.0 < self.native_return <= 1.0:
            raise ContractError("A successful episode pays a positive return <= 1.")
        if not self.success and self.native_return != 0.0:
            raise ContractError("A failed episode pays nothing.")


class XLandMiniGridEnv(BaseEnv):
    """Public five-key packet over one ``small-1m`` ruleset, played K times."""

    label: ClassVar[str] = "XLand-MiniGrid"
    state_schema: ClassVar[str] = XLAND_STATE_SCHEMA
    protocol = XLAND_PROTOCOL

    def __init__(
        self,
        *,
        physical_horizon: int = XLAND_HORIZON,
        attempts: int = XLAND_ATTEMPTS,
        scored_from: int = XLAND_SCORED_FROM,
        split: str = "train",
        source_indices: Sequence[int] | range | None = None,
        fixed_task_index: int | None = None,
        initial_seed: int = 0,
    ) -> None:
        if physical_horizon < 1 or attempts < 1 or not 1 <= scored_from <= attempts:
            raise ContractError("XLand horizon, attempts and scored band are invalid.")
        super().__init__(
            split=split,
            roster=xland_task_sources(split)
            if source_indices is None
            else source_indices,
            fixed_task_index=fixed_task_index,
            initial_seed=initial_seed,
        )
        self._check_roster()
        self.horizon = physical_horizon
        self.attempts = attempts
        self.scored_from = scored_from
        self.size = XLAND_GRID
        self._jax, self._jnp, self._xminigrid = _import_simulator()
        env, params = self._xminigrid.make(XLAND_ENVIRONMENT_ID)
        self._env = env
        self._params = params.replace(max_steps=physical_horizon)
        goal_length = int(_benchmark().get_ruleset(0).goal.shape[-1])
        if goal_length != XLAND_GOAL_LENGTH:
            raise ContractError("The benchmark's goal encoding changed length.")
        self._install_contract(public_fields(goal_length), XLAND_ACTIONS)
        self._low = np.concatenate([np.asarray(f.low) for f in self.fields])
        self._high = np.concatenate([np.asarray(f.high) for f in self.fields])
        # JAX_PLATFORMS=cpu (set before the import) keeps both on the CPU.
        self._reset_fn = self._jax.jit(self._env.reset)
        self._step_fn = self._jax.jit(self._env.step)
        self._ruleset_id: int | None = None
        self._timestep: Any = None
        self._goal: np.ndarray | None = None
        self._pending_reset = False
        self._attempt = 0
        self._attempt_step = 0
        self._attempt_first_step = 0
        self._attempt_return = 0.0
        self._success_step: int | None = None
        self._global_step = 0
        self._task_return = 0.0
        self._records: list[XLandAttempt] = []

    def _check_roster(self) -> None:
        """Every roster endpoint must map to a ruleset of the pinned permutation."""
        for source in (
            self.source_indices
            if isinstance(self.source_indices, tuple)
            else (self.source_indices[0], self.source_indices[-1])
        ):
            ruleset_index(int(source))

    # ------------------------------------------------------------------
    # Evaluator-only accessors
    # ------------------------------------------------------------------

    @property
    def completed_attempts(self) -> tuple[XLandAttempt, ...]:
        """Every episode that reached its boundary in the live task."""
        return tuple(self._records)

    @property
    def task_return(self) -> float:
        """Unscaled native return accumulated over every episode."""
        return self._task_return

    @property
    def partial_attempt(self) -> XLandAttempt | None:
        """The outer task ends with a complete episode; never a partial one."""
        return None

    @property
    def ruleset_id(self) -> int:
        """The hidden ruleset of the live task. Evaluator-only; never a policy input."""
        if self._ruleset_id is None:
            raise ContractError(f"{self.label} must be reset before use.")
        return self._ruleset_id

    def public_fields(self, packet: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Decode one packet's ``current`` token back to the named native values."""
        current = np.asarray(packet["current"], dtype=np.float32)
        if current.shape != (self.width,):
            raise ContractError("Packet width disagrees with the public contract.")
        values = (current.astype(np.float64) + 1.0) / 2.0 * np.maximum(
            self._high - self._low, 1.0
        ) + self._low
        decoded: dict[str, np.ndarray] = {}
        offset = 0
        for field in self.fields:
            width = len(field.low)
            decoded[field.name] = values[offset : offset + width]
            offset += width
        return decoded

    # ------------------------------------------------------------------
    # Simulator plumbing
    # ------------------------------------------------------------------

    def _episode_key(self, attempt: int) -> Any:
        jax = self._jax
        key = jax.random.key(self.ruleset_id)
        key = jax.random.fold_in(key, attempt)
        return jax.random.fold_in(key, self.initial_seed)

    def _begin_episode(self, attempt: int) -> np.ndarray:
        self._timestep = self._reset_fn(self._params, self._episode_key(attempt))
        self._attempt = attempt
        self._attempt_step = 0
        self._attempt_first_step = 0
        self._attempt_return = 0.0
        self._success_step = None
        self._pending_reset = False
        return self._project(done=False)

    def _project(self, *, done: bool) -> np.ndarray:
        view = np.asarray(self._timestep.observation, dtype=np.int64)
        if view.shape != (XLAND_VIEW, XLAND_VIEW, 2):
            raise ContractError("The XLand view changed shape.")
        direction = np.zeros(XLAND_DIRECTIONS, dtype=np.float64)
        direction[int(self._timestep.state.agent.direction)] = 1.0
        assert self._goal is not None
        values = np.concatenate(
            [
                view[..., 0].reshape(-1).astype(np.float64),
                view[..., 1].reshape(-1).astype(np.float64),
                direction,
                np.array([float(done)]),
                self._goal.astype(np.float64),
                np.array([self._attempt_step / self.horizon]),
            ]
        )
        return normalize_fields(values, self._low, self._high)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _attempt_record(self) -> XLandAttempt:
        return XLandAttempt(
            index=self._attempt,
            first_step=self._attempt_first_step,
            last_step=self._global_step,
            steps=self._attempt_step,
            success=self._success_step is not None,
            native_return=self._attempt_return,
            complete=True,
            success_step=self._success_step,
        )

    def _begin_task(self, source: int) -> np.ndarray:
        self._ruleset_id = ruleset_index(source)
        ruleset = _benchmark().get_ruleset(self._ruleset_id)
        self._params = self._params.replace(ruleset=ruleset)
        goal = np.asarray(ruleset.goal, dtype=np.int64).reshape(-1)
        if (
            goal.shape != (XLAND_GOAL_LENGTH,)
            or goal.min() < 0
            or goal.max() > XLAND_GOAL_HIGH
        ):
            raise ContractError("The ruleset goal left its declared bounds.")
        self._goal = goal
        self._global_step = 0
        self._task_return = 0.0
        self._records = []
        return self._begin_episode(0)

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
        """Advance one decision and build its causal public record."""
        selected = self._validate_action(action)
        self._global_step += 1
        reset_only = self._pending_reset
        terminated = truncated = False
        if reset_only:
            # The episode ended on the previous step; this command is ignored.
            self._count_reset_only_step()
            current = self._begin_episode(self._attempt + 1)
            reward = 0.0
            attempt_done = False
            decision = PublicDecision(current=current, new_attempt=True)
        else:
            self._attempt_step += 1
            if self._attempt_step == 1:
                self._attempt_first_step = self._global_step
            self._timestep = self._step_fn(
                self._params, self._timestep, self._jnp.asarray(selected)
            )
            reward = float(self._timestep.reward)
            if reward < 0.0 or reward > 1.0:
                raise ContractError("Native XLand rewards lie in [0, 1].")
            self._attempt_return += reward
            self._task_return += reward
            ended = bool(self._timestep.last())
            if reward > 0.0:
                if not ended or self._success_step is not None:
                    raise ContractError("XLand pays exactly once, at the goal.")
                self._success_step = self._attempt_step
            timed_out = self._attempt_step >= self.horizon
            if timed_out and not ended:
                raise ContractError("The native step limit disagrees with the horizon.")
            attempt_done = ended
            current = self._project(done=attempt_done)
            if attempt_done:
                self._records.append(self._attempt_record())
                if self._attempt + 1 >= self.attempts:
                    terminated = truncated = True
                else:
                    self._pending_reset = True
            decision = PublicDecision(
                current=current,
                previous=self._current,
                outcome=current,
                executed_action=selected,
                reward=reward,
                physical_done=attempt_done,
                outer_terminated=terminated,
                outer_truncated=truncated,
            )
        self._current = current
        self._done = bool(terminated or truncated)
        info = {
            "evaluator_task_index": self.evaluator_task_index,
            "attempt_index": self._attempt,
            "attempt_done": attempt_done,
            "attempt_success": attempt_done and self._success_step is not None,
            "attempt_return": self._attempt_return if attempt_done else 0.0,
            "attempt_steps": self._attempt_step if attempt_done else 0,
            "reset_only": reset_only,
            "step_in_task": self._global_step,
            "task_return": self._task_return,
        }
        return self._packet(decision), reward, terminated, truncated, info

    def close(self) -> None:
        self._timestep = None

    # ------------------------------------------------------------------
    # Restorable state
    # ------------------------------------------------------------------

    def _identity(self) -> tuple[object, ...]:
        return (
            XLAND_ENVIRONMENT_ID,
            XLAND_BENCHMARK,
            XLAND_SPLIT_KEY,
            self.horizon,
            self.attempts,
            self.scored_from,
        )

    def _leaf_to_record(self, leaf: Any) -> dict[str, object]:
        """One simulator leaf as JSON: typed PRNG keys travel as their key data."""
        jax = self._jax
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            data = np.asarray(jax.random.key_data(leaf))
            return {"key": True, "dtype": str(data.dtype), "value": data.tolist()}
        array = np.asarray(leaf)
        return {"key": False, "dtype": str(array.dtype), "value": array.tolist()}

    def _record_to_leaf(self, record: Mapping[str, Any], template: Any) -> Any:
        jax = self._jax
        value = np.asarray(record["value"], dtype=np.dtype(str(record["dtype"])))
        if bool(record.get("key")):
            if not jax.dtypes.issubdtype(template.dtype, jax.dtypes.prng_key):
                raise ContractError("XLand-MiniGrid checkpoint key leaf misplaced.")
            expected = tuple(np.asarray(jax.random.key_data(template)).shape)
            if value.shape != expected:
                raise ContractError("XLand-MiniGrid checkpoint key shape mismatch.")
            return jax.random.wrap_key_data(self._jnp.asarray(value))
        if value.shape != tuple(np.asarray(template).shape):
            raise ContractError("XLand-MiniGrid checkpoint leaf shape mismatch.")
        return self._jnp.asarray(value)

    def _episode_state(self) -> dict[str, object]:
        """The attempt counters, records and simulator leaves of the live task."""
        leaves, _ = self._jax.tree_util.tree_flatten(self._timestep)
        return {
            "pending_reset": self._pending_reset,
            "attempt": self._attempt,
            "attempt_step": self._attempt_step,
            "attempt_first_step": self._attempt_first_step,
            "attempt_return": self._attempt_return,
            "success_step": self._success_step,
            "global_step": self._global_step,
            "task_return": self._task_return,
            "records": [asdict(record) for record in self._records],
            "timestep": [self._leaf_to_record(leaf) for leaf in leaves],
        }

    def _state(self) -> dict[str, object]:
        return {"ruleset_id": self._ruleset_id, **self._episode_state()}

    def _load_timestep(self, stored: Sequence[object]) -> None:
        """Rebuild the simulator timestep against the structure of a fresh reset."""
        template = self._reset_fn(self._params, self._episode_key(0))
        template_leaves, treedef = self._jax.tree_util.tree_flatten(template)
        if len(template_leaves) != len(stored):
            raise ContractError(
                "XLand-MiniGrid checkpoint changed the simulator state."
            )
        leaves = [
            self._record_to_leaf(cast(Mapping[str, Any], saved), leaf)
            for leaf, saved in zip(template_leaves, stored, strict=True)
        ]
        self._timestep = self._jax.tree_util.tree_unflatten(treedef, leaves)

    def _load_state(self, state: Mapping[str, object]) -> None:
        records = state.get("records")
        stored = state.get("timestep")
        ruleset_id = state.get("ruleset_id")
        if (
            not isinstance(records, Sequence)
            or not isinstance(stored, Sequence)
            or type(ruleset_id) is not int
        ):
            raise ContractError("XLand-MiniGrid checkpoint is malformed.")
        restored = [XLandAttempt(**cast(Any, row)) for row in records]
        ruleset = _benchmark().get_ruleset(ruleset_id)
        self._params = self._params.replace(ruleset=ruleset)
        self._ruleset_id = ruleset_id
        self._goal = np.asarray(ruleset.goal, dtype=np.int64).reshape(-1)
        self._load_timestep(stored)
        self._load_episode_state(state, restored)

    def _load_episode_state(
        self, state: Mapping[str, object], restored: Sequence[XLandAttempt]
    ) -> None:
        self._pending_reset = bool(state["pending_reset"])
        self._attempt = int(cast(Any, state["attempt"]))
        self._attempt_step = int(cast(Any, state["attempt_step"]))
        self._attempt_first_step = int(cast(Any, state["attempt_first_step"]))
        self._attempt_return = float(cast(Any, state["attempt_return"]))
        success_step = state.get("success_step")
        self._success_step = (
            None if success_step is None else int(cast(Any, success_step))
        )
        self._global_step = int(cast(Any, state["global_step"]))
        self._task_return = float(cast(Any, state["task_return"]))
        self._records = list(deepcopy(restored))


__all__ = [
    "XLAND_ACTIONS",
    "XLAND_ATTEMPTS",
    "XLAND_BENCHMARK",
    "XLAND_DATA_VARIABLE",
    "XLAND_ENVIRONMENT_ID",
    "XLAND_FIELDS",
    "XLAND_GOAL_HIGH",
    "XLAND_GOAL_LENGTH",
    "XLAND_HORIZON",
    "XLAND_PROTOCOL",
    "XLAND_SCORED_FROM",
    "XLAND_SPLIT_KEY",
    "XLAND_TRAINING_TASKS",
    "BenchmarkCorpus",
    "XLandAttempt",
    "XLandMiniGridEnv",
    "benchmark_arrays",
    "benchmark_corpus",
    "public_fields",
    "ruleset_index",
    "xland_task_sources",
]
