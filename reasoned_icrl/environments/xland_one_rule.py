"""The XLand one-rule task: fixed-layout, five-attempt lifetimes over the manifest.

Same simulator, packet and attempt records as the broad-corpus adapter in
:mod:`reasoned_icrl.environments.xland_minigrid` (the partial ``5 x 5`` view,
direction and done flags, the visible goal, the attempt clock), with the
protocol of :mod:`reasoned_icrl.experiments.xland_one_rule` in place of the
broad corpus:

* one manifest task per outer reset, selected from the split's roster of
  study identities; its ruleset is rebuilt from the manifest row (goal, the
  single hidden rule, three initial objects), never read from a packet;
* **five attempts of at most 128 physical actions**, each restarting the
  **same** initial layout: the layout key is fixed for the lifetime, so
  objects and the agent return to their cells at every attempt; the next
  lifetime draws a fresh layout;
* a reset-only decision between attempts, as before; at most 644 charged
  calls and 645 records per lifetime;
* the matched curriculum on the training split: a new lifetime is a
  **warmup** row (rule removed, the goal object in the precursor's slot) or a
  **primary** row according to the actor's charged-call phase, decided at the
  outer reset; evaluation splits are primary only;
* the pool of the live lifetime is exposed as ``replay_pool`` so the AMAGO
  adapter can tag the trajectory file, and the realised exposure per pool is
  counted beside the shared collection counters.

Layouts: training draws layout indices at or above the protocol's training
offset from the actor's roster RNG (resumable); any other split uses its
``initial_seed`` as the layout index. The evaluator passes the declared rollout
roots 7 / 17 / 27, below the offset, so scored replicates never share a layout
with training; AMAGO's in-training validation actors pass their actor seeds.
Hidden rules, layouts, the pool and the task identity never reach a packet;
they are evaluator-side records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from reasoned_icrl.environments.xland_minigrid import (
    XLandAttempt,
    XLandMiniGridEnv,
)
from reasoned_icrl.experiments.contracts import ContractError

if TYPE_CHECKING:
    from reasoned_icrl.experiments.xland_one_rule import CurriculumSchedule

XLAND_ONE_RULE_PROTOCOL = "xland-r1-9x9-one-rule-goal-visible-fixed-layout-5x128-v1"
XLAND_ONE_RULE_STATE_SCHEMA = "xland-one-rule-state.v1"
XLAND_ONE_RULE_HORIZON = 128
"""Physical actions per attempt; the native limit is 243 for this room."""
XLAND_ONE_RULE_ATTEMPTS = 5
XLAND_ONE_RULE_SCORED_FROM = 4
XLAND_ONE_RULE_OUTER_LENGTH = XLAND_ONE_RULE_ATTEMPTS * (XLAND_ONE_RULE_HORIZON + 1) - 1
"""644 charged calls: five attempts of at most 128 actions and four reset-only
decisions between them; 645 records with the initial one."""
LAYOUT_SEED = 20260913
LAYOUT_FIXTURES = (7, 17, 27)
"""Declared layout indices: the evaluation rollout roots, reused as witness
fixtures. Development scoring uses the first two, final scoring all three."""
LAYOUT_TRAINING_OFFSET = 1000
"""Training layout indices start here so they never coincide with a fixture."""
LAYOUT_INDEX_LIMIT = 2**31 - 1
POOLS = ("warmup", "primary")
"""The curriculum's lifetime pools; the tag every training replay file carries."""
POOL_COUNTERS = tuple(
    f"{pool}_{name}" for pool in POOLS for name in ("tasks_started", "charged_calls")
)
"""Realised exposure per pool, beside the shared collection counters."""
RULE_ROWS = 4
RULE_WIDTH = 7
GOAL_WIDTH = 5
INIT_SLOTS = 10


def layout_key(jax: Any, source: int, layout_index: int) -> Any:
    """The declared layout key of one corpus row and layout index."""
    if layout_index < 0 or source < 0:
        raise ContractError("Layout keys take non-negative indices.")
    key = jax.random.key(LAYOUT_SEED)
    key = jax.random.fold_in(key, int(source))
    return jax.random.fold_in(key, int(layout_index))


def manifest_ruleset(jnp: Any, row: Mapping[str, Any], *, rules: bool) -> Any:
    """The native ruleset of one manifest row (or warmup entry).

    ``rules=False`` leaves every rule row empty: the warmup variant, and the
    rule-necessity control of the witnesses.
    """
    from xminigrid.types import RuleSet

    goal = jnp.asarray(
        list(row["goal"]) + [0] * (GOAL_WIDTH - len(row["goal"])), dtype=jnp.uint8
    )
    rule_rows = np.zeros((RULE_ROWS, RULE_WIDTH), dtype=np.uint8)
    if rules:
        rule_rows[0, : len(row["rule"])] = row["rule"]
    init = np.zeros((INIT_SLOTS, 2), dtype=np.uint8)
    for slot, tile in enumerate(row["init_tiles"]):
        init[slot] = tile
    return RuleSet(
        goal=goal, rules=jnp.asarray(rule_rows), init_tiles=jnp.asarray(init)
    )


class XLandOneRuleEnv(XLandMiniGridEnv):
    """One manifest task per lifetime, replayed on a fixed layout five times."""

    label: ClassVar[str] = "XLand one-rule"
    state_schema: ClassVar[str] = XLAND_ONE_RULE_STATE_SCHEMA
    protocol = XLAND_ONE_RULE_PROTOCOL

    def __init__(
        self,
        *,
        physical_horizon: int = XLAND_ONE_RULE_HORIZON,
        attempts: int = XLAND_ONE_RULE_ATTEMPTS,
        scored_from: int = XLAND_ONE_RULE_SCORED_FROM,
        split: str = "train",
        source_indices: Sequence[int] | range | None = None,
        fixed_task_index: int | None = None,
        initial_seed: int = 0,
        curriculum: CurriculumSchedule | None = None,
    ) -> None:
        if physical_horizon > XLAND_ONE_RULE_HORIZON:
            raise ContractError("The one-rule attempt is at most 128 physical actions.")
        if curriculum is not None and split != "train":
            raise ContractError("The curriculum applies to the training split only.")
        if initial_seed < 0:
            raise ContractError("The layout index of an evaluation actor is its seed.")
        self.curriculum = curriculum
        self._pool = "primary"
        self._layout_index = 0
        self._row: dict[str, Any] | None = None
        self._pool_counters = dict.fromkeys(POOL_COUNTERS, 0)
        from reasoned_icrl.experiments.xland_one_rule import (
            xland_one_rule_task_sources,
        )

        super().__init__(
            physical_horizon=physical_horizon,
            attempts=attempts,
            scored_from=scored_from,
            split=split,
            source_indices=xland_one_rule_task_sources(split)
            if source_indices is None
            else source_indices,
            fixed_task_index=fixed_task_index,
            initial_seed=initial_seed,
        )

    def _check_roster(self) -> None:
        """Every roster identity must be a manifest task of this split."""
        from reasoned_icrl.experiments.xland_one_rule import (
            xland_one_rule_task_sources,
        )

        allowed = set(xland_one_rule_task_sources(self.split))
        for source in self.source_indices:
            if int(source) not in allowed:
                raise ContractError(
                    f"Task {source} is not in the one-rule {self.split} roster."
                )

    # ------------------------------------------------------------------
    # Evaluator-only accessors
    # ------------------------------------------------------------------

    @property
    def ruleset_id(self) -> int:
        raise ContractError("One-rule tasks are manifest rows, not ruleset ids.")

    @property
    def replay_pool(self) -> str:
        """``warmup`` or ``primary`` for the live lifetime; tags replay files."""
        return self._pool

    @property
    def layout_index(self) -> int:
        """The layout index of the live lifetime. Evaluator-only."""
        return self._layout_index

    @property
    def task_manifest_row(self) -> Mapping[str, Any]:
        """The manifest row (or warmup entry) behind the live lifetime."""
        if self._row is None:
            raise ContractError(f"{self.label} must be reset before use.")
        return self._row

    def collection_counters(self) -> dict[str, int]:
        return {**super().collection_counters(), **self._pool_counters}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _episode_key(self, attempt: int) -> Any:
        """Every attempt of a lifetime restarts the same layout."""
        del attempt
        return layout_key(self._jax, int(self._row_source()), self._layout_index)

    def _row_source(self) -> int:
        if self._row is None:
            raise ContractError(f"{self.label} must be reset before use.")
        return int(self._row["source"])

    def _select_pool(self) -> str:
        if self.curriculum is None:
            return "primary"
        return self.curriculum.pool(self._charged_calls, self.np_random)

    def _select_layout(self) -> int:
        if self.split == "train":
            return int(
                self.np_random.integers(LAYOUT_TRAINING_OFFSET, LAYOUT_INDEX_LIMIT)
            )
        return self.initial_seed

    def _install_row(self, row: Mapping[str, Any], *, rules: bool) -> None:
        self._row = dict(row)
        ruleset = manifest_ruleset(self._jnp, row, rules=rules)
        self._params = self._params.replace(ruleset=ruleset)
        goal = np.asarray(ruleset.goal, dtype=np.int64).reshape(-1)
        self._goal = goal
        self._ruleset_id = None

    def _begin_task(self, source: int) -> np.ndarray:
        # The manifest is protocol data; the environment reads it lazily so the
        # environments package never imports the protocol module at load time.
        from reasoned_icrl.experiments.xland_one_rule import task_row, warmup_row

        self._pool = self._select_pool()
        self._layout_index = self._select_layout()
        if self._pool == "warmup":
            self._install_row(warmup_row(source), rules=False)
        else:
            self._install_row(task_row(source), rules=True)
        self._pool_counters[f"{self._pool}_tasks_started"] += 1
        self._global_step = 0
        self._task_return = 0.0
        self._records = []
        return self._begin_episode(0)

    def _validate_action(self, action: Any) -> int:
        selected = super()._validate_action(action)
        self._pool_counters[f"{self._pool}_charged_calls"] += 1
        return selected

    def _reset_info(self) -> dict[str, Any]:
        return {**super()._reset_info(), **self._lifetime_info()}

    def _lifetime_info(self) -> dict[str, Any]:
        return {"pool": self._pool, "layout_index": self._layout_index}

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        packet, reward, terminated, truncated, info = super().step(action)
        info.update(self._lifetime_info())
        return packet, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Restorable state
    # ------------------------------------------------------------------

    def _identity(self) -> tuple[object, ...]:
        return (
            *super()._identity(),
            None
            if self.curriculum is None
            else tuple(self.curriculum.as_dict().items()),
        )

    def _state(self) -> dict[str, object]:
        return {
            "pool": self._pool,
            "layout_index": self._layout_index,
            "row": None if self._row is None else dict(self._row),
            "pool_counters": dict(self._pool_counters),
            **self._episode_state(),
        }

    def _load_state(self, state: Mapping[str, object]) -> None:
        records = state.get("records")
        stored = state.get("timestep")
        row = state.get("row")
        pool = state.get("pool")
        layout_index = state.get("layout_index")
        counters = state.get("pool_counters")
        if (
            not isinstance(records, Sequence)
            or not isinstance(stored, Sequence)
            or not isinstance(row, Mapping)
            or pool not in POOLS
            or type(layout_index) is not int
            or not isinstance(counters, Mapping)
        ):
            raise ContractError("XLand one-rule checkpoint is malformed.")
        restored = [XLandAttempt(**cast(Any, entry)) for entry in records]
        self._pool = str(pool)
        self._layout_index = int(layout_index)
        self._install_row(row, rules=self._pool == "primary")
        self._pool_counters = {
            key: int(cast(int, counters.get(key, 0))) for key in POOL_COUNTERS
        }
        self._load_timestep(stored)
        self._load_episode_state(state, restored)


__all__ = [
    "GOAL_WIDTH",
    "INIT_SLOTS",
    "LAYOUT_FIXTURES",
    "LAYOUT_INDEX_LIMIT",
    "LAYOUT_SEED",
    "LAYOUT_TRAINING_OFFSET",
    "POOLS",
    "POOL_COUNTERS",
    "RULE_ROWS",
    "RULE_WIDTH",
    "XLAND_ONE_RULE_ATTEMPTS",
    "XLAND_ONE_RULE_HORIZON",
    "XLAND_ONE_RULE_OUTER_LENGTH",
    "XLAND_ONE_RULE_PROTOCOL",
    "XLAND_ONE_RULE_SCORED_FROM",
    "XLAND_ONE_RULE_STATE_SCHEMA",
    "XLandOneRuleEnv",
    "layout_key",
    "manifest_ruleset",
]
