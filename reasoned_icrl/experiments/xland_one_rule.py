"""The XLand one-rule protocol: task manifest and rule-necessity witnesses.

The conditional application of ``the XLand protocol``
selects released ``small-1m`` rulesets whose goal is visible and whose single
hidden rule is necessary: ``AgentHoldGoal(B)`` with ``AgentHoldRule(A -> B)``
(family ``hold-produce-hold``) or ``AgentNearGoal(B)`` with ``AgentNearRule(A -> B)``
(family ``near-produce-near``), with exactly three distinct pickable objects
placed initially — the precursor ``A`` and two distractors — and the goal
object ``B`` absent. This module owns everything about that roster that is
decided before a fit:

* the corpus filter and the canonical semantic task (goal plus active rule;
  distractors do not make a new task), hashed with SHA-256 over canonical JSON;
* deduplication by canonical hash keeping the lowest source index, and the
  exclusion of every semantic task the retired broad-corpus protocol inspected;
* the hash-sorted selection of :data:`GOAL_COUNT` goal objects that are
  eligible in both families and the hash-sorted allocation of each goal x
  family stratum to train / development / final under :data:`QUOTA`;
* the study-band identities of the selected tasks, interleaved over strata so
  that every roster prefix is balanced, and the warmup entries derived from
  the training rows (rule removed, ``A`` replaced by ``B`` in its slot);
* the coverage and overlap report the protocol asks for;
* the rule-necessity witnesses: for every task and declared layout fixture a
  planned native action sequence of at most :data:`WITNESS_ACTION_LIMIT`
  actions that succeeds under the ruleset, and the same sequence under an
  empty rule set that does not.

The corpus can fall short of the declared quota; when it does the builder
raises :class:`~reasoned_icrl.experiments.contracts.ContractError` and never
relaxes the filter. Witnesses, hidden rules and every field of the manifest
are evaluator-side records: nothing here reaches a policy packet.

Layout keys are declared here so the fixed-layout lifecycle and the witnesses
draw the same layouts: a layout is ``fold_in(fold_in(key(LAYOUT_SEED), source),
layout_index)`` where ``source`` is the corpus row and ``layout_index`` is the
rollout root of an evaluation replicate (7, 17, 27, also the witness fixtures)
or, in training, an index the actor draws at or above
:data:`LAYOUT_TRAINING_OFFSET`, so evaluation layouts never coincide with
training layouts. The native reset samples object and agent positions from
the key alone, so two tasks with the same layout index place their slot-*i*
objects on the same cells.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np

from reasoned_icrl.environments.base import (
    DEVELOPMENT_TASKS,
    FINAL_TASKS,
    REVISED_FINAL_TASKS,
    TRAINING_TASKS,
)
from reasoned_icrl.environments.xland_minigrid import (
    XLAND_BENCHMARK,
    XLAND_ENVIRONMENT_ID,
    XLAND_GRID,
    BenchmarkCorpus,
    benchmark_corpus,
    ruleset_index,
)
from reasoned_icrl.environments.xland_one_rule import (
    GOAL_WIDTH,
    LAYOUT_FIXTURES,
    LAYOUT_SEED,
    LAYOUT_TRAINING_OFFSET,
    POOLS,
    RULE_ROWS,
    RULE_WIDTH,
    XLAND_ONE_RULE_ATTEMPTS,
    XLAND_ONE_RULE_HORIZON,
    XLAND_ONE_RULE_OUTER_LENGTH,
    XLAND_ONE_RULE_PROTOCOL,
    XLAND_ONE_RULE_SCORED_FROM,
    layout_key,
    manifest_ruleset,
)
from reasoned_icrl.experiments.contracts import ContractError

MANIFEST_SCHEMA = "xland-one-rule-manifest.v1"
TASK_SCHEMA = "xland-one-rule-task.v1"
WITNESS_SCHEMA = "xland-one-rule-witnesses.v1"
SELECTION_SEED = 20260913
"""The declared selection seed: enters every goal and task sort key."""
GOAL_COUNT = 16
QUOTA: Mapping[str, int] = {"train": 7, "development": 2, "final": 8}
"""Tasks per goal x family stratum and split (17 eligible precursors needed).

The protocol first declared 8/2/8; the released corpus holds only 13 goal
objects with 18 eligible precursors in both families, so the
training quota was lowered to 7 rather than relax the filter. Sixteen
goals x two families give 224 / 64 / 256 tasks in 32 strata."""
SPLIT_ORDER = ("train", "development", "final")
"""Allocation order within a hash-sorted stratum: the frozen tie-break."""
SPLIT_BANDS: Mapping[str, range] = {
    "train": TRAINING_TASKS,
    "development": DEVELOPMENT_TASKS,
    "final": REVISED_FINAL_TASKS,
}
"""Study-band identities per split; ``final`` is the untouched ``final-revised``
band because the broad-corpus protocol declared ``final`` for its own roster."""
OBJECT_COUNT = 3
PICKABLE_TILES = (3, 4, 5, 7, 11, 12)
"""``BALL, SQUARE, PYRAMID, KEY, HEX, STAR``: xminigrid's pickable tile ids."""
WALKABLE_TILES = (1, 6, 10)
"""``FLOOR, GOAL, DOOR_OPEN``: the cells a forward move enters."""
EMPTY_TILE = (0, 0)
WITNESS_ACTION_LIMIT = 64
WITNESS_HORIZON = XLAND_ONE_RULE_HORIZON
"""The protocol's physical horizon per attempt; the witness runs within it."""
CURRICULUM_WARMUP_CALLS = 1_000_000
CURRICULUM_MIXED_CALLS = 2_000_000
CURRICULUM_MIXED_PROBABILITY = 0.5
FORWARD, TURN_RIGHT, TURN_LEFT, PICK_UP, PUT_DOWN, TOGGLE = range(6)
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_INSPECTED_BANDS = (("development", DEVELOPMENT_TASKS), ("final", FINAL_TASKS))
"""The bands the broad-corpus adapter maps to rulesets; it never mapped the
``final-revised`` band, so no ruleset there was ever inspectable."""


@dataclass(frozen=True, slots=True)
class Family:
    """One equal-weight goal/rule family of the protocol."""

    name: str
    goal_id: int
    rule_id: int
    goal_class: str
    rule_class: str


FAMILIES = (
    Family("hold-produce-hold", 1, 1, "AgentHoldGoal", "AgentHoldRule"),
    Family("near-produce-near", 3, 2, "AgentNearGoal", "AgentNearRule"),
)
FAMILY_NAMES = tuple(family.name for family in FAMILIES)


def canonical_json(payload: Any) -> str:
    """The frozen serialisation behind every hash: sorted keys, no whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_of(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticTask:
    """One eligible corpus row: the semantic task plus its concrete objects."""

    family: str
    goal: tuple[int, int, int]
    rule: tuple[int, int, int, int, int]
    init_tiles: tuple[tuple[int, int], ...]
    source: int

    @property
    def product(self) -> tuple[int, int]:
        return (self.goal[1], self.goal[2])

    @property
    def precursor(self) -> tuple[int, int]:
        return (self.rule[1], self.rule[2])

    @property
    def distractors(self) -> tuple[tuple[int, int], ...]:
        return tuple(tile for tile in self.init_tiles if tile != self.precursor)

    @property
    def canonical(self) -> dict[str, Any]:
        """Goal plus active rule; distractors do not make a new semantic task."""
        return {
            "schema": TASK_SCHEMA,
            "family": self.family,
            "goal": list(self.goal),
            "rule": list(self.rule),
        }

    @property
    def sha256(self) -> str:
        return sha256_of(self.canonical)


def _tile(row: Sequence[int]) -> tuple[int, int]:
    return (int(row[0]), int(row[1]))


def eligible_task(
    corpus: BenchmarkCorpus, source: int, family: Family
) -> SemanticTask | str:
    """Classify one corpus row: the semantic task, or the first failed test."""
    if int(corpus.num_rules[source]) != 1:
        return "num_rules != 1"
    goal = corpus.goals[source]
    rules = corpus.rules[source]
    if int(goal[0]) != family.goal_id:
        return "goal class"
    if int(rules[0, 0]) != family.rule_id:
        return "rule class"
    if goal.shape != (GOAL_WIDTH,) or rules.shape != (RULE_ROWS, RULE_WIDTH):
        raise ContractError("The corpus encodings changed shape.")
    if (rules[1:] != 0).any() or (rules[0, 5:] != 0).any() or (goal[3:] != 0).any():
        return "padding not canonical"
    product = _tile(goal[1:3])
    precursor = _tile(rules[0, 1:3])
    if _tile(rules[0, 3:5]) != product:
        return "rule product != goal object"
    if precursor == product:
        return "precursor == goal object"
    slots = [_tile(tile) for tile in corpus.init_tiles[source]]
    objects = tuple(tile for tile in slots if tile != EMPTY_TILE)
    if slots[: len(objects)] != list(objects):
        raise ContractError("Empty init slots are not trailing padding.")
    if len(objects) != OBJECT_COUNT:
        return f"{len(objects)} objects"
    if len(set(objects)) != OBJECT_COUNT:
        return "duplicate objects"
    if any(tile[0] not in PICKABLE_TILES for tile in objects):
        return "non-pickable object"
    if precursor not in objects:
        return "precursor absent"
    if product in objects:
        return "goal object present"
    if product[0] not in PICKABLE_TILES:
        return "goal object not pickable"
    return SemanticTask(
        family=family.name,
        goal=(int(goal[0]), *product),
        rule=(int(rules[0, 0]), *precursor, *product),
        init_tiles=objects,
        source=int(source),
    )


def filter_corpus(
    corpus: BenchmarkCorpus,
) -> tuple[list[SemanticTask], dict[str, dict[str, int]]]:
    """Every eligible row of every family, and the rejection ledger.

    The coarse tests (rule count, goal and rule class) run vectorised over the
    million rows; the remaining tests run per candidate row.
    """
    eligible: list[SemanticTask] = []
    ledger: dict[str, dict[str, int]] = {}
    one_rule = corpus.num_rules == 1
    for family in FAMILIES:
        counts: dict[str, int] = {}
        candidates = np.flatnonzero(
            one_rule
            & (corpus.goals[:, 0] == family.goal_id)
            & (corpus.rules[:, 0, 0] == family.rule_id)
        )
        counts["candidate rows"] = len(candidates)
        for source in candidates.tolist():
            outcome = eligible_task(corpus, int(source), family)
            if isinstance(outcome, str):
                counts[outcome] = counts.get(outcome, 0) + 1
            else:
                eligible.append(outcome)
        counts["eligible rows"] = sum(
            1 for task in eligible if task.family == family.name
        )
        ledger[family.name] = counts
    return eligible, ledger


def deduplicate(tasks: Iterable[SemanticTask]) -> dict[str, tuple[SemanticTask, int]]:
    """Group by canonical hash; keep the lowest source index, count the rest."""
    kept: dict[str, tuple[SemanticTask, int]] = {}
    for task in sorted(tasks, key=lambda t: t.source):
        digest = task.sha256
        if digest in kept:
            first, duplicates = kept[digest]
            kept[digest] = (first, duplicates + 1)
        else:
            kept[digest] = (task, 1)
    return kept


def inspected_rulesets() -> dict[str, list[int]]:
    """Rulesets the broad-corpus protocol reserved for its development and
    final bands (its failed pilot read the development band; the final band
    is excluded as declared, whether or not it was opened)."""
    return {
        name: [ruleset_index(task) for task in band] for name, band in _INSPECTED_BANDS
    }


def inspected_task_hashes(corpus: BenchmarkCorpus) -> dict[str, list[int]]:
    """Canonical hashes of every inspected ruleset that is itself a one-rule
    task of either family, mapped to the inspected rulesets carrying them."""
    excluded: dict[str, list[int]] = {}
    for rulesets in inspected_rulesets().values():
        for source in rulesets:
            for family in FAMILIES:
                outcome = eligible_task(corpus, source, family)
                if isinstance(outcome, SemanticTask):
                    excluded.setdefault(outcome.sha256, []).append(source)
    return excluded


def goal_sort_key(goal: tuple[int, int]) -> str:
    return sha256_of({"role": "goal", "seed": SELECTION_SEED, "goal": list(goal)})


def task_sort_key(task_sha256: str) -> str:
    return sha256_of({"role": "task", "seed": SELECTION_SEED, "task": task_sha256})


def select_goals(
    tasks: Mapping[str, tuple[SemanticTask, int]],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """The hash-sorted goal objects eligible in both families at the quota.

    Raises when fewer than :data:`GOAL_COUNT` goals qualify: the corpus, not
    the filter, decides whether the recipe is feasible.
    """
    needed = sum(QUOTA.values())
    precursors: dict[tuple[int, int], dict[str, set[tuple[int, int]]]] = {}
    for task, _ in tasks.values():
        per_family = precursors.setdefault(task.product, {})
        per_family.setdefault(task.family, set()).add(task.precursor)
    qualifying = sorted(
        (
            goal
            for goal, per_family in precursors.items()
            if all(len(per_family.get(name, ())) >= needed for name in FAMILY_NAMES)
        ),
        key=lambda goal: (goal_sort_key(goal), goal),
    )
    census = {
        "goals_with_any_eligible_task": len(precursors),
        "goals_qualifying_in_both_families": len(qualifying),
        "precursors_needed_per_family": needed,
        "qualifying_goals": [list(goal) for goal in qualifying],
    }
    if len(qualifying) < GOAL_COUNT:
        raise ContractError(
            f"Only {len(qualifying)} goal objects have {needed} eligible precursors "
            f"in both families; the protocol needs {GOAL_COUNT}. The filter is "
            "not relaxed."
        )
    return qualifying[:GOAL_COUNT], census


@dataclass(frozen=True, slots=True)
class ManifestTask:
    """One allocated task with its study identity."""

    task_id: int
    split: str
    stratum: int
    rank: int
    semantic: SemanticTask
    duplicates: int

    def as_dict(self) -> dict[str, Any]:
        s = self.semantic
        return {
            "id": self.task_id,
            "split": self.split,
            "stratum": self.stratum,
            "rank": self.rank,
            "family": s.family,
            "goal": list(s.goal),
            "rule": list(s.rule),
            "precursor": list(s.precursor),
            "init_tiles": [list(tile) for tile in s.init_tiles],
            "source": s.source,
            "sha256": s.sha256,
            "duplicates": self.duplicates,
        }


def stratum_index(goal_rank: int, family: str) -> int:
    return goal_rank * len(FAMILIES) + FAMILY_NAMES.index(family)


def allocate(
    tasks: Mapping[str, tuple[SemanticTask, int]],
    goals: Sequence[tuple[int, int]],
) -> tuple[list[ManifestTask], list[dict[str, Any]]]:
    """Hash-sort every selected stratum and cut it into the splits in order.

    Identities interleave strata (position ``rank * strata + stratum``) so any
    prefix of a split's roster is balanced over goals and families.
    """
    strata = len(goals) * len(FAMILIES)
    allocated: list[ManifestTask] = []
    unallocated: list[dict[str, Any]] = []
    for goal_rank, goal in enumerate(goals):
        for family in FAMILY_NAMES:
            stratum = stratum_index(goal_rank, family)
            members = sorted(
                (
                    entry
                    for entry in tasks.values()
                    if entry[0].product == goal and entry[0].family == family
                ),
                key=lambda entry: (task_sort_key(entry[0].sha256), entry[0].source),
            )
            cursor = 0
            for split in SPLIT_ORDER:
                for rank in range(QUOTA[split]):
                    task, duplicates = members[cursor]
                    cursor += 1
                    band = SPLIT_BANDS[split]
                    position = rank * strata + stratum
                    allocated.append(
                        ManifestTask(
                            task_id=band[position],
                            split=split,
                            stratum=stratum,
                            rank=rank,
                            semantic=task,
                            duplicates=duplicates,
                        )
                    )
            for task, duplicates in members[cursor:]:
                unallocated.append(
                    {
                        "stratum": stratum,
                        "family": family,
                        "object": list(goal),
                        "precursor": list(task.precursor),
                        "source": task.source,
                        "sha256": task.sha256,
                        "duplicates": duplicates,
                    }
                )
    allocated.sort(key=lambda t: (SPLIT_ORDER.index(t.split), t.task_id))
    return allocated, unallocated


def warmup_entry(task: ManifestTask) -> dict[str, Any]:
    """The curriculum row derived from one training task: no rule, ``B`` in
    ``A``'s slot, the distractors untouched. Same source, same layouts."""
    s = task.semantic
    init_tiles = [
        list(s.product if tile == s.precursor else tile) for tile in s.init_tiles
    ]
    return {
        "derived_from": task.task_id,
        "family": s.family,
        "goal": list(s.goal),
        "init_tiles": init_tiles,
        "source": s.source,
    }


def coverage_report(tasks: Sequence[ManifestTask]) -> dict[str, Any]:
    """Held-out goal / tile / colour coverage and the cross-family overlap."""
    by_split: dict[str, list[ManifestTask]] = {split: [] for split in SPLIT_ORDER}
    for task in tasks:
        by_split[task.split].append(task)

    def objects(rows: Iterable[ManifestTask]) -> set[tuple[int, int]]:
        found: set[tuple[int, int]] = set()
        for row in rows:
            found.update(row.semantic.init_tiles)
            found.add(row.semantic.product)
        return found

    def pairs(
        rows: Iterable[ManifestTask],
    ) -> set[tuple[str, tuple[int, int], tuple[int, int]]]:
        return {
            (r.semantic.family, r.semantic.product, r.semantic.precursor) for r in rows
        }

    training = by_split["train"]
    train_objects = objects(training)
    train_goals = {t.semantic.product for t in training}
    train_pairs = pairs(training)
    report: dict[str, Any] = {}
    for split in ("development", "final"):
        rows = by_split[split]
        held_objects = objects(rows)
        held_pairs = pairs(rows)
        report[split] = {
            "goals_seen_in_training": all(
                t.semantic.product in train_goals for t in rows
            ),
            "tiles_seen_in_training": sorted(
                {tile for tile, _ in held_objects} - {tile for tile, _ in train_objects}
            )
            == [],
            "colours_seen_in_training": sorted(
                {colour for _, colour in held_objects}
                - {colour for _, colour in train_objects}
            )
            == [],
            "objects_unseen_in_training": sorted(
                [list(tile) for tile in held_objects - train_objects]
            ),
            "goal_precursor_pairs_unseen_in_training": len(held_pairs - train_pairs)
            == len(held_pairs),
            "tasks": len(rows),
        }
    hold = {
        (t.semantic.product, t.semantic.precursor): t.split
        for t in tasks
        if t.semantic.family == FAMILY_NAMES[0]
    }
    near = {
        (t.semantic.product, t.semantic.precursor): t.split
        for t in tasks
        if t.semantic.family == FAMILY_NAMES[1]
    }
    shared = sorted(set(hold) & set(near))
    report["cross_family"] = {
        "goal_precursor_pairs_in_both_families": len(shared),
        "shared_pairs_in_different_splits": sum(
            1 for pair in shared if hold[pair] != near[pair]
        ),
        "shared_pairs_by_split_pair": {
            f"{hold[pair]}/{near[pair]}": sum(
                1 for p in shared if (hold[p], near[p]) == (hold[pair], near[pair])
            )
            for pair in shared
        },
    }
    report["strata"] = {"count": len({t.stratum for t in tasks}), "quota": dict(QUOTA)}
    return report


def build_manifest(corpus: BenchmarkCorpus | None = None) -> dict[str, Any]:
    """Filter, deduplicate, exclude, select, allocate and describe the roster."""
    if corpus is None:
        corpus = benchmark_corpus()
    eligible, ledger = filter_corpus(corpus)
    deduplicated = deduplicate(eligible)
    excluded = inspected_task_hashes(corpus)
    dropped = sorted(digest for digest in deduplicated if digest in excluded)
    remaining = {k: v for k, v in deduplicated.items() if k not in excluded}
    goals, census = select_goals(remaining)
    allocated, unallocated = allocate(remaining, goals)
    from reasoned_icrl.environments.xland_minigrid import _import_simulator

    _, _, xminigrid = _import_simulator()
    counts = {
        "eligible_rows": len(eligible),
        "semantic_tasks": len(deduplicated),
        "semantic_tasks_dropped_as_inspected": len(dropped),
        "semantic_tasks_after_exclusion": len(remaining),
        "allocated": len(allocated),
        "unallocated_in_selected_strata": len(unallocated),
        **{
            split: sum(1 for t in allocated if t.split == split)
            for split in SPLIT_ORDER
        },
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "protocol": XLAND_ONE_RULE_PROTOCOL,
        "corpus": {
            "benchmark": XLAND_BENCHMARK,
            "environment": XLAND_ENVIRONMENT_ID,
            "file": corpus.file,
            "sha256": corpus.sha256,
            "rulesets": len(corpus.goals),
            "xminigrid": str(xminigrid.__version__),
        },
        "selection_seed": SELECTION_SEED,
        "filter": {
            "families": [
                {
                    "name": f.name,
                    "goal": [f.goal_id, f.goal_class],
                    "rule": [f.rule_id, f.rule_class],
                }
                for f in FAMILIES
            ],
            "rules": "exactly one active rule at row 0, canonical empty padding",
            "objects": (
                f"exactly {OBJECT_COUNT} distinct pickable objects: the precursor "
                "and two distractors; the goal object absent and pickable"
            ),
            "pickable_tiles": list(PICKABLE_TILES),
            "dedupe": "canonical goal + active rule, lowest source index kept",
            "exclusions": "inspected broad-corpus bands, matched by canonical hash",
        },
        "quota": dict(QUOTA),
        "goal_count": GOAL_COUNT,
        "split_bands": {
            split: [band[0], band[-1]] for split, band in SPLIT_BANDS.items()
        },
        "layouts": {
            "seed": LAYOUT_SEED,
            "fixtures": list(LAYOUT_FIXTURES),
            "training_offset": LAYOUT_TRAINING_OFFSET,
            "key": "fold_in(fold_in(key(seed), source), layout_index)",
        },
        "ledger": ledger,
        "census": census,
        "goals": [
            {"rank": rank, "object": list(goal), "sort_key": goal_sort_key(goal)}
            for rank, goal in enumerate(goals)
        ],
        "strata": [
            {
                "index": stratum_index(rank, family),
                "goal_rank": rank,
                "object": list(goal),
                "family": family,
            }
            for rank, goal in enumerate(goals)
            for family in FAMILY_NAMES
        ],
        "counts": counts,
        "exclusions": {
            "bands": {
                name: {"rulesets": len(rulesets)}
                for name, rulesets in inspected_rulesets().items()
            },
            "one_rule_inspected_tasks": len(excluded),
            "dropped": [
                {"sha256": digest, "inspected_rulesets": excluded[digest]}
                for digest in dropped
            ],
        },
        "tasks": [task.as_dict() for task in allocated],
        "warmup": [warmup_entry(task) for task in allocated if task.split == "train"],
        "unallocated": unallocated,
        "coverage": coverage_report(allocated),
    }


ROW_SECTIONS = ("tasks", "warmup", "unallocated")
"""Manifest sections serialised one row per line."""


def manifest_text(manifest: Mapping[str, Any]) -> str:
    """The committed serialisation: sorted keys, one-space indent, and the row
    sections at one compact row per line so the file stays diffable."""
    head = {k: v for k, v in manifest.items() if k not in ROW_SECTIONS}
    text = json.dumps(head, sort_keys=True, indent=1)
    assert text.endswith("\n}")
    pieces = [text[:-2]]
    for section in ROW_SECTIONS:
        rows = ",\n".join(
            "  " + canonical_json(row) for row in manifest.get(section, ())
        )
        pieces.append(
            f',\n "{section}": [\n{rows}\n ]' if rows else f',\n "{section}": []'
        )
    return "".join(pieces) + "\n}\n"


def manifest_path(repository: Path | None = None) -> Path:
    from reasoned_icrl.utils import repository_root

    root = repository_root() if repository is None else Path(repository)
    return root / "configs" / "manifests" / "xland-one-rule-v1.json"


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_text(manifest), encoding="utf-8")


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    target = manifest_path() if path is None else path
    manifest = cast(dict[str, Any], json.loads(target.read_text(encoding="utf-8")))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Structural invariants a reader relies on; raises on any violation."""
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ContractError("Not a one-rule manifest.")
    if manifest.get("protocol") != XLAND_ONE_RULE_PROTOCOL:
        raise ContractError("The manifest names another protocol.")
    tasks = manifest["tasks"]
    strata = GOAL_COUNT * len(FAMILIES)
    ids: set[int] = set()
    hashes: set[str] = set()
    per_split: dict[str, int] = {split: 0 for split in SPLIT_ORDER}
    per_stratum: dict[tuple[str, int], int] = {}
    for row in tasks:
        split = row["split"]
        if split not in SPLIT_ORDER:
            raise ContractError(f"Unknown split {split!r}.")
        band = SPLIT_BANDS[split]
        if row["id"] not in band:
            raise ContractError(f"Task {row['id']} lies outside the {split} band.")
        expected = band[row["rank"] * strata + row["stratum"]]
        if row["id"] != expected:
            raise ContractError(f"Task {row['id']} is not at its interleaved position.")
        if row["id"] in ids or row["sha256"] in hashes:
            raise ContractError("Duplicate task identity or semantic task.")
        ids.add(row["id"])
        hashes.add(row["sha256"])
        per_split[split] += 1
        key = (split, int(row["stratum"]))
        per_stratum[key] = per_stratum.get(key, 0) + 1
    for split in SPLIT_ORDER:
        if per_split[split] != QUOTA[split] * strata:
            raise ContractError(f"The {split} split does not hold its quota.")
        for stratum in range(strata):
            if per_stratum.get((split, stratum), 0) != QUOTA[split]:
                raise ContractError(f"Stratum {stratum} is unbalanced in {split}.")
    if len(manifest["goals"]) != GOAL_COUNT:
        raise ContractError("The manifest does not hold the declared goal count.")
    if len(manifest["warmup"]) != per_split["train"]:
        raise ContractError("Every training task derives exactly one warmup entry.")


def verify_manifest(path: Path | None = None) -> dict[str, Any]:
    """Rebuild from the corpus and require byte identity with the saved file."""
    target = manifest_path() if path is None else path
    saved = target.read_text(encoding="utf-8")
    rebuilt = manifest_text(build_manifest())
    if saved != rebuilt:
        raise ContractError(f"{target} does not reproduce from the pinned corpus.")
    return load_manifest(target)


def manifest_roster(manifest: Mapping[str, Any], split: str) -> tuple[int, ...]:
    """The ordered task identities of one split."""
    if split not in SPLIT_ORDER:
        raise ContractError(f"Unknown one-rule split {split!r}.")
    return tuple(
        int(row["id"])
        for row in sorted(manifest["tasks"], key=lambda r: int(r["id"]))
        if row["split"] == split
    )


ONE_RULE_SPLIT_SOURCES: Mapping[str, str] = {
    "train": "train",
    "development": "development",
    "final-revised": "final",
}
"""Environment split name -> manifest split; ``final`` is the ``final-revised``
band, and the other study bands hold no one-rule tasks."""


@lru_cache(maxsize=1)
def committed_manifest() -> dict[str, Any]:
    """The frozen manifest, loaded and validated once per process."""
    return load_manifest()


def xland_one_rule_task_sources(split: str) -> tuple[int, ...]:
    """The ordered task roster of one environment split."""
    try:
        return manifest_roster(committed_manifest(), ONE_RULE_SPLIT_SOURCES[split])
    except KeyError:
        raise ContractError(f"The one-rule protocol has no {split!r} split.") from None


@lru_cache(maxsize=1)
def _rows_by_id() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    manifest = committed_manifest()
    tasks = {int(row["id"]): row for row in manifest["tasks"]}
    warmup = {int(row["derived_from"]): row for row in manifest["warmup"]}
    return tasks, warmup


def task_row(task_id: int) -> dict[str, Any]:
    """The manifest row of one task identity (evaluator-only fields)."""
    try:
        return _rows_by_id()[0][int(task_id)]
    except KeyError:
        raise ContractError(
            f"Task {task_id} is not in the one-rule manifest."
        ) from None


def warmup_row(task_id: int) -> dict[str, Any]:
    """The warmup entry derived from one training task identity."""
    try:
        return _rows_by_id()[1][int(task_id)]
    except KeyError:
        raise ContractError(f"Task {task_id} derives no warmup entry.") from None


@dataclass(frozen=True, slots=True)
class CurriculumSchedule:
    """The matched curriculum: which pool a new lifetime draws from, and which
    pools the learner may sample, both by charged calls.

    ``warmup_calls`` and ``mixed_calls`` are global charged-call boundaries
    (1M and 2M). The actors step in lockstep, so an actor applies them to its
    own counter divided by ``actors``; the phase changes at the actor's next
    outer reset. Replay eligibility follows the same boundaries on the
    learner's measured total: warmup only, both, then primary only.
    """

    warmup_calls: int = CURRICULUM_WARMUP_CALLS
    mixed_calls: int = CURRICULUM_MIXED_CALLS
    mixed_probability: float = CURRICULUM_MIXED_PROBABILITY
    actors: int = 16

    def __post_init__(self) -> None:
        if not 0 <= self.warmup_calls <= self.mixed_calls:
            raise ContractError("Curriculum boundaries must be ordered.")
        if not 0.0 <= self.mixed_probability <= 1.0 or self.actors < 1:
            raise ContractError("Curriculum mixture and actor count are invalid.")

    @property
    def warmup_calls_per_actor(self) -> int:
        return self.warmup_calls // self.actors

    @property
    def mixed_calls_per_actor(self) -> int:
        return self.mixed_calls // self.actors

    def phase(self, actor_calls: int) -> str:
        if actor_calls < self.warmup_calls_per_actor:
            return "warmup"
        if actor_calls < self.mixed_calls_per_actor:
            return "mixed"
        return "primary"

    def pool(self, actor_calls: int, rng: np.random.Generator) -> str:
        """The pool of the lifetime an actor starts at ``actor_calls``."""
        phase = self.phase(actor_calls)
        if phase == "mixed":
            return "warmup" if rng.random() < self.mixed_probability else "primary"
        return phase

    def eligible(self, learner_calls: int) -> frozenset[str]:
        """The pools the learner may sample after ``learner_calls`` collected."""
        if learner_calls < self.warmup_calls:
            return frozenset({"warmup"})
        if learner_calls < self.mixed_calls:
            return frozenset(POOLS)
        return frozenset({"primary"})

    def as_dict(self) -> dict[str, Any]:
        return {
            "warmup_calls": self.warmup_calls,
            "mixed_calls": self.mixed_calls,
            "mixed_probability": self.mixed_probability,
            "actors": self.actors,
            "warmup_calls_per_actor": self.warmup_calls_per_actor,
            "mixed_calls_per_actor": self.mixed_calls_per_actor,
        }


# ----------------------------------------------------------------------
# Rule-necessity witnesses
# ----------------------------------------------------------------------


def plan_witness(
    grid: np.ndarray,
    position: tuple[int, int],
    direction: int,
    target: tuple[int, int],
    family: str,
) -> list[int] | None:
    """Shortest native action sequence that fires the family's rule on ``target``.

    A breadth-first search over ``(row, column, direction)`` with the native
    move semantics (a blocked forward move keeps the position). The hold
    family ends by picking up the precursor from the cell in front; the near
    family ends with a forward action whose resulting position is
    four-adjacent to the precursor, which is when ``AgentNearRule`` and
    ``AgentNearGoal`` are both checked. ``None`` when no sequence exists.
    """
    if grid.shape != (XLAND_GRID, XLAND_GRID, 2):
        raise ContractError("Witness planning expects the 9 x 9 native grid.")
    walkable = np.isin(grid[..., 0], WALKABLE_TILES)

    def ahead(state: tuple[int, int, int]) -> tuple[int, int]:
        row, column, facing = state
        d_row, d_column = _DIRECTIONS[facing]
        return (
            min(max(row + d_row, 0), XLAND_GRID - 1),
            min(max(column + d_column, 0), XLAND_GRID - 1),
        )

    def adjacent(cell: tuple[int, int]) -> bool:
        return abs(cell[0] - target[0]) + abs(cell[1] - target[1]) == 1

    def forward(state: tuple[int, int, int]) -> tuple[int, int, int]:
        cell = ahead(state)
        if walkable[cell]:
            return (cell[0], cell[1], state[2])
        return state

    start = (position[0], position[1], direction)
    if family == FAMILY_NAMES[0]:
        if ahead(start) == target:
            return [PICK_UP]
    elif family == FAMILY_NAMES[1]:
        if adjacent(forward(start)[:2]):
            return [FORWARD]
    else:
        raise ContractError(f"Unknown family {family!r}.")
    parents: dict[tuple[int, int, int], tuple[tuple[int, int, int], int]] = {}
    queue: deque[tuple[int, int, int]] = deque([start])
    seen = {start}
    while queue:
        state = queue.popleft()
        moves = (
            (FORWARD, forward(state)),
            (TURN_RIGHT, (state[0], state[1], (state[2] + 1) % 4)),
            (TURN_LEFT, (state[0], state[1], (state[2] - 1) % 4)),
        )
        for action, following in moves:
            if following in seen:
                continue
            seen.add(following)
            parents[following] = (state, action)
            if family == FAMILY_NAMES[0]:
                done = ahead(following) == target
                closing = [PICK_UP]
            else:
                done = action == FORWARD and adjacent(following[:2])
                closing = []
            if done:
                actions: list[int] = []
                cursor = following
                while cursor != start:
                    cursor, taken = parents[cursor]
                    actions.append(taken)
                actions.reverse()
                return actions + closing
            queue.append(following)
    return None


@dataclass(frozen=True, slots=True)
class Witness:
    """One task x fixture: the planned actions and both native verdicts."""

    task_id: int
    layout_index: int
    actions: tuple[int, ...]
    success_step: int | None
    rule_removed_success: bool
    agent: tuple[int, int, int]
    precursor_cell: tuple[int, int]

    @property
    def passed(self) -> bool:
        return (
            self.success_step is not None
            and self.success_step <= WITNESS_ACTION_LIMIT
            and not self.rule_removed_success
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "layout_index": self.layout_index,
            "actions": list(self.actions),
            "success_step": self.success_step,
            "rule_removed_success": self.rule_removed_success,
            "agent": list(self.agent),
            "precursor_cell": list(self.precursor_cell),
            "passed": self.passed,
        }


def run_witnesses(
    manifest: Mapping[str, Any],
    *,
    fixtures: Sequence[int] = LAYOUT_FIXTURES,
    task_ids: Sequence[int] | None = None,
) -> list[Witness]:
    """Plan and natively verify a witness for every task x fixture.

    Runs the pinned simulator on the CPU: reset under the declared layout key,
    plan on the evaluator-visible grid, step the plan under the ruleset (the
    last action must pay the native reward and end the episode within the
    action limit), then step the same plan under an empty rule set (no step
    may pay). The plan and the grid never reach a policy.
    """
    from reasoned_icrl.environments.xland_minigrid import _import_simulator

    jax, jnp, xminigrid = _import_simulator()
    env, params = xminigrid.make(XLAND_ENVIRONMENT_ID)
    params = params.replace(max_steps=WITNESS_HORIZON)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    wanted = None if task_ids is None else set(task_ids)
    witnesses: list[Witness] = []
    for row in manifest["tasks"]:
        if wanted is not None and int(row["id"]) not in wanted:
            continue
        precursor = tuple(row["precursor"])
        for layout_index in fixtures:
            key = layout_key(jax, int(row["source"]), int(layout_index))
            with_rule = params.replace(ruleset=manifest_ruleset(jnp, row, rules=True))
            timestep = reset(with_rule, key)
            grid = np.asarray(timestep.state.grid)
            cells = np.argwhere(
                (grid[..., 0] == precursor[0]) & (grid[..., 1] == precursor[1])
            )
            if len(cells) != 1:
                raise ContractError("The precursor must occupy exactly one cell.")
            target = (int(cells[0][0]), int(cells[0][1]))
            agent = (
                int(timestep.state.agent.position[0]),
                int(timestep.state.agent.position[1]),
                int(timestep.state.agent.direction),
            )
            plan = plan_witness(grid, agent[:2], agent[2], target, str(row["family"]))
            if plan is None:
                witnesses.append(
                    Witness(
                        int(row["id"]),
                        int(layout_index),
                        (),
                        None,
                        False,
                        agent,
                        target,
                    )
                )
                continue
            success_step: int | None = None
            for index, action in enumerate(plan, start=1):
                timestep = step(with_rule, timestep, jnp.asarray(action))
                if float(timestep.reward) > 0.0:
                    success_step = index
                    break
            without_rule = params.replace(
                ruleset=manifest_ruleset(jnp, row, rules=False)
            )
            control = reset(without_rule, key)
            removed_success = False
            for action in plan:
                control = step(without_rule, control, jnp.asarray(action))
                if float(control.reward) > 0.0 or bool(control.last()):
                    removed_success = True
                    break
            witnesses.append(
                Witness(
                    int(row["id"]),
                    int(layout_index),
                    tuple(plan),
                    success_step,
                    removed_success,
                    agent,
                    target,
                )
            )
    return witnesses


def witness_summary(witnesses: Sequence[Witness]) -> dict[str, Any]:
    lengths = [w.success_step for w in witnesses if w.success_step is not None]
    return {
        "schema": WITNESS_SCHEMA,
        "witnesses": len(witnesses),
        "passed": sum(1 for w in witnesses if w.passed),
        "unplanned": sum(1 for w in witnesses if not w.actions),
        "failed_native": sum(
            1 for w in witnesses if w.actions and w.success_step is None
        ),
        "over_limit": sum(
            1
            for w in witnesses
            if w.success_step is not None and w.success_step > WITNESS_ACTION_LIMIT
        ),
        "rule_removed_success": sum(1 for w in witnesses if w.rule_removed_success),
        "action_limit": WITNESS_ACTION_LIMIT,
        "actions": {
            "min": min(lengths) if lengths else None,
            "median": float(np.median(lengths)) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
    }


__all__ = [
    "CURRICULUM_MIXED_CALLS",
    "CURRICULUM_MIXED_PROBABILITY",
    "CURRICULUM_WARMUP_CALLS",
    "FAMILIES",
    "FAMILY_NAMES",
    "GOAL_COUNT",
    "LAYOUT_FIXTURES",
    "LAYOUT_SEED",
    "LAYOUT_TRAINING_OFFSET",
    "MANIFEST_SCHEMA",
    "ONE_RULE_SPLIT_SOURCES",
    "PICKABLE_TILES",
    "POOLS",
    "QUOTA",
    "SELECTION_SEED",
    "SPLIT_BANDS",
    "SPLIT_ORDER",
    "WITNESS_ACTION_LIMIT",
    "XLAND_ONE_RULE_ATTEMPTS",
    "XLAND_ONE_RULE_HORIZON",
    "XLAND_ONE_RULE_OUTER_LENGTH",
    "XLAND_ONE_RULE_PROTOCOL",
    "XLAND_ONE_RULE_SCORED_FROM",
    "CurriculumSchedule",
    "Family",
    "ManifestTask",
    "SemanticTask",
    "Witness",
    "allocate",
    "build_manifest",
    "canonical_json",
    "committed_manifest",
    "coverage_report",
    "deduplicate",
    "eligible_task",
    "filter_corpus",
    "goal_sort_key",
    "inspected_task_hashes",
    "layout_key",
    "load_manifest",
    "manifest_path",
    "manifest_roster",
    "manifest_ruleset",
    "manifest_text",
    "plan_witness",
    "run_witnesses",
    "select_goals",
    "stratum_index",
    "task_row",
    "task_sort_key",
    "validate_manifest",
    "verify_manifest",
    "warmup_entry",
    "warmup_row",
    "witness_summary",
    "write_manifest",
    "xland_one_rule_task_sources",
]
