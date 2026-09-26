"""The XLand one-rule manifest: filter, selection, allocation and witnesses.

Pure checks on a synthetic corpus first (every rejection reason, the
canonical hash, the hash-sorted selection and allocation, the interleaved
identities, the warmup derivation, the coverage report and the manifest
validator), the witness planner on hand-built grids, then the committed
manifest against the pinned corpus: it reproduces byte for byte, and a
sample of its witnesses replays natively with and without the rule. The full
witness sweep is the slow test; the script did it once and saved the record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reasoned_icrl.environments.base import (
    DEVELOPMENT_TASKS,
    REVISED_FINAL_TASKS,
    TRAINING_TASKS,
)
from reasoned_icrl.environments.xland_minigrid import BenchmarkCorpus
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.xland_one_rule import (
    FAMILIES,
    FAMILY_NAMES,
    FORWARD,
    GOAL_COUNT,
    LAYOUT_FIXTURES,
    PICK_UP,
    QUOTA,
    SPLIT_ORDER,
    TURN_LEFT,
    TURN_RIGHT,
    WITNESS_ACTION_LIMIT,
    XLAND_ONE_RULE_PROTOCOL,
    SemanticTask,
    allocate,
    build_manifest,
    coverage_report,
    deduplicate,
    eligible_task,
    filter_corpus,
    goal_sort_key,
    load_manifest,
    manifest_path,
    manifest_roster,
    manifest_text,
    plan_witness,
    select_goals,
    stratum_index,
    task_sort_key,
    validate_manifest,
    warmup_entry,
)

ROOT = Path(__file__).resolve().parents[2]
HOLD, NEAR = FAMILIES
BALL, SQUARE, PYRAMID, KEY, HEX, STAR = 3, 4, 5, 7, 11, 12
GOAL_TILE = 6
FLOOR, WALL = 1, 2


def row(
    *,
    family: Any = HOLD,
    goal_id: int | None = None,
    rule_id: int | None = None,
    product: tuple[int, int] = (KEY, 9),
    precursor: tuple[int, int] = (BALL, 1),
    objects: tuple[tuple[int, int], ...] | None = None,
    rule_product: tuple[int, int] | None = None,
    num_rules: int = 1,
    extra_rule: bool = False,
) -> dict[str, Any]:
    objects = ((BALL, 1), (SQUARE, 2), (PYRAMID, 3)) if objects is None else objects
    rule_product = product if rule_product is None else rule_product
    goal = [family.goal_id if goal_id is None else goal_id, *product, 0, 0]
    rules = np.zeros((4, 7), dtype=np.int64)
    rules[0, :5] = [
        family.rule_id if rule_id is None else rule_id,
        *precursor,
        *rule_product,
    ]
    if extra_rule:
        rules[1, :5] = [family.rule_id, *precursor, *rule_product]
    init = np.zeros((10, 2), dtype=np.int64)
    for slot, tile in enumerate(objects):
        init[slot] = tile
    return {"goal": goal, "rules": rules, "init": init, "num_rules": num_rules}


def corpus(rows: list[dict[str, Any]]) -> BenchmarkCorpus:
    return BenchmarkCorpus(
        goals=np.asarray([r["goal"] for r in rows], dtype=np.int64),
        rules=np.stack([r["rules"] for r in rows]).astype(np.int64),
        init_tiles=np.stack([r["init"] for r in rows]).astype(np.int64),
        num_rules=np.asarray([r["num_rules"] for r in rows], dtype=np.int64),
        sha256="synthetic",
        file="synthetic",
    )


def test_eligible_task_names_every_rejection() -> None:
    cases: list[tuple[dict[str, Any], str]] = [
        (row(num_rules=2, extra_rule=True), "num_rules != 1"),
        (row(goal_id=NEAR.goal_id), "goal class"),
        (row(rule_id=NEAR.rule_id), "rule class"),
        (row(extra_rule=True), "padding not canonical"),
        (row(rule_product=(STAR, 4)), "rule product != goal object"),
        (row(product=(BALL, 1)), "precursor == goal object"),
        (row(objects=((BALL, 1), (SQUARE, 2))), "2 objects"),
        (row(objects=((BALL, 1), (SQUARE, 2), (SQUARE, 2))), "duplicate objects"),
        (row(objects=((BALL, 1), (SQUARE, 2), (GOAL_TILE, 3))), "non-pickable object"),
        (row(objects=((HEX, 1), (SQUARE, 2), (PYRAMID, 3))), "precursor absent"),
        (row(objects=((BALL, 1), (KEY, 9), (PYRAMID, 3))), "goal object present"),
        (row(product=(GOAL_TILE, 5)), "goal object not pickable"),
    ]
    for source, (entry, reason) in enumerate(cases):
        assert eligible_task(corpus([entry]), 0, HOLD) == reason, source
    task = eligible_task(corpus([row()]), 0, HOLD)
    assert isinstance(task, SemanticTask)
    assert task.product == (KEY, 9) and task.precursor == (BALL, 1)
    assert task.distractors == ((SQUARE, 2), (PYRAMID, 3))
    assert task.goal == (1, KEY, 9) and task.rule == (1, BALL, 1, KEY, 9)
    assert task.canonical == {
        "schema": "xland-one-rule-task.v1",
        "family": "hold-produce-hold",
        "goal": [1, KEY, 9],
        "rule": [1, BALL, 1, KEY, 9],
    }
    twin = eligible_task(
        corpus([row(objects=((BALL, 1), (HEX, 7), (STAR, 8)))]), 0, HOLD
    )
    assert isinstance(twin, SemanticTask) and twin.sha256 == task.sha256
    near = eligible_task(corpus([row(family=NEAR)]), 0, NEAR)
    assert isinstance(near, SemanticTask) and near.sha256 != task.sha256


def test_deduplicate_keeps_the_lowest_source_and_counts_the_rest() -> None:
    rows = [
        row(objects=((BALL, 1), (HEX, 7), (STAR, 8))),
        row(precursor=(SQUARE, 2)),
        row(),
    ]
    eligible, ledger = filter_corpus(corpus(rows))
    assert len(eligible) == 3 and ledger[HOLD.name]["eligible rows"] == 3
    assert ledger[NEAR.name]["candidate rows"] == 0
    kept = deduplicate(eligible)
    assert len(kept) == 2
    first, duplicates = next(v for v in kept.values() if v[0].precursor == (BALL, 1))
    assert first.source == 0 and duplicates == 2


def synthetic_study(goals: int, precursors: int) -> BenchmarkCorpus:
    """``goals`` goal objects x both families x ``precursors`` precursors."""
    rows: list[dict[str, Any]] = []
    products = [
        (tile, colour) for tile in (KEY, HEX, STAR, BALL) for colour in range(1, 12)
    ]
    candidates = [
        (tile, colour) for tile in (SQUARE, PYRAMID, BALL) for colour in range(1, 12)
    ]
    for product in products[:goals]:
        for family in FAMILIES:
            chosen = [tile for tile in candidates if tile != product][:precursors]
            for precursor in chosen:
                distractors = [t for t in candidates if t not in (precursor, product)][
                    :2
                ]
                rows.append(
                    row(
                        family=family,
                        product=product,
                        precursor=precursor,
                        objects=(precursor, *distractors),
                    )
                )
    return corpus(rows)


def test_select_goals_refuses_a_short_corpus_and_hash_sorts_a_full_one() -> None:
    needed = sum(QUOTA.values())
    eligible, _ = filter_corpus(synthetic_study(GOAL_COUNT - 1, needed))
    with pytest.raises(ContractError, match="not relaxed"):
        select_goals(deduplicate(eligible))
    eligible, _ = filter_corpus(synthetic_study(GOAL_COUNT + 2, needed - 1))
    with pytest.raises(ContractError, match="Only 0 goal objects"):
        select_goals(deduplicate(eligible))
    eligible, _ = filter_corpus(synthetic_study(GOAL_COUNT + 2, needed + 1))
    goals, census = select_goals(deduplicate(eligible))
    assert len(goals) == GOAL_COUNT
    assert census["goals_qualifying_in_both_families"] == GOAL_COUNT + 2
    assert goals == sorted(goals, key=goal_sort_key)
    assert goals == [tuple(g) for g in census["qualifying_goals"][:GOAL_COUNT]]


def test_allocation_is_hash_ordered_balanced_and_interleaved() -> None:
    needed = sum(QUOTA.values())
    eligible, _ = filter_corpus(synthetic_study(GOAL_COUNT, needed + 1))
    tasks = deduplicate(eligible)
    goals, _ = select_goals(tasks)
    allocated, unallocated = allocate(tasks, goals)
    strata = GOAL_COUNT * len(FAMILIES)
    assert len(allocated) == needed * strata and len(unallocated) == strata
    for split in SPLIT_ORDER:
        rows = [t for t in allocated if t.split == split]
        assert len(rows) == QUOTA[split] * strata
        for stratum in range(strata):
            members = sorted(
                (t for t in rows if t.stratum == stratum), key=lambda t: t.rank
            )
            assert [t.rank for t in members] == list(range(QUOTA[split]))
    # Within a stratum the hash order runs train -> development -> final.
    stratum = allocated[0].stratum
    ordered = sorted(
        (t for t in allocated if t.stratum == stratum),
        key=lambda t: task_sort_key(t.semantic.sha256),
    )
    expected = [s for s in SPLIT_ORDER for _ in range(QUOTA[s])]
    assert [t.split for t in ordered] == expected
    leftover = next(u for u in unallocated if u["stratum"] == stratum)
    assert task_sort_key(leftover["sha256"]) > task_sort_key(
        ordered[-1].semantic.sha256
    )
    # Interleaved identities: position = rank * strata + stratum in the band.
    train = [t for t in allocated if t.split == "train"]
    assert {t.task_id for t in train} == set(TRAINING_TASKS[: len(train)])
    assert all(t.task_id == t.rank * strata + t.stratum for t in train)
    development = [t for t in allocated if t.split == "development"]
    assert {t.task_id for t in development} == set(DEVELOPMENT_TASKS)
    final = [t for t in allocated if t.split == "final"]
    assert {t.task_id for t in final} == set(REVISED_FINAL_TASKS)
    assert stratum_index(3, NEAR.name) == 7
    # Warmup rows: same source, no rule, B in A's slot, distractors kept.
    task = train[0]
    warmup = warmup_entry(task)
    assert warmup["derived_from"] == task.task_id
    assert warmup["source"] == task.semantic.source
    assert warmup["init_tiles"][0] == list(task.semantic.product)
    assert warmup["init_tiles"][1:] == [list(t) for t in task.semantic.distractors]
    report = coverage_report(allocated)
    assert report["strata"]["count"] == strata
    for split in ("development", "final"):
        assert report[split]["goals_seen_in_training"]
        assert report[split]["goal_precursor_pairs_unseen_in_training"]


def test_validate_manifest_refuses_tampering() -> None:
    manifest = load_manifest()
    validate_manifest(manifest)
    tampered = json.loads(json.dumps(manifest))
    tampered["tasks"][0]["id"] += 1
    with pytest.raises(ContractError):
        validate_manifest(tampered)
    tampered = json.loads(json.dumps(manifest))
    tampered["tasks"][0]["split"] = "final"
    with pytest.raises(ContractError):
        validate_manifest(tampered)
    tampered = json.loads(json.dumps(manifest))
    tampered["protocol"] = "other"
    with pytest.raises(ContractError, match="another protocol"):
        validate_manifest(tampered)
    with pytest.raises(ContractError, match="Unknown one-rule split"):
        manifest_roster(manifest, "final-ood")


def grid(objects: dict[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    cells = np.zeros((9, 9, 2), dtype=np.int64)
    cells[..., 0] = FLOOR
    cells[0, :, 0] = cells[-1, :, 0] = cells[:, 0, 0] = cells[:, -1, 0] = WALL
    for (y, x), tile in objects.items():
        cells[y, x] = tile
    return cells


def test_witness_planner_mirrors_native_moves() -> None:
    target = (4, 4)
    cells = grid({target: (BALL, 1), (2, 2): (SQUARE, 2), (6, 6): (PYRAMID, 3)})
    # Facing the precursor: pick it up.
    assert plan_witness(cells, (5, 4), 0, target, HOLD.name) == [PICK_UP]
    # Facing away: turn twice, then pick up.
    plan = plan_witness(cells, (5, 4), 2, target, HOLD.name)
    assert plan is not None and len(plan) == 3 and plan[-1] == PICK_UP
    assert plan[:2] in ([TURN_RIGHT, TURN_RIGHT], [TURN_LEFT, TURN_LEFT])
    # Objects block movement: the path around the precursor turns.
    plan = plan_witness(cells, (1, 4), 2, target, HOLD.name)
    assert plan is not None and plan == [FORWARD, FORWARD, PICK_UP]
    # Near family: a forward move that lands next to the precursor.
    assert plan_witness(cells, (5, 5), 0, target, NEAR.name) == [FORWARD]
    # A blocked forward move keeps the position and still counts natively.
    assert plan_witness(cells, (5, 4), 0, target, NEAR.name) == [FORWARD]
    # Far away: the plan is short and ends with a forward move.
    plan = plan_witness(cells, (1, 1), 1, target, NEAR.name)
    assert plan is not None and plan[-1] == FORWARD and len(plan) <= 10
    # Walled off: no witness.
    boxed = grid({target: (BALL, 1)})
    boxed[3:6, 3, 0] = boxed[3:6, 5, 0] = boxed[3, 3:6, 0] = boxed[5, 3:6, 0] = WALL
    assert plan_witness(boxed, (1, 1), 0, target, HOLD.name) is None
    with pytest.raises(ContractError, match="Unknown family"):
        plan_witness(cells, (1, 1), 0, target, "other")


def test_committed_manifest_reproduces_from_the_pinned_corpus() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("xminigrid")
    path = manifest_path()
    assert path == ROOT / "configs" / "manifests" / "xland-one-rule-v1.json"
    saved = path.read_text(encoding="utf-8")
    manifest = build_manifest()
    assert manifest_text(manifest) == saved
    assert manifest["protocol"] == XLAND_ONE_RULE_PROTOCOL
    assert manifest["counts"]["train"] == 224
    assert manifest["counts"]["development"] == 64
    assert manifest["counts"]["final"] == 256
    assert manifest["census"]["goals_qualifying_in_both_families"] >= GOAL_COUNT
    assert manifest["exclusions"]["dropped"] == []
    for split in ("development", "final"):
        assert manifest["coverage"][split]["objects_unseen_in_training"] == []
    rosters = {split: manifest_roster(manifest, split) for split in SPLIT_ORDER}
    assert rosters["train"] == tuple(TRAINING_TASKS[:224])
    assert rosters["development"] == tuple(DEVELOPMENT_TASKS)
    assert rosters["final"] == tuple(REVISED_FINAL_TASKS)
    assert len(manifest["warmup"]) == 224
    assert all(w["derived_from"] in rosters["train"] for w in manifest["warmup"])
    families = {t["family"] for t in manifest["tasks"]}
    assert families == set(FAMILY_NAMES)


def test_witnesses_replay_natively_with_and_without_the_rule() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("xminigrid")
    from reasoned_icrl.experiments.xland_one_rule import run_witnesses

    manifest = load_manifest()
    by_family = {t["family"]: t["id"] for t in manifest["tasks"]}
    witnesses = run_witnesses(
        manifest, fixtures=LAYOUT_FIXTURES[:1], task_ids=list(by_family.values())
    )
    assert len(witnesses) == len(FAMILIES)
    for witness in witnesses:
        assert witness.passed
        assert witness.success_step == len(witness.actions) <= WITNESS_ACTION_LIMIT
        assert not witness.rule_removed_success
        assert witness.layout_index == LAYOUT_FIXTURES[0]


@pytest.mark.slow
def test_every_task_has_a_witness_on_every_fixture() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("xminigrid")
    from reasoned_icrl.experiments.xland_one_rule import run_witnesses, witness_summary

    summary = witness_summary(run_witnesses(load_manifest()))
    assert summary["witnesses"] == 544 * len(LAYOUT_FIXTURES)
    assert summary["passed"] == summary["witnesses"]
    assert summary["rule_removed_success"] == 0
    assert summary["actions"]["max"] <= WITNESS_ACTION_LIMIT
