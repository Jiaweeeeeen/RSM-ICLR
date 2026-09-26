"""XLand one-rule: fixed layouts, boundaries, curriculum, leakage and replay tags.

Execution-integrity checks on the manifest-driven environment behind the
public packet: the roster, the witness that solves an attempt natively, the
same layout restored at every attempt of a lifetime and a fresh one at the
next, the reset-only step, the 644-call bound, restoration mid-task, the
curriculum's pool choice at outer resets with its exposure counters, the
warmup rows, what the packet carries, the pool-tagged AMAGO name and the
phase-aware replay sampler. Nothing here trains to convergence or claims
learnability; the pilot decides that on the cluster.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("xminigrid")

from reasoned_icrl.environments.base import DEVELOPMENT_TASKS, REVISED_FINAL_TASKS
from reasoned_icrl.environments.xland_minigrid import XLAND_FIELDS
from reasoned_icrl.environments.xland_one_rule import (
    POOL_COUNTERS,
    XLAND_ONE_RULE_STATE_SCHEMA,
    XLandOneRuleEnv,
)
from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.environments import build_environment, roster
from reasoned_icrl.experiments.summary_memory.configs import (
    load_xland_one_rule_study,
)
from reasoned_icrl.experiments.xland_one_rule import (
    LAYOUT_FIXTURES,
    LAYOUT_TRAINING_OFFSET,
    XLAND_ONE_RULE_OUTER_LENGTH,
    XLAND_ONE_RULE_PROTOCOL,
    CurriculumSchedule,
    task_row,
    warmup_row,
    xland_one_rule_task_sources,
)
from reasoned_icrl.runtime.environments import amago_environment
from reasoned_icrl.runtime.replay import create_replay_dataset, replay_pool

ROOT = Path(__file__).resolve().parents[2]
FORWARD, RIGHT, LEFT, PICK_UP, PUT_DOWN, TOGGLE = range(6)
FLOOR, WALL, BALL, KEY, HEX, STAR = 1, 2, 3, 7, 11, 12
WITNESSES = ROOT / "outputs" / "xland-one-rule-8m" / "qualification"


def witness_actions(task_id: int, layout_index: int) -> list[int]:
    """The saved witness if the record exists, else a fresh native plan."""
    record = WITNESSES / "witnesses.json"
    if record.is_file():
        rows = json.loads(record.read_text())["rows"]
        for row in rows:
            if row["id"] == task_id and row["layout_index"] == layout_index:
                return [int(a) for a in row["actions"]]
    from reasoned_icrl.experiments.xland_one_rule import load_manifest, run_witnesses

    (witness,) = run_witnesses(
        load_manifest(), fixtures=(layout_index,), task_ids=[task_id]
    )
    assert witness.passed
    return list(witness.actions)


def simulator_layout(env: XLandOneRuleEnv) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Evaluator-only: the grid and agent pose of the live simulator state."""
    state = env._timestep.state
    return np.asarray(state.grid).copy(), (
        int(state.agent.position[0]),
        int(state.agent.position[1]),
        int(state.agent.direction),
    )


def run_until_done(env: XLandOneRuleEnv, action: int) -> int:
    calls = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(action)
        calls += 1
        done = terminated or truncated
    return calls


def test_the_contract_declares_the_one_rule_protocol() -> None:
    study = load_xland_one_rule_study()
    contract = study.contract("xland_one_rule")
    env = contract.environment
    assert contract.protocol == XLAND_ONE_RULE_PROTOCOL
    assert env.name == "xland_one_rule" and env.encoder == "xland"
    assert (env.attempts, env.scored_from, env.horizon) == (5, 4, 128)
    assert env.outer_length == XLAND_ONE_RULE_OUTER_LENGTH == 644
    assert env.curriculum == {
        "warmup_calls": 1_000_000,
        "mixed_calls": 2_000_000,
        "mixed_probability": 0.5,
    }
    assert contract.evaluation.rollout_seeds == LAYOUT_FIXTURES
    assert contract.evaluation.primary_metric == "success_last2"
    assert contract.evaluation.retention == "complete"
    assert contract.memory == {
        "summary": {"segment_length": 64, "memory_tokens": 4},
        "window": {"segment_length": 72},
    }
    assert contract.training.max_sequence_length == 644
    assert contract.training.trajectory_length == 645
    assert contract.training.epsilon_anneal_steps == 62_500
    assert contract.training.start_learning_epoch == 3
    assert contract.roster("development") == tuple(DEVELOPMENT_TASKS)
    assert contract.roster("final") == tuple(REVISED_FINAL_TASKS)
    tier = study.tier("xland_one_rule")
    assert tier.tier == 3 and len(tier.primary) == 6
    # The figure-set supplements are declared, never primary.
    assert tier.supplementary == (
        "full_gru",
        "raw_summary",
        "raw_segment",
        "memo",
        "memo_fixed",
    )
    assert tier.qualification_reference == "full_context"
    config = experiment_config(
        contract,
        study,
        condition="fixed_summary",
        seed=42,
        repository=ROOT,
        device="cpu",
    )
    mapping = config.as_runtime_mapping()
    assert mapping["model"]["summary"]["segment_length"] == 64
    assert mapping["environment"]["curriculum"]["warmup_calls"] == 1_000_000
    assert config.model.critic is None
    window = experiment_config(
        contract,
        study,
        condition="fixed_window",
        seed=42,
        repository=ROOT,
        device="cpu",
    )
    assert window.as_runtime_mapping()["model"]["window"]["segment_length"] == 72


def test_rosters_are_the_manifest_bands() -> None:
    assert xland_one_rule_task_sources("train") == tuple(range(224))
    assert xland_one_rule_task_sources("development") == tuple(DEVELOPMENT_TASKS)
    assert xland_one_rule_task_sources("final-revised") == tuple(REVISED_FINAL_TASKS)
    with pytest.raises(ContractError, match="no 'final' split"):
        xland_one_rule_task_sources("final")
    study = load_xland_one_rule_study()
    config = experiment_config(
        study.contract("xland_one_rule"),
        study,
        condition="full_context",
        seed=0,
        repository=ROOT,
        device="cpu",
    ).as_runtime_mapping()
    assert roster(config, "train") == tuple(range(224))
    assert roster(config, "development", task_count=3) == tuple(DEVELOPMENT_TASKS[:3])
    with pytest.raises(ContractError, match="not in the one-rule"):
        XLandOneRuleEnv(split="development", source_indices=(0,), initial_seed=7)
    with pytest.raises(ContractError, match="training split only"):
        XLandOneRuleEnv(split="development", curriculum=CurriculumSchedule())
    with pytest.raises(ContractError, match="at most 128"):
        XLandOneRuleEnv(split="development", physical_horizon=129)
    trained = build_environment(config, split="train", seed=3)
    assert isinstance(trained, XLandOneRuleEnv) and trained.curriculum is not None
    assert trained.curriculum.actors == 16
    scored = build_environment(config, split="development", seed=7)
    assert isinstance(scored, XLandOneRuleEnv) and scored.curriculum is None


def test_a_witness_solves_the_attempt_and_the_layout_is_fixed_per_lifetime() -> None:
    env = XLandOneRuleEnv(split="development", initial_seed=LAYOUT_FIXTURES[0])
    task = env.source_indices[0]
    env.set_task(task)
    packet, info = env.reset(seed=0)
    assert set(packet) == {"current", "previous", "outcome", "event", "valid"}
    assert info["pool"] == "primary" and info["layout_index"] == LAYOUT_FIXTURES[0]
    assert env.layout_index == LAYOUT_FIXTURES[0] and env.replay_pool == "primary"
    row = task_row(task)
    assert env.task_manifest_row["sha256"] == row["sha256"]
    with pytest.raises(ContractError, match="manifest rows"):
        _ = env.ruleset_id
    first_grid, first_pose = simulator_layout(env)
    plan = witness_actions(task, LAYOUT_FIXTURES[0])
    reward = 0.0
    taken = 0
    for action in plan:
        packet, reward, terminated, truncated, info = env.step(action)
        taken += 1
        assert not terminated and not truncated
        if info["attempt_done"]:
            break
    assert info["attempt_success"] and info["attempt_steps"] == len(plan) == taken
    assert reward == pytest.approx(1.0 - 0.9 * len(plan) / 128)
    assert packet["event"].tolist() == [1.0, 0.0, 1.0]
    # The reset-only step ignores its action, pays nothing and restores the layout.
    packet, reward, terminated, truncated, info = env.step(TOGGLE)
    assert info["reset_only"] and reward == 0.0 and info["attempt_index"] == 1
    assert packet["event"].tolist() == [0.0, 1.0, 0.0]
    grid, pose = simulator_layout(env)
    assert np.array_equal(grid, first_grid) and pose == first_pose
    # The same plan solves attempt 2 on the restored layout.
    for action in plan:
        packet, reward, terminated, truncated, info = env.step(action)
        if info["attempt_done"]:
            break
    assert info["attempt_success"]
    # Fail the remaining attempts by turning in place: the lifetime ends at
    # the fifth attempt's limit, with every attempt recorded.
    env.step(TOGGLE)
    calls = run_until_done(env, RIGHT)
    assert calls == 3 * 128 + 2
    attempts = env.completed_attempts
    assert [a.success for a in attempts] == [True, True, False, False, False]
    assert [a.steps for a in attempts] == [len(plan), len(plan), 128, 128, 128]
    assert env.task_return == pytest.approx(2 * (1.0 - 0.9 * len(plan) / 128))
    assert env.partial_attempt is None
    counters = env.collection_counters()
    assert counters["reset_only_steps"] == 4 and counters["tasks_completed"] == 1
    assert counters["primary_tasks_started"] == 1
    assert counters["primary_charged_calls"] == counters["charged_calls"]
    assert counters["warmup_tasks_started"] == 0
    # A different rollout root is a different layout of the same task.
    other = XLandOneRuleEnv(split="development", initial_seed=LAYOUT_FIXTURES[1])
    other.set_task(task)
    other.reset(seed=0)
    other_grid, _ = simulator_layout(other)
    assert not np.array_equal(other_grid, first_grid)
    # The same root reproduces the layout exactly, whatever the roster RNG did.
    again = XLandOneRuleEnv(split="development", initial_seed=LAYOUT_FIXTURES[0])
    again.set_task(task)
    again.reset(seed=99)
    again_grid, again_pose = simulator_layout(again)
    assert np.array_equal(again_grid, first_grid) and again_pose == first_pose


def test_the_lifetime_is_bounded_by_644_calls_and_645_records() -> None:
    env = XLandOneRuleEnv(split="development", initial_seed=LAYOUT_FIXTURES[2])
    env.set_task(env.source_indices[5])
    env.reset(seed=0)
    calls = run_until_done(env, LEFT)
    assert calls == XLAND_ONE_RULE_OUTER_LENGTH == 644
    assert env.collection_counters()["physical_actions"] == 640
    assert all(not a.success and a.steps == 128 for a in env.completed_attempts)
    with pytest.raises(ContractError, match="outer reset"):
        env.step(LEFT)


def test_the_packet_carries_only_the_declared_public_values() -> None:
    env = XLandOneRuleEnv(split="development", initial_seed=LAYOUT_FIXTURES[0])
    env.set_task(env.source_indices[1])
    packet, _ = env.reset(seed=0)
    assert [f.name for f in env.fields] == list(XLAND_FIELDS)
    assert env.width == 25 + 25 + 4 + 1 + 5 + 1
    decoded = env.public_fields(packet)
    row = task_row(env.source_indices[1])
    assert np.rint(decoded["goal"]).astype(int).tolist() == [*row["goal"], 0, 0]
    assert decoded["attempt_time"].tolist() == [0.0]
    # Nothing in the packet encodes the hidden rule, the precursor, the layout
    # index, the pool or the task identity: the fields are exactly the six
    # public ones and their widths.
    assert sum(len(f.low) for f in env.fields) == env.width
    contract = env.contract
    assert contract["environment_protocol"] == XLAND_ONE_RULE_PROTOCOL
    assert {f["name"] for f in contract["fields"]} == set(XLAND_FIELDS)


def test_state_restores_mid_task_and_refuses_the_wrong_contract() -> None:
    env = XLandOneRuleEnv(split="development", initial_seed=LAYOUT_FIXTURES[0])
    env.set_task(env.source_indices[2])
    env.reset(seed=4)
    for action in (FORWARD, RIGHT, FORWARD, PICK_UP):
        env.step(action)
    snapshot = env.state_dict()
    assert snapshot["schema"] == XLAND_ONE_RULE_STATE_SCHEMA
    assert snapshot["pool"] == "primary" and snapshot["layout_index"] == 7
    twin = XLandOneRuleEnv(split="development", initial_seed=LAYOUT_FIXTURES[0])
    twin.load_state_dict(snapshot)
    for action in (LEFT, FORWARD, FORWARD, PUT_DOWN, FORWARD):
        left = env.step(action)
        right = twin.step(action)
        assert all(np.array_equal(left[0][k], right[0][k]) for k in left[0])
        assert left[1:4] == right[1:4] and left[4] == right[4]
    assert twin.collection_counters() == env.collection_counters()
    other_root = XLandOneRuleEnv(split="development", initial_seed=LAYOUT_FIXTURES[1])
    other_root.load_state_dict(snapshot)  # the root is not part of the contract
    assert other_root.layout_index == 7
    broad = XLandOneRuleEnv(split="train", initial_seed=0)
    with pytest.raises(ContractError, match="contract does not match"):
        broad.load_state_dict(snapshot)
    bad = json.loads(json.dumps(snapshot))
    bad["pool"] = "other"
    with pytest.raises(ContractError, match="malformed"):
        twin.load_state_dict(bad)


def test_the_curriculum_chooses_the_pool_at_outer_resets() -> None:
    schedule = CurriculumSchedule(
        warmup_calls=16 * 700, mixed_calls=16 * 1400, actors=16
    )
    assert schedule.warmup_calls_per_actor == 700
    assert schedule.mixed_calls_per_actor == 1400
    assert schedule.phase(699) == "warmup" and schedule.phase(700) == "mixed"
    assert schedule.phase(1400) == "primary"
    assert schedule.eligible(0) == {"warmup"}
    assert schedule.eligible(16 * 700) == {"warmup", "primary"}
    assert schedule.eligible(16 * 1400) == {"primary"}
    env = XLandOneRuleEnv(split="train", initial_seed=11, curriculum=schedule)
    pools: list[tuple[str, int, int]] = []
    for _ in range(5):
        env.reset()
        pools.append(
            (
                env.replay_pool,
                env.layout_index,
                env.collection_counters()["charged_calls"],
            )
        )
        run_until_done(env, RIGHT)  # 644 calls: never solves anything
    assert [p[0] for p in pools][:2] == ["warmup", "warmup"]  # 0 and 644 < 700
    assert pools[2][0] in ("warmup", "primary")  # 1288: mixed
    assert [p[0] for p in pools][3:] == ["primary", "primary"]  # 1932, 2576
    assert all(p[1] >= LAYOUT_TRAINING_OFFSET for p in pools)
    assert len({p[1] for p in pools}) == 5
    counters = env.collection_counters()
    assert set(POOL_COUNTERS) <= set(counters)
    assert counters["warmup_tasks_started"] + counters["primary_tasks_started"] == 5
    assert (
        counters["warmup_charged_calls"] + counters["primary_charged_calls"] == 5 * 644
    )
    # A warmup lifetime is the training row without its rule and with the goal
    # object in the precursor's slot; the roster identity is unchanged.
    env = XLandOneRuleEnv(split="train", initial_seed=12, curriculum=schedule)
    env.reset()
    assert env.replay_pool == "warmup"
    task = env.evaluator_task_index
    assert env.task_manifest_row == warmup_row(task)
    primary = task_row(task)
    assert env.task_manifest_row["derived_from"] == task
    tiles = env.task_manifest_row["init_tiles"]
    assert primary["goal"][1:3] in tiles and primary["precursor"] not in tiles
    # The pool and the counters survive a snapshot.
    snapshot = env.state_dict()
    twin = XLandOneRuleEnv(split="train", initial_seed=11, curriculum=schedule)
    twin.load_state_dict(snapshot)
    assert (
        twin.replay_pool == "warmup"
        and twin.collection_counters() == env.collection_counters()
    )
    # Evaluation splits never draw warmup lifetimes.
    scored = XLandOneRuleEnv(split="development", initial_seed=LAYOUT_FIXTURES[0])
    scored.reset()
    assert scored.replay_pool == "primary"


def test_a_warmup_lifetime_is_solvable_without_the_rule() -> None:
    schedule = CurriculumSchedule(
        warmup_calls=16 * 10**6, mixed_calls=16 * 10**6, actors=16
    )
    env = XLandOneRuleEnv(split="train", initial_seed=5, curriculum=schedule)
    env.set_task(0)
    env.reset()
    assert env.replay_pool == "warmup"
    goal = tuple(task_row(0)["goal"][1:3])
    grid, pose = simulator_layout(env)
    from reasoned_icrl.experiments.xland_one_rule import plan_witness

    cells = np.argwhere((grid[..., 0] == goal[0]) & (grid[..., 1] == goal[1]))
    assert len(cells) == 1
    family = task_row(0)["family"]
    plan = plan_witness(
        grid, pose[:2], pose[2], (int(cells[0][0]), int(cells[0][1])), family
    )
    assert plan is not None
    for action in plan:
        _, reward, _, _, info = env.step(action)
        if info["attempt_done"]:
            break
    assert info["attempt_success"] and reward > 0.0


def test_the_amago_adapter_tags_the_pool_and_replay_samples_by_phase(
    tmp_path: Path,
) -> None:
    schedule = CurriculumSchedule(
        warmup_calls=16 * 10**6, mixed_calls=16 * 10**6, actors=16
    )
    env = XLandOneRuleEnv(split="train", initial_seed=2, curriculum=schedule)
    wrapped = amago_environment(env, name="XLandOneRuleEnv-train", seed=2)
    wrapped.reset(seed=2)
    assert wrapped.env_name == "XLandOneRuleEnv-train-warmup"
    scored = amago_environment(
        XLandOneRuleEnv(split="development", initial_seed=7),
        name="XLandOneRuleEnv-development",
        seed=7,
    )
    scored.reset(seed=7)
    assert scored.env_name == "XLandOneRuleEnv-development-primary"
    assert replay_pool("XLandOneRuleEnv-train-warmup_ab12cd34_1.5.npz") == "warmup"
    assert replay_pool("XLandOneRuleEnv-train-primary_ab12cd34_1.5.npz") == "primary"
    assert replay_pool("DarkKeyToDoorEnv-train_ab12cd34_1.5.npz") is None
    dataset = create_replay_dataset(
        tmp_path, capacity=8, curriculum=CurriculumSchedule()
    )
    fifo = Path(dataset.fifo_path)
    for name in (
        "XLandOneRuleEnv-train-warmup_00000001_1.0.npz",
        "XLandOneRuleEnv-train-warmup_00000002_2.0.npz",
        "XLandOneRuleEnv-train-primary_00000003_3.0.npz",
    ):
        (fifo / name).write_bytes(b"")
    dataset._refresh_files()
    dataset.learner_calls = 0
    assert {Path(f).name for f in dataset.eligible_filenames()} == {
        "XLandOneRuleEnv-train-warmup_00000001_1.0.npz",
        "XLandOneRuleEnv-train-warmup_00000002_2.0.npz",
    }
    dataset.learner_calls = 1_500_000
    assert len(dataset.eligible_filenames()) == 3
    dataset.learner_calls = 2_000_000
    assert [Path(f).name for f in dataset.eligible_filenames()] == [
        "XLandOneRuleEnv-train-primary_00000003_3.0.npz"
    ]
    (fifo / "Untagged_00000004_4.0.npz").write_bytes(b"")
    dataset._refresh_files()
    with pytest.raises(ContractError, match="no curriculum pool"):
        dataset.eligible_filenames()
    plain = create_replay_dataset(tmp_path / "plain", capacity=8)
    assert plain.curriculum is None and plain.eligible_filenames() == []


def test_the_learner_phase_follows_the_measured_counters(tmp_path: Path) -> None:
    dataset = create_replay_dataset(
        tmp_path, capacity=8, curriculum=CurriculumSchedule()
    )

    class FakeExperiment:
        collection_counters: ClassVar[dict[str, int]] = {"charged_calls": 2_500_000}
        accelerator: Any = None

    (
        Path(dataset.fifo_path) / "XLandOneRuleEnv-train-primary_0000000a_1.0.npz"
    ).write_bytes(b"")
    dataset.has_edit_rights = False
    log = dataset.on_end_of_collection(FakeExperiment())
    assert dataset.learner_calls == 2_500_000
    assert log["Curriculum Eligible Trajectory Files"] == 1
    assert log["Trajectory Files In Pool primary"] == 1
    assert log["Trajectory Files In Pool warmup"] == 0


def test_the_reactive_reference_reads_only_the_current_view() -> None:
    from reasoned_icrl.experiments.evaluation import (
        DEVELOPMENT_REPLICATES,
        evaluation_replicates,
        xland_one_rule_references,
        xland_reactive_action,
    )

    def fields(tiles: np.ndarray) -> dict[str, np.ndarray]:
        return {"grid_tile": tiles.reshape(-1).astype(np.float64)}

    floor = np.ones((5, 5)) * FLOOR
    # Nothing visible: forward when the cell ahead is walkable, else turn.
    assert xland_reactive_action(fields(floor)) == FORWARD
    blocked = floor.copy()
    blocked[3, 2] = WALL
    assert xland_reactive_action(fields(blocked)) == RIGHT
    # An object directly ahead is picked up; one to the side turns the agent.
    ahead = floor.copy()
    ahead[3, 2] = BALL
    assert xland_reactive_action(fields(ahead)) == PICK_UP
    left = floor.copy()
    left[4, 0] = KEY
    assert xland_reactive_action(fields(left)) == LEFT
    right = floor.copy()
    right[4, 4] = KEY
    assert xland_reactive_action(fields(right)) == RIGHT
    # A distant object straight ahead: walk towards it; the nearest wins ties
    # by row then column.
    far = floor.copy()
    far[0, 2] = HEX
    far[2, 4] = STAR
    assert xland_reactive_action(fields(far)) == FORWARD
    two = floor.copy()
    two[2, 1] = HEX
    two[2, 3] = STAR
    assert xland_reactive_action(fields(two)) == FORWARD
    study = load_xland_one_rule_study()
    contract = study.contract("xland_one_rule")
    assert evaluation_replicates(contract, "development") == LAYOUT_FIXTURES[:2]
    assert evaluation_replicates(contract, "final") == LAYOUT_FIXTURES
    assert DEVELOPMENT_REPLICATES == 2
    environments = [
        XLandOneRuleEnv(split="development", initial_seed=root)
        for root in LAYOUT_FIXTURES[:2]
    ]
    tasks = environments[0].source_indices[:4]
    random, reactive = xland_one_rule_references(
        environments, task_ids=tasks, generator_seeds=(0,)
    )
    assert random.attempt_success is not None and reactive.attempt_success is not None
    assert set(random.attempt_success) == set(reactive.attempt_success) == set(tasks)
    assert random.generator_seeds == (0,) and reactive.generator_seeds == ()
    assert 0.0 <= random.mean_attempt_success <= 1.0
    assert 0.0 <= reactive.mean_attempt_success <= 1.0
    assert reactive.charged_calls <= 2 * len(tasks) * XLAND_ONE_RULE_OUTER_LENGTH
    assert random.physical_actions <= random.charged_calls
    from dataclasses import replace

    other = replace(
        contract, environment=replace(contract.environment, name="xland_minigrid")
    )
    with pytest.raises(ContractError, match="several roots"):
        evaluation_replicates(other, "development")
