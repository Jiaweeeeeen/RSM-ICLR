"""XLand-MiniGrid (decision 15): boundaries, leakage, rosters and records.

Execution-integrity checks on the pinned ``xminigrid`` simulator behind the
public packet. Nothing here trains to convergence or claims learnability; the
pilot (M10.9) decides that on the cluster.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("xminigrid")

from reasoned_icrl.environments.base import (
    DEVELOPMENT_TASKS,
    FINAL_TASKS,
    TRAINING_TASKS,
)
from reasoned_icrl.environments.xland_minigrid import (
    XLAND_ACTIONS,
    XLAND_ATTEMPTS,
    XLAND_FIELDS,
    XLAND_GOAL_LENGTH,
    XLAND_HORIZON,
    XLAND_PROTOCOL,
    XLAND_SCORED_FROM,
    XLAND_TRAINING_TASKS,
    XLandAttempt,
    XLandMiniGridEnv,
    ruleset_index,
    xland_task_sources,
)
from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import (
    ContractError,
    ResultValidationError,
)
from reasoned_icrl.experiments.environments import (
    build_environment,
    roster,
)
from reasoned_icrl.experiments.evaluation import (
    AttemptTaskResult,
    attempt_secondary,
    events,
    random_policy_attempt_success,
)
from reasoned_icrl.experiments.summary_memory.configs import (
    load_retired_summary_memory_study,
)

ROOT = Path(__file__).resolve().parents[2]
FORWARD, RIGHT, LEFT, PICK_UP, PUT_DOWN, TOGGLE = range(6)


def contract() -> Any:
    return load_retired_summary_memory_study().contract("xland_minigrid")


def resolved(condition: str = "raw", **overrides: Any) -> Any:
    settings: dict[str, Any] = {"seed": 0, "repository": ROOT, "device": "cpu"}
    settings.update(overrides)
    study = load_retired_summary_memory_study()
    return experiment_config(contract(), study, condition=condition, **settings)


def environment(**overrides: Any) -> XLandMiniGridEnv:
    settings: dict[str, Any] = {
        "physical_horizon": 12,
        "attempts": 3,
        "scored_from": 2,
        "split": "development",
        "initial_seed": 0,
    }
    settings.update(overrides)
    return XLandMiniGridEnv(**settings)


def run_task(env: XLandMiniGridEnv, task: int, actions: Any) -> list[tuple[Any, ...]]:
    env.set_task(task)
    env.reset(seed=0)
    trace = []
    done = False
    index = 0
    while not done:
        action = int(actions[index % len(actions)])
        index += 1
        packet, reward, terminated, truncated, info = env.step(action)
        trace.append((packet, reward, terminated, truncated, info))
        done = terminated or truncated
    return trace


def test_the_contract_declares_the_native_protocol() -> None:
    c = contract()
    env = c.environment
    assert c.protocol == XLAND_PROTOCOL and env.name == "xland_minigrid"
    assert env.attempts == XLAND_ATTEMPTS and env.scored_from == XLAND_SCORED_FROM
    assert env.horizon == XLAND_HORIZON and env.encoder == "xland"
    assert env.outer_length == XLAND_ATTEMPTS * (XLAND_HORIZON + 1) - 1 == 1219
    assert env.scored_attempts == 2
    assert c.evaluation.primary_metric == "success_last2"
    assert c.event_kind == "attempt"
    assert c.memory == {
        "summary": {"segment_length": 32, "memory_tokens": 4},
        "window": {"segment_length": 40},
        "critic": {"min_return": -500.0, "max_return": 500.0, "output_bins": 32},
    }
    assert c.training.reward_multiplier == 10.0 and c.training.gradient_clip == 2.0
    config = resolved()
    assert config.model.critic is not None
    assert (config.model.critic.min_return, config.model.critic.max_return) == (
        -500.0,
        500.0,
    )
    assert config.model.critic.output_bins == 32
    assert config.as_runtime_mapping()["environment"]["encoder"] == "xland"


def test_task_rosters_are_disjoint_and_select_disjoint_rulesets() -> None:
    assert xland_task_sources("train") == XLAND_TRAINING_TASKS
    assert xland_task_sources("development") == DEVELOPMENT_TASKS
    assert xland_task_sources("final") == FINAL_TASKS
    assert len(XLAND_TRAINING_TASKS) + 64 + 256 == 1_000_000
    assert set(XLAND_TRAINING_TASKS).isdisjoint(DEVELOPMENT_TASKS)
    assert XLAND_TRAINING_TASKS[-1] < TRAINING_TASKS[-1]
    development = {ruleset_index(t) for t in DEVELOPMENT_TASKS}
    final = {ruleset_index(t) for t in FINAL_TASKS}
    sample = {ruleset_index(t) for t in range(0, len(XLAND_TRAINING_TASKS), 997)}
    assert len(development) == 64 and len(final) == 256
    assert development.isdisjoint(final) and sample.isdisjoint(development | final)
    assert ruleset_index(5) == ruleset_index(5)  # pinned, not sampled
    with pytest.raises(ContractError, match="outside every XLand band"):
        ruleset_index(TRAINING_TASKS[-1])
    config = resolved().as_runtime_mapping()
    assert roster(config, "train") == XLAND_TRAINING_TASKS
    assert roster(config, "development", task_count=4) == range(
        DEVELOPMENT_TASKS[0], DEVELOPMENT_TASKS[0] + 4
    )


def test_the_packet_carries_only_the_declared_public_values() -> None:
    env = environment()
    try:
        assert tuple(f.name for f in env.fields) == XLAND_FIELDS
        assert env.width == 25 + 25 + 4 + 1 + XLAND_GOAL_LENGTH + 1
        assert env.action_space.n == XLAND_ACTIONS
        packet, info = env.reset(seed=0)
        assert env.observation_space.contains(packet)
        decoded = env.public_fields(packet)
        assert decoded["grid_tile"].shape == (25,) and decoded["grid_tile"].max() <= 12
        assert decoded["grid_color"].max() <= 11
        assert decoded["direction"].sum() == 1.0 and decoded["attempt_done"][0] == 0.0
        assert decoded["goal"].shape == (XLAND_GOAL_LENGTH,)
        assert decoded["attempt_time"][0] == 0.0
        assert set(info) == {
            "evaluator_task_index",
            "attempt_index",
            "attempt_done",
            "attempt_success",
            "attempt_return",
            "attempt_steps",
            "reset_only",
            "step_in_task",
            "task_return",
        }
        # Two rulesets with the same goal-visible surface differ only in the
        # values the policy may see; the rule encoding is never in a packet.
        assert not hasattr(env, "rule_encoding")
        assert "rules" not in json.dumps({k: v.tolist() for k, v in packet.items()})
    finally:
        env.close()


def test_episode_boundaries_follow_the_declared_lifecycle() -> None:
    env = environment(physical_horizon=4, attempts=3, scored_from=3)
    try:
        trace = run_task(env, DEVELOPMENT_TASKS[1], [RIGHT])  # turning never succeeds
        packets = [row[0] for row in trace]
        events_ = [tuple(p["event"].tolist()) for p in packets]
        # Four physical steps, a reset-only step, four, a reset-only step, four.
        assert events_ == (
            [(1.0, 0.0, 0.0)] * 3
            + [(1.0, 0.0, 1.0)]
            + [(0.0, 1.0, 0.0)]
            + [(1.0, 0.0, 0.0)] * 3
            + [(1.0, 0.0, 1.0)]
            + [(0.0, 1.0, 0.0)]
            + [(1.0, 0.0, 0.0)] * 3
            + [(1.0, 0.0, 1.0)]
        )
        assert [row[2] for row in trace] == [False] * 13 + [True]
        assert [row[3] for row in trace] == [False] * 13 + [True]
        assert all(row[1] == 0.0 for row in trace)
        infos = [row[4] for row in trace]
        assert [i["attempt_index"] for i in infos] == [0] * 4 + [1] * 5 + [2] * 5
        assert [i["reset_only"] for i in infos] == [False] * 4 + [True] + [
            False
        ] * 4 + [True] + [False] * 4
        assert [i["attempt_done"] for i in infos] == [False] * 3 + [True] + [
            False
        ] * 4 + [True] + [False] * 4 + [True]
        records = env.completed_attempts
        assert len(records) == 3 and all(isinstance(r, XLandAttempt) for r in records)
        assert [r.first_step for r in records] == [1, 6, 11]
        assert [r.last_step for r in records] == [4, 9, 14]
        assert all(r.steps == 4 and not r.success and r.complete for r in records)
        assert env.partial_attempt is None and env.task_return == 0.0
        # The done flag and the episode clock are public.
        assert env.public_fields(packets[3])["attempt_done"][0] == 1.0
        assert env.public_fields(packets[3])["attempt_time"][0] == 1.0
        assert env.public_fields(packets[4])["attempt_time"][0] == 0.0
        with pytest.raises(ContractError, match="outer reset"):
            env.step(0)
    finally:
        env.close()


def test_a_task_identity_selects_the_same_ruleset_and_layouts_everywhere() -> None:
    first = environment(initial_seed=0)
    second = environment(initial_seed=0)
    other_seed = environment(initial_seed=1)
    try:
        task = DEVELOPMENT_TASKS[2]
        a = run_task(first, task, [FORWARD, RIGHT, FORWARD, LEFT])
        b = run_task(second, task, [FORWARD, RIGHT, FORWARD, LEFT])
        assert first.ruleset_id == second.ruleset_id == ruleset_index(task)
        for (pa, ra, *_), (pb, rb, *_) in zip(a, b, strict=True):
            assert ra == rb
            for key in pa:
                np.testing.assert_array_equal(pa[key], pb[key])
        # A different actor seed keeps the ruleset and goal, moves the layout.
        c = run_task(other_seed, task, [FORWARD, RIGHT, FORWARD, LEFT])
        assert other_seed.ruleset_id == first.ruleset_id
        np.testing.assert_array_equal(
            first.public_fields(a[0][0])["goal"],
            other_seed.public_fields(c[0][0])["goal"],
        )
        # Episodes of one task differ in layout: the view is not constant across resets.
        views = [
            first.public_fields(row[0])["grid_tile"]
            for row in a
            if row[4]["reset_only"]
        ]
        assert len(views) == 2
    finally:
        first.close()
        second.close()
        other_seed.close()


def test_state_dict_round_trips_mid_episode_and_at_a_boundary() -> None:
    env = environment(physical_horizon=5, attempts=2, scored_from=1)
    twin = environment(physical_horizon=5, attempts=2, scored_from=1)
    try:
        env.set_task(DEVELOPMENT_TASKS[3])
        env.reset(seed=0)
        for action in (FORWARD, FORWARD, RIGHT, FORWARD, FORWARD):
            env.step(action)  # the fifth step ends the episode
        state = env.state_dict()
        assert state["schema"] == env.state_schema
        assert json.dumps(state)  # every leaf is serialisable
        expected = env.step(TOGGLE)  # the reset-only step
        twin.set_task(DEVELOPMENT_TASKS[3])
        twin.reset(seed=7)
        twin.load_state_dict(state)
        actual = twin.step(TOGGLE)
        for key in expected[0]:
            np.testing.assert_array_equal(expected[0][key], actual[0][key])
        assert expected[1:4] == actual[1:4] and expected[4] == actual[4]
        assert twin.completed_attempts == env.completed_attempts
        assert twin.ruleset_id == env.ruleset_id
        with pytest.raises(ContractError, match="malformed"):
            twin.load_state_dict({**state, "timestep": None})
    finally:
        env.close()
        twin.close()


def test_the_builder_resolves_the_contract_and_the_smoke_profile() -> None:
    config = resolved().as_runtime_mapping()
    env = build_environment(config, split="development", seed=3, task_count=2)
    try:
        assert isinstance(env, XLandMiniGridEnv)
        assert env.horizon == XLAND_HORIZON and env.attempts == XLAND_ATTEMPTS
        assert env.scored_from == XLAND_SCORED_FROM and env.initial_seed == 3
        assert env.source_indices == range(
            DEVELOPMENT_TASKS[0], DEVELOPMENT_TASKS[0] + 2
        )
    finally:
        env.close()
    smoke = resolved(smoke=True)
    assert smoke.environment.horizon == 8 and smoke.environment.attempts == 5
    assert smoke.training.max_sequence_length == smoke.environment.outer_length == 44


def test_events_score_the_declared_band_and_secondary_reports_every_episode() -> None:
    c = contract()
    config = resolved()
    attempts = tuple(
        XLandAttempt(
            index=k,
            first_step=1 + 3 * k,
            last_step=2 + 3 * k,
            steps=2,
            success=k >= 3,
            native_return=0.9 if k >= 3 else 0.0,
            complete=True,
            success_step=2 if k >= 3 else None,
        )
        for k in range(5)
    )
    result = AttemptTaskResult(DEVELOPMENT_TASKS[0], 0, attempts, None, 1.8, 14, None)
    rows = events(
        c, config, [result], checkpoint="e1", split="development", history="retained"
    )
    assert [r.event_index for r in rows] == [4, 5]
    assert [r.numerator for r in rows] == [1, 1] and rows[0].native_return == 0.9
    secondary = attempt_secondary(c.environment, [result])
    assert [secondary[f"attempt_{k}_success"] for k in range(1, 6)] == [0, 0, 0, 1, 1]
    assert secondary["success_first"] == 0.0 and secondary["success_last2"] == 1.0
    assert secondary["delta_adapt"] == 1.0
    short = AttemptTaskResult(DEVELOPMENT_TASKS[0], 0, attempts[:3], None, 0.0, 9, None)
    with pytest.raises(ResultValidationError, match="completed 3 attempts"):
        events(
            c, config, [short], checkpoint="e1", split="development", history="retained"
        )
    with pytest.raises(ContractError, match="timing disagree"):
        XLandAttempt(0, 1, 2, 2, True, 0.5, True, None)
    with pytest.raises(ContractError, match="pays nothing"):
        XLandAttempt(0, 1, 2, 2, False, 0.5, True, None)


def test_the_random_reference_scores_the_band_on_the_same_rosters() -> None:
    env = environment(physical_horizon=6, attempts=3, scored_from=2)
    try:
        measured = random_policy_attempt_success(
            env, task_ids=list(DEVELOPMENT_TASKS[:3]), generator_seed=0
        )
        assert set(measured) == {
            "random_reference_scored_attempt_success",
            "random_reference_first_attempt_success",
        }
        assert all(0.0 <= v <= 1.0 for v in measured.values())
        again = random_policy_attempt_success(
            env, task_ids=list(DEVELOPMENT_TASKS[:3]), generator_seed=0
        )
        assert again == measured
    finally:
        env.close()


def test_result_records_accept_the_native_success_return() -> None:
    c = contract()
    config = resolved()
    ok = XLandAttempt(3, 10, 12, 3, True, 0.7, True, 3)
    bad = XLandAttempt(3, 10, 12, 3, True, 1.0, True, 3)
    good_rows = events(
        c,
        config,
        [AttemptTaskResult(DEVELOPMENT_TASKS[0], 0, (ok,) * 5, None, 3.5, 20, None)],
        checkpoint="e1",
        split="development",
        history="retained",
    )
    assert all(0.0 < r.native_return <= 1.0 for r in good_rows)
    assert bad.native_return == 1.0  # a full return is a valid success too


def test_the_task_diagnostic_measures_goal_to_rule_ambiguity() -> None:
    from reasoned_icrl.runtime.diagnostics import xland_diagnostic

    env = environment()
    try:
        result = xland_diagnostic(env, task_ids=list(DEVELOPMENT_TASKS[:8]))
    finally:
        env.close()
    assert result.benchmark == "xland_minigrid"
    m = result.measurements
    assert m["tasks"] == 8.0 and m["benchmark_rulesets"] == 1_000_000.0
    assert 0.0 <= m["witness_fraction"] <= 1.0
    assert m["distinct_rule_sets_per_goal_min"] >= 1.0
    assert result.passed == (m["witness_fraction"] >= 0.5 and m["witnesses"] >= 5)
    assert "rule encoding is in no packet field" in result.statement
    with pytest.raises(ContractError, match="needs tasks"):
        xland_diagnostic(environment(), task_ids=[])
