"""R4 acceptance: complete event retention, metric membership and the endpoint rule.

A complete Key-to-Door record keeps every attempt of the 500-call task, dates
each key acquisition and marks the attempt the budget cut short; the count
metric reads all of them while the legacy first-eight rate reads only its
band; the endpoint panel refuses a run that stopped early; and the declared
reference policies are evaluation-only rollouts on the same roster.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from reasoned_icrl.environments.base import DEVELOPMENT_TASKS
from reasoned_icrl.experiments.artifacts import (
    checkpoint_labels,
    collected_at_label,
    endpoint_label,
    endpoint_training_epoch,
)
from reasoned_icrl.experiments.benchmarks import (
    EvaluationSplit,
    experiment_config,
    load_contract,
)
from reasoned_icrl.experiments.contracts import ContractError, ResultValidationError
from reasoned_icrl.experiments.evaluation import (
    KEY_TO_DOOR_MOVES,
    AttemptTaskResult,
    evaluation_directory,
    events,
    random_policy_door_counts,
    retained_attempts,
    sweep_policy_action,
    sweep_policy_door_counts,
)
from reasoned_icrl.experiments.records import (
    BenchmarkEvent,
    BenchmarkRun,
    cell_values,
    metric_events,
    primary_value,
    validate_benchmark_results,
)
from reasoned_icrl.experiments.summary_memory.configs import (
    load_retired_summary_memory_study,
    load_summary_memory_study,
)
from reasoned_icrl.utils import repository_root
from tests.environments.test_dark_key_to_door import (
    DOWN,
    RIGHT,
    STAY,
    native_environment,
    pin_layout,
)

ROOT = repository_root()


def _revised(task_count: int = 1) -> tuple[Any, Any, Any]:
    study = load_summary_memory_study()
    contract = study.contract("dark_key_to_door")
    contract = replace(
        contract,
        evaluation=replace(
            contract.evaluation,
            splits={
                name: EvaluationSplit(split.source, task_count, split.offset)
                for name, split in contract.evaluation.splits.items()
            },
        ),
    )
    config = experiment_config(
        contract, study, condition="full_context", seed=42, repository=ROOT
    )
    return study, contract, config


def _drive(actions: list[int], task: int) -> AttemptTaskResult:
    """Roll one development task under a repeated scripted action cycle."""
    env = native_environment()
    try:
        env.reset(options={"task_index": task})
        if actions != [STAY]:
            pin_layout(env, start=(0, 0), key=(0, 2), goal=(2, 2))
        done = False
        index = 0
        while not done:
            _, _, terminated, truncated, info = env.step(actions[index % len(actions)])
            index = 0 if info["attempt_done"] or info["reset_only"] else index + 1
            done = terminated or truncated
        return AttemptTaskResult(
            task_id=task,
            rollout_seed=0,
            attempts=env.completed_attempts,
            partial=env.partial_attempt,
            task_return=env.task_return,
            decisions=500,
        )
    finally:
        env.close()


def test_the_revised_contract_retains_every_attempt_and_counts_doors() -> None:
    _, contract, _ = _revised()
    assert contract.evaluation.retention == "complete"
    assert contract.evaluation.primary_metric == "doors_completed"
    assert contract.evaluation.panel_rules == ("endpoint", "selected")
    retired = load_retired_summary_memory_study().contract("dark_key_to_door")
    assert retired.evaluation.retention == "scored-band"
    assert retired.evaluation.primary_metric == "door_success_first8"
    assert retired.evaluation.panel_rules == ("selected", "final-epoch")


def test_contracts_refuse_a_count_metric_without_complete_retention(
    tmp_path: Path,
) -> None:
    source = ROOT / "configs/environments/8m/dark_key_to_door.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["evaluation"]["retention"] = "scored-band"
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="retain complete events"):
        load_contract(path)
    raw["evaluation"]["retention"] = "everything"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="Unknown event retention"):
        load_contract(path)
    # Concentration admits complete retention since R7 (terminal episode plus
    # every public flip); an episode task without a flip record still refuses.
    maze = yaml.safe_load((ROOT / "configs/environments/mazerunner.yaml").read_text())
    maze["evaluation"]["retention"] = "complete"
    path.write_text(yaml.safe_dump(maze, sort_keys=False))
    with pytest.raises(ContractError, match="complete retention applies"):
        load_contract(path)


def test_complete_events_keep_successes_after_attempt_eight_and_date_keys() -> None:
    """A four-step route repeats a hundred times inside the 500-call task."""
    _, contract, config = _revised()
    task = DEVELOPMENT_TASKS[0]
    result = _drive([RIGHT, RIGHT, DOWN, DOWN], task)
    assert len(result.attempts) == 100 and result.partial is None
    rows = events(
        contract,
        config,
        [result],
        checkpoint="policy_epoch_1000",
        split="development",
        history="retained",
        checkpoint_rule="endpoint",
    )
    assert len(rows) == 100
    assert all(row.complete is True and row.numerator == 1 for row in rows)
    assert [row.event_index for row in rows] == list(range(1, 101))
    for row in rows:
        assert row.start_step is not None and row.end_step is not None
        assert row.end_step - row.start_step + 1 == row.step == 4
        assert row.key_step == row.start_step + 1  # the second move finds the key
        assert row.native_return == 2.0 and row.checkpoint_rule == "endpoint"
    assert primary_value(rows, "doors_completed") == 100.0
    assert primary_value(rows, "door_success_first8") == 1.0
    assert len(metric_events(rows, "door_success_first8")) == 8
    assert len(metric_events(rows, "doors_completed")) == 100
    run = BenchmarkRun(
        contract.protocol,
        "dark_key_to_door",
        "full_context",
        42,
        "policy_epoch_1000",
        "development",
        "retained",
        "completed",
        checkpoint_rule="endpoint",
        retention="complete",
        metric="doors_completed",
        charged_calls=500,
        physical_actions=400,
        reset_only_steps=100,
    )
    validate_benchmark_results([contract], [run], rows)
    assert evaluation_directory("development", "retained", "endpoint") == (
        "development-retained-endpoint"
    )


def test_complete_events_mark_the_partial_attempt_and_score_no_success() -> None:
    """Standing still times out nine attempts and leaves a 41-step partial one."""
    _, contract, config = _revised()
    result = _drive([STAY], DEVELOPMENT_TASKS[0])
    assert len(result.attempts) == 9 and result.partial is not None
    assert len(retained_attempts(result)) == 10
    rows = events(
        contract,
        config,
        [result],
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    assert len(rows) == 10
    assert [row.complete for row in rows] == [True] * 9 + [False]
    assert all(row.numerator == 0 and row.key_step is None for row in rows)
    assert rows[-1].step == 41 and rows[-1].end_step == 500
    assert primary_value(rows, "doors_completed") == 0.0
    assert primary_value(rows, "door_success_first8") == 0.0
    run = BenchmarkRun(
        contract.protocol,
        "dark_key_to_door",
        "full_context",
        42,
        "checkpoint.pt",
        "development",
        "retained",
        "completed",
        retention="complete",
        metric="doors_completed",
    )
    validate_benchmark_results([contract], [run], rows)

    def refused(
        rows_: list[BenchmarkEvent], match: str, run_: BenchmarkRun = run
    ) -> None:
        with pytest.raises(ResultValidationError, match=match):
            validate_benchmark_results([contract], [run_], rows_)

    refused([*rows[:-1], replace(rows[-1], complete=None)], "says whether it finished")
    refused(
        [*rows[:-1], replace(rows[-1], numerator=1, native_return=2.0)],
        "partial attempt cannot",
    )
    refused([replace(rows[0], complete=False), *rows[1:]], "only the last attempt may")
    refused([replace(rows[0], key_step=3), *rows[1:]], "key acquisition")
    refused(rows, "disagrees with the contract", replace(run, retention="scored-band"))
    refused(rows, "not one the environment names", replace(run, metric="pair_fraction"))
    refused(
        rows,
        "charged_calls must equal",
        replace(run, charged_calls=10, physical_actions=5, reset_only_steps=4),
    )
    with pytest.raises(ResultValidationError, match=r"span the outer task"):
        validate_benchmark_results([contract], [run], rows[:8])
    with pytest.raises(ResultValidationError, match=r"finished 7 attempts"):
        validate_benchmark_results(
            [contract], [run], [*rows[:7], replace(rows[-1], event_index=8)]
        )


def test_cell_values_separate_counts_from_scored_band_rates() -> None:
    _, contract, config = _revised()
    result = _drive([RIGHT, RIGHT, DOWN, DOWN], DEVELOPMENT_TASKS[0])
    rows = events(
        contract,
        config,
        [result],
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    # Fail attempts 3 and 20, and make the last one partial.
    edited = [
        replace(row, numerator=0, native_return=1.0)
        if row.event_index in (3, 20)
        else row
        for row in rows
    ]
    edited[-1] = replace(edited[-1], complete=False, numerator=0, native_return=0.0)
    edited[-1] = replace(edited[-1], key_step=None)
    cell = (42, DEVELOPMENT_TASKS[0], 0)
    assert cell_values(edited, "doors_completed") == {cell: 97.0}
    assert cell_values(edited, "door_success_first8") == {cell: 7 / 8}
    assert cell_values(edited)[cell] == pytest.approx(97 / 100)  # legacy mean rate
    assert primary_value(edited, "doors_completed") == 97.0


def test_the_endpoint_rule_refuses_a_run_that_stopped_early(tmp_path: Path) -> None:
    run = tmp_path / "seed-42"
    weights = run / "ckpts" / "policy_weights"
    weights.mkdir(parents=True)
    (weights / "policy_epoch_950.pt").write_bytes(b"")
    with pytest.raises(ContractError, match="incomplete"):
        endpoint_training_epoch(run, epochs=1000)
    # AMAGO numbers epochs from zero: the endpoint weights are label 999.
    (weights / "policy_epoch_999.pt").write_bytes(b"")
    with pytest.raises(ContractError, match=r"missing checkpoint\.pt, metrics\.json"):
        endpoint_training_epoch(run, epochs=1000)
    (run / "checkpoint.pt").write_bytes(b"")
    (run / "metrics.json").write_text("{}")
    assert endpoint_training_epoch(run, epochs=1000) == 999
    assert endpoint_label(1000) == 999
    assert checkpoint_labels(1000, 50, start_learning=1) == (*range(50, 1000, 50), 999)
    assert checkpoint_labels(1000, 50) == (*range(0, 1000, 50), 999)
    assert checkpoint_labels(2, 1, start_learning=1) == (1,)  # the smoke profile
    assert collected_at_label(999, timesteps_per_epoch=500, actors=16) == 8_000_000
    assert collected_at_label(950, timesteps_per_epoch=500, actors=16) == 7_608_000
    assert collected_at_label(0, timesteps_per_epoch=500, actors=16) == 8_000


def test_the_sweep_is_a_hamiltonian_cycle_from_any_start() -> None:
    size = 8
    for start in ((0, 0), (3, 5), (7, 7), (1, 0), (0, 7)):
        position = start
        seen = {position}
        for _ in range(size * size - 1):
            move = KEY_TO_DOOR_MOVES[sweep_policy_action(position, size)]
            position = (position[0] + move[0], position[1] + move[1])
            assert 0 <= position[0] < size and 0 <= position[1] < size
            seen.add(position)
        assert len(seen) == size * size
        move = KEY_TO_DOOR_MOVES[sweep_policy_action(position, size)]
        assert (position[0] + move[0], position[1] + move[1]) == start
    with pytest.raises(ContractError, match="even room side"):
        sweep_policy_action((0, 0), 1)
    with pytest.raises(ContractError, match="even room side"):
        sweep_policy_action((8, 8), 9)


def test_reference_policies_roll_the_declared_roster_evaluation_only() -> None:
    env = native_environment()
    try:
        roster = tuple(DEVELOPMENT_TASKS[:3])
        random = random_policy_door_counts(
            env, task_ids=roster, generator_seeds=(0, 1, 2)
        )
        assert random.generator_seeds == (0, 1, 2)
        assert set(random.doors_completed) == set(roster)
        assert all(0.0 <= v <= 1.0 for v in random.door_success_first8.values())
        assert random.charged_calls == 3 * 3 * 500
        assert 0 < random.physical_actions < random.charged_calls
        sweep = sweep_policy_door_counts(env, task_ids=roster)
        again = sweep_policy_door_counts(env, task_ids=roster)
        assert sweep == again  # deterministic in the public position alone
        assert sweep.generator_seeds == ()
        assert sweep.charged_calls == 3 * 500
        assert all(v >= 0.0 for v in sweep.doors_completed.values())
        assert random.mean_doors_completed >= 0.0 and sweep.mean_doors_completed >= 0.0
    finally:
        env.close()
