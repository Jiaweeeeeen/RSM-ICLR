"""The extended-horizon evaluation adapter (EXPERIMENTS section 5)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import evaluation_directory
from reasoned_icrl.experiments.horizon import (
    check_horizon_prefix,
    extended_horizon,
    window_door_counts,
)
from reasoned_icrl.experiments.records import BenchmarkEvent
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.experiments import resolve

ROOT = Path(__file__).resolve().parents[2]


def _key_to_door(tmp_path: Path):
    study = load_summary_memory_study(None)  # the 8M study fits every paper cell
    return resolve(
        study,
        benchmark="dark_key_to_door",
        condition="full_gru",
        seed=42,
        device="cpu",
        output_root=tmp_path,
    )


def test_the_extended_horizon_changes_only_the_budget_and_the_context(
    tmp_path: Path,
) -> None:
    contract, config = _key_to_door(tmp_path)
    assert contract.environment.outer_length == 500
    longer, extended = extended_horizon(contract, config, 2000)
    assert longer.environment.outer_length == 2000
    assert longer.environment.meta_horizon == 2000
    assert extended.environment == longer.environment
    assert extended.training.max_sequence_length == 2000
    assert extended.training.trajectory_length == 2001
    assert (
        replace(extended.training, max_sequence_length=500, trajectory_length=501)
        == config.training
    )
    assert extended.model == config.model
    assert extended.condition == config.condition and extended.seed == config.seed
    assert longer.evaluation == contract.evaluation
    assert extended.run_directory == config.run_directory
    same, same_config = extended_horizon(contract, config, 500)
    assert same.environment == contract.environment
    assert same_config == config


def test_the_adapter_refuses_shorter_budgets_and_other_tasks(tmp_path: Path) -> None:
    contract, config = _key_to_door(tmp_path)
    with pytest.raises(ContractError, match="below the trained budget"):
        extended_horizon(contract, config, 499)
    with pytest.raises(ContractError, match="positive integer"):
        extended_horizon(contract, config, 0)
    # Concentration keeps no task past its board: no adapter.
    other = replace(
        contract, environment=replace(contract.environment, name="concentration")
    )
    with pytest.raises(ContractError, match="no extended-horizon adapter"):
        extended_horizon(other, replace(config, environment=other.environment), 2000)


def test_horizon_panels_get_their_own_directory() -> None:
    assert evaluation_directory("confirmation", "retained", "endpoint") == (
        "confirmation-retained-endpoint"
    )
    assert evaluation_directory("confirmation", "retained", "endpoint", 2000) == (
        "confirmation-retained-endpoint-h2000"
    )
    assert evaluation_directory("development", "retained") == "development-retained"
    with pytest.raises(ContractError, match="positive integer"):
        evaluation_directory("confirmation", "retained", "endpoint", 0)


def _attempt(
    task: int,
    index: int,
    start: int,
    end: int,
    *,
    success: bool = True,
    complete: bool = True,
    outer_length: int = 500,
) -> BenchmarkEvent:
    return BenchmarkEvent(
        protocol="native-keydoor-fixed500-first8",
        benchmark="dark_key_to_door",
        condition="raw_summary",
        training_seed=42,
        checkpoint="policy_epoch_999",
        split="confirmation",
        history="retained",
        task_id=task,
        cluster_id=task,
        rollout_seed=0,
        kind="attempt",
        event_index=index,
        step=end - start + 1,
        numerator=int(success),
        denominator=1,
        native_return=2.0 if success else 0.0,
        start_step=start,
        end_step=end,
        checkpoint_rule="endpoint",
        complete=complete,
        key_step=start + 3 if success else None,
        outer_length=outer_length,
    )


def test_the_prefix_check_accepts_a_reproduced_prefix_and_rejects_a_changed_one() -> (
    None
):
    base = [
        _attempt(1, 1, 1, 40),
        _attempt(1, 2, 41, 300, success=False),
        _attempt(1, 3, 301, 500, success=False, complete=False),
        _attempt(2, 1, 1, 500, success=False, complete=False),
    ]
    extended = [
        _attempt(1, 1, 1, 40, outer_length=1000),
        _attempt(1, 2, 41, 300, success=False, outer_length=1000),
        _attempt(1, 3, 301, 620, outer_length=1000),
        _attempt(1, 4, 621, 1000, success=False, complete=False, outer_length=1000),
        _attempt(2, 1, 1, 700, outer_length=1000),
        _attempt(2, 2, 701, 1000, success=False, complete=False, outer_length=1000),
    ]
    counts = check_horizon_prefix(base, extended, native=500)
    assert counts == {"compared": 2, "partial": 2, "units": 2}
    changed = list(extended)
    changed[0] = _attempt(1, 1, 1, 41, outer_length=1000)
    with pytest.raises(ContractError, match="differs inside the first 500"):
        check_horizon_prefix(base, changed, native=500)
    with pytest.raises(ContractError, match="missing from the extended panel"):
        check_horizon_prefix(base, [*extended[:2], *extended[4:]], native=500)
    extra = [*extended, _attempt(3, 1, 1, 20, outer_length=1000)]
    with pytest.raises(ContractError, match="different tasks"):
        check_horizon_prefix(base, extra, native=500)


def test_window_door_counts_follow_the_completing_call() -> None:
    events = [
        _attempt(1, 1, 1, 40, outer_length=1000),
        _attempt(1, 2, 41, 300, success=False, outer_length=1000),
        _attempt(1, 3, 301, 620, outer_length=1000),
        _attempt(1, 4, 621, 1000, success=False, complete=False, outer_length=1000),
        _attempt(2, 1, 1, 500, outer_length=1000),
        _attempt(2, 2, 501, 1000, success=True, complete=False, outer_length=1000),
    ]
    assert window_door_counts(events, outer_length=1000) == {
        (1, 0): (1, 1),
        (2, 0): (1, 0),
    }
    with pytest.raises(ContractError, match="whole number of windows"):
        window_door_counts(events, outer_length=1234)


def test_event_spans_are_validated_against_the_run_outer_budget() -> None:
    """The acceptance check reads an extended panel with the native contract:
    the span rule must use the run's declared budget, not the contract's."""
    from dataclasses import replace as dc_replace

    from reasoned_icrl.experiments.records import (
        ResultValidationError,
        validate_benchmark_results,
    )
    from tests.experiments.fixtures import fixture_results

    study, runs, events = fixture_results()
    contract = next(c for c in study.contracts if c.name == "dark_key_to_door")
    assert contract.environment.outer_length == 500
    run = next(r for r in runs if r.benchmark == "dark_key_to_door")
    rows = [e for e in events if e.run_identity == run.identity]
    late = [
        dc_replace(rows[0], start_step=651, end_step=700, step=50, outer_length=1000),
        *rows[1:],
    ]
    extended = dc_replace(run, outer_length=1000)
    validate_benchmark_results([contract], [extended], late)
    with pytest.raises(ResultValidationError, match="outer task length"):
        validate_benchmark_results(
            [contract], [run], [dc_replace(late[0], outer_length=None), *rows[1:]]
        )
    with pytest.raises(ResultValidationError, match="disagrees with its run"):
        validate_benchmark_results([contract], [run], late)
    with pytest.raises(ResultValidationError, match="positive integer"):
        validate_benchmark_results([contract], [dc_replace(run, outer_length=0)], [])


def test_window_matrix_gives_every_roster_unit_a_row() -> None:
    from reasoned_icrl.experiments.horizon import window_matrix

    events = [
        _attempt(1, 1, 1, 40, outer_length=1000),
        _attempt(1, 2, 41, 300, success=False, outer_length=1000),
        _attempt(1, 3, 301, 620, outer_length=1000),
        _attempt(1, 4, 621, 1000, success=False, complete=False, outer_length=1000),
        _attempt(2, 1, 1, 1000, success=False, complete=False, outer_length=1000),
    ]
    matrix = window_matrix(events, units=[(2, 0), (1, 0), (3, 0)], outer_length=1000)
    assert matrix.tolist() == [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    with pytest.raises(ContractError, match="outside the roster"):
        window_matrix(events, units=[(2, 0)], outer_length=1000)
    with pytest.raises(ContractError, match="distinct"):
        window_matrix(events, units=[(1, 0), (1, 0), (2, 0)], outer_length=1000)


def test_layout_matrix_bins_doors_by_the_attempts_recorded_layout() -> None:
    from dataclasses import replace as dc_replace

    from reasoned_icrl.experiments.horizon import layout_matrix

    # The old layout persists to the first attempt boundary at or after the
    # period, so the door completed at call 520 belongs to layout 0 although
    # it lies in the second 500-call window.
    events = [
        dc_replace(_attempt(1, 1, 1, 480, outer_length=1000), layout_index=0),
        dc_replace(_attempt(1, 2, 481, 520, outer_length=1000), layout_index=0),
        dc_replace(_attempt(1, 3, 521, 900, outer_length=1000), layout_index=1),
        dc_replace(
            _attempt(1, 4, 901, 1000, success=False, complete=False, outer_length=1000),
            layout_index=1,
        ),
        dc_replace(
            _attempt(2, 1, 1, 1000, success=False, complete=False, outer_length=1000),
            layout_index=0,
        ),
    ]
    matrix = layout_matrix(events, units=[(2, 0), (1, 0)], layouts=2)
    assert matrix.tolist() == [[0.0, 0.0], [2.0, 1.0]]
    with pytest.raises(ContractError, match="names no layout"):
        layout_matrix(
            [dc_replace(events[0], layout_index=None)], units=[(1, 0)], layouts=2
        )
    with pytest.raises(ContractError, match="outside the 1 layouts"):
        layout_matrix(events, units=[(2, 0), (1, 0)], layouts=1)
    with pytest.raises(ContractError, match="outside the roster"):
        layout_matrix(events, units=[(2, 0)], layouts=2)


def test_layout_change_panels_are_named_and_validated() -> None:
    from dataclasses import replace as dc_replace

    from reasoned_icrl.experiments.records import (
        ResultValidationError,
        validate_benchmark_results,
    )
    from tests.experiments.fixtures import fixture_results

    assert evaluation_directory(
        "confirmation", "retained", "endpoint", 4000, layout_period=500
    ) == ("confirmation-retained-endpoint-h4000-relayout500")
    with pytest.raises(ContractError, match="positive number of calls"):
        evaluation_directory(
            "confirmation", "retained", "endpoint", 4000, layout_period=0
        )
    study, runs, events = fixture_results()
    contract = next(c for c in study.contracts if c.name == "dark_key_to_door")
    run = next(r for r in runs if r.benchmark == "dark_key_to_door")
    rows = [e for e in events if e.run_identity == run.identity]
    changing = dc_replace(run, layout_period=500)
    with pytest.raises(ResultValidationError, match="names its layout"):
        validate_benchmark_results([contract], [changing], rows)
    labelled = [dc_replace(e, layout_index=0) for e in rows]
    validate_benchmark_results([contract], [changing], labelled)
    with pytest.raises(ResultValidationError, match="layout-change panel carry"):
        validate_benchmark_results([contract], [run], labelled)
    with pytest.raises(ResultValidationError, match="int >= 0"):
        validate_benchmark_results(
            [contract],
            [changing],
            [dc_replace(labelled[0], layout_index=-1), *labelled[1:]],
        )


def _query(task: int, step: int, *, true_count: int, answer: int) -> BenchmarkEvent:
    correct = answer == true_count
    return BenchmarkEvent(
        protocol="count-recall-medium",
        benchmark="count_recall",
        condition="full_gru",
        training_seed=42,
        checkpoint="policy_epoch_999",
        split="confirmation",
        history="retained",
        task_id=task,
        cluster_id=task,
        rollout_seed=0,
        kind="query",
        event_index=step,
        step=step,
        numerator=int(correct),
        denominator=1,
        native_return=(1.0 if correct else -1.0) / 103,
        true_count=true_count,
        answer=answer,
        checkpoint_rule="endpoint",
        outer_length=103,
    )


def _flip(event: BenchmarkEvent, *keys: tuple[int, int]) -> BenchmarkEvent:
    if (event.task_id, event.step) not in keys:
        return event
    return replace(event, answer=9, numerator=0, native_return=-1 / 103)


def test_the_stream_check_bounds_answer_flips_and_rejects_changed_queries() -> None:
    from reasoned_icrl.experiments.horizon import (
        STREAM_FLIP_TOLERANCE,
        check_stream_prefix,
    )

    base = [
        _query(t, s, true_count=s % 5, answer=s % 5)
        for t in (1, 2)
        for s in range(1, 104)
    ]
    same = [replace(e, outer_length=207) for e in base]
    assert check_stream_prefix(base, same, native=104) == {
        "compared": 206,
        "units": 2,
        "decision_flips": 0,
    }
    # One different answer to the same query (a near-tie flipped between two GPU
    # kernels) is counted, not refused: the paper's read stays the adapter's panel.
    counts = check_stream_prefix(base, [_flip(e, (2, 46)) for e in same], native=104)
    assert counts["decision_flips"] == 1 and counts["compared"] == 206
    assert STREAM_FLIP_TOLERANCE == 0.001
    # Two flips exceed the bound on 206 decisions (max(1, 0) allowed).
    twice = [_flip(e, (2, 46), (1, 7)) for e in same]
    with pytest.raises(ContractError, match="2 of 206 decisions"):
        check_stream_prefix(base, twice, native=104)
    # A different query or true count at any decision is an adapter defect, not a flip.
    changed = [
        replace(e, true_count=9) if (e.task_id, e.step) == (2, 46) else e for e in same
    ]
    with pytest.raises(ContractError, match="differs inside the native stream"):
        check_stream_prefix(base, changed, native=104)


# ----------------------------------------------------------------------
# MazeRunner: the larger-maze axis
# ----------------------------------------------------------------------


def _mazerunner(tmp_path: Path):
    study = load_summary_memory_study(Path("configs/mazerunner_8m.yaml"))
    return resolve(
        study,
        benchmark="mazerunner",
        condition="full_gru",
        seed=42,
        device="cpu",
        output_root=tmp_path,
    )


def test_the_maze_adapter_enlarges_the_maze_under_the_area_scaled_timer(
    tmp_path: Path,
) -> None:
    from reasoned_icrl.environments.mazerunner import MazeRunnerEnv
    from reasoned_icrl.experiments.environments import build_environment
    from reasoned_icrl.experiments.horizon import (
        MAZE_SIZES,
        horizon_kind,
        maze_size_ladder,
        maze_timer,
        native_horizon,
    )

    contract, config = _mazerunner(tmp_path)
    assert horizon_kind(contract) == "maze" and native_horizon(contract) == 15
    assert maze_size_ladder(15) == MAZE_SIZES == (15, 17, 19, 21, 25)
    assert maze_size_ladder(11) == (11, 13, 15, 17, 21)
    assert [
        maze_timer(s, trained_size=11, trained_horizon=250)
        for s in (11, 13, 15, 17, 21)
    ] == [
        250,
        349,
        465,
        597,
        911,
    ]
    with pytest.raises(ContractError, match="odd maze"):
        maze_size_ladder(12)
    assert contract.environment.horizon == 500
    # A constant budget per cell of the maze: 500 * (S / 15)^2, whole steps.
    assert [
        maze_timer(s, trained_size=15, trained_horizon=500) for s in MAZE_SIZES
    ] == [
        500,
        642,
        802,
        980,
        1389,
    ]
    longer, extended = extended_horizon(contract, config, 25)
    environment = longer.environment
    assert (environment.size, environment.horizon, environment.protocol_size) == (
        25,
        1389,
        15,
    )
    assert environment.outer_length == 1389
    assert environment.benchmark == contract.environment.benchmark
    assert extended.environment == environment
    assert extended.training.max_sequence_length == 1389
    assert extended.training.trajectory_length == 1390
    assert extended.model == config.model
    assert extended.run_directory == config.run_directory
    assert longer.evaluation == contract.evaluation
    built = build_environment(
        extended.as_runtime_mapping(), split="confirmation", seed=0
    )
    try:
        assert isinstance(built, MazeRunnerEnv)
        assert built.size == 25 and built.horizon == 1389
        assert built.protocol == contract.environment.benchmark
        assert built.protocol_size == 15
    finally:
        built.close()
    # The trained size is the plain contract: the acceptance panel's identity.
    same, same_config = extended_horizon(contract, config, 15)
    assert same.environment == contract.environment and same_config == config
    with pytest.raises(ContractError, match="odd"):
        extended_horizon(contract, config, 16)
    with pytest.raises(ContractError, match="below the trained budget"):
        extended_horizon(contract, config, 13)
    with pytest.raises(ContractError, match="starts from the trained contract"):
        extended_horizon(longer, extended, 31)


def _episode(task: int, completed: int, goals: int, decisions: int) -> BenchmarkEvent:
    return BenchmarkEvent(
        protocol="mazerunner-15-randomized-actions",
        benchmark="mazerunner",
        condition="raw_summary_residual",
        training_seed=42,
        checkpoint="policy_epoch_999",
        split="confirmation",
        history="retained",
        task_id=task,
        cluster_id=task,
        rollout_seed=0,
        kind="episode",
        event_index=1,
        step=decisions,
        numerator=completed,
        denominator=goals,
        native_return=float(completed),
        start_step=1,
        end_step=decisions,
        checkpoint_rule="endpoint",
        outer_length=1389,
    )


def test_the_goal_fraction_estimand_averages_units_and_pools_the_rate() -> None:
    from reasoned_icrl.experiments.horizon import maze_goal_fraction

    events = [_episode(1, 3, 3, 120), _episode(2, 1, 3, 500), _episode(3, 0, 3, 500)]
    got = maze_goal_fraction(events)
    assert got["goal_fraction"] == pytest.approx((1.0 + 1.0 / 3.0 + 0.0) / 3.0)
    assert got["full_sequence"] == pytest.approx(1.0 / 3.0)
    assert got["mean_decisions"] == pytest.approx(1120.0 / 3.0)
    assert got["goals_per_500_steps"] == pytest.approx(500.0 * 4.0 / 1120.0)
    assert got["units"] == 3.0
    with pytest.raises(ContractError, match="episode events"):
        maze_goal_fraction([e for e in events if e.kind != "episode"])


def test_the_laps_adapter_replays_the_trained_task_until_the_budget(
    tmp_path: Path,
) -> None:
    from reasoned_icrl.environments.mazerunner import MazeRunnerEnv
    from reasoned_icrl.experiments.environments import build_environment
    from reasoned_icrl.experiments.evaluation import continued_history_modes
    from reasoned_icrl.experiments.horizon import LAP_BUDGETS, continued_laps

    contract, config = _mazerunner(tmp_path)
    assert LAP_BUDGETS == (1000, 2000, 4000)
    longer, extended = continued_laps(contract, config, 4000)
    environment = longer.environment
    assert (
        environment.meta_horizon,
        environment.size,
        environment.horizon,
        environment.protocol_size,
        environment.outer_length,
    ) == (4000, 15, 500, None, 4000)
    assert environment.benchmark == contract.environment.benchmark
    assert longer.event_kind == "attempt" and contract.event_kind == "episode"
    assert longer.evaluation.retention == "complete"
    assert contract.evaluation.retention == "scored-band"
    assert longer.evaluation.splits == contract.evaluation.splits
    assert extended.environment == environment
    assert extended.training.max_sequence_length == 4000
    assert extended.training.trajectory_length == 4001
    assert extended.model == config.model
    assert extended.run_directory == config.run_directory
    # The lap-cleared companion exists on the laps task only.
    assert continued_history_modes(environment) == (
        "retained",
        "goal-cleared",
        "summary-cleared",
        "attempt-cleared",
    )
    assert continued_history_modes(contract.environment) == (
        "retained",
        "goal-cleared",
        "summary-cleared",
    )
    built = build_environment(
        extended.as_runtime_mapping(), split="confirmation", seed=0
    )
    try:
        assert isinstance(built, MazeRunnerEnv)
        assert built.meta_horizon == 4000 and built.size == 15 and built.horizon == 500
        assert built.protocol == contract.environment.benchmark
    finally:
        built.close()
    with pytest.raises(ContractError, match="does not exceed the trained episode"):
        continued_laps(contract, config, 500)
    with pytest.raises(ContractError, match="starts from the trained one-episode"):
        continued_laps(longer, extended, 8000)
    larger, larger_config = extended_horizon(contract, config, 17)
    with pytest.raises(ContractError, match="starts from the trained one-episode"):
        continued_laps(larger, larger_config, 4000)
    with pytest.raises(ContractError, match="starts from the trained contract"):
        extended_horizon(longer, extended, 17)
    keydoor, keydoor_config = _key_to_door(tmp_path)
    with pytest.raises(ContractError, match="Only MazeRunner"):
        continued_laps(keydoor, keydoor_config, 4000)
    # Neither evaluation-only contract ever trains.
    from reasoned_icrl.runtime.training import train_experiment

    with pytest.raises(ContractError, match="evaluation-only"):
        train_experiment(extended)
    with pytest.raises(ContractError, match="evaluation-only"):
        train_experiment(larger_config)
    assert (
        evaluation_directory("confirmation", "retained", "endpoint", laps=4000)
        == "confirmation-retained-endpoint-calls4000"
    )
    assert (
        evaluation_directory("confirmation", "attempt-cleared", "endpoint", laps=2000)
        == "confirmation-attempt-cleared-endpoint-calls2000"
    )
    with pytest.raises(ContractError, match="not a horizon or streams"):
        evaluation_directory(
            "confirmation", "retained", "endpoint", horizon=4000, laps=4000
        )
    with pytest.raises(ContractError, match="names its budget"):
        evaluation_directory("confirmation", "retained", "endpoint", laps=0)


def _lap(
    task: int,
    index: int,
    start: int,
    end: int,
    goal_steps: tuple[int, ...],
    *,
    complete: bool = True,
    outer_length: int = 1000,
) -> BenchmarkEvent:
    return BenchmarkEvent(
        protocol="mazerunner-15-randomized-actions",
        benchmark="mazerunner",
        condition="raw_summary_residual",
        training_seed=42,
        checkpoint="policy_epoch_999",
        split="confirmation",
        history="retained",
        task_id=task,
        cluster_id=task,
        rollout_seed=0,
        kind="attempt",
        event_index=index,
        step=end - start + 1,
        numerator=len(goal_steps),
        denominator=3,
        native_return=float(len(goal_steps)),
        start_step=start,
        end_step=end,
        complete=complete,
        goal_steps=goal_steps,
        checkpoint_rule="endpoint",
        outer_length=outer_length,
    )


def test_lap_goals_are_binned_by_their_calls_and_the_first_lap_is_checked() -> None:
    from reasoned_icrl.experiments.horizon import (
        check_laps_prefix,
        laps_per_unit,
        window_goal_counts,
        window_goal_matrix,
    )

    base = [_episode(1, 3, 3, 120), _episode(2, 1, 3, 500)]
    laps = [
        # Map 1: lap 1 = the plain episode, lap 2 straddles the window edge,
        # the partial lap 3 still counts its goal.
        _lap(1, 1, 1, 120, (40, 80, 120)),
        _lap(1, 2, 122, 620, (300, 501, 620)),
        _lap(1, 3, 622, 1000, (900,), complete=False),
        # Map 2: the timer ends lap 1 with one goal; lap 2 is cut at the budget.
        _lap(2, 1, 1, 500, (250,)),
        _lap(2, 2, 502, 1000, (), complete=False),
    ]
    counts = window_goal_counts(laps, outer_length=1000)
    assert counts == {(1, 0): (4, 3), (2, 0): (1, 0)}
    matrix = window_goal_matrix(laps, units=[(2, 0), (1, 0)], outer_length=1000)
    assert matrix.tolist() == [[1.0, 0.0], [4.0, 3.0]]
    assert laps_per_unit(laps) == {(1, 0): 2, (2, 0): 1}
    assert check_laps_prefix(base, laps) == {"compared": 2, "units": 2}
    with pytest.raises(ContractError, match="whole number of windows"):
        window_goal_counts(laps, outer_length=1200)
    with pytest.raises(ContractError, match="outside the roster"):
        window_goal_matrix(laps, units=[(1, 0)], outer_length=1000)
    changed = [_lap(1, 1, 1, 120, (40, 80)), *laps[1:]]
    with pytest.raises(ContractError, match="differs from the plain episode"):
        check_laps_prefix(base, changed)
    with pytest.raises(ContractError, match="different maps"):
        check_laps_prefix(base, laps[3:])
    with pytest.raises(ContractError, match="needs the plain episode panel"):
        check_laps_prefix([], laps)
    # A lap record is validated with its goal steps when it is read back.
    from reasoned_icrl.experiments.records import ResultValidationError

    with pytest.raises(ContractError, match="lie inside it, in order"):
        from reasoned_icrl.environments.mazerunner import MazeLap

        MazeLap(
            index=0,
            first_step=1,
            last_step=10,
            steps=10,
            success=False,
            native_return=2.0,
            complete=True,
            goals=2,
            goal_steps=(5, 3),
        )
    assert ResultValidationError is not None
