"""Bounded train/evaluate lifecycles for the summary-memory study.

Every condition of the study roster runs through the shared trainer and the
shared evaluator on CPU with the smoke profile, as a library call and, for the
reference cell, through the two scripts. Passing establishes execution
integrity only, never learning.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from reasoned_icrl.experiments.artifacts import latest_training_epoch
from reasoned_icrl.experiments.benchmarks import Study, saved_config
from reasoned_icrl.experiments.config import architecture_id
from reasoned_icrl.experiments.contracts import ALL_CONDITIONS, ContractError
from reasoned_icrl.experiments.evaluation import (
    RESULTS_FILE,
    history_modes,
    task_intervention,
)
from reasoned_icrl.experiments.records import BenchmarkEvent
from reasoned_icrl.experiments.summary_memory import experiments as summary_memory
from reasoned_icrl.experiments.summary_memory.configs import (
    load_retired_summary_memory_study,
    load_summary_memory_study,
    reference_condition,
)
from reasoned_icrl.runtime.amago import BOUND_CARRIERS
from reasoned_icrl.runtime.rollout import evaluate as evaluate_checkpoint
from reasoned_icrl.runtime.training import (
    close_experiment,
    load_experiment,
    load_selected_checkpoint,
)

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = "dark_key_to_door"
_STUDY = load_summary_memory_study()
CONDITIONS = _STUDY.cells(_STUDY.contract(BENCHMARK))
"""The cells tier 0 declares (the seven revised names): every bound cell runs
the Key-to-Door smoke lifecycle; a cell whose carrier is not bound would be
skipped, not faked (every cell binds since R3). The union roster also carries
the legacy ordinary-writer pair added to tier 2, which is
outside tier 0's scope and has its own Medium lifecycle below. The legacy-named
tests further down read the retired 4M roster explicitly."""
TIER2_BACKBONE_CELLS = ("raw_summary", "raw_segment")
"""Tier 2's supplementary ordinary-writer pair (`amago-summary-v1`, route
`causal_prefix`), run on the active Medium contract."""
COST_FIELDS = (
    "persistent_state_bytes",
    "decision_latency_seconds",
    "decision_latency_p95_seconds",
    "boundary_latency_seconds",
    "boundary_latency_p95_seconds",
    "boundary_latency_max_seconds",
)
BOUNDARY_FIELDS = COST_FIELDS[3:]


@pytest.mark.slow
@pytest.mark.parametrize("condition", TIER2_BACKBONE_CELLS)
def test_cpu_lifecycle_of_the_ordinary_writer_pair_on_medium(
    condition: str, tmp_path: Path
) -> None:
    """The legacy ordinary-attention writer pair trains, checkpoints and
    evaluates under every declared history mode on the active
    CountRecallMedium contract, as a tier-2 supplementary cell."""
    study = load_summary_memory_study()
    benchmark = "count_recall"
    contract = study.contract(benchmark)
    assert study.tier(benchmark).group_of(condition) == "supplementary"
    assert study.tier(BENCHMARK).group_of(condition) in {"primary", "supplementary"}
    run = summary_memory.train(
        study,
        benchmark=benchmark,
        condition=condition,
        seed=0,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    assert isinstance(run, Path) and (run / "checkpoint.pt").is_file()
    assert run.parent.parent.name == contract.protocol
    config = saved_config(
        summary_memory.resolve(
            study,
            benchmark=benchmark,
            condition=condition,
            seed=0,
            output_root=tmp_path,
        )[1]
    )
    assert config.model.architecture_id == architecture_id(condition)
    assert config.model.dat is None
    assert config.model.summary is not None
    assert (
        config.model.summary.segment_length,
        config.model.summary.memory_tokens,
    ) == (32, 4)
    assert config.model.summary.relational_sources == "causal_prefix"
    systems = json.loads((run / "systems.json").read_text())
    assert all(field in systems for field in COST_FIELDS)
    for field in BOUNDARY_FIELDS:
        assert systems[field] is not None, field
    assert systems["persistent_state_bytes"] == systems["cache_bytes"] > 0
    measured = systems["measured"]
    assert measured["charged_calls"] == (
        measured["physical_actions"] + measured["reset_only_steps"]
    )
    modes = ("retained", "current-token") + (
        ("summary-cleared",) if condition == "raw_summary" else ()
    )
    for history in modes:
        destination = summary_memory.evaluate(
            study,
            benchmark=benchmark,
            condition=condition,
            seed=0,
            device="cpu",
            output_root=tmp_path,
            split="development",
            history=history,
            task_cap=1,
        )
        assert destination.name == f"development-{history}"
        results = json.loads((destination / RESULTS_FILE).read_text())
        assert results["runs"][0]["status"] == "completed"
        assert results["runs"][0]["condition"] == condition
        assert results["runs"][0]["history"] == history
        rows = results["events"]
        assert len(rows) == contract.environment.horizon
        assert all(row["kind"] == "query" for row in rows)


SECONDARY_CELLS = (
    ("concentration", "raw"),
    ("concentration", "raw_summary"),
    ("concentration", "raw_dat_summary"),
    ("xland_minigrid", "raw"),
    ("xland_minigrid", "raw_summary"),
    ("xland_minigrid", "raw_dat_summary"),
    ("count_recall", "raw"),
    ("count_recall", "raw_summary"),
    ("mazerunner", "raw"),
    ("mazerunner", "raw_summary"),
)
"""The study's other protocols, checked with the reference and the summary cells
(Concentration also with the dual-attention summary cell at its 24-slot capacity)."""


def _development_events(
    study: Study,
    root: Path,
    condition: str,
    *,
    batch: int,
    task_cap: int,
    history: str = "retained",
) -> tuple[BenchmarkEvent, ...]:
    """Score the first ``task_cap`` development tasks ``batch`` at a time, in memory."""
    contract, config = summary_memory.resolve(
        study,
        benchmark=BENCHMARK,
        condition=condition,
        seed=0,
        device="cpu",
        output_root=root,
    )
    config = saved_config(config)
    experiment = load_experiment(config, work_directory=config.run_directory)
    try:
        load_selected_checkpoint(experiment, "checkpoint.pt")
        _, events, summary = evaluate_checkpoint(
            contract,
            config,
            experiment,
            checkpoint="checkpoint.pt",
            split="development",
            history=history,
            task_cap=task_cap,
            batch_size=batch,
        )
    finally:
        close_experiment(experiment)
    assert summary["evaluation_batch"] == float(min(batch, task_cap))
    return events


@pytest.mark.slow
@pytest.mark.parametrize("condition", CONDITIONS)
def test_cpu_lifecycle(condition: str, tmp_path: Path) -> None:
    study = load_summary_memory_study()
    assert condition in study.conditions
    assert study.tier(BENCHMARK).group_of(condition) in ("primary", "supplementary")
    if architecture_id(condition) not in BOUND_CARRIERS:
        pytest.skip(f"{condition}: carrier {architecture_id(condition)} is unbound")
    reference = reference_condition(study, BENCHMARK)
    run = summary_memory.train(
        study,
        benchmark=BENCHMARK,
        condition=condition,
        seed=0,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    assert isinstance(run, Path) and (run / "checkpoint.pt").is_file()
    assert (run / "packet_spec.json").is_file()
    systems = json.loads((run / "systems.json").read_text())
    assert systems["gradient_steps"] == 1
    assert systems["cache_bytes"] > 0
    if ALL_CONDITIONS[condition].trajectory_encoder == "gru":
        assert systems["attention"] == {"backend_map": {}}
    spec = ALL_CONDITIONS[condition]
    bounded = spec.memory in ("segment", "summary")
    assert all(field in systems for field in COST_FIELDS)
    assert systems["persistent_state_bytes"] > 0
    assert systems["decision_latency_seconds"] > 0
    assert (
        systems["decision_latency_seconds"] <= systems["decision_latency_p95_seconds"]
    )
    for field in BOUNDARY_FIELDS:
        assert (systems[field] is not None) == bounded, field
    if bounded:
        assert (
            0
            < systems["boundary_latency_seconds"]
            <= systems["boundary_latency_p95_seconds"]
            <= systems["boundary_latency_max_seconds"]
        )
        assert systems["latency_probe"]["boundaries_timed"] == 8
        assert systems["latency_probe"]["probes"] == 8
        # Allocated state: every slot up to capacity, filled or not.
        assert systems["persistent_state_bytes"] == systems["cache_bytes"]
    window = spec.memory == "window"
    if window:
        # R3: the band allocates every W slot per layer, filled or not, and
        # counts FLOPs on a steady-state decision; it has no boundary to time or
        # count. The dual-attention band records the summary reference its W was
        # matched against; the ordinary window (`raw_window`)
        # keeps the legacy identity, whose W = 40 match was measured in the 4M
        # study and is restated in the contract, so it records none.
        assert systems["persistent_state_bytes"] == systems["cache_bytes"]
        if spec.attention == "ordinary":
            assert systems["state_match"] is None
        else:
            assert systems["state_match"]["window_length"] == 40
            assert systems["state_match"]["reference_segment_length"] == 32
            assert systems["state_match"]["residual_bytes"] > 0
        assert systems["latency_probe"]["steps"] == 40 + 8
        assert systems["decision_flops_counted"] > 0
        assert systems["boundary_flops_counted"] is None
    else:
        assert systems["state_match"] is None
        assert systems["decision_flops_counted"] > 0
        assert (systems["boundary_flops_counted"] is not None) == bounded
    assert sum(systems["state_tensor_bytes"].values()) == systems["cache_bytes"]
    # R4: validation interactions are charged to their own clock; the smoke
    # profile validates every epoch over one whole outer task.
    measured = systems["measured"]
    assert measured["charged_calls"] == (
        measured["physical_actions"] + measured["reset_only_steps"]
    )
    validation = measured["validation"]
    assert validation["charged_calls"] >= 72 and validation["tasks_started"] >= 1
    assert validation["charged_calls"] == (
        validation["physical_actions"] + validation["reset_only_steps"]
    )
    if spec.dat_mode is not None:
        assert systems["attention"]["cache_capacity"] == (
            40 if bounded or window else 73
        )
    assert list((run / "ckpts/policy_weights").glob("policy_epoch_*.pt"))
    destination = summary_memory.evaluate(
        study,
        benchmark=BENCHMARK,
        condition=condition,
        seed=0,
        device="cpu",
        output_root=tmp_path,
        split="development",
        task_cap=2,
    )
    results = json.loads((destination / RESULTS_FILE).read_text())
    assert results["partial_task_cap"] == 2
    assert results["runs"][0]["status"] == "completed"
    assert results["runs"][0]["condition"] == condition
    assert results["events"]
    tracked = [event["writes_before_decision"] for event in results["events"]]
    if bounded:
        # Smoke Key-to-Door runs 72 decisions over 32-record segments: the
        # later attempts of a task start after one or two boundaries.
        assert all(isinstance(value, int) for value in tracked)
        assert max(tracked) >= 1 and min(tracked) == 0
        ages = [event["evidence_age_writes"] for event in results["events"]]
        assert all(
            age is None or 0 <= age <= writes
            for age, writes in zip(ages, tracked, strict=True)
        )
        for event in results["events"]:
            recent = event["evidence_age_writes_recent"]
            assert (recent is None) == (event["evidence_age_writes"] is None)
            if recent is not None:
                assert 0 <= recent <= event["evidence_age_writes"]
                assert event["evidence_in_current_segment"] is (recent == 0)
    else:
        assert all(value is None for value in tracked)
        assert all(
            event["evidence_age_writes_recent"] is None
            and event["evidence_in_current_segment"] is None
            for event in results["events"]
        )
    for event in results["events"]:
        assert 1 <= event["start_step"] <= event["end_step"]
        assert event["end_step"] - event["start_step"] + 1 == event["step"]
    assert results["runs"][0]["checkpoint_rule"] == "selected"
    # The batched rollout: every carrier keeps its rows independent, so the
    # same tasks scored one at a time and four at a time (one chunk, and a
    # chunk of three plus one) yield identical events, interventions included.
    single = _development_events(study, tmp_path, condition, batch=1, task_cap=4)
    assert len({event.task_id for event in single}) == 4
    assert (
        _development_events(study, tmp_path, condition, batch=4, task_cap=4) == single
    )
    assert (
        _development_events(study, tmp_path, condition, batch=3, task_cap=4) == single
    )
    intervention = (
        "summary-cleared" if spec.memory == "summary" else task_intervention(BENCHMARK)
    )
    cleared = _development_events(
        study, tmp_path, condition, batch=1, task_cap=4, history=intervention
    )
    assert (
        _development_events(
            study, tmp_path, condition, batch=4, task_cap=4, history=intervention
        )
        == cleared
    )
    if condition == reference:
        # The fixed-final-checkpoint supplement: its own directory and rule.
        final = summary_memory.evaluate(
            study,
            benchmark=BENCHMARK,
            condition=condition,
            seed=0,
            device="cpu",
            output_root=tmp_path,
            split="development",
            task_cap=2,
            checkpoint_rule="final-epoch",
        )
        assert final.name == "development-retained-final-epoch"
        supplement = json.loads((final / RESULTS_FILE).read_text())
        assert supplement["runs"][0]["checkpoint_rule"] == "final-epoch"
        assert supplement["runs"][0]["checkpoint"] == (
            f"policy_epoch_{latest_training_epoch(run)}"
        )
        assert all(e["checkpoint_rule"] == "final-epoch" for e in supplement["events"])
        assert (destination / RESULTS_FILE).is_file()  # the selected panel is kept
        with pytest.raises(ContractError, match="resolves the checkpoint"):
            summary_memory.evaluate(
                study,
                benchmark=BENCHMARK,
                condition=condition,
                seed=0,
                device="cpu",
                output_root=tmp_path,
                split="development",
                checkpoint="policy_epoch_1",
                task_cap=1,
                checkpoint_rule="final-epoch",
            )
    if spec.memory == "summary":
        cleared = summary_memory.evaluate(
            study,
            benchmark=BENCHMARK,
            condition=condition,
            seed=0,
            device="cpu",
            output_root=tmp_path,
            split="development",
            history="summary-cleared",
            task_cap=2,
        )
        assert cleared.name == "development-summary-cleared"
        rows = json.loads((cleared / RESULTS_FILE).read_text())
        assert rows["runs"][0]["history"] == "summary-cleared"
        assert len(rows["events"]) == len(results["events"])
    else:
        with pytest.raises(ContractError, match="memory regime is 'summary'"):
            summary_memory.evaluate(
                study,
                benchmark=BENCHMARK,
                condition=condition,
                seed=0,
                device="cpu",
                output_root=tmp_path,
                split="development",
                history="summary-cleared",
                task_cap=1,
            )


@pytest.mark.slow
@pytest.mark.parametrize(("benchmark", "condition"), SECONDARY_CELLS)
def test_cpu_lifecycle_on_the_secondary_protocols(
    benchmark: str, condition: str, tmp_path: Path
) -> None:
    """`raw` and `raw_summary` train, evaluate and intervene on the other protocols."""
    study = load_retired_summary_memory_study()
    contract = study.contract(benchmark)
    run = summary_memory.train(
        study,
        benchmark=benchmark,
        condition=condition,
        seed=0,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    assert isinstance(run, Path) and (run / "checkpoint.pt").is_file()
    assert run.parent.parent.name == contract.protocol
    systems = json.loads((run / "systems.json").read_text())
    assert all(field in systems for field in COST_FIELDS)
    bounded = ALL_CONDITIONS[condition].memory == "summary"
    kind = contract.event_kind
    segment = (
        (contract.memory or {}).get("summary", {}).get("segment_length", 32)
        if bounded
        else None
    )
    expected = {
        "query": contract.environment.horizon,
        "episode": 1,
        "attempt": contract.environment.scored_attempts,
    }[kind]

    def evaluate(history: str) -> list[dict[str, Any]]:
        destination = summary_memory.evaluate(
            study,
            benchmark=benchmark,
            condition=condition,
            seed=0,
            device="cpu",
            output_root=tmp_path,
            split="development",
            history=history,
            task_cap=1,
        )
        assert destination.name == f"development-{history}"
        results = json.loads((destination / RESULTS_FILE).read_text())
        assert results["partial_task_cap"] == 1
        assert results["runs"][0]["status"] == "completed"
        assert results["runs"][0]["protocol"] == contract.protocol
        assert results["runs"][0]["history"] == history
        rows: list[dict[str, Any]] = results["events"]
        assert len(rows) == expected and all(row["kind"] == kind for row in rows)
        return rows

    events = evaluate("retained")
    outer = contract.environment.outer_length
    for event in events:
        assert 1 <= event["start_step"] <= event["end_step"] <= outer
    if kind == "query":
        assert all(0 <= event["true_count"] <= 16 for event in events)
        assert [event["event_index"] for event in events] == list(
            range(1, expected + 1)
        )
    if kind == "attempt":
        # XLand scores the declared band (episodes 4-5 of five) as its events.
        first = contract.environment.scored_from
        assert [event["event_index"] for event in events] == list(
            range(first, contract.environment.attempts + 1)
        )
    tracked = [event["writes_before_decision"] for event in events]
    if bounded:
        assert all(isinstance(value, int) for value in tracked)
        if kind == "query":
            # 207 decisions over 32-record segments: a boundary before decisions
            # 33, 65, ..., 193, so the last queries follow six writes.
            assert tracked == [(index - 1) // 32 for index in range(1, expected + 1)]
            for event in events:
                before = event["count_before_current_segment"]
                inside = event["count_in_current_segment"]
                assert isinstance(before, int) and isinstance(inside, int)
                assert before + inside == event["true_count"]
                age = event["evidence_age_writes"]
                assert age is None or 0 <= age <= event["writes_before_decision"]
        elif kind == "attempt":
            # Scored episodes start after earlier ones: their write counts are
            # the boundaries crossed before each episode's first decision.
            assert isinstance(segment, int)
            assert tracked == [(event["start_step"] - 1) // segment for event in events]
        else:
            # One episode event: the boundaries before its last decision (none
            # on the shortened MazeRunner smoke; six on the fixed 104-flip deck).
            assert isinstance(segment, int)
            assert tracked == [(events[0]["step"] - 1) // segment]
            if benchmark == "concentration":
                ledger = events[0]
                assert isinstance(ledger["retrieval_opportunities"], int)
                assert isinstance(ledger["evicted_opportunities"], int)
                assert (
                    ledger["evicted_opportunities"] <= ledger["retrieval_opportunities"]
                )
        cleared = evaluate("summary-cleared")
        assert [event["true_count"] for event in cleared] == [
            event["true_count"] for event in events
        ]
        if kind == "attempt":
            assert [event["event_index"] for event in cleared] == [
                event["event_index"] for event in events
            ]
    else:
        assert all(value is None for value in tracked)
        assert all(
            event["count_before_current_segment"] is None
            and event["evidence_age_writes"] is None
            for event in events
        )
        if benchmark == "concentration":
            assert isinstance(events[0]["retrieval_opportunities"], int)
            assert events[0]["evicted_opportunities"] is None
        with pytest.raises(ContractError, match="memory regime is 'summary'"):
            evaluate("summary-cleared")
    # The task's own intervention (current-token, goal-cleared) on either cell.
    evaluate(history_modes(benchmark)[1])


@pytest.mark.slow
def test_the_scripts_run_the_reference_cell_end_to_end(tmp_path: Path) -> None:
    """The scripts default to the active study, whose reference is full_context."""
    common = [
        "summary_memory",
        "--benchmark",
        BENCHMARK,
        "--condition",
        reference_condition(load_summary_memory_study(), BENCHMARK),
        "--seed",
        "0",
        "--device",
        "cpu",
        "--output-root",
        str(tmp_path),
    ]
    train = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train.py"), *common, "--smoke"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=900,
    )
    assert train.returncode == 0, train.stderr
    run = Path(train.stdout.strip().splitlines()[-1])
    assert run.is_relative_to(tmp_path) and (run / "checkpoint.pt").is_file()
    evaluate = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate.py"),
            *common,
            "--split",
            "development",
            "--task-cap",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=900,
    )
    assert evaluate.returncode == 0, evaluate.stderr
    destination = Path(evaluate.stdout.strip().splitlines()[-1])
    assert (destination / RESULTS_FILE).is_file()


@pytest.mark.slow
def test_gru_exact_resume_with_live_prefix(tmp_path: Path) -> None:
    """An interrupted `raw_gru` fit resumes to the uninterrupted weights and replay.

    The interruption lands mid-task, so the resumed run restores the GRU state
    tensor through the runtime checkpoint and rebuilds it after the next update.
    """
    from dataclasses import replace

    import torch

    from reasoned_icrl.experiments.benchmarks import experiment_config
    from reasoned_icrl.runtime.training import close_experiment

    from ..test_experiment import _started_experiment

    study = load_retired_summary_memory_study()
    cfg = experiment_config(
        study.contract(BENCHMARK),
        study,
        condition="raw_gru",
        seed=0,
        repository=ROOT,
        device="cpu",
        output_root=tmp_path / "whole",
        smoke=True,
    )
    outer = cfg.environment.outer_length
    cfg = replace(
        cfg,
        environment=replace(cfg.environment, parallel_envs=2),
        training=replace(
            cfg.training,
            epochs=3,
            start_learning_epoch=0,
            timesteps_per_epoch=outer + 3,
            batches_per_epoch=1,
            validation_timesteps=outer,
            validation_interval=1,
            checkpoint_interval=1,
            batch_size=2,
            replay_capacity=64,
            epsilon_anneal_steps=3 * (outer + 3),
        ),
    )
    whole = _started_experiment(cfg)
    whole.learn()
    expected = {
        k: v.detach().cpu().clone() for k, v in whole.policy.state_dict().items()
    }
    expected_replay = [
        Path(p).read_bytes() for p in whole.reasoned_dataset.all_filenames
    ]
    expected_updates = whole.grad_update_counter
    assert isinstance(whole.hidden_state, torch.Tensor)
    close_experiment(whole)
    cfg = replace(cfg, output_root=tmp_path / "resume")
    partial = _started_experiment(cfg, epoch_limit=1)
    partial.learn()
    assert any(
        int(t.time_idx[0]) > 0
        for seq in partial.train_envs.envs
        for t in seq.active_trajs[0].timesteps
    )
    close_experiment(partial)
    resumed = _started_experiment(cfg)
    resumed.load_checkpoint(0, resume_training_state=True)
    assert isinstance(resumed.hidden_state, torch.Tensor)
    resumed.epoch = 1
    resumed.learn()
    for name, value in expected.items():
        torch.testing.assert_close(
            resumed.policy.state_dict()[name].cpu(), value, rtol=1e-5, atol=1e-6
        )
    assert expected_updates == resumed.grad_update_counter
    assert expected_replay == [
        Path(p).read_bytes() for p in resumed.reasoned_dataset.all_filenames
    ]
    close_experiment(resumed)


@pytest.mark.slow
def test_summary_exact_resume_across_a_boundary_and_an_outer_reset(
    tmp_path: Path,
) -> None:
    """An interrupted `raw_summary` fit resumes to the uninterrupted weights.

    The interruption lands eight records into the second segment of the second
    task, so the resumed run restores a summary written at one boundary, a
    partially filled cache and a crossed outer reset through the runtime
    checkpoint, then rebuilds the state after the next update.
    """
    from dataclasses import replace

    import torch

    from reasoned_icrl.experiments.benchmarks import experiment_config
    from reasoned_icrl.model.summary_transformer import SummaryHiddenState
    from reasoned_icrl.runtime.training import close_experiment

    from ..test_experiment import _started_experiment

    study = load_retired_summary_memory_study()
    cfg = experiment_config(
        study.contract(BENCHMARK),
        study,
        condition="raw_summary",
        seed=0,
        repository=ROOT,
        device="cpu",
        output_root=tmp_path / "whole",
        smoke=True,
    )
    outer = cfg.environment.outer_length
    assert cfg.model.summary is not None and cfg.model.summary.segment_length == 32
    interruption = outer + 40  # one boundary and eight records into segment two
    cfg = replace(
        cfg,
        environment=replace(cfg.environment, parallel_envs=2),
        training=replace(
            cfg.training,
            epochs=3,
            start_learning_epoch=0,
            timesteps_per_epoch=interruption,
            batches_per_epoch=1,
            validation_timesteps=outer,
            validation_interval=1,
            checkpoint_interval=1,
            batch_size=2,
            replay_capacity=64,
            epsilon_anneal_steps=3 * interruption,
        ),
    )
    whole = _started_experiment(cfg)
    whole.learn()
    expected = {
        k: v.detach().cpu().clone() for k, v in whole.policy.state_dict().items()
    }
    expected_replay = [
        Path(p).read_bytes() for p in whole.reasoned_dataset.all_filenames
    ]
    expected_updates = whole.grad_update_counter
    assert isinstance(whole.hidden_state, SummaryHiddenState)
    close_experiment(whole)
    cfg = replace(cfg, output_root=tmp_path / "resume")
    partial = _started_experiment(cfg, epoch_limit=1)
    partial.learn()
    state = partial.hidden_state
    assert isinstance(state, SummaryHiddenState)
    assert state.segment.tolist() == [1, 1] and state.lengths.tolist() == [12, 12]
    close_experiment(partial)
    resumed = _started_experiment(cfg)
    resumed.load_checkpoint(0, resume_training_state=True)
    assert isinstance(resumed.hidden_state, SummaryHiddenState)
    assert resumed.hidden_state.segment.tolist() == [1, 1]
    resumed.epoch = 1
    resumed.learn()
    for name, value in expected.items():
        torch.testing.assert_close(
            resumed.policy.state_dict()[name].cpu(), value, rtol=1e-5, atol=1e-6
        )
    assert expected_updates == resumed.grad_update_counter
    assert expected_replay == [
        Path(p).read_bytes() for p in resumed.reasoned_dataset.all_filenames
    ]
    close_experiment(resumed)
