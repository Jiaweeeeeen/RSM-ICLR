"""Bounded train/evaluate lifecycles of the Memo comparator (ME1 acceptance).

The ``memo`` cell of the comparator study runs through the shared trainer and
the shared evaluator on CPU with the smoke profile: it trains, checkpoints,
rebuilds its state after the learner update, writes its cost record with the
state-growth schedule, evaluates under every declared history mode with
independent rows, and resumes an interrupted fit exactly across a boundary
and an outer reset. Passing establishes execution integrity only, never
learning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasoned_icrl.experiments.benchmarks import Study, saved_config
from reasoned_icrl.experiments.config import architecture_id
from reasoned_icrl.experiments.contracts import (
    MEMO_ARCHITECTURE_ID,
    ContractError,
    memo_live_slots,
)
from reasoned_icrl.experiments.evaluation import RESULTS_FILE
from reasoned_icrl.experiments.records import BenchmarkEvent
from reasoned_icrl.experiments.summary_memory import experiments as summary_memory
from reasoned_icrl.experiments.summary_memory.configs import load_memo_study
from reasoned_icrl.runtime.rollout import evaluate as evaluate_checkpoint
from reasoned_icrl.runtime.training import (
    close_experiment,
    load_experiment,
    load_selected_checkpoint,
)

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = "dark_key_to_door"
CONDITIONS = ("memo", "memo_fixed")
"""The jittered recipe (ME2) and the fixed-segment cell (ME4); the plan asks
for the ME1 checks to be re-run on the new cell before its launch."""
COST_FIELDS = (
    "persistent_state_bytes",
    "decision_latency_seconds",
    "decision_latency_p95_seconds",
    "boundary_latency_seconds",
    "boundary_latency_p95_seconds",
    "boundary_latency_max_seconds",
)


def _development_events(
    study: Study,
    root: Path,
    condition: str,
    *,
    batch: int,
    task_cap: int,
    history: str = "retained",
) -> tuple[BenchmarkEvent, ...]:
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
@pytest.mark.parametrize("CONDITION", CONDITIONS)
def test_memo_cpu_lifecycle(CONDITION: str, tmp_path: Path) -> None:
    study = load_memo_study()
    contract = study.contract(BENCHMARK)
    assert study.tier(BENCHMARK).group_of(CONDITION) == "primary"
    assert architecture_id(CONDITION) == MEMO_ARCHITECTURE_ID
    run = summary_memory.train(
        study,
        benchmark=BENCHMARK,
        condition=CONDITION,
        seed=0,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    assert isinstance(run, Path) and (run / "checkpoint.pt").is_file()
    assert run.parent.parent.name == contract.protocol
    assert (run / "packet_spec.json").is_file()
    config = saved_config(
        summary_memory.resolve(
            study,
            benchmark=BENCHMARK,
            condition=CONDITION,
            seed=0,
            output_root=tmp_path,
        )[1]
    )
    assert config.model.architecture_id == MEMO_ARCHITECTURE_ID
    assert config.model.dat is None and config.model.summary is None
    assert config.model.memo is not None
    assert (config.model.memo.segment_length, config.model.memo.summary_tokens) == (
        32,
        4,
    )
    assert config.model.memo.training_segment_jitter == (
        0.2 if CONDITION == "memo" else 0.0
    )
    spec = config.model.memo
    outer = config.environment.outer_length
    systems = json.loads((run / "systems.json").read_text())
    assert systems["gradient_steps"] == 1
    assert all(field in systems for field in COST_FIELDS)
    assert systems["attention"] == {
        "backend_map": {
            str(index): {"content": "vanilla"} for index in range(config.model.layers)
        }
    }
    assert systems["state_match"] is None
    # Allocated for the longest task; the live schedule is recorded beside it.
    assert systems["persistent_state_bytes"] == systems["cache_bytes"] > 0
    assert sum(systems["state_tensor_bytes"].values()) == systems["cache_bytes"]
    assert systems["shared_state_bytes"] == 0
    growth = systems["state_growth"]
    records = outer + 1  # the initial record plus one per charged call
    assert growth["longest_task_records"] == records
    assert growth["capacity_slots"] == spec.capacity(outer)
    slots = growth["live_slots_by_prefix"]
    assert slots["1"] == 1 and slots["32"] == 32 and slots["33"] == 5
    assert slots[str(records)] == memo_live_slots(spec, records)
    assert growth["live_bytes_by_prefix"][str(records)] == (
        slots[str(records)] * growth["bytes_per_slot"] + growth["counter_bytes"]
    )
    assert growth["live_bytes_by_prefix"]["32"] < systems["persistent_state_bytes"]
    # The probe crosses as many boundaries as the smoke task allows (two).
    probes = spec.summaries_before(outer)
    assert systems["latency_probe"]["probes"] == probes == 2
    assert systems["latency_probe"]["boundaries_timed"] == probes
    assert (
        0
        < systems["boundary_latency_seconds"]
        <= systems["boundary_latency_p95_seconds"]
        <= systems["boundary_latency_max_seconds"]
    )
    assert (
        systems["decision_flops_counted"] > 0 and systems["boundary_flops_counted"] > 0
    )
    assert (
        systems["decision_latency_seconds"] <= systems["decision_latency_p95_seconds"]
    )
    measured = systems["measured"]
    assert measured["charged_calls"] == (
        measured["physical_actions"] + measured["reset_only_steps"]
    )
    assert list((run / "ckpts/policy_weights").glob("policy_epoch_*.pt"))
    # Every declared history mode of the task and of the carrier evaluates.
    for history in ("retained", "attempt-cleared", "summary-cleared"):
        destination = summary_memory.evaluate(
            study,
            benchmark=BENCHMARK,
            condition=CONDITION,
            seed=0,
            device="cpu",
            output_root=tmp_path,
            split="development",
            history=history,
            task_cap=2,
        )
        assert destination.name == f"development-{history}"
        results = json.loads((destination / RESULTS_FILE).read_text())
        assert results["partial_task_cap"] == 2
        assert results["runs"][0]["status"] == "completed"
        assert results["runs"][0]["condition"] == CONDITION
        assert results["runs"][0]["history"] == history
        tracked = [event["writes_before_decision"] for event in results["events"]]
        if history == "attempt-cleared":
            assert all(value is None for value in tracked)
        else:
            # Smoke Key-to-Door runs 72 decisions over 32-record segments: the
            # later attempts start after one or two boundaries, retained and
            # cleared alike (the cleared rollout still counts its boundaries).
            assert all(isinstance(value, int) for value in tracked)
            assert max(tracked) >= 1 and min(tracked) == 0
    with pytest.raises(ContractError, match="history modes"):
        summary_memory.evaluate(
            study,
            benchmark=BENCHMARK,
            condition=CONDITION,
            seed=0,
            device="cpu",
            output_root=tmp_path,
            split="development",
            history="current-token",
            task_cap=1,
        )
    # Rows are independent: the same tasks scored one at a time and in chunks
    # yield identical events, and so does the cleared intervention.
    single = _development_events(study, tmp_path, CONDITION, batch=1, task_cap=4)
    assert len({event.task_id for event in single}) == 4
    assert (
        _development_events(study, tmp_path, CONDITION, batch=4, task_cap=4) == single
    )
    assert (
        _development_events(study, tmp_path, CONDITION, batch=3, task_cap=4) == single
    )
    cleared = _development_events(
        study, tmp_path, CONDITION, batch=1, task_cap=4, history="summary-cleared"
    )
    assert (
        _development_events(
            study, tmp_path, CONDITION, batch=4, task_cap=4, history="summary-cleared"
        )
        == cleared
    )
    assert cleared != single


@pytest.mark.slow
@pytest.mark.parametrize("CONDITION", CONDITIONS)
def test_memo_exact_resume_across_a_boundary_and_an_outer_reset(
    CONDITION: str, tmp_path: Path
) -> None:
    """An interrupted ``memo`` fit resumes to the uninterrupted weights.

    The interruption lands eight records into the second segment of the
    second task, so the resumed run restores one accumulated summary, a
    partially filled cache and a crossed outer reset through the runtime
    checkpoint (and the torch RNG that draws the dense segment length), then
    rebuilds the state after the next update.
    """
    from dataclasses import replace

    import torch

    from reasoned_icrl.experiments.benchmarks import experiment_config
    from reasoned_icrl.model.memo_transformer import MemoHiddenState

    from ..test_experiment import _started_experiment

    study = load_memo_study()
    cfg = experiment_config(
        study.contract(BENCHMARK),
        study,
        condition=CONDITION,
        seed=0,
        repository=ROOT,
        device="cpu",
        output_root=tmp_path / "whole",
        smoke=True,
    )
    outer = cfg.environment.outer_length
    assert cfg.model.memo is not None and cfg.model.memo.segment_length == 32
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
    assert isinstance(whole.hidden_state, MemoHiddenState)
    close_experiment(whole)
    cfg = replace(cfg, output_root=tmp_path / "resume")
    partial = _started_experiment(cfg, epoch_limit=1)
    partial.learn()
    state = partial.hidden_state
    assert isinstance(state, MemoHiddenState)
    assert state.segment.tolist() == [1, 1] and state.lengths.tolist() == [4 + 8, 4 + 8]
    close_experiment(partial)
    resumed = _started_experiment(cfg)
    resumed.load_checkpoint(0, resume_training_state=True)
    assert isinstance(resumed.hidden_state, MemoHiddenState)
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


@pytest.mark.slow
@pytest.mark.parametrize("CONDITION", CONDITIONS)
def test_memo_cpu_lifecycle_on_count_recall(CONDITION: str, tmp_path: Path) -> None:
    """ME5: both Memo cells train, checkpoint and evaluate under every declared
    history mode on the tier-2 CountRecallMedium contract, with the scripted
    count probe the tier's panels require."""
    study = load_memo_study("count_recall")
    benchmark = "count_recall"
    contract = study.contract(benchmark)
    assert study.tier(benchmark).group_of(CONDITION) == "primary"
    run = summary_memory.train(
        study,
        benchmark=benchmark,
        condition=CONDITION,
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
            condition=CONDITION,
            seed=0,
            output_root=tmp_path,
        )[1]
    )
    assert config.model.architecture_id == MEMO_ARCHITECTURE_ID
    assert config.model.memo is not None and config.model.dat is None
    assert config.model.memo.training_segment_jitter == (
        0.2 if CONDITION == "memo" else 0.0
    )
    systems = json.loads((run / "systems.json").read_text())
    assert systems["persistent_state_bytes"] == systems["cache_bytes"] > 0
    growth = systems["state_growth"]
    assert growth["longest_task_records"] == config.environment.outer_length + 1
    assert growth["capacity_slots"] == config.model.memo.capacity(
        config.environment.outer_length
    )
    assert (
        systems["latency_probe"]["boundaries_timed"]
        == systems["latency_probe"]["probes"]
    )
    measured = systems["measured"]
    assert measured["charged_calls"] == (
        measured["physical_actions"] + measured["reset_only_steps"]
    )
    for history in ("retained", "current-token", "summary-cleared"):
        destination = summary_memory.evaluate(
            study,
            benchmark=benchmark,
            condition=CONDITION,
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
        assert results["runs"][0]["condition"] == CONDITION
        rows = results["events"]
        assert len(rows) == contract.environment.horizon
        assert all(row["kind"] == "query" for row in rows)
        tracked = [row["writes_before_decision"] for row in rows]
        if history == "current-token":
            assert all(value is None for value in tracked)
        else:
            assert all(isinstance(value, int) for value in tracked)
            assert min(tracked) == 0 and max(tracked) >= 1
