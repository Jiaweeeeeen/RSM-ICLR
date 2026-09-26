"""R8 acceptance: official Medium semantics, public instruments and nine-fit gates."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from reasoned_icrl.analysis.statistics import count_recall_query_curves
from reasoned_icrl.analysis.tier import write_tier_report
from reasoned_icrl.environments.base import FINAL_TASKS, TRAINING_TASKS
from reasoned_icrl.environments.count_recall import CountRecallEnv, PublicStreamCounter
from reasoned_icrl.experiments.benchmarks import saved_config
from reasoned_icrl.experiments.contracts import ContractError, ResultValidationError
from reasoned_icrl.experiments.count_recall_controls import (
    fit_count_time_prior,
    score_count_time_prior,
)
from reasoned_icrl.experiments.evaluation import (
    CountRecallStreamResult,
    events,
    query_write_fields,
    run_record,
)
from reasoned_icrl.experiments.records import validate_benchmark_results
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.experiments import (
    evaluate,
    resolve,
    train,
)
from reasoned_icrl.experiments.summary_memory.jobs import (
    completed_training_budget,
    plan_fits,
    require_primary_training_complete,
)
from reasoned_icrl.experiments.summary_memory.revised import finalize_revised_cell
from reasoned_icrl.runtime.diagnostics import count_recall_diagnostic
from reasoned_icrl.runtime.rollout import scripted_count_recall
from reasoned_icrl.runtime.training import (
    close_experiment,
    load_experiment,
)
from reasoned_icrl.utils import repository_root

CELLS = ("full_dual_relational", "fixed_summary", "fixed_segment")


def test_official_medium_roster_budget_and_public_contract(tmp_path: Path) -> None:
    study = load_summary_memory_study()
    contract = study.contract("count_recall")
    plans = plan_fits(
        study, contract, repository=repository_root(), output_root=tmp_path
    )
    assert [(p.condition, p.seed) for p in plans] == [
        (c, s) for c in CELLS for s in (42, 100, 2026)
    ]
    assert contract.protocol == "count-recall-medium"
    assert contract.evaluation.retention == "complete"
    for cell in CELLS:
        _, config = resolve(
            study,
            benchmark="count_recall",
            condition=cell,
            seed=42,
            device="cpu",
            output_root=tmp_path,
            wandb=True,
        )
        assert (
            config.training.epochs
            * config.training.timesteps_per_epoch
            * config.environment.parallel_envs
            == 8_000_000
        )
        assert (config.environment.size, config.environment.horizon) == (4, 103)
        assert config.training.trajectory_length == 104
        assert config.training.max_sequence_length == 103
        assert config.training.mixed_precision == "no" and config.tracking.wandb
        if config.model.summary:
            spec = config.model.summary
            assert (spec.segment_length, spec.memory_tokens, spec.detach) == (
                32,
                4,
                "none",
            )
    assert contract.memory["window"]["segment_length"] == 40
    assert study.primary_cells(contract) == CELLS
    # The ICLR-plan additions are tier-2 supplementary cells; they
    # never gate the primary roster, its completeness or its report. The last
    # two joined later: the truncated-gradient ablation
    # (commit 593f3be) and the residual rewrite (798f2df),
    # which this expectation had not followed.
    assert study.tier("count_recall").supplementary == (
        "full_gru",
        "fixed_window",
        "full_context",
        "full_dual_content",
        "raw_summary",
        "raw_segment",
        "raw_summary_detach",
        "raw_summary_residual",
    )
    for supplementary in study.tier("count_recall").supplementary:
        _, config = resolve(
            study, benchmark="count_recall", condition=supplementary, seed=42
        )
        assert config.condition == supplementary
        assert study.tier("count_recall").group_of(supplementary) == "supplementary"
    bands = [
        set(contract.roster("development")),
        set(contract.roster("final")),
        set(FINAL_TASKS),
    ]
    assert all(not (a & b) for i, a in enumerate(bands) for b in bands[i + 1 :])
    assert all(t not in TRAINING_TASKS for band in bands for t in band)


def test_medium_matches_pinned_native_every_action_and_restores() -> None:
    from amago.envs.builtin.popgym_envs import POPGym

    native = POPGym("popgym-CountRecallMedium-v0")
    env = CountRecallEnv(variant="medium", split="development")
    try:
        packet, _ = env.reset(options={"task_index": 1_000_001})
        observation, _ = native.reset(seed=1_000_001)
        assert native.unwrapped.num_distinct_cards == 4
        assert native.unwrapped.max_card_count == 26
        assert native.action_space.n == 27 and packet["current"].shape == (9,)
        counter = PublicStreamCounter(4)
        values = []
        saved = None
        next_packet = None
        for index in range(1, 104):
            np.testing.assert_allclose(
                (packet["current"] + 1) / 2, observation, atol=1e-7
            )
            value, query, timer = env.decode(packet)
            values.append(value)
            truth = counter.observe(value, query)
            assert timer == pytest.approx((index - 1) / 1000, abs=1e-6)
            assert set(packet) == {"current", "previous", "outcome", "event", "valid"}
            answer = (index * 7) % 27
            if index == 33:
                saved = env.state_dict()
            packet, reward, term, trunc, info = env.step(answer)
            observation, native_reward, native_term, native_trunc, _ = native.step(
                answer
            )
            assert (
                reward
                == native_reward
                == pytest.approx((1 if answer == truth else -1) / 103)
            )
            assert (term, trunc) == (native_term, native_trunc) == (index == 103, False)
            assert info["true_count"] == truth and info["query_index"] == index
            assert not {"counts", "query_counts", "deck", "get_state"} & set(info)
            if index == 33:
                next_packet = packet
        values.append(env.decode(packet)[0])
        assert [values.count(value) for value in range(4)] == [26] * 4
        assert len(env.scored_queries) == 103
        accuracy = np.mean([q.correct for q in env.scored_queries])
        assert env.stream_return == pytest.approx(2 * accuracy - 1)
        assert env.collection_counters()["charged_calls"] == 103
        assert saved is not None and next_packet is not None
        env.load_state_dict(saved)
        actual, _, _, _, _ = env.step((33 * 7) % 27)
        for field in actual:
            np.testing.assert_array_equal(actual[field], next_packet[field])
    finally:
        env.close()
        native.close()


def test_medium_query_boundaries_error_retention_and_stream_weighting() -> None:
    study = load_summary_memory_study()
    contract, config = resolve(
        study, benchmark="count_recall", condition="fixed_summary", seed=42
    )
    contract = replace(
        contract,
        evaluation=replace(
            contract.evaluation,
            splits={
                "development": replace(
                    contract.evaluation.splits["development"], count=2
                )
            },
        ),
    )
    env = CountRecallEnv(variant="medium", split="development")
    results = []
    try:
        for stream in contract.roster("development"):
            packet, _ = env.reset(options={"task_index": stream})
            counter = PublicStreamCounter(4)
            values = []
            for index in range(1, 104):
                value, query, _ = env.decode(packet)
                values.append(value)
                truth = counter.observe(value, query)
                answer = truth if index % 2 else (truth + 1) % 27
                packet, _, _, _, _ = env.step(answer)
            result = CountRecallStreamResult(
                stream,
                0,
                env.scored_queries,
                env.stream_return,
                103,
                tuple((i - 1) // 32 for i in range(1, 104)),
                tuple(values),
            )
            results.append(result)
            for query in result.queries:
                fields = query_write_fields(result, query)
                assert fields.writes == (query.index - 1) // 32
                assert (
                    fields.count_before_segment + fields.count_in_segment
                    == query.true_count
                )
                if query.index in (33, 65, 97):
                    assert fields.count_in_segment == int(
                        values[query.index - 1] == query.query
                    )
    finally:
        env.close()
    rows = events(
        contract,
        config,
        results,
        checkpoint="fixture",
        split="development",
        history="retained",
    )
    run = run_record(
        contract, config, checkpoint="fixture", split="development", history="retained"
    )
    validate_benchmark_results([contract], [run], list(rows))
    assert len(rows) == 206 and all(e.answer is not None for e in rows)
    curve = count_recall_query_curves(rows, samples=0)
    assert curve[-1].estimate == pytest.approx(52 / 103) and curve[-1].tasks == 2
    error = count_recall_query_curves(rows, measure="absolute_error", samples=0)
    assert error[0].estimate == 0 and error[1].estimate >= 1
    with pytest.raises(ResultValidationError, match="all 103"):
        count_recall_query_curves(rows[1:], samples=0)
    with pytest.raises(ResultValidationError, match="answer/correctness"):
        validate_benchmark_results(
            [contract], [run], [replace(rows[0], answer=None), *rows[1:]]
        )


def test_prior_fits_training_only_and_freezes_before_held_out_scoring() -> None:
    prior = fit_count_time_prior(tuple(range(12)))
    before = prior.as_dict()
    timed, equality = score_count_time_prior(
        prior, (1_000_000, 1_000_001), split="development"
    )
    assert prior.as_dict() == before
    assert len(timed) == len(equality) == 2 and prior.charged_calls == 12 * 103
    assert all(0 <= value <= 1 for value in [*timed.values(), *equality.values()])
    with pytest.raises(ContractError, match="training-only"):
        fit_count_time_prior((1_000_000,))
    with pytest.raises(ContractError, match="overlaps"):
        score_count_time_prior(prior, (0,), split="train")


def test_medium_c3_reports_complete_input_pair_and_task_coverage() -> None:
    env = CountRecallEnv(variant="medium", split="development")
    try:
        result = count_recall_diagnostic(
            env, stream_ids=tuple(range(1_000_000, 1_000_016))
        )
    finally:
        env.close()
    assert result.measurements["verified_pairs"] > 0
    assert result.measurements["tasks_in_verified_pairs"] == 16


def test_all_final_and_report_entry_points_refuse_one_incomplete_fit(
    tmp_path: Path,
) -> None:
    study = load_summary_memory_study()
    contract, config = resolve(
        study,
        benchmark="count_recall",
        condition=CELLS[0],
        seed=42,
        output_root=tmp_path,
    )
    with pytest.raises(ContractError, match="Every primary fit"):
        evaluate(
            study,
            benchmark="count_recall",
            condition=CELLS[0],
            seed=42,
            split="final",
            output_root=tmp_path,
        )
    with pytest.raises(ContractError, match="Every primary fit"):
        finalize_revised_cell(study, contract, config, split="final")
    for split in ("development", "final"):
        with pytest.raises(ContractError, match="Every primary fit"):
            write_tier_report(study, contract, tmp_path, split=split)
    plans = plan_fits(
        study, contract, repository=repository_root(), output_root=tmp_path
    )
    for plan in plans:
        directory = plan.run_directory
        weights = directory / "ckpts/policy_weights/policy_epoch_999.pt"
        weights.parent.mkdir(parents=True)
        weights.touch()
        (directory / "checkpoint.pt").touch()
        (directory / "metrics.json").write_text(
            json.dumps({"status": "trained", "scalar_training_transitions": 8_000_000})
        )
        (directory / "systems.json").write_text(
            json.dumps(
                {
                    "measured": {
                        "charged_calls": 8_000_000,
                        "physical_actions": 8_000_000,
                        "reset_only_steps": 0,
                    }
                }
            )
        )
        assert completed_training_budget(directory, contract) == 8_000_000
    require_primary_training_complete(study, contract, tmp_path)
    (plans[-1].run_directory / "systems.json").write_text(
        json.dumps(
            {
                "measured": {
                    "charged_calls": 7_992_000,
                    "physical_actions": 7_992_000,
                    "reset_only_steps": 0,
                }
            }
        )
    )
    with pytest.raises(ContractError, match="Every primary fit"):
        require_primary_training_complete(study, contract, tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("condition", CELLS)
def test_medium_cpu_lifecycle_with_three_writes_and_shared_probe(
    condition: str, tmp_path: Path
) -> None:
    study = load_summary_memory_study()
    directory = train(
        study,
        benchmark="count_recall",
        condition=condition,
        seed=42,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    assert isinstance(directory, Path)
    measured = json.loads((directory / "systems.json").read_text())["measured"]
    assert measured["charged_calls"] == measured["physical_actions"] > 0
    panel = evaluate(
        study,
        benchmark="count_recall",
        condition=condition,
        seed=42,
        device="cpu",
        output_root=tmp_path,
        task_cap=2,
    )
    raw = json.loads((panel / "benchmark_results.json").read_text())
    assert len(raw["events"]) == 206
    if condition != "full_dual_relational":
        assert [row["writes_before_decision"] for row in raw["events"][:103]] == [
            (i - 1) // 32 for i in range(1, 104)
        ]
    contract, config = resolve(
        study,
        benchmark="count_recall",
        condition=condition,
        seed=42,
        device="cpu",
        output_root=tmp_path,
    )
    config = saved_config(config)
    experiment = load_experiment(config, work_directory=directory)
    modes = ("retained", "current-token") + (
        ("summary-cleared",) if condition == "fixed_summary" else ()
    )
    signatures = []
    try:
        for history in modes:
            probe = scripted_count_recall(
                experiment,
                contract,
                config,
                split="development",
                checkpoint="checkpoint.pt",
                history=history,
                task_cap=2,
            )
            assert probe["charged_calls"] == 206
            signatures.append([row["input_sha256"] for row in probe["queries"]])
    finally:
        close_experiment(experiment)
    assert all(signature == signatures[0] for signature in signatures)


def test_medium_references_roundtrip_both_frozen_priors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reasoned_icrl.experiments import count_recall_controls
    from reasoned_icrl.experiments.summary_memory.revised import (
        measure_references,
        read_references,
        write_references,
    )

    prior = fit_count_time_prior(tuple(range(12)))
    monkeypatch.setattr(
        count_recall_controls, "frozen_count_time_prior", lambda root: prior
    )
    study = load_summary_memory_study()
    contract, config = resolve(
        study,
        benchmark="count_recall",
        condition=CELLS[0],
        seed=42,
        output_root=tmp_path,
    )
    panel = measure_references(contract, config, task_cap=2)
    assert panel.random_level == pytest.approx(1 / 27)
    assert panel.reactive is not None and len(panel.additional) == 1
    assert panel.level == max(
        panel.random_level, panel.reactive_level, panel._mean(panel.additional[0])
    )
    assert (
        panel.reactive.charged_calls == 206 and panel.additional[0].charged_calls == 0
    )
    write_references(tmp_path, panel)
    assert read_references(tmp_path, contract.protocol, "development") == panel
