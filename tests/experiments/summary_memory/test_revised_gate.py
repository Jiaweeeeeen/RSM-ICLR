"""R5 acceptance: references, the revised C2/C3 rules, development series and
panels of the 8M study.

Fixture series exercise the decision rules; a CPU smoke run exercises the
incremental development record, the endpoint refusal for a run that has not
reached its final epoch, the endpoint/selected panels and the tier dashboard.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from reasoned_icrl.experiments.artifacts import endpoint_label
from reasoned_icrl.experiments.benchmarks import EvaluationSplit, experiment_config
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import ReferenceRollout
from reasoned_icrl.experiments.qualification import TaskDiagnostic
from reasoned_icrl.experiments.summary_memory import experiments as summary_memory
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.revised import (
    C2_CHECKPOINTS,
    EARLY_WINDOW,
    LATE_WINDOW,
    MINIMUM_PAIRED_TASKS,
    MINIMUM_VERIFIED_PAIRS,
    ReferencePanel,
    collect_developments,
    decide_revised_c2,
    decide_revised_c3,
    develop_run,
    development_modes,
    finalize_revised_cell,
    gate_tier,
    measure_references,
    read_references,
    read_run_development,
    write_references,
)
from reasoned_icrl.utils import repository_root
from tests.analysis.test_revised_statistics import PROTOCOL, SEEDS, TASKS, _panel

ROOT = repository_root()
BENCHMARK = "dark_key_to_door"


def _small(task_count: int = 2) -> tuple[Any, Any]:
    study = load_summary_memory_study()
    contract = study.contract(BENCHMARK)
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
    return study, contract


def _rollout(name: str, doors: float, tasks: tuple[int, ...]) -> ReferenceRollout:
    return ReferenceRollout(
        name,
        (0, 1, 2),
        dict.fromkeys(tasks, doors),
        dict.fromkeys(tasks, min(doors / 8, 1.0)),
        len(tasks) * 500,
        len(tasks) * 400,
    )


def _references(random: float, reactive: float | None) -> ReferencePanel:
    return ReferencePanel(
        PROTOCOL,
        "development",
        "doors_completed",
        TASKS,
        _rollout("random policy (uniform actions)", random, TASKS),
        None if reactive is None else _rollout("sweep policy", reactive, TASKS),
        "2000-01-01T00:00:00+0000",
    )


def test_the_reference_level_is_the_larger_mean_and_round_trips(tmp_path: Path) -> None:
    panel = _references(0.5, 2.0)
    assert panel.random_level == 0.5 and panel.reactive_level == 2.0
    assert panel.level == 2.0
    assert _references(0.5, None).level == 0.5
    write_references(tmp_path, panel)
    again = read_references(tmp_path, PROTOCOL, "development")
    assert again == panel
    assert read_references(tmp_path, PROTOCOL, "final") is None
    assert C2_CHECKPOINTS == (900, 950, 999) and endpoint_label(1000) == 999
    assert LATE_WINDOW == (850, 900, 950, 999) and EARLY_WINDOW == (650, 700, 750, 800)


def test_revised_c2_needs_delta_over_the_level_at_every_late_checkpoint() -> None:
    _, contract = _small()
    references = _references(0.5, 2.0)  # level 2.0, delta 1.0 -> required 3.0
    good = {s: {900: 3.4, 950: 3.6, 999: 3.8} for s in SEEDS}
    decision = decide_revised_c2(
        contract, "full_context", good, references=references, delta=1.0
    )
    assert decision.passed and decision.complete and not decision.provisional
    assert decision.reference_level == 2.0 and decision.reasons == ()
    assert decision.seeds_clearing == {900: 3, 950: 3, 999: 3}
    one_low = {**good, 2026: {900: 2.0, 950: 2.0, 999: 2.0}}
    decision = decide_revised_c2(
        contract, "full_context", one_low, references=references, delta=1.0
    )
    # Across-seed means 2.93/3.07/3.2: epoch 900 misses 3.0; two seeds clear.
    assert not decision.passed and any("epoch 900" in r for r in decision.reasons)
    two_low = {**one_low, 100: {900: 2.0, 950: 2.0, 999: 2.0}}
    decision = decide_revised_c2(
        contract, "full_context", two_low, references=references, delta=1.0
    )
    assert not decision.passed and any("only 1 seed" in r for r in decision.reasons)
    pilot = {42: good[42]}
    decision = decide_revised_c2(
        contract, "full_context", pilot, references=references, delta=1.0
    )
    assert decision.provisional and not decision.complete and not decision.passed
    partial = {**good, 100: {900: 3.4}}
    decision = decide_revised_c2(
        contract, "full_context", partial, references=references, delta=1.0
    )
    assert not decision.complete and "only 2 of 3 seeds" in decision.reasons[0]
    payload = json.dumps(decision.as_dict())
    assert '"reference_level": 2.0' in payload


def _diagnostic(pairs: int, tasks: int) -> TaskDiagnostic:
    return TaskDiagnostic(
        BENCHMARK,
        "q",
        True,
        {"verified_pairs": float(pairs), "tasks_in_verified_pairs": float(tasks)},
        "fixture",
    )


def test_revised_c3_needs_coverage_and_a_conditional_loss_above_zero() -> None:
    _, contract = _small()
    retained = [
        replace(e, condition="full_context") for e in _panel({"full_context": 8})
    ]
    intervened = [
        replace(e, history="attempt-cleared") for e in _panel({"full_context": 6})
    ]
    decision = decide_revised_c3(
        contract,
        "full_context",
        retained=retained,
        intervened=intervened,
        diagnostic=_diagnostic(MINIMUM_VERIFIED_PAIRS, MINIMUM_PAIRED_TASKS),
        delta=1.0,
        samples=50,
    )
    assert decision.passed and decision.dependence == 2.0
    assert decision.positive_seeds == 3 and decision.coverage_passed
    assert decision.conditional_lower == 2.0 == decision.conditional_upper
    thin = decide_revised_c3(
        contract,
        "full_context",
        retained=retained,
        intervened=intervened,
        diagnostic=_diagnostic(10, 5),
        delta=1.0,
        samples=10,
    )
    assert not thin.passed and "coverage 10 verified pairs" in thin.reasons[0]
    no_loss = decide_revised_c3(
        contract,
        "full_context",
        retained=retained,
        intervened=[replace(e, history="attempt-cleared") for e in retained],
        diagnostic=_diagnostic(40, 20),
        delta=1.0,
        samples=10,
    )
    assert not no_loss.passed and no_loss.dependence == 0.0
    assert any("below delta/2" in r for r in no_loss.reasons)
    none = decide_revised_c3(
        contract,
        "full_context",
        retained=retained,
        intervened=intervened,
        diagnostic=TaskDiagnostic(BENCHMARK, "q", True, {}, "no coverage"),
        delta=1.0,
        samples=10,
    )
    assert not none.passed and "no verified-pair coverage" in none.reasons[0]
    with pytest.raises(ContractError, match="one seed/task roster"):
        decide_revised_c3(
            contract,
            "full_context",
            retained=retained,
            intervened=intervened[: len(intervened) // 2],
            diagnostic=_diagnostic(40, 20),
            delta=1.0,
            samples=10,
        )


def test_measured_references_are_evaluation_only_and_recorded(tmp_path: Path) -> None:
    study, contract = _small()
    config = experiment_config(
        contract, study, condition="full_context", seed=42, repository=ROOT
    )
    panel = measure_references(contract, config, task_cap=2, generator_seeds=(0, 1, 2))
    assert (
        panel.metric == "doors_completed"
        and panel.task_ids == contract.roster("development")[:2]
    )
    assert panel.random.generator_seeds == (0, 1, 2)
    assert panel.reactive is not None and panel.reactive.name.startswith("sweep")
    assert panel.level >= max(panel.random_level, 0.0)
    path = write_references(tmp_path, panel)
    assert path.name == f"{contract.protocol}-development.json"
    assert read_references(tmp_path, contract.protocol, "development") == panel


@pytest.mark.slow
def test_development_series_endpoint_and_panels_on_a_smoke_run(tmp_path: Path) -> None:
    """A smoke fit of the reference cell: the incremental development record,
    the endpoint panel after completion, the selected panel and the gate."""
    study, contract = _small()
    run = summary_memory.train(
        study,
        benchmark=BENCHMARK,
        condition="full_context",
        seed=42,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    assert isinstance(run, Path)
    config = experiment_config(
        contract,
        study,
        condition="full_context",
        seed=42,
        repository=ROOT,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    assert development_modes(contract, "full_context") == (
        "retained",
        "attempt-cleared",
    )
    assert development_modes(contract, "fixed_summary") == (
        "retained",
        "attempt-cleared",
        "summary-cleared",
    )
    record = develop_run(study, contract, config, task_cap=2)
    assert record.metric == "doors_completed" and record.task_cap == 2
    assert [s.epoch for s in record.series] == [1]  # label 0 precedes learning
    assert record.endpoint_epoch == 1 and record.endpoint_reached
    assert set(record.modes) == {"retained", "attempt-cleared"}
    assert read_run_development(run) == record
    # A second pass evaluates nothing new and keeps the selection.
    again = develop_run(study, contract, config, task_cap=2)
    assert (
        again.series == record.series and again.selected_epoch == record.selected_epoch
    )
    assert again.development_evaluation_seconds >= record.development_evaluation_seconds
    assert collect_developments(study, contract, tmp_path) == (again,)
    written = finalize_revised_cell(
        study, contract, config, split="development", task_cap=2
    )
    assert set(written) == {
        "retained-endpoint",
        "attempt-cleared-endpoint",
        "retained-selected",
    }
    assert written["retained-endpoint"].name == "development-retained-endpoint"
    payload = json.loads(
        (written["retained-endpoint"] / "benchmark_results.json").read_text()
    )
    assert payload["runs"][0]["checkpoint"] == "policy_epoch_1"
    assert payload["runs"][0]["checkpoint_rule"] == "endpoint"
    assert payload["runs"][0]["retention"] == "complete"
    assert payload["runs"][0]["charged_calls"] == 2 * 72
    assert all("complete" in e for e in payload["events"])
    kept = finalize_revised_cell(
        study, contract, config, split="development", task_cap=2
    )
    assert kept == written
    # A run whose endpoint weights are missing is incomplete for that panel.
    (run / "ckpts" / "policy_weights" / "policy_epoch_1.pt").rename(
        run / "ckpts" / "policy_weights" / "policy_epoch_1.bak"
    )
    with pytest.raises(ContractError, match="incomplete"):
        finalize_revised_cell(
            study, contract, config, split="development", task_cap=2, refresh=True
        )
    (run / "ckpts" / "policy_weights" / "policy_epoch_1.bak").rename(
        run / "ckpts" / "policy_weights" / "policy_epoch_1.pt"
    )
    gate = gate_tier(
        study,
        contract,
        tmp_path,
        repository=ROOT,
        device="cpu",
        task_cap=2,
        checkpoints=(1,),
        late=(1,),
        early=(1,),
        samples=10,
    )
    assert gate.reference_condition == "full_context"
    assert gate.c2.provisional and not gate.c2.complete
    # The unscored windows leave the plateau incomplete and the JSON finite.
    assert all(not v.complete for v in gate.plateau.values())
    assert gate.status == "C2 provisional" and gate.c3 is None  # capped roster
    dashboard = tmp_path / f"qualification-{contract.protocol}-8m.md"
    assert dashboard.is_file() and "Tier qualification" in dashboard.read_text()
    assert json.loads(dashboard.with_suffix(".json").read_text())["status"] == (
        "C2 provisional"
    )
    # A capped pass measures its references in memory and never overwrites the
    # declared full-roster file.
    assert read_references(tmp_path, contract.protocol, "development") is None


@pytest.mark.slow
def test_a_summary_cell_writes_every_mode_with_valid_records(tmp_path: Path) -> None:
    """The intervention panels of a summary cell validate on the full smoke
    roster: the summary write bookkeeping is recorded under retained and
    summary-cleared rollouts and left out under attempt-cleared, whose cache
    resets restart the boundary counter."""
    study, contract = _small()
    summary_memory.train(
        study,
        benchmark=BENCHMARK,
        condition="fixed_summary",
        seed=42,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    config = experiment_config(
        contract,
        study,
        condition="fixed_summary",
        seed=42,
        repository=ROOT,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    # The smoke profile shortens the outer task to 72 calls; the validator
    # reads the outer length from the contract, so the contract follows it.
    contract = replace(contract, environment=config.environment)
    record = develop_run(study, contract, config)  # the whole 64-task roster
    assert set(record.modes) == {"retained", "attempt-cleared", "summary-cleared"}
    from reasoned_icrl.experiments.summary_memory.revised import panel_events

    run = config.run_directory
    retained = panel_events(
        run, contract, split="development", history="retained", rule="selected"
    )
    cleared = panel_events(
        run, contract, split="development", history="attempt-cleared", rule="selected"
    )
    summary_cleared = panel_events(
        run, contract, split="development", history="summary-cleared", rule="selected"
    )
    assert len(retained) == len(cleared) == len(summary_cleared) > 0
    assert all(e.writes_before_decision is not None for e in retained)
    assert all(e.writes_before_decision is not None for e in summary_cleared)
    assert all(
        e.writes_before_decision is None and e.evidence_age_writes is None
        for e in cleared
    )
    # An untrained smoke policy rarely finds the key, so ages may all be absent;
    # whenever one exists it lies within the writes before the decision.
    assert all(
        e.evidence_age_writes is None
        or (
            e.writes_before_decision is not None
            and 0 <= e.evidence_age_writes <= e.writes_before_decision
        )
        for e in retained
    )
