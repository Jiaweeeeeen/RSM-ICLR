"""R7 acceptance: public Easy traces, references, full budgets and four carriers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from reasoned_icrl.analysis.statistics import concentration_flip_curves
from reasoned_icrl.environments.concentration import ConcentrationEnv
from reasoned_icrl.experiments.contracts import ContractError, ResultValidationError
from reasoned_icrl.experiments.evaluation import (
    ConcentrationResult,
    events,
    visible_board_action,
)
from reasoned_icrl.experiments.records import (
    BenchmarkRun,
    read_benchmark_results,
    write_benchmark_results,
)
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
from reasoned_icrl.experiments.summary_memory.revised import (
    measure_references,
    read_references,
    write_references,
)

ROOT = Path(__file__).resolve().parents[3]
CELLS = ("full_context", "full_dual_relational", "full_dual_content", "full_gru")


def test_r7_roster_budget_and_official_deck(tmp_path: Path) -> None:
    study = load_summary_memory_study()
    contract = study.contract("concentration")
    plans = plan_fits(study, contract, repository=ROOT, output_root=tmp_path)
    assert [(p.condition, p.seed) for p in plans] == [
        (c, s) for c in CELLS for s in (42, 100, 2026)
    ]
    for condition in CELLS:
        _, config = resolve(
            study, benchmark="concentration", condition=condition, seed=42, wandb=True
        )
        assert config.tracking.wandb
        assert (
            config.training.epochs
            * config.training.timesteps_per_epoch
            * config.environment.parallel_envs
            == 8_000_000
        )
        assert (config.environment.size, config.environment.horizon) == (52, 104)
        assert config.training.mixed_precision == "no"
    assert contract.memory == {
        "summary": {"segment_length": 32, "memory_tokens": 4},
        "window": {"segment_length": 40},
    }
    assert contract.evaluation.retention == "complete"
    assert not set(contract.roster("development")) & set(contract.roster("final"))
    assert min(contract.roster("final")) == 4_000_000
    _, smoke = resolve(
        study,
        benchmark="concentration",
        condition="full_context",
        seed=42,
        smoke=True,
        wandb=True,
    )
    assert smoke.tracking.wandb and smoke.tracking.group.endswith("/smoke")


def test_references_keep_actual_board_scores_and_clocks(tmp_path: Path) -> None:
    study = load_summary_memory_study()
    contract, config = resolve(
        study, benchmark="concentration", condition="full_context", seed=42
    )
    panel = measure_references(contract, config, task_cap=4)
    assert panel.reactive is not None
    for reference in (panel.random, panel.reactive):
        assert reference.generator_seeds == (0, 1, 2)
        assert (
            reference.pair_fractions is not None and len(reference.pair_fractions) == 4
        )
        assert 0 < reference.charged_calls <= 4 * 3 * 104
        assert reference.physical_actions == reference.charged_calls
        assert not reference.doors_completed
    write_references(tmp_path, panel)
    assert read_references(tmp_path, contract.protocol, "development") == panel
    assert read_references(tmp_path, contract.protocol, "final") is None
    assert (
        visible_board_action(
            np.array([0, 1, 2, 0]), sentinel=2, generator=np.random.default_rng(0)
        )
        == 2
    )


def test_complete_flip_roundtrip_curve_and_tamper_refusal(tmp_path: Path) -> None:
    study = load_summary_memory_study()
    contract, config = resolve(
        study, benchmark="concentration", condition="full_context", seed=42
    )
    split = replace(contract.evaluation.splits["development"], count=2)
    contract = replace(
        contract, evaluation=replace(contract.evaluation, splits={"development": split})
    )
    results = []
    env = ConcentrationEnv(variant="easy", split="development")
    try:
        for task in contract.roster("development"):
            env.set_task(task)
            env.reset()
            # Public-history solver: first sweep, then known hidden partners.
            done = False
            for position in range(52):
                _, _, term, trunc, _ = env.step(position)
                done = term or trunc
                if done:
                    break
            while not done:
                known = env._board.known_hidden()
                first, second = next(
                    (a, b)
                    for a in known
                    for b in known
                    if a != b and known[a] == known[b]
                )
                env.step(first)
                _, _, term, trunc, _ = env.step(second)
                done = term or trunc
            results.append(
                ConcentrationResult(
                    task,
                    0,
                    env.flips,
                    env.matched_pairs,
                    env.pairs,
                    env.episode_return,
                    len(env.flips),
                )
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
    run = BenchmarkRun(
        contract.protocol,
        "concentration",
        "full_context",
        42,
        "fixture",
        "development",
        "retained",
        "completed",
        retention="complete",
        metric="pair_fraction",
    )
    path = tmp_path / "results.json"
    write_benchmark_results(path, [contract], [run], rows)
    assert read_benchmark_results(path, [contract])[1] == tuple(rows)
    curve = concentration_flip_curves(rows, samples=0)
    assert curve[0].estimate == 0 and curve[-1].estimate == 1
    latest = max(e.step for e in rows)
    assert latest < 104
    assert all(p.estimate == 1 for p in curve if p.step >= latest)
    assert all(p.tasks == 2 for p in curve)
    with pytest.raises(ResultValidationError, match="cover every decision"):
        write_benchmark_results(
            path,
            [contract],
            [run],
            [replace(rows[0], flips=rows[0].flips[:-1]), rows[1]],
        )
    with pytest.raises(ResultValidationError, match="complete flip traces"):
        concentration_flip_curves([replace(rows[0], flips=None)])
    # Equal and opposite corruption keeps the terminal native return intact;
    # validation must still reject the incorrect per-flip reward timing.
    flips = list(rows[0].flips)
    flips[0] = replace(flips[0], native_reward=flips[0].native_reward + 0.1)
    flips[1] = replace(flips[1], native_reward=flips[1].native_reward - 0.1)
    with pytest.raises(ResultValidationError, match="Flip reward"):
        write_benchmark_results(
            path, [contract], [run], [replace(rows[0], flips=tuple(flips)), rows[1]]
        )


def test_endpoint_requires_all_twelve_measured_budgets(tmp_path: Path) -> None:
    study = load_summary_memory_study()
    contract = study.contract("concentration")
    with pytest.raises(ContractError, match="Every primary fit"):
        require_primary_training_complete(study, contract, tmp_path)
    for plan in plan_fits(study, contract, repository=ROOT, output_root=tmp_path):
        root = plan.run_directory
        weights = root / "ckpts/policy_weights/policy_epoch_999.pt"
        weights.parent.mkdir(parents=True)
        weights.touch()
        (root / "checkpoint.pt").touch()
        (root / "metrics.json").write_text(
            json.dumps({"status": "trained", "scalar_training_transitions": 8_000_000})
        )
        (root / "systems.json").write_text(
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
        assert completed_training_budget(root, contract) == 8_000_000
    require_primary_training_complete(study, contract, tmp_path)
    (root / "systems.json").write_text(
        json.dumps(
            {
                "measured": {
                    "charged_calls": 7_999_999,
                    "physical_actions": 7_999_999,
                    "reset_only_steps": 0,
                }
            }
        )
    )
    with pytest.raises(ContractError, match="inconsistent measured"):
        require_primary_training_complete(study, contract, tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("condition", CELLS)
def test_r7_cpu_lifecycle(condition: str, tmp_path: Path) -> None:
    study = load_summary_memory_study()
    directory = train(
        study,
        benchmark="concentration",
        condition=condition,
        seed=42,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    assert isinstance(directory, Path)
    measured = json.loads((directory / "systems.json").read_text())["measured"]
    assert measured["charged_calls"] > 0
    panel = evaluate(
        study,
        benchmark="concentration",
        condition=condition,
        seed=42,
        device="cpu",
        output_root=tmp_path,
        split="development",
        task_cap=2,
    )
    assert panel.is_dir()
    raw = json.loads((panel / "benchmark_results.json").read_text())
    assert all(len(row["flips"]) == row["step"] for row in raw["events"])


@pytest.mark.slow
def test_scripted_probe_preserves_identical_public_histories(tmp_path: Path) -> None:
    from reasoned_icrl.experiments.benchmarks import saved_config
    from reasoned_icrl.runtime.rollout import scripted_concentration_retrieval
    from reasoned_icrl.runtime.training import (
        close_experiment,
        load_experiment,
        load_selected_checkpoint,
    )

    study = load_summary_memory_study()
    observed = []
    for condition in ("full_context", "full_gru"):
        train(
            study,
            benchmark="concentration",
            condition=condition,
            seed=42,
            device="cpu",
            output_root=tmp_path,
            smoke=True,
        )
        contract, config = resolve(
            study,
            benchmark="concentration",
            condition=condition,
            seed=42,
            device="cpu",
            output_root=tmp_path,
        )
        config = saved_config(config)
        experiment = load_experiment(config, work_directory=config.run_directory)
        try:
            load_selected_checkpoint(experiment, "checkpoint.pt")
            probe = scripted_concentration_retrieval(
                experiment,
                contract,
                config,
                split="development",
                checkpoint="checkpoint.pt",
                task_cap=2,
            )
        finally:
            close_experiment(experiment)
        assert probe["charged_calls"] == 208
        assert probe["opportunity_count"] > 0
        observed.append(
            [
                {
                    k: row[k]
                    for k in ("task_id", "flip", "eligible", "evidence_age_flips")
                }
                for row in probe["opportunities"]
            ]
        )
    assert observed[0] == observed[1]
