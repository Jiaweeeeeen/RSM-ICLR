"""Bounded CPU training lifecycles of the XLand one-rule application.

Two cells of the one-rule roster run through the shared trainer on CPU with
the smoke profile: they train over the curriculum (a warmup lifetime, then a
primary one), tag their replay files by pool, checkpoint, and record the
per-pool exposure beside the shared counters; an interrupted fit resumes to
the uninterrupted weights across a lifetime boundary and a pool change. The
evaluator stage is not exercised here: the contract's three rollout roots and
its sampled-policy endpoint wait for that stage. Passing establishes execution
integrity only, never learning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasoned_icrl.experiments.summary_memory import experiments as summary_memory
from reasoned_icrl.experiments.summary_memory.configs import (
    load_xland_one_rule_study,
)
from reasoned_icrl.experiments.xland_one_rule import POOLS
from reasoned_icrl.runtime.replay import replay_pool

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = "xland_one_rule"


@pytest.mark.slow
@pytest.mark.parametrize(
    "condition",
    (
        "full_context",
        "fixed_summary",
        # The figure-set supplements run the same smoke on the contract before
        # any of them is launched.
        "full_gru",
        "raw_summary",
        "raw_segment",
        "memo",
        "memo_fixed",
    ),
)
def test_xland_one_rule_cpu_smoke_lifecycle(condition: str, tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("xminigrid")
    study = load_xland_one_rule_study()
    run = summary_memory.train(
        study,
        benchmark=BENCHMARK,
        condition=condition,
        seed=0,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    assert isinstance(run, Path)
    assert run.parent.parent.name == study.contract(BENCHMARK).protocol
    assert Path(run, "provenance.json").is_file()
    systems = json.loads(Path(run, "systems.json").read_text())
    measured = systems["measured"]
    outer = 5 * (8 + 1) - 1  # the smoke profile shortens attempts to 8 actions
    assert measured["charged_calls"] == 2 * outer
    assert measured["warmup_tasks_started"] == 1
    assert measured["primary_tasks_started"] >= 1
    assert measured["warmup_charged_calls"] == outer
    assert (
        measured["warmup_charged_calls"] + measured["primary_charged_calls"]
        == (measured["charged_calls"])
    )
    assert measured["validation"]["warmup_tasks_started"] == 0
    files = sorted(Path(run, "replay").rglob("*.npz"))
    pools = sorted(replay_pool(str(f)) or "" for f in files)
    assert set(pools) == set(POOLS)
    assert (run / "ckpts" / "policy_weights").is_dir()
    assert (run / "config.yaml").is_file() and (run / "metrics.json").is_file()
    saved = Path(run, "config.yaml").read_text()
    assert "curriculum" in saved and "xland-r1-9x9-one-rule" in saved


@pytest.mark.slow
def test_xland_one_rule_exact_resume_across_a_lifetime_and_a_pool_change(
    tmp_path: Path,
) -> None:
    """An interrupted fit resumes to the uninterrupted weights and replay.

    Two actors, an epoch that ends inside the second lifetime (after a
    warmup lifetime completed and a mixed-phase lifetime began), and a
    curriculum whose eligible pools include both, so the learner always has
    a file to sample. The resumed run restores the environment (its pool,
    layout, row and counters), the replay order and the learner state.
    """
    pytest.importorskip("jax")
    pytest.importorskip("xminigrid")
    from dataclasses import replace

    import torch

    from reasoned_icrl.experiments.benchmarks import experiment_config
    from reasoned_icrl.runtime.training import close_experiment

    from ..test_experiment import _started_experiment

    study = load_xland_one_rule_study()
    cfg = experiment_config(
        study.contract(BENCHMARK),
        study,
        condition="fixed_summary",
        seed=0,
        repository=ROOT,
        device="cpu",
        output_root=tmp_path / "whole",
        smoke=True,
    )
    outer = cfg.environment.outer_length
    interruption = outer + 20
    cfg = replace(
        cfg,
        environment=replace(
            cfg.environment,
            parallel_envs=2,
            curriculum={
                "warmup_calls": 2 * outer,  # one warmup lifetime per actor
                "mixed_calls": 40 * outer,  # then mixed lifetimes: both pools eligible
                "mixed_probability": 0.5,
            },
        ),
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
    expected_counters = dict(whole.collection_counters)
    expected_updates = whole.grad_update_counter
    # Each actor's first lifetime is warmup; a warmup lifetime the random
    # policy solves early lets a second warmup lifetime start before the
    # actor's threshold, so the count is at least one per actor.
    assert expected_counters["warmup_tasks_started"] >= 2
    assert (
        expected_counters["primary_tasks_started"]
        + expected_counters["warmup_tasks_started"]
        == expected_counters["tasks_started"]
    )
    close_experiment(whole)
    cfg = replace(cfg, output_root=tmp_path / "resume")
    partial = _started_experiment(cfg, epoch_limit=1)
    partial.learn()
    partial_counters = dict(partial.collection_counters)
    assert partial_counters["charged_calls"] == 2 * interruption
    close_experiment(partial)
    resumed = _started_experiment(cfg)
    resumed.load_checkpoint(0, resume_training_state=True)
    resumed.epoch = 1
    resumed.learn()
    for name, value in expected.items():
        torch.testing.assert_close(
            resumed.policy.state_dict()[name].cpu(), value, rtol=1e-5, atol=1e-6
        )
    assert expected_updates == resumed.grad_update_counter
    assert expected_counters == dict(resumed.collection_counters)
    assert expected_replay == [
        Path(p).read_bytes() for p in resumed.reasoned_dataset.all_filenames
    ]
    close_experiment(resumed)
