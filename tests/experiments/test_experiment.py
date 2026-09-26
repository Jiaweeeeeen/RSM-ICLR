"""Regression tests for the official-style AMAGO integration boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import gin
import pytest
import torch

from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.config import CriticConfig
from reasoned_icrl.experiments.contracts import HISTORY_CONDITIONS, ContractError
from reasoned_icrl.model.step_encoder import TransitionTstepEncoder
from reasoned_icrl.model.trajectory_encoder import HistoryTrajEncoder
from reasoned_icrl.runtime.amago import configure_amago
from reasoned_icrl.runtime.environments import environment_builders
from reasoned_icrl.runtime.experiment import (
    ReasonedExperiment,
    build_experiment,
    persist_operative_gin,
    start_experiment,
)
from reasoned_icrl.runtime.replay import create_replay_dataset
from reasoned_icrl.runtime.training import (
    close_experiment,
    seed_everything,
    train_experiment,
)
from tests.experiments.fixtures import load_fixture_study

ROOT = Path(__file__).resolve().parents[2]


def darkroom_config(condition: str, **overrides: object):
    study = load_fixture_study("stage1")
    return experiment_config(
        study.contract("darkroom"),
        study,
        condition=condition,
        seed=0,
        repository=ROOT,
        smoke=True,
        **overrides,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("device_type", "expected"),
    (("cpu", False), ("mps", False), ("cuda", True)),
)
def test_replay_loader_pins_memory_only_for_cuda(
    device_type: str, expected: bool
) -> None:
    experiment = object.__new__(ReasonedExperiment)
    experiment.dataset = [object()]
    experiment.batch_size = 1
    experiment.dloader_workers = 0
    experiment.accelerator = SimpleNamespace(
        device=torch.device(device_type), prepare=lambda loader: loader
    )

    loader = experiment.init_dloaders()

    assert loader.pin_memory is expected


def test_config_translation_uses_history_switches_and_optional_critic() -> None:
    standard = darkroom_config("transition_bypass").as_runtime_mapping()
    standard["model"]["public_contract"] = {"schema": "test-public-contract.v1"}
    components = configure_amago(standard["model"], standard["training"])
    assert components.timestep_encoder is TransitionTstepEncoder
    assert components.architecture_id == "amago-history-v1"
    assert components.trajectory_encoder is HistoryTrajEncoder
    # Every baseline condition now uses component-isolated initialization.
    assert components.agent.__name__ == "MatchedBaselineAgent"
    assert components.exploration.__module__ == "reasoned_icrl.runtime.amago"
    assert (
        gin.query_parameter(
            "reasoned_icrl.runtime.amago.SeededBilevelEpsilonGreedy.rollout_horizon"
        )
        == standard["training"]["exploration_rollout_horizon"]
    )
    assert (
        gin.query_parameter(
            "reasoned_icrl.runtime.amago.SeededBilevelEpsilonGreedy.steps_anneal"
        )
        == standard["training"]["epsilon_anneal_steps"]
    )
    with pytest.raises(ValueError, match="no bound parameters"):
        gin.query_parameter("amago.nets.actor_critic.NCriticsTwoHot.output_bins")

    feedforward = darkroom_config("feedforward").as_runtime_mapping()
    components = configure_amago(feedforward["model"], feedforward["training"])
    assert components.timestep_encoder.__name__ == "FFTstepEncoder"
    assert components.trajectory_encoder.__name__ == "FFTrajEncoder"

    with_critic = darkroom_config("raw")
    with_critic = replace(
        with_critic,
        model=replace(with_critic.model, critic=CriticConfig(-1500.0, 1500.0, 32)),
    ).as_runtime_mapping()
    with_critic["model"]["public_contract"] = {"schema": "test-public-contract.v1"}
    components = configure_amago(with_critic["model"], with_critic["training"])
    assert gin.query_parameter(
        "amago.nets.actor_critic.NCriticsTwoHot.min_return"
    ) == pytest.approx(-1500.0)
    assert gin.query_parameter(
        "amago.nets.actor_critic.NCriticsTwoHot.max_return"
    ) == pytest.approx(1500.0)
    assert (
        gin.query_parameter("amago.nets.actor_critic.NCriticsTwoHot.output_bins") == 32
    )

    gin.clear_config()


def test_ordered_replay_roundtrip_quarantines_post_checkpoint_files(
    tmp_path: Path,
) -> None:
    dataset = create_replay_dataset(tmp_path, capacity=2)
    fifo = Path(dataset.fifo_path)
    for name in ("Env_a_3.0.npz", "Env_b_1.0.npz"):
        (fifo / name).write_bytes(b"trajectory")
    dataset._refresh_files()
    assert [Path(value).name for value in dataset.all_filenames] == [
        "Env_b_1.0.npz",
        "Env_a_3.0.npz",
    ]
    checkpoint = dataset.state_dict()

    extra = fifo / "Env_c_4.0.npz"
    extra.write_bytes(b"partial-next-epoch")
    dataset._refresh_files()
    dataset.load_state_dict(checkpoint)
    assert [Path(value).name for value in dataset.all_filenames] == [
        "Env_b_1.0.npz",
        "Env_a_3.0.npz",
    ]
    assert not extra.exists()
    assert (
        tmp_path / "replay" / "orphaned-after-checkpoint" / extra.name
    ).read_bytes() == b"partial-next-epoch"


def test_ordered_replay_refuses_evicted_files_unless_the_deviation_is_recorded(
    tmp_path: Path,
) -> None:
    """A pack that dies after its last training state has let the FIFO evict
    that state's oldest files: the exact resume is refused with the count, and
    the explicit lenient resume drops them, keeps the survivors' order,
    quarantines the newer files and records the deviation (R6)."""
    dataset = create_replay_dataset(tmp_path, capacity=3)
    fifo = Path(dataset.fifo_path)
    for name in ("Env_a_1.0.npz", "Env_b_2.0.npz", "Env_c_3.0.npz"):
        (fifo / name).write_bytes(b"trajectory")
    dataset._refresh_files()
    checkpoint = dataset.state_dict()
    (fifo / "Env_a_1.0.npz").unlink()  # evicted after the checkpoint was saved
    newer = fifo / "Env_d_4.0.npz"
    newer.write_bytes(b"collected-after-the-checkpoint")
    dataset._refresh_files()
    with pytest.raises(ContractError, match=r"missing \(1 of 3\)"):
        dataset.load_state_dict(checkpoint)
    assert dataset.resume_deviation is None
    assert newer.exists()
    dataset.load_state_dict(checkpoint, allow_missing=True)
    assert [Path(value).name for value in dataset.all_filenames] == [
        "Env_b_2.0.npz",
        "Env_c_3.0.npz",
    ]
    assert dataset.resume_deviation == {
        "schema": "replay-resume-deviation.v1",
        "expected_files": 3,
        "missing_files": 1,
        "missing": ["Env_a_1.0.npz"],
    }
    assert not newer.exists()
    assert (tmp_path / "replay" / "orphaned-after-checkpoint" / newer.name).exists()
    # An exact resume leaves no deviation behind.
    dataset.load_state_dict(dataset.state_dict(), allow_missing=True)
    assert dataset.resume_deviation is None


def test_ordered_replay_capacity_removes_the_oldest_trajectory(tmp_path: Path) -> None:
    dataset = create_replay_dataset(tmp_path, capacity=2)
    fifo = Path(dataset.fifo_path)
    for index in range(3):
        (fifo / f"Env_{index}_{float(index)}.npz").write_bytes(b"trajectory")
    dataset._refresh_files()
    dataset._filter()
    assert [Path(value).name for value in dataset.all_filenames] == [
        "Env_1_1.0.npz",
        "Env_2_2.0.npz",
    ]


def test_invalid_baseline_critic_is_rejected() -> None:
    with pytest.raises(ContractError, match="smaller"):
        CriticConfig(1500.0, -1500.0, 32)


def test_the_wandb_switch_names_the_group_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`wandb=True` mirrors telemetry under `<study>/<protocol>`; the project comes
    from the environment; explicit smoke opt-in uses a separate group; the
    resolved config round-trips unchanged."""
    from reasoned_icrl.experiments.config import dump_config, load_resolved_config

    study = load_fixture_study("stage1")
    contract = study.contract("darkroom")
    off = experiment_config(
        contract, study, condition="raw", seed=0, repository=ROOT, output_root=tmp_path
    )
    assert off.tracking.wandb is False
    assert off.tracking.group == f"stage1/{contract.protocol}"
    monkeypatch.setenv("REASONED_ICRL_WANDB_PROJECT", "reasoned-icrl-test")
    on = experiment_config(
        contract,
        study,
        condition="raw",
        seed=0,
        repository=ROOT,
        output_root=tmp_path,
        wandb=True,
    )
    assert on.tracking.wandb is True
    assert on.tracking.project == "reasoned-icrl-test"
    assert on.tracking.group == f"stage1/{contract.protocol}"
    assert on.as_runtime_mapping()["tracking"]["wandb"] is True
    path = dump_config(on, tmp_path / "config.yaml")
    assert load_resolved_config(path, repository=ROOT) == on
    smoke = darkroom_config("raw", output_root=tmp_path, wandb=True)
    assert smoke.tracking.wandb is True
    assert smoke.tracking.group == f"stage1/{contract.protocol}/smoke"
    assert darkroom_config("raw", output_root=tmp_path).tracking.wandb is False


def test_the_wandb_run_is_named_by_protocol_condition_and_seed(
    tmp_path: Path,
) -> None:
    config = darkroom_config("feedforward", output_root=tmp_path, device="cpu")
    mapping = config.as_runtime_mapping()
    training, validation = environment_builders(mapping)
    experiment = build_experiment(
        config=mapping,
        run_directory=config.run_directory,
        make_train_env=training,
        make_validation_env=validation,
    )
    try:
        assert isinstance(experiment, ReasonedExperiment)
        assert experiment.wandb_run_name == (
            f"{config.environment.benchmark}/feedforward/seed-0"
        )
        assert experiment.wandb_tags == ("feedforward",)
        assert experiment.wandb_config["reasoned_icrl/seed"] == 0
        assert experiment.wandb_group_name == config.tracking.group
    finally:
        close_experiment(experiment)


@pytest.mark.slow
def test_a_tracked_lifecycle_records_the_run_in_its_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the tracker on (in wandb's disabled mode, so no network or login),
    the run starts, trains, finishes and names its wandb run in provenance."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")
    config = _tiny_training_config(tmp_path / "tracked")
    config = replace(config, tracking=replace(config.tracking, wandb=True))
    train_experiment(config)
    run = config.run_directory
    payload = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    assert payload["host"]
    assert payload["gpu"] is None
    assert isinstance(payload["wandb"], dict)
    assert set(payload["wandb"]) == {"id", "name", "project", "url"}
    assert (run / "wandb_logs").is_dir()
    assert (run / "checkpoint.pt").is_file()


def _tiny_training_config(output_root: Path):
    config = replace(darkroom_config("feedforward", output_root=output_root), seed=7)
    return replace(
        config,
        environment=replace(config.environment, attempts=1, horizon=4, parallel_envs=2),
        training=replace(
            config.training,
            epochs=2,
            start_learning_epoch=0,
            timesteps_per_epoch=16,
            batches_per_epoch=1,
            validation_timesteps=4,
            validation_interval=1,
            checkpoint_interval=1,
            batch_size=2,
            max_sequence_length=4,
            trajectory_length=4,
            replay_capacity=64,
            epsilon_anneal_steps=32,
            warmup_steps=1,
        ),
    )


def _started_experiment(config, *, epoch_limit: int | None = None):
    mapping = config.as_runtime_mapping()
    seed_everything(config.seed)
    training, validation = environment_builders(mapping)
    experiment = build_experiment(
        config=mapping,
        run_directory=config.run_directory,
        make_train_env=training,
        make_validation_env=validation,
    )
    start_experiment(experiment, mapping, checkpoint_runtime=True)
    if epoch_limit is not None:
        experiment.epochs = epoch_limit
    return experiment


@pytest.mark.slow
def test_cpu_checkpoint_resume_restores_training_and_replay_state(
    tmp_path: Path,
) -> None:
    uninterrupted_config = _tiny_training_config(tmp_path / "uninterrupted")
    uninterrupted = _started_experiment(uninterrupted_config)
    operative = persist_operative_gin(uninterrupted, uninterrupted_config.run_directory)
    operative_text = operative.read_text(encoding="utf-8")
    assert "# Encoder architecture: amago-ff-tstep-v3.4.0" in operative_text
    assert "FFTstepEncoder.d_hidden = 128" in operative_text
    assert "FFTstepEncoder.d_output = 64" in operative_text
    assert (
        uninterrupted.encoder_architecture_id
        == uninterrupted_config.model.architecture_id
    )
    uninterrupted.learn()
    telemetry = [
        json.loads(line)
        for line in (uninterrupted_config.run_directory / "training_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["panel"] for row in telemetry} >= {
        "dataset",
        "train-rollout",
        "train-update",
        "val",
    }
    expected_policy = {
        name: value.detach().cpu().clone()
        for name, value in uninterrupted.policy.state_dict().items()
    }
    expected_scheduler = uninterrupted.lr_schedule.state_dict()
    expected_updates = uninterrupted.grad_update_counter
    expected_replay = [
        Path(filename).read_bytes()
        for filename in uninterrupted.reasoned_dataset.all_filenames
    ]
    close_experiment(uninterrupted)

    resumed_config = _tiny_training_config(tmp_path / "resumed")
    interrupted = _started_experiment(resumed_config, epoch_limit=1)
    interrupted.learn()
    training_state = (
        resumed_config.run_directory
        / "ckpts/training_states"
        / f"{resumed_config.run_directory.name}_epoch_0"
    )
    assert (training_state / "custom_checkpoint_0.pkl").is_file()
    epoch_policy = {
        name: value.detach().clone()
        for name, value in interrupted.policy.state_dict().items()
    }
    with torch.no_grad():
        next(interrupted.policy.parameters()).add_(1.0)
    interrupted.load_checkpoint(0, resume_training_state=False)
    for name, expected in epoch_policy.items():
        torch.testing.assert_close(
            interrupted.policy.state_dict()[name], expected, rtol=0, atol=0
        )
    close_experiment(interrupted)

    resumed = _started_experiment(resumed_config)
    resumed.load_checkpoint(0, resume_training_state=True)
    resumed.epoch = 1
    resumed.learn()

    for name, expected in expected_policy.items():
        # PyTorch's macOS CPU kernels can vary in low-order bits across fresh
        # allocations even with deterministic algorithms enabled. The restored
        # state, batches, and update counts remain exact.
        torch.testing.assert_close(
            resumed.policy.state_dict()[name].cpu(),
            expected,
            rtol=1e-6,
            atol=1e-7,
        )
    assert resumed.lr_schedule.state_dict() == expected_scheduler
    assert resumed.grad_update_counter == expected_updates
    actual_replay = [
        Path(filename).read_bytes()
        for filename in resumed.reasoned_dataset.all_filenames
    ]
    assert actual_replay == expected_replay
    close_experiment(resumed)


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_runtime")
def test_cuda_smoke_supports_bf16_compile(tmp_path: Path) -> None:
    config = darkroom_config("transition_bypass", device="cuda", output_root=tmp_path)
    config = replace(
        config,
        training=replace(config.training, mixed_precision="bf16", torch_compile=True),
    )
    mapping = config.as_runtime_mapping()
    seed_everything(config.seed)
    training, validation = environment_builders(mapping)
    experiment = build_experiment(
        config=mapping,
        run_directory=config.run_directory,
        make_train_env=training,
        make_validation_env=validation,
    )
    assert start_experiment(experiment, mapping, checkpoint_runtime=False) == "cuda"
    value = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(lambda tensor: tensor @ tensor)
    assert compiled(value).shape == value.shape
    torch.cuda.synchronize()
    close_experiment(experiment)


# The Stage-1 history conditions (kept as contract rows) resume exactly from a
# live prefix: policy, replay files, update counter and schedule all match a
# whole run. Moved here from the retired Stage-1 lifecycle suite.


@pytest.mark.slow
@pytest.mark.parametrize("condition", HISTORY_CONDITIONS)
def test_transition_history_exact_resume_with_live_prefix(
    condition: str, tmp_path: Path
) -> None:
    _resume_case(condition, tmp_path, "cpu")


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.usefixtures("cuda_runtime")
def test_transition_history_cuda_resume_with_live_prefix(tmp_path: Path) -> None:
    _resume_case("transition_bypass", tmp_path, "cuda")


def _resume_case(condition: str, tmp_path: Path, device: str) -> None:
    study = load_fixture_study("stage1")
    cfg = experiment_config(
        study.contract("darkroom"),
        study,
        condition=condition,
        seed=0,
        repository=ROOT,
        device=device,
        output_root=tmp_path / "whole",
        smoke=True,
    )
    cfg = replace(
        cfg,
        environment=replace(cfg.environment, horizon=2, parallel_envs=2),
        training=replace(
            cfg.training,
            epochs=3,
            start_learning_epoch=0,
            timesteps_per_epoch=13,
            batches_per_epoch=1,
            validation_timesteps=10,
            validation_interval=1,
            checkpoint_interval=1,
            batch_size=2,
            max_sequence_length=10,
            trajectory_length=10,
            replay_capacity=64,
            epsilon_anneal_steps=39,
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
    expected_schedule = whole.lr_schedule.state_dict()
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
    resumed.epoch = 1
    resumed.learn()
    for name, value in expected.items():
        torch.testing.assert_close(
            resumed.policy.state_dict()[name].cpu(), value, rtol=1e-5, atol=1e-6
        )
    assert expected_updates == resumed.grad_update_counter
    assert expected_schedule == resumed.lr_schedule.state_dict()
    assert expected_replay == [
        Path(p).read_bytes() for p in resumed.reasoned_dataset.all_filenames
    ]
    close_experiment(resumed)
