"""CPU development defaults and explicit accelerator contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.config import (
    ExperimentConfig,
    dump_config,
    load_resolved_config,
)
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.runtime.devices import (
    RuntimeCapabilities,
    require_cuda_runtime,
    resolve_runtime,
)
from tests.experiments.fixtures import load_fixture_study

ROOT = Path(__file__).resolve().parents[2]
CPU = RuntimeCapabilities(cuda=False, mps=True, flash=False, bf16=False)
CUDA = RuntimeCapabilities(cuda=True, mps=False, flash=True, bf16=True)


def request(
    condition: str = "transition_bypass",
    *,
    device: str = "auto",
    attention: str = "flash",
    precision: str = "bf16",
    compile: bool = False,
    smoke: bool = False,
) -> ExperimentConfig:
    """A DarkRoom run asking for the accelerated backend and precision."""
    study = load_fixture_study("stage1")
    config = experiment_config(
        study.contract("darkroom"),
        study,
        condition=condition,
        seed=0,
        repository=ROOT,
        device=device,
        smoke=smoke,
    )
    return replace(
        config,
        model=replace(config.model, attention_backend=attention),  # type: ignore[arg-type]
        training=replace(
            config.training, mixed_precision=precision, torch_compile=compile
        ),
    )


@pytest.mark.parametrize(
    "capabilities", (CPU, replace(CUDA, flash=False), replace(CUDA, bf16=False))
)
def test_auto_fallback_preserves_scientific_settings_and_roundtrips(
    capabilities: RuntimeCapabilities, tmp_path: Path
) -> None:
    requested = request("feedforward")
    selection = resolve_runtime(requested, capabilities=capabilities)
    effective = selection.config
    assert effective.device == "cpu"
    assert effective.model.attention_backend == "vanilla"
    assert effective.training.mixed_precision == "no"
    assert not effective.training.torch_compile
    assert effective.environment == requested.environment
    assert effective.model == replace(requested.model, attention_backend="vanilla")
    assert effective.training == replace(
        requested.training, mixed_precision="no", torch_compile=False
    )
    assert selection.metadata()["requested_attention"] == "flash"
    assert selection.metadata()["requested_precision"] == "bf16"
    assert selection.metadata()["precision"] == "no"
    assert "fallback" in selection.reason
    saved = dump_config(effective, tmp_path / "config.yaml")
    assert load_resolved_config(saved, repository=ROOT) == effective


def test_explicit_cpu_normalizes_a_gpu_request_without_probing_hardware() -> None:
    selected = resolve_runtime(request(device="cpu")).config
    assert selected.device == "cpu"
    assert selected.model.attention_backend == "vanilla"
    assert selected.training.mixed_precision == "no"


@pytest.mark.parametrize(
    ("capabilities", "missing"),
    (
        (CPU, "CUDA device"),
        (replace(CUDA, flash=False), "FlashAttention"),
        (replace(CUDA, bf16=False), "BF16"),
    ),
)
def test_explicit_cuda_never_silently_falls_back(
    capabilities: RuntimeCapabilities, missing: str
) -> None:
    with pytest.raises(ContractError, match=missing):
        resolve_runtime(request(device="cuda"), capabilities=capabilities)


def test_available_cuda_preserves_requested_backend_and_precision() -> None:
    requested = request()
    selected = resolve_runtime(requested, capabilities=CUDA).config
    assert selected == replace(requested, device="cuda")


def test_vanilla_cuda_does_not_require_flashattention() -> None:
    requested = request(attention="vanilla", precision="no")
    selected = resolve_runtime(
        requested, capabilities=replace(CUDA, flash=False)
    ).config
    assert selected.device == "cuda"
    assert selected.model.attention_backend == "vanilla"


@pytest.mark.parametrize(
    ("capabilities", "missing"),
    (
        (replace(CUDA, cuda=False), "CUDA device"),
        (replace(CUDA, flash=False), "FlashAttention"),
        (replace(CUDA, bf16=False), "BF16"),
    ),
)
def test_native_qualification_requires_every_capability(
    capabilities: RuntimeCapabilities, missing: str
) -> None:
    with pytest.raises(ContractError, match=missing):
        require_cuda_runtime(capabilities)


def test_native_qualification_accepts_complete_capabilities() -> None:
    require_cuda_runtime(CUDA)


def test_cpu_provenance_retains_requested_compilation_and_effective_settings() -> None:
    selection = resolve_runtime(request("raw", device="cpu", compile=True))
    metadata = selection.metadata()
    assert metadata["requested_device"] == metadata["device"] == "cpu"
    assert metadata["requested_attention"] == "flash"
    assert metadata["attention"] == "vanilla"
    assert metadata["requested_precision"] == "bf16"
    assert metadata["precision"] == "no"
    assert metadata["requested_torch_compile"] is True
    assert metadata["torch_compile"] is False


@pytest.mark.parametrize(
    "condition", ("feedforward", "raw", "raw_bypass", "transition", "transition_bypass")
)
def test_smoke_preserves_all_attempts_and_a_complete_replay_prefix(
    condition: str,
) -> None:
    full: Any = request(condition)
    smoke: Any = request(condition, smoke=True)
    assert smoke.environment.attempts == full.environment.attempts
    assert smoke.environment.parallel_envs == 1
    assert smoke.model.width == 32
    assert smoke.model.layers == 1
    outer_length = smoke.environment.attempts * smoke.environment.horizon
    assert smoke.training.max_sequence_length == outer_length
    assert smoke.training.trajectory_length == outer_length
    assert smoke.training.timesteps_per_epoch == outer_length
    assert not smoke.tracking.wandb
