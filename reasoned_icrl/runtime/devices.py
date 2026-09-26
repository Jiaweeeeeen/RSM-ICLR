"""Resolve hardware once, before recording config or constructing AMAGO.

Automatic selection never chooses MPS. Explicit accelerator requests are strict;
CPU requests and automatic CPU fallback use eager FP32 and VanillaAttention.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import torch

from reasoned_icrl.experiments.contracts import AttentionBackend, ContractError, Device

if TYPE_CHECKING:
    from reasoned_icrl.experiments.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Capabilities needed by the requested workload, independently testable."""

    cuda: bool
    mps: bool
    flash: bool
    bf16: bool


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    config: ExperimentConfig
    requested_device: Device
    requested_attention: AttentionBackend
    requested_precision: str
    requested_compile: bool
    reason: str

    def metadata(self) -> dict[str, str | bool]:
        """Record requested and effective execution settings alongside artifacts."""
        return {
            "requested_device": self.requested_device,
            "requested_attention": self.requested_attention,
            "requested_precision": self.requested_precision,
            "requested_torch_compile": self.requested_compile,
            "device": self.config.device,
            "attention": self.config.model.attention_backend,
            "precision": self.config.training.mixed_precision,
            "torch_compile": self.config.training.torch_compile,
            "reason": self.reason,
        }


def runtime_capabilities() -> RuntimeCapabilities:
    cuda = torch.cuda.is_available()
    flash = False
    if cuda:
        try:
            importlib.import_module("flash_attn")
        except (ImportError, OSError):
            pass
        else:
            flash = torch.cuda.get_device_capability()[0] >= 8
    return RuntimeCapabilities(
        cuda=cuda,
        mps=torch.backends.mps.is_available(),
        flash=flash,
        bf16=cuda and torch.cuda.is_bf16_supported(),
    )


def _missing_cuda_requirements(
    available: RuntimeCapabilities, *, attention: AttentionBackend, precision: str
) -> list[str]:
    missing: list[str] = []
    if not available.cuda:
        missing.append("CUDA device")
    if attention == "flash":
        if not available.flash:
            missing.append(
                "FlashAttention (install with uv sync --frozen --extra cuda)"
            )
        if precision not in ("bf16", "fp16"):
            missing.append("FlashAttention mixed precision (bf16 or fp16)")
    if precision == "bf16" and not available.bf16:
        missing.append("CUDA BF16 support")
    return missing


def require_cuda_runtime(capabilities: RuntimeCapabilities | None = None) -> None:
    """Fail native qualification unless the complete Flash/BF16 stack is available.

    This checks availability only. The CUDA test suite must still execute kernels,
    optimizer updates, checkpoint reloads and evaluation to qualify the runtime.
    """
    available = capabilities or runtime_capabilities()
    missing = _missing_cuda_requirements(available, attention="flash", precision="bf16")
    if missing:
        raise ContractError("CUDA qualification is unavailable: " + "; ".join(missing))


def resolve_runtime(
    config: ExperimentConfig,
    *,
    capabilities: RuntimeCapabilities | None = None,
) -> RuntimeSelection:
    """Return an effective immutable config; never change scientific dimensions.

    ``auto`` falls back to CPU if any accelerator requirement is unavailable.
    Explicit CUDA/MPS requests fail with an actionable error. Choosing CPU replaces
    precision, attention and compilation settings but preserves model/task sizes.
    """
    requested = config.device
    attention = config.model.attention_backend
    precision = config.training.mixed_precision
    reason = "explicit CPU runtime"
    if requested == "cpu":
        selected: Device = "cpu"
    else:
        available = capabilities or runtime_capabilities()
        if requested == "mps":
            if not available.mps:
                raise ContractError("MPS was requested but is unavailable.")
            if attention == "flash" or precision != "no":
                raise ContractError("MPS requires vanilla attention and FP32.")
            selected, reason = "mps", "explicit MPS runtime"
        else:
            missing = _missing_cuda_requirements(
                available, attention=attention, precision=precision
            )
            if missing and requested == "cuda":
                raise ContractError(
                    "CUDA runtime is unavailable: " + "; ".join(missing)
                )
            if missing:
                selected = "cpu"
                reason = "automatic CPU fallback: " + "; ".join(missing)
            else:
                selected, reason = "cuda", "CUDA workload requirements available"
    effective = replace(config, device=selected)
    if selected == "cpu":
        effective = replace(
            effective,
            model=replace(config.model, attention_backend="vanilla"),
            training=replace(
                config.training,
                mixed_precision="no",
                torch_compile=False,
            ),
        )
    return RuntimeSelection(
        effective,
        requested,
        attention,
        precision,
        config.training.torch_compile,
        reason,
    )


def validate_requested_device(config: Mapping[str, Any]) -> None:
    """Fail early when an explicitly requested accelerator is unavailable."""
    model = cast(Mapping[str, Any], config["model"])
    training = cast(Mapping[str, Any], config["training"])
    device = str(config.get("device", "auto"))
    attention_backend = model.get("attention_backend")
    mixed_precision = str(training.get("mixed_precision", "no"))
    if device in ("cpu", "mps") and mixed_precision != "no":
        raise ContractError("Mixed precision requires CUDA with AMAGO 3.4.0.")
    if attention_backend == "flash":
        if device in ("cpu", "mps"):
            raise ContractError("FlashAttention requires a CUDA device.")
        if mixed_precision == "no":
            raise ContractError(
                "FlashAttention requires CUDA mixed precision (bf16 or fp16)."
            )
    if device == "cuda" and not torch.cuda.is_available():
        raise ContractError("CUDA was requested but is unavailable.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ContractError("MPS was requested but is unavailable.")
