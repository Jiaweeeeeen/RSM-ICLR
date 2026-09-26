"""Shared numerical helpers for policy encoders."""

import hashlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch
from torch import nn


def freeze_unoptimized_parameters(
    module: nn.Module, optimized: Iterable[nn.Parameter]
) -> tuple[str, ...]:
    """Exclude copy-network parameter gradients without detaching their inputs.

    Native all-parameter clipping otherwise includes gradients that optimizer
    zero_grad never clears. Input/action differentiation through frozen modules
    remains valid; optimizer-owned online parameters keep their existing flags.
    """
    owned = {id(parameter) for parameter in optimized}
    frozen = []
    for name, parameter in module.named_parameters():
        if id(parameter) not in owned:
            parameter.requires_grad_(False)
            parameter.grad = None
            frozen.append(name)
    return tuple(frozen)


@contextmanager
def module_seed(seed: int, namespace: str) -> Iterator[None]:
    """Isolate CPU initialization RNG by a stable component name (v1)."""
    digest = hashlib.sha256(f"darkroom-init.v1/{seed}/{namespace}".encode()).digest()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(
            torch.Generator(device="cpu")
            .manual_seed(int.from_bytes(digest[:8], "big") % (2**63))
            .get_state()
        )
        yield


def reset_named_linears(module: nn.Module, seed: int, namespace: str) -> None:
    """Use native leaf initializers with independent streams for unequal widths."""
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear | nn.LayerNorm):
            with module_seed(seed, f"{namespace}/{name}"):
                child.reset_parameters()


__all__ = ["freeze_unoptimized_parameters", "module_seed", "reset_named_linears"]
