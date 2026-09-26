"""Shared test options and fixtures."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="Fail unless CUDA, FlashAttention and BF16 are available.",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    if session.config.getoption("--require-cuda"):
        from reasoned_icrl.experiments.contracts import ContractError
        from reasoned_icrl.runtime.devices import require_cuda_runtime

        try:
            require_cuda_runtime()
        except ContractError as error:
            raise pytest.UsageError(str(error)) from error


@pytest.fixture(scope="session")
def cuda_runtime() -> None:
    """Skip optional native checks on CPU installs; --require-cuda fails earlier."""
    from reasoned_icrl.experiments.contracts import ContractError
    from reasoned_icrl.runtime.devices import require_cuda_runtime

    try:
        require_cuda_runtime()
    except ContractError as error:
        pytest.skip(str(error))
