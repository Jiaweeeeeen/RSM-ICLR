"""The two Key-to-Door controls: the in-distribution long-horizon control
trained at 1,000 calls and the M = 16 capacity ablation, each its own roster
and root."""

from __future__ import annotations

from pathlib import Path

import pytest

from reasoned_icrl.experiments.config import (
    KEY_TO_DOOR_PROTOCOL_1000,
    environment_config,
)
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.experiments import resolve

ROOT = Path(__file__).resolve().parents[3]


def test_the_capacity_ablation_hashes_apart_and_reads_frozen_references(
    tmp_path: Path,
) -> None:
    capacity = load_summary_memory_study(ROOT / "configs/keydoor_capacity_8m.yaml")
    paper = load_summary_memory_study(None)
    contract = capacity.contract("dark_key_to_door")
    assert contract.protocol == paper.contract("dark_key_to_door").protocol
    assert contract.memory is not None
    assert contract.memory["summary"] == {"segment_length": 32, "memory_tokens": 16}
    tier = capacity.tier("dark_key_to_door")
    # The ablation moved to the method RSM-O; the
    # RSM-R fits stay as the supplementary row.
    assert tier.primary == ("raw_summary",)
    assert tier.supplementary == ("raw_summary_residual",)
    assert tier.reference_cells == ("full_context", "full_gru")
    assert tier.reference_root == "outputs/summary-memory-8m"
    _, wide = resolve(
        capacity,
        benchmark="dark_key_to_door",
        condition="raw_summary_residual",
        seed=42,
        device="cpu",
        output_root=tmp_path,
    )
    _, narrow = resolve(
        paper,
        benchmark="dark_key_to_door",
        condition="raw_summary_residual",
        seed=42,
        device="cpu",
        output_root=tmp_path,
    )
    assert wide.model.summary is not None and narrow.model.summary is not None
    assert (wide.model.summary.memory_tokens, narrow.model.summary.memory_tokens) == (
        16,
        4,
    )
    assert (
        wide.model.summary.segment_length == narrow.model.summary.segment_length == 32
    )
    assert wide.model.summary.sha256 != narrow.model.summary.sha256
    assert wide.model.architecture_id == narrow.model.architecture_id
    # The frozen references carry no summary identity, so the M = 16 block cannot
    # re-resolve them; they are read from the 8M root through the comparator path
    # and are refused as fits of this root.
    with pytest.raises(ContractError, match="frozen reference cell"):
        resolve(
            capacity,
            benchmark="dark_key_to_door",
            condition="full_context",
            seed=42,
            device="cpu",
            output_root=tmp_path,
        )
    saved = "outputs/summary-memory-8m/native-keydoor-fixed500-first8"
    if not (ROOT / saved / "full_context/seed-42/config.yaml").is_file():
        pytest.skip("reading the frozen references needs the saved 8M study root")
    from scripts.develop_summary_memory import reference_configs

    references = reference_configs(capacity, contract, repository=ROOT, device="cpu")
    assert sorted({r.condition for r in references}) == ["full_context", "full_gru"]
    for reference in references:
        assert reference.model.summary is None
        assert "summary-memory-8m" in str(reference.run_directory)


def test_key_to_door_admits_exactly_the_two_declared_protocols() -> None:
    base = {
        "name": "dark_key_to_door",
        "benchmark": KEY_TO_DOOR_PROTOCOL_1000,
        "size": 8,
        "attempts": 8,
        "horizon": 50,
        "meta_horizon": 1000,
        "randomized_actions": False,
        "parallel_envs": 16,
    }
    assert environment_config(base).outer_length == 1000
    with pytest.raises(ContractError, match="requires one of"):
        environment_config({**base, "benchmark": "native-keydoor-fixed750-first8"})
    with pytest.raises(ContractError, match="cannot guarantee"):
        environment_config({**base, "meta_horizon": 400})


@pytest.mark.parametrize(
    ("study_file", "memory_tokens", "supplementary"),
    [
        ("configs/keydoor_capacity_m1_8m.yaml", 1, ("raw_summary_residual",)),
        ("configs/keydoor_capacity_m8_8m.yaml", 8, ()),
        ("configs/keydoor_capacity_8m.yaml", 16, ("raw_summary_residual",)),
    ],
)
def test_the_method_summary_length_ladder_hashes_apart_on_key_to_door(
    tmp_path: Path,
    study_file: str,
    memory_tokens: int,
    supplementary: tuple[str, ...],
) -> None:
    """The summary-length ablation on the method: RSM-O at M in {1, 8, 16} on the
    paper's Key-to-Door task, each arm its own root with C = 32, the primary cell
    RSM-O, the frozen full-history and GRU endpoints read from the 8M root, and a
    summary identity that hashes apart from the paper's M = 4 cell and from every
    other arm."""
    arm = load_summary_memory_study(ROOT / study_file)
    paper = load_summary_memory_study(None)
    contract = arm.contract("dark_key_to_door")
    assert contract.protocol == paper.contract("dark_key_to_door").protocol
    assert contract.memory is not None
    assert contract.memory["summary"] == {
        "segment_length": 32,
        "memory_tokens": memory_tokens,
    }
    tier = arm.tier("dark_key_to_door")
    assert tier.primary == ("raw_summary",)
    assert tier.supplementary == supplementary
    assert tier.reference_cells == ("full_context", "full_gru")
    assert tier.reference_root == "outputs/summary-memory-8m"
    assert arm.output_root != paper.output_root
    _, method = resolve(
        arm,
        benchmark="dark_key_to_door",
        condition="raw_summary",
        seed=42,
        device="cpu",
        output_root=tmp_path,
    )
    _, base = resolve(
        paper,
        benchmark="dark_key_to_door",
        condition="raw_summary",
        seed=42,
        device="cpu",
        output_root=tmp_path,
    )
    assert method.model.summary is not None and base.model.summary is not None
    assert (method.model.summary.memory_tokens, base.model.summary.memory_tokens) == (
        memory_tokens,
        4,
    )
    assert method.model.summary.segment_length == 32
    assert base.model.summary.segment_length == 32
    assert method.model.summary.sha256 != base.model.summary.sha256
    assert method.model.architecture_id == base.model.architecture_id
    assert method.environment.benchmark == base.environment.benchmark
    with pytest.raises(ContractError, match="frozen reference cell"):
        resolve(
            arm,
            benchmark="dark_key_to_door",
            condition="full_context",
            seed=42,
            device="cpu",
            output_root=tmp_path,
        )


def test_the_method_summary_length_arms_hash_apart_from_each_other(
    tmp_path: Path,
) -> None:
    """No two arms of the ladder share a summary identity, so no arm can load
    another arm's checkpoint."""
    hashes = set()
    for study_file in (
        "configs/keydoor_capacity_m1_8m.yaml",
        "configs/keydoor_capacity_m8_8m.yaml",
        "configs/keydoor_capacity_8m.yaml",
    ):
        _, config = resolve(
            load_summary_memory_study(ROOT / study_file),
            benchmark="dark_key_to_door",
            condition="raw_summary",
            seed=42,
            device="cpu",
            output_root=tmp_path,
        )
        assert config.model.summary is not None
        hashes.add(config.model.summary.sha256)
    assert len(hashes) == 3
