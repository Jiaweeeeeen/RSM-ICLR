"""The summary-memory study's condition rows, identities and refusals.

These pin the contract layer that every later task builds on: the two new
``ConditionSpec`` switches, the ``raw`` attention variants, the
``SUMMARY_CONDITIONS`` table, the four new architecture identities and the
rule that a registered row whose carrier is not bound is refused at
configuration time rather than silently downgraded.
"""

from __future__ import annotations

from pathlib import Path

import gin
import pytest
import yaml

from reasoned_icrl.experiments.benchmarks import experiment_config, load_study
from reasoned_icrl.experiments.config import (
    architecture_id,
    dump_config,
    load_resolved_config,
)
from reasoned_icrl.experiments.contracts import (
    ALL_CONDITION_LABELS,
    ALL_CONDITIONS,
    ARCHITECTURE_LABELS,
    CONDITIONS,
    DAT_ARCHITECTURE_ID,
    DAT_CONDITIONS,
    DAT_SUMMARY_ARCHITECTURE_ID,
    DUAL_CONTENT_ARCHITECTURE_ID,
    DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID,
    GRU_HISTORY_ARCHITECTURE_ID,
    HISTORY_ARCHITECTURE_ID,
    SUMMARY_ARCHITECTURE_ID,
    SUMMARY_CONDITIONS,
    WINDOW_ARCHITECTURE_ID,
    ConditionSpec,
    ContractError,
    architecture_uses_dat,
    architecture_uses_history_packet,
    architecture_uses_summary,
    condition_label,
    condition_spec,
    summary_architecture_id,
)
from reasoned_icrl.experiments.summary_memory.configs import RETIRED_STUDY
from reasoned_icrl.runtime.amago import BOUND_CARRIERS, configure_amago
from tests.experiments.fixtures import fixture_study_path

ROOT = Path(__file__).resolve().parents[2]
DAT_STUDY = fixture_study_path("dat_benchmarks")

EXPECTED_ARCHITECTURES = {
    "raw_dat": DAT_ARCHITECTURE_ID,
    "raw_dual_content": DUAL_CONTENT_ARCHITECTURE_ID,
    "raw_gru": GRU_HISTORY_ARCHITECTURE_ID,
    "raw_segment": SUMMARY_ARCHITECTURE_ID,
    "raw_summary": SUMMARY_ARCHITECTURE_ID,
    "raw_summary_detach": SUMMARY_ARCHITECTURE_ID,
    "raw_summary_residual": SUMMARY_ARCHITECTURE_ID,
    "raw_summary_gated": SUMMARY_ARCHITECTURE_ID,
    "raw_dat_segment": DAT_SUMMARY_ARCHITECTURE_ID,
    "raw_dat_summary": DAT_SUMMARY_ARCHITECTURE_ID,
    "raw_dat_summary_relational_write_off": DAT_SUMMARY_ARCHITECTURE_ID,
    "raw_dual_content_summary": DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID,
    "raw_window": WINDOW_ARCHITECTURE_ID,
}
NEW_ROWS = tuple(EXPECTED_ARCHITECTURES)


def _study_with(tmp_path: Path, conditions: list[str]) -> Path:
    """The DAT study's backbone and contracts with a different condition roster.

    The summary block of the summary-memory study is added so that its bounded
    rows resolve; full-prefix rows ignore it exactly as they ignore ``dat``.
    """
    raw = yaml.safe_load(DAT_STUDY.read_text())
    raw["contracts"] = [
        str((DAT_STUDY.parent / entry).resolve()) for entry in raw["contracts"]
    ]
    raw["conditions"] = conditions
    raw.pop("control_conditions")
    raw.pop("pilot_seeds")
    summary_model = yaml.safe_load((ROOT / RETIRED_STUDY).read_text())["model"]
    raw["model"]["summary"] = summary_model["summary"]
    raw["model"]["window"] = summary_model["window"]
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _config(tmp_path: Path, condition: str):
    study = load_study(_study_with(tmp_path, [condition]))
    return experiment_config(
        study.contract("dark_key_to_door"),
        study,
        condition=condition,
        seed=0,
        repository=ROOT,
        device="cpu",
        output_root=tmp_path,
    )


# --------------------------------------------------------------------------
# Rows and labels
# --------------------------------------------------------------------------


def test_every_existing_row_keeps_full_memory_and_the_same_writer() -> None:
    for name, spec in CONDITIONS.items():
        assert spec.memory == "full" and spec.writer == "same", name
        assert spec.trajectory_encoder in ("feedforward", "transformer"), name
    for name in ("transition_dat", "transition_dat_symbol_only"):
        assert DAT_CONDITIONS[name].memory == "full"  # type: ignore[index]


def test_the_new_rows_share_the_raw_packet_without_a_bypass() -> None:
    for name in NEW_ROWS:
        spec = condition_spec(name)
        assert spec.evidence == "raw" and not spec.bypass, name
        assert spec.uses_history_packet
    assert SUMMARY_CONDITIONS["raw_gru"] == ConditionSpec("gru", "raw", False)
    assert (
        SUMMARY_CONDITIONS["raw_dat_summary_relational_write_off"].writer
        == "relational_off"
    )
    assert all(
        spec.bounded for name, spec in SUMMARY_CONDITIONS.items() if name != "raw_gru"
    )
    assert not SUMMARY_CONDITIONS["raw_gru"].bounded
    assert "raw_dat_symbol_only" not in ALL_CONDITIONS


def test_labels_and_identities_exist_for_every_new_row() -> None:
    assert set(ALL_CONDITION_LABELS) == set(ALL_CONDITIONS)
    assert condition_label("raw_gru") == "Raw history, GRU (AMAGO recurrent baseline)"
    assert condition_label("raw_dat") == "Raw history, dual attention"
    for identity in (
        GRU_HISTORY_ARCHITECTURE_ID,
        SUMMARY_ARCHITECTURE_ID,
        DAT_SUMMARY_ARCHITECTURE_ID,
        DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID,
        WINDOW_ARCHITECTURE_ID,
    ):
        assert identity in ARCHITECTURE_LABELS
        assert architecture_uses_history_packet(identity)
    assert not architecture_uses_dat(GRU_HISTORY_ARCHITECTURE_ID)
    assert not architecture_uses_dat(SUMMARY_ARCHITECTURE_ID)
    assert architecture_uses_dat(DAT_SUMMARY_ARCHITECTURE_ID)
    assert architecture_uses_dat(DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID)
    assert architecture_uses_summary(SUMMARY_ARCHITECTURE_ID)
    assert architecture_uses_summary(DAT_SUMMARY_ARCHITECTURE_ID)
    assert architecture_uses_summary(DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID)
    assert not architecture_uses_summary(HISTORY_ARCHITECTURE_ID)
    assert not architecture_uses_summary(WINDOW_ARCHITECTURE_ID)
    with pytest.raises(ContractError, match="Unknown architecture"):
        architecture_uses_summary("amago-invented-v1")


@pytest.mark.parametrize("condition", NEW_ROWS)
def test_each_new_row_resolves_to_its_declared_architecture(condition: str) -> None:
    assert architecture_id(condition) == EXPECTED_ARCHITECTURES[condition]


def test_summary_architecture_resolution_rejects_other_regimes() -> None:
    assert summary_architecture_id("ordinary", "segment") == SUMMARY_ARCHITECTURE_ID
    assert summary_architecture_id("dat", "summary") == DAT_SUMMARY_ARCHITECTURE_ID
    assert (
        summary_architecture_id("dual_content", "summary")
        == DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID
    )
    with pytest.raises(ContractError, match="segment or summary"):
        summary_architecture_id("ordinary", "full")
    with pytest.raises(ContractError, match="No summary carrier"):
        summary_architecture_id("dat_symbol_only", "summary")


# --------------------------------------------------------------------------
# ConditionSpec refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments,message",
    [
        (
            {"trajectory_encoder": "gru", "evidence": "raw", "memory": "summary"},
            "Transformer carrier only",
        ),
        (
            {"trajectory_encoder": "feedforward", "memory": "segment"},
            "Transformer carrier only",
        ),
        (
            {"trajectory_encoder": "gru", "evidence": "raw", "attention": "dat"},
            "no attention variant",
        ),
        (
            {"trajectory_encoder": "gru", "evidence": "raw", "bypass": True},
            "no state bypass",
        ),
        (
            {
                "trajectory_encoder": "transformer",
                "evidence": "raw",
                "writer": "relational_off",
            },
            "relational-write-off",
        ),
        (
            {
                "trajectory_encoder": "transformer",
                "evidence": "raw",
                "attention": "ordinary",
                "memory": "summary",
                "writer": "relational_off",
            },
            "relational-write-off",
        ),
        (
            {
                "trajectory_encoder": "transformer",
                "evidence": "raw",
                "attention": "dat",
                "memory": "segment",
                "writer": "relational_off",
            },
            "relational-write-off",
        ),
        (
            # A dual-attention band is the revised `fixed_window`; without the
            # revised route it would be the legacy window under a DAT name.
            {
                "trajectory_encoder": "transformer",
                "evidence": "raw",
                "attention": "dat",
                "memory": "window",
            },
            "timestep-record relational route",
        ),
        (
            {
                "trajectory_encoder": "transformer",
                "evidence": "raw",
                "attention": "dual_content",
                "memory": "window",
            },
            "ordinary or dual attention only",
        ),
        ({"trajectory_encoder": "feedforward", "evidence": "raw"}, "no history token"),
        ({"trajectory_encoder": "transformer"}, "evidence selection"),
        ({"trajectory_encoder": "gru"}, "evidence selection"),
    ],
)
def test_inconsistent_switch_combinations_are_refused(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        ConditionSpec(**arguments)  # type: ignore[arg-type]


def test_dual_attention_rows_are_free_in_evidence_but_not_in_bypass() -> None:
    """The old transition-only check is now bypass-only: raw DAT rows resolve."""
    assert architecture_id("raw_dat") == architecture_id("transition_dat")
    assert architecture_id("raw_dual_content") == architecture_id(
        "transition_dual_content"
    )


# --------------------------------------------------------------------------
# Configuration: resolved and refused
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "condition",
    tuple(name for name in NEW_ROWS if EXPECTED_ARCHITECTURES[name] in BOUND_CARRIERS),
)
def test_bound_rows_resolve_and_round_trip(tmp_path: Path, condition: str) -> None:
    config = _config(tmp_path, condition)
    assert config.model.architecture_id == EXPECTED_ARCHITECTURES[condition]
    mode = ALL_CONDITIONS[condition].dat_mode
    if mode is None:
        assert config.model.dat is None
    else:
        assert config.model.dat is not None and config.model.dat.mode == mode
    bounded = ALL_CONDITIONS[condition].memory in ("segment", "summary")
    assert (config.model.summary is not None) == bounded
    mapping = config.as_runtime_mapping()
    assert mapping["model"]["evidence"] == "raw" and mapping["model"]["bypass"] is False
    path = dump_config(config, tmp_path / condition / "config.yaml")
    assert load_resolved_config(path, repository=ROOT) == config
    mapping["model"]["public_contract"] = {"schema": "test-public-contract.v1"}
    components = configure_amago(mapping["model"], mapping["training"])
    assert components.architecture_id == EXPECTED_ARCHITECTURES[condition]
    gin.clear_config()


def test_the_raw_variants_share_the_raw_backbone_recipe(tmp_path: Path) -> None:
    raw = _config(tmp_path, "raw")
    assert (
        raw.model.architecture_id == HISTORY_ARCHITECTURE_ID and raw.model.dat is None
    )
    variant = _config(tmp_path, "raw_dat")
    assert variant.environment == raw.environment
    assert variant.training == raw.training
    assert (variant.model.width, variant.model.layers, variant.model.heads) == (
        raw.model.width,
        raw.model.layers,
        raw.model.heads,
    )


def test_every_new_row_has_a_bound_carrier() -> None:
    """Every registered row binds a carrier (M0.3, M2.3, M2.6, M2.7); a row
    whose carrier were missing would be refused by ``configure_amago``."""
    for name in NEW_ROWS:
        assert EXPECTED_ARCHITECTURES[name] in BOUND_CARRIERS, name
    assert WINDOW_ARCHITECTURE_ID in BOUND_CARRIERS
    assert DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID in BOUND_CARRIERS


# --------------------------------------------------------------------------
# The truncated-gradient ablation
# --------------------------------------------------------------------------


def test_the_truncated_gradient_cell_derives_its_detach_rule_from_its_row(
    tmp_path: Path,
) -> None:
    """``raw_summary_detach`` shares the ordinary carrier's identity string but
    hashes apart from ``raw_summary`` through its detach rule, which comes from
    the condition row rather than the study block; a block that names the
    other rule for a cell is refused."""
    for name in ("full", "truncated", "bad"):
        (tmp_path / name).mkdir()
    full = _config(tmp_path / "full", "raw_summary")
    truncated = _config(tmp_path / "truncated", "raw_summary_detach")
    assert full.model.summary is not None and truncated.model.summary is not None
    assert full.model.summary.detach == "none"
    assert truncated.model.summary.detach == "boundary"
    assert truncated.model.summary.sha256 != full.model.summary.sha256
    assert truncated.model.architecture_id == SUMMARY_ARCHITECTURE_ID
    assert truncated.model.architecture_id == full.model.architecture_id
    path = _study_with(tmp_path / "bad", ["raw_summary"])
    raw = yaml.safe_load(path.read_text())
    raw["model"]["summary"]["detach"] = "boundary"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="disagrees"):
        load_study(path)  # the loader resolves every roster cell's model block


def test_the_residual_rewrite_cell_derives_its_rule_from_its_row(
    tmp_path: Path,
) -> None:
    """``raw_summary_residual`` hashes apart from
    ``raw_summary`` through its rewrite rule, which comes from the condition
    row; the replacing write is unmarked in the hash, so every identity
    recorded before the field existed is unchanged; a block that names the
    other rule for a cell is refused."""
    import hashlib
    import json
    from dataclasses import asdict

    for name in ("replacing", "residual", "bad"):
        (tmp_path / name).mkdir()
    replacing = _config(tmp_path / "replacing", "raw_summary")
    residual = _config(tmp_path / "residual", "raw_summary_residual")
    assert replacing.model.summary is not None
    assert residual.model.summary is not None
    assert replacing.model.summary.rewrite == "replace"
    assert residual.model.summary.rewrite == "residual"
    assert residual.model.summary.sha256 != replacing.model.summary.sha256
    assert residual.model.architecture_id == SUMMARY_ARCHITECTURE_ID
    assert residual.model.architecture_id == replacing.model.architecture_id
    unmarked = asdict(replacing.model.summary)
    del unmarked["rewrite"], unmarked["relational_sources"]  # legacy route
    assert (
        replacing.model.summary.sha256
        == hashlib.sha256(json.dumps(unmarked, sort_keys=True).encode()).hexdigest()
    )
    path = _study_with(tmp_path / "bad", ["raw_summary"])
    raw = yaml.safe_load(path.read_text())
    raw["model"]["summary"]["rewrite"] = "residual"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="disagrees"):
        load_study(path)


def test_the_gated_rewrite_cell_derives_its_rule_from_its_row(tmp_path: Path) -> None:
    """``raw_summary_gated`` (T3) hashes apart from both the
    replacing and the residual carrier through its rewrite rule, keeps the
    summary architecture id, and a block that names another rule is refused."""
    for name in ("residual", "gated", "bad"):
        (tmp_path / name).mkdir()
    residual = _config(tmp_path / "residual", "raw_summary_residual")
    gated = _config(tmp_path / "gated", "raw_summary_gated")
    assert residual.model.summary is not None and gated.model.summary is not None
    assert gated.model.summary.rewrite == "gated"
    assert gated.model.summary.sha256 != residual.model.summary.sha256
    assert gated.model.architecture_id == SUMMARY_ARCHITECTURE_ID
    assert gated.model.summary._hashed()["rewrite"] == "gated"
    path = _study_with(tmp_path / "bad", ["raw_summary_residual"])
    raw = yaml.safe_load(path.read_text())
    raw["model"]["summary"]["rewrite"] = "gated"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="disagrees"):
        load_study(path)
