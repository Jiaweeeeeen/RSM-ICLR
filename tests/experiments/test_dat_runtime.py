"""Configuration, registration, checkpoint and refresh integration for DAT.

These cover the seams where a dual-attention run has to behave like any other
run: it must resolve from a preset, select the right carrier, survive a
checkpoint round trip, refuse a checkpoint whose attention differs, and rebuild
its rollout caches after a learner update.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import gin
import pytest
import torch
import yaml
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.experiments.benchmarks import experiment_config, load_study
from reasoned_icrl.experiments.config import dump_config, load_resolved_config
from reasoned_icrl.experiments.contracts import (
    ALL_CONDITIONS,
    CONDITIONS,
    DAT_ARCHITECTURE_ID,
    DAT_CONDITIONS,
    DUAL_CONTENT_ARCHITECTURE_ID,
    MEMO_CONDITIONS,
    REVISED_CONDITIONS,
    SUMMARY_CONDITIONS,
    ContractError,
    DATSpec,
)
from reasoned_icrl.model.dat_transformer import EMPTY_TIME, DATHiddenState
from reasoned_icrl.model.trajectory_encoder import DATTrajEncoder, HistoryTrajEncoder
from reasoned_icrl.runtime.amago import configure_amago
from reasoned_icrl.runtime.checkpointing import (
    _hidden_state,
    _restore_hidden_state,
    _runtime_contract,
    policy_checkpoint,
    read_policy_checkpoint,
    validate_checkpoint_architecture,
)
from tests.experiments.fixtures import fixture_study_path

ROOT = Path(__file__).resolve().parents[2]
STUDY = fixture_study_path("dat_benchmarks")
BASELINE = "transition"
ARMS = ("transition_dat", "transition_dat_symbol_only", "transition_dual_content")
"""The DAT study's arms; the raw-evidence rows belong to the summary-memory study."""


def _config(condition: str, study_path: Path = STUDY, **overrides: object):
    study = load_study(study_path)
    return experiment_config(
        study.contract("dark_key_to_door"),
        study,
        condition=condition,
        seed=0,
        repository=ROOT,
        device="cpu",
        **overrides,  # type: ignore[arg-type]
    )


def _study_copy(tmp_path: Path) -> tuple[dict, Path]:
    """A writable copy of the study next to its contracts."""
    raw = yaml.safe_load(STUDY.read_text())
    raw["contracts"] = [
        str((STUDY.parent / entry).resolve()) for entry in raw["contracts"]
    ]
    return raw, tmp_path / "study.yaml"


def _carrier(spec: DATSpec, max_seq_len: int = 8) -> DATTrajEncoder:
    torch.compiler.set_stance("force_eager")
    torch.set_num_threads(1)
    gin.clear_config()
    target = "amago.nets.traj_encoders.TformerTrajEncoder"
    for name, value in {
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 3,
        "d_ff": 1024,
        "attention_type": VanillaAttention,
        "dropout_ff": 0.0,
        "dropout_emb": 0.0,
        "dropout_attn": 0.0,
        "dropout_qkv": 0.0,
    }.items():
        gin.bind_parameter(f"{target}.{name}", value)
    torch.manual_seed(0)
    return DATTrajEncoder(65, max_seq_len, spec=spec).eval()


def _spec(**overrides: object) -> DATSpec:
    settings: dict[str, object] = {"layer_indices": (2,), "max_relative_distance": 6}
    settings.update(overrides)
    return DATSpec(**settings)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The Stage-1 factorial stays frozen
# --------------------------------------------------------------------------


def test_the_five_condition_factorial_is_not_extended() -> None:
    """Analysis and lifecycle paths require exactly the five Stage-1 cells."""
    assert tuple(CONDITIONS) == (
        "feedforward",
        "raw",
        "raw_bypass",
        "transition",
        "transition_bypass",
    )
    assert set(DAT_CONDITIONS).isdisjoint(CONDITIONS)
    assert set(SUMMARY_CONDITIONS).isdisjoint(CONDITIONS | DAT_CONDITIONS)
    assert set(REVISED_CONDITIONS).isdisjoint(
        CONDITIONS | DAT_CONDITIONS | SUMMARY_CONDITIONS
    )
    assert set(MEMO_CONDITIONS).isdisjoint(
        CONDITIONS | DAT_CONDITIONS | SUMMARY_CONDITIONS | REVISED_CONDITIONS
    )
    assert set(ALL_CONDITIONS) == (
        set(CONDITIONS)
        | set(DAT_CONDITIONS)
        | set(SUMMARY_CONDITIONS)
        | set(REVISED_CONDITIONS)
        | set(MEMO_CONDITIONS)
    )
    for name, spec in DAT_CONDITIONS.items():
        assert not spec.bypass and spec.memory == "full" and spec.writer == "same"
        assert spec.evidence == (
            "transition" if name.startswith("transition") else "raw"
        )
    assert tuple(DAT_CONDITIONS)[:3] == ARMS


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_every_arm_resolves_to_its_own_attention_identity() -> None:
    identities = {}
    for condition in ARMS:
        config = _config(condition)
        spec = config.model.dat
        assert spec is not None
        identities[condition] = spec.sha256
        expected = (
            DUAL_CONTENT_ARCHITECTURE_ID
            if spec.mode == "dual_content"
            else DAT_ARCHITECTURE_ID
        )
        assert config.model.architecture_id == expected
    assert len(set(identities.values())) == len(ARMS)


def test_the_baseline_arm_carries_no_dat_settings_anywhere() -> None:
    config = _config(BASELINE)
    assert config.model.dat is None
    assert "dat" not in config.as_runtime_mapping()["model"]


def test_symbol_only_and_full_dat_share_an_architecture_but_not_an_identity() -> None:
    full = _config("transition_dat").model
    symbols = _config("transition_dat_symbol_only").model
    assert full.architecture_id == symbols.architecture_id == DAT_ARCHITECTURE_ID
    assert full.dat is not None and symbols.dat is not None
    assert full.dat.sha256 != symbols.dat.sha256


def test_a_dat_condition_without_settings_is_rejected_not_downgraded(
    tmp_path: Path,
) -> None:
    raw, path = _study_copy(tmp_path)
    del raw["model"]["dat"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match=r"requires a model\.dat"):
        _config(ARMS[0], path)


@pytest.mark.parametrize(
    "override,message",
    [
        ({"layer_indices": []}, "at least one"),
        ({"layer_indices": [2, 2]}, "unique"),
        ({"layer_indices": [9]}, "outside a 3-layer"),
        ({"relational_heads": 8}, "at least one content head"),
        ({"relation_projection_dim": 15}, "relation_channels"),
        ({"max_relative_distance": 0}, "clipping distance"),
        ({"relation_activation": "softmax"}, "identity relation activation"),
        ({"relational_backend": "sparse"}, "dense relational backend"),
        ({"position_method": "rope"}, "fixed positional"),
        ({"cache_dtype": "float64"}, "cache dtype"),
        ({"symbol_dim": 0}, "symbol width"),
    ],
)
def test_unqualified_attention_settings_are_rejected(
    tmp_path: Path, override: dict, message: str
) -> None:
    raw, path = _study_copy(tmp_path)
    raw["model"]["dat"].update(override)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match=message):
        _config(ARMS[0], path)


def test_unknown_and_derived_dat_keys_are_rejected(tmp_path: Path) -> None:
    for override, message in (
        ({"invented": 1}, "Unknown model.dat"),
        ({"d_model": 128}, "disagrees with the condition"),
        ({"mode": "symbol_only"}, "disagrees with the condition"),
    ):
        raw, path = _study_copy(tmp_path)
        raw["model"]["dat"].update(override)
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        with pytest.raises(ContractError, match=message):
            _config(ARMS[0], path)


def test_resolved_configuration_round_trips_with_its_identity(tmp_path: Path) -> None:
    for condition in (BASELINE, *ARMS):
        original = _config(condition, output_root=tmp_path)
        path = dump_config(original, tmp_path / condition / "config.yaml")
        restored = load_resolved_config(path, repository=ROOT)
        assert restored == original
        recorded = yaml.safe_load(path.read_text())["model"].get("dat")
        if original.model.dat is None:
            assert recorded is None
        else:
            assert recorded["sha256"] == original.model.dat.sha256


def test_a_tampered_recorded_identity_is_rejected(tmp_path: Path) -> None:
    original = _config(ARMS[0], output_root=tmp_path)
    path = dump_config(original, tmp_path / "run" / "config.yaml")
    raw = yaml.safe_load(path.read_text())
    raw["model"]["dat"]["sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="identity does not match"):
        load_resolved_config(path, repository=ROOT)


# --------------------------------------------------------------------------
# Component selection
# --------------------------------------------------------------------------


def test_the_factory_selects_the_dat_carrier_and_binds_its_width() -> None:
    for condition in ARMS:
        mapping = _config(condition).as_runtime_mapping()
        mapping["model"]["public_contract"] = {"schema": "test-public-contract.v1"}
        components = configure_amago(mapping["model"], mapping["training"])
        assert components.trajectory_encoder is DATTrajEncoder
        target = "reasoned_icrl.model.trajectory_encoder.DATTrajEncoder"
        # switch_traj_encoder would have bound `memory_size` here instead.
        assert gin.query_parameter(f"{target}.d_model") == 256
        assert (
            gin.query_parameter(f"{target}.spec").sha256
            == (mapping["model"]["dat"]["sha256"])
        )
    gin.clear_config()


def test_the_baseline_arm_still_selects_the_ordinary_carrier() -> None:
    mapping = _config(BASELINE).as_runtime_mapping()
    mapping["model"]["public_contract"] = {"schema": "test-public-contract.v1"}
    components = configure_amago(mapping["model"], mapping["training"])
    assert components.trajectory_encoder is HistoryTrajEncoder
    gin.clear_config()


def test_the_carrier_refuses_a_bypass_packet() -> None:
    with pytest.raises(ContractError, match="without the state bypass"):
        _carrier(_spec()).__class__(129, 8, spec=_spec())


def test_the_carrier_requires_a_resolved_spec() -> None:
    with pytest.raises(ContractError, match="resolved attention spec"):
        _carrier(_spec()).__class__(65, 8)


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def test_the_attention_identity_travels_with_the_weights() -> None:
    carrier = _carrier(_spec())
    other = _carrier(_spec(max_relative_distance=5))
    assert not torch.equal(
        carrier.attention_protocol_identity, other.attention_protocol_identity
    )
    wrapped = policy_checkpoint(
        carrier.state_dict(),
        condition=ARMS[0],
        architecture_id=DAT_ARCHITECTURE_ID,
    )
    restored = read_policy_checkpoint(
        wrapped, condition=ARMS[0], architecture_id=DAT_ARCHITECTURE_ID
    )
    validate_checkpoint_architecture(
        restored, DAT_ARCHITECTURE_ID, expected_state=carrier.state_dict()
    )
    # Same tensor shapes, different attention semantics: still rejected.
    with pytest.raises(ContractError):
        validate_checkpoint_architecture(
            restored, DAT_ARCHITECTURE_ID, expected_state=other.state_dict()
        )


def test_the_resume_contract_names_the_attention_not_only_the_packet() -> None:
    carrier = _carrier(_spec())
    experiment = SimpleNamespace(
        encoder_architecture_id=DAT_ARCHITECTURE_ID,
        policy_condition=ARMS[0],
        policy=SimpleNamespace(
            traj_encoder=carrier,
            tstep_encoder=SimpleNamespace(spec=SimpleNamespace(sha256="packet")),
        ),
        learner_contract="amago-optimizer-ownership.v1",
        reasoned_training_settings={},
    )
    contract = _runtime_contract(experiment)
    assert contract["attention_sha256"] == carrier.spec.sha256
    assert contract["packet_sha256"] == "packet"


def _rollout(carrier: DATTrajEncoder, batch: int, steps: int) -> DATHiddenState:
    torch.manual_seed(3)
    seq = torch.randn(batch, steps, 65)
    seq[..., -1] = 1.0
    times = torch.arange(steps).view(1, steps, 1).expand(batch, steps, 1).contiguous()
    hidden = carrier.init_hidden_state(batch, torch.device("cpu"))
    with torch.no_grad():
        for step in range(steps):
            carrier(seq[:, step : step + 1], times[:, step : step + 1], hidden)
    return hidden


def _experiment(carrier: DATTrajEncoder) -> SimpleNamespace:
    return SimpleNamespace(
        policy=SimpleNamespace(traj_encoder=carrier), DEVICE=torch.device("cpu")
    )


def test_dat_hidden_state_round_trips_through_a_checkpoint() -> None:
    carrier = _carrier(_spec())
    hidden = _rollout(carrier, 2, 4)
    payload = _hidden_state(hidden)
    assert payload["schema"] == "amago-dat-hidden-state.v1"
    restored = _restore_hidden_state(_experiment(carrier), payload)
    assert isinstance(restored, DATHiddenState)
    assert restored.lengths.tolist() == hidden.lengths.tolist()
    torch.testing.assert_close(restored.times, hidden.times, rtol=0, atol=0)
    for source, target in zip(hidden.layers, restored.layers, strict=True):
        assert source.variant == target.variant
        left, right = source.tensors(), target.tensors()
        assert set(left) == set(right)
        for name in left:
            torch.testing.assert_close(
                torch.nan_to_num(left[name]),
                torch.nan_to_num(right[name]),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_relative_distance", 5),
        ("layer_indices", (1,)),
        ("mode", "symbol_only"),
        ("cache_dtype", "bfloat16"),
    ],
)
def test_a_hidden_state_from_a_different_attention_is_rejected(
    field: str, value: object
) -> None:
    carrier = _carrier(_spec())
    payload = _hidden_state(_rollout(carrier, 1, 3))
    other = _carrier(replace(_spec(), **{field: value}))  # type: ignore[arg-type]
    with pytest.raises(ContractError):
        _restore_hidden_state(_experiment(other), payload)


def test_invalid_hidden_states_are_rejected() -> None:
    carrier = _carrier(_spec())
    hidden = _rollout(carrier, 2, 4)

    over_capacity = _hidden_state(hidden)
    over_capacity["lengths"] = torch.full((2,), carrier.capacity, dtype=torch.int32)
    with pytest.raises(ContractError, match="outside the cache capacity"):
        _restore_hidden_state(_experiment(carrier), over_capacity)

    unordered = _hidden_state(hidden)
    times = unordered["times"].clone()  # type: ignore[union-attr]
    times[0, 0], times[0, 1] = times[0, 1].clone(), times[0, 0].clone()
    unordered["times"] = times
    with pytest.raises(ContractError, match="strictly increase"):
        _restore_hidden_state(_experiment(carrier), unordered)

    beyond = _hidden_state(hidden)
    times = beyond["times"].clone()  # type: ignore[union-attr]
    times[0, -1] = 99
    beyond["times"] = times
    with pytest.raises(ContractError, match="beyond its retained length"):
        _restore_hidden_state(_experiment(carrier), beyond)

    missing = _hidden_state(hidden)
    missing.pop("capacity")
    with pytest.raises(ContractError, match="DAT hidden state"):
        _restore_hidden_state(_experiment(carrier), missing)


def test_an_empty_slot_inside_the_retained_window_is_rejected() -> None:
    carrier = _carrier(_spec())
    payload = _hidden_state(_rollout(carrier, 1, 3))
    times = payload["times"].clone()  # type: ignore[union-attr]
    times[0, 1] = EMPTY_TIME
    payload["times"] = times
    with pytest.raises(ContractError, match="no source position"):
        _restore_hidden_state(_experiment(carrier), payload)


# --------------------------------------------------------------------------
# Cache refresh after a learner update
# --------------------------------------------------------------------------


def test_rebuild_reproduces_a_live_rollout_under_the_same_weights() -> None:
    carrier = _carrier(_spec())
    batch, steps = 3, 5
    torch.manual_seed(5)
    seq = torch.randn(batch, steps, 65)
    seq[..., -1] = 1.0
    times = torch.arange(steps).view(1, steps, 1).expand(batch, steps, 1).contiguous()
    lengths = [5, 3, 1]
    with torch.no_grad():
        live = carrier.init_hidden_state(batch, torch.device("cpu"))
        for step in range(steps):
            active = [row for row in range(batch) if step < lengths[row]]
            if len(active) != batch:
                continue
            carrier(seq[:, step : step + 1], times[:, step : step + 1], live)
        rebuilt = carrier.rebuild_hidden_state(seq, times, lengths)
    assert rebuilt.lengths.tolist() == lengths
    # Row 0 ran the full prefix live, so its rebuilt state must match exactly.
    for source, target in zip(live.layers, rebuilt.layers, strict=True):
        left, right = source.tensors(), target.tensors()
        for name in left:
            torch.testing.assert_close(
                left[name][0, :1], right[name][0, :1], rtol=1e-5, atol=2e-6
            )


def test_rebuild_never_processes_padded_decisions() -> None:
    """A short row keeps exactly its true prefix, with the rest poisoned."""
    carrier = _carrier(_spec())
    torch.manual_seed(6)
    seq = torch.randn(2, 6, 65)
    seq[..., -1] = 1.0
    times = torch.arange(6).view(1, 6, 1).expand(2, 6, 1).contiguous()
    with torch.no_grad():
        rebuilt = carrier.rebuild_hidden_state(seq, times, [4, 0])
    assert rebuilt.lengths.tolist() == [4, 0]
    assert rebuilt.times[0, :4].tolist() == [0, 1, 2, 3]
    assert (rebuilt.times[0, 4:] == EMPTY_TIME).all()
    assert (rebuilt.times[1] == EMPTY_TIME).all()
    for cache in rebuilt.layers:
        for tensor in cache.tensors().values():
            assert torch.isfinite(tensor[0, :4]).all()
            assert torch.isnan(tensor[0, 4:]).all()
            assert torch.isnan(tensor[1]).all()


def test_rebuild_rejects_a_prefix_longer_than_the_cache() -> None:
    carrier = _carrier(_spec(), max_seq_len=3)
    seq = torch.zeros(1, 9, 65)
    seq[..., -1] = 1.0
    times = torch.arange(9).view(1, 9, 1)
    with torch.no_grad(), pytest.raises(ContractError, match="exceeds the cache"):
        carrier.rebuild_hidden_state(seq, times, [9])
