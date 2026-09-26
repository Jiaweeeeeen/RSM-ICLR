"""The GRU carrier: AMAGO's recurrent baseline over the history packet.

Full-sequence training forward and step-by-step rollout agree; the batched
rebuild after a learner update equals a live rollout of the same prefix; resets
touch only the selected rows; the hidden state round-trips through a checkpoint
and refuses every disagreement; and the study's configuration binds the carrier
explicitly at the declared width and depth.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gin
import numpy as np
import pytest
import torch
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import (
    GRU_HISTORY_ARCHITECTURE_ID,
    ContractError,
)
from reasoned_icrl.experiments.summary_memory.configs import (
    load_retired_summary_memory_study,
)
from reasoned_icrl.model.trajectory_encoder import (
    DATTrajEncoder,
    GRUHistoryTrajEncoder,
    HistoryTrajEncoder,
)
from reasoned_icrl.runtime.amago import configure_amago
from reasoned_icrl.runtime.checkpointing import (
    GRU_HIDDEN_STATE_SCHEMA,
    _hidden_state,
    _restore_hidden_state,
    _runtime_contract,
)
from reasoned_icrl.runtime.experiment import cache_measurements

ROOT = Path(__file__).resolve().parents[2]
TOKEN = 16
WIDTH = 24
LAYERS = 2


def _carrier(n_layers: int = LAYERS) -> GRUHistoryTrajEncoder:
    torch.compiler.set_stance("force_eager")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    return GRUHistoryTrajEncoder(
        TOKEN + 1, 8, token_dim=TOKEN, d_model=WIDTH, n_layers=n_layers
    ).eval()


def _packet(batch: int, length: int, valid_lengths: list[int] | None = None):
    torch.manual_seed(1)
    seq = torch.randn(batch, length, TOKEN + 1)
    valid = torch.zeros(batch, length, 1)
    for row in range(batch):
        keep = length if valid_lengths is None else valid_lengths[row]
        valid[row, :keep] = 1.0
    seq[..., -1:] = valid
    times = (
        torch.arange(length).view(1, length, 1).expand(batch, length, 1).contiguous()
    )
    return seq, times, valid


def _rollout(
    carrier: GRUHistoryTrajEncoder, seq: torch.Tensor, times: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = carrier.init_hidden_state(seq.shape[0], torch.device("cpu"))
    outputs = []
    with torch.no_grad():
        for step in range(seq.shape[1]):
            out, hidden = carrier(
                seq[:, step : step + 1], times[:, step : step + 1], hidden
            )
            outputs.append(out)
    return torch.cat(outputs, 1), hidden


def _experiment(carrier: torch.nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        policy=SimpleNamespace(traj_encoder=carrier), DEVICE=torch.device("cpu")
    )


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_the_carrier_is_amago_s_gru_at_the_declared_width_and_depth() -> None:
    carrier = _carrier()
    assert carrier.emb_dim == WIDTH and carrier.n_layers == LAYERS
    assert carrier.backbone.rnn.hidden_size == WIDTH
    assert carrier.backbone.rnn.num_layers == LAYERS
    assert carrier.backbone.rnn.input_size == TOKEN
    assert carrier.backbone.emb_dim == WIDTH
    assert not hasattr(carrier, "spec") and not hasattr(carrier, "fusion")
    hidden = carrier.init_hidden_state(3, torch.device("cpu"))
    assert hidden.shape == (LAYERS, 3, WIDTH) and hidden.dtype == torch.float32
    assert not hidden.any()


def test_the_carrier_refuses_a_bypass_packet_and_bad_geometry() -> None:
    with pytest.raises(ContractError, match="without the state bypass"):
        GRUHistoryTrajEncoder(2 * TOKEN + 1, 8, token_dim=TOKEN, d_model=WIDTH)
    with pytest.raises(ContractError, match="positive"):
        GRUHistoryTrajEncoder(TOKEN + 1, 8, token_dim=TOKEN, d_model=WIDTH, n_layers=0)


# --------------------------------------------------------------------------
# Full sequence, step-by-step and rebuild agree
# --------------------------------------------------------------------------


def test_full_sequence_forward_equals_the_stepwise_rollout() -> None:
    carrier = _carrier()
    seq, times, _ = _packet(3, 7)
    with torch.no_grad():
        full, none = carrier(seq, times)
    assert none is None
    stepwise, hidden = _rollout(carrier, seq, times)
    torch.testing.assert_close(full, stepwise, rtol=1e-6, atol=1e-6)
    assert hidden.shape == (LAYERS, 3, WIDTH)
    with torch.no_grad():
        _, final = carrier.backbone.rnn(seq[..., :TOKEN])
    torch.testing.assert_close(hidden, final, rtol=1e-6, atol=1e-6)


def test_padded_rows_are_zero_and_do_not_change_valid_outputs() -> None:
    carrier = _carrier()
    seq, times, _ = _packet(2, 6, valid_lengths=[6, 3])
    with torch.no_grad():
        full, _ = carrier(seq, times)
        altered = seq.clone()
        altered[1, 3:, :TOKEN] += 50.0
        changed, _ = carrier(altered, times)
    assert not full[1, 3:].any()
    torch.testing.assert_close(full, changed, rtol=0, atol=0)
    torch.testing.assert_close(full[0], changed[0], rtol=0, atol=0)


def test_future_tokens_cannot_change_earlier_outputs() -> None:
    carrier = _carrier()
    seq, times, _ = _packet(1, 5)
    with torch.no_grad():
        base, _ = carrier(seq, times)
        changed = seq.clone()
        changed[:, -1, :TOKEN] += 20.0
        later, _ = carrier(changed, times)
    torch.testing.assert_close(later[:, :-1], base[:, :-1], rtol=0, atol=0)
    assert not torch.equal(later[:, -1], base[:, -1])


def test_rebuild_equals_the_stepwise_state_for_unequal_lengths() -> None:
    carrier = _carrier()
    seq, times, _ = _packet(4, 6)
    lengths = [6, 1, 0, 4]
    rebuilt = carrier.rebuild_hidden_state(seq, times, lengths)
    assert rebuilt.shape == (LAYERS, 4, WIDTH)
    for row, length in enumerate(lengths):
        if length == 0:
            assert not rebuilt[:, row].any()
            continue
        _, live = _rollout(
            carrier, seq[row : row + 1, :length], times[row : row + 1, :length]
        )
        torch.testing.assert_close(rebuilt[:, row], live[:, 0], rtol=1e-6, atol=1e-6)


def test_rebuild_ignores_everything_beyond_the_true_prefix() -> None:
    carrier = _carrier()
    seq, times, _ = _packet(2, 5)
    rebuilt = carrier.rebuild_hidden_state(seq, times, [3, 2])
    tampered = seq.clone()
    tampered[0, 3:, :TOKEN] = 99.0
    tampered[1, 2:, :TOKEN] = -99.0
    again = carrier.rebuild_hidden_state(tampered, times, [3, 2])
    torch.testing.assert_close(rebuilt, again, rtol=0, atol=0)


def test_rebuild_refuses_mismatched_lengths_and_long_prefixes() -> None:
    carrier = _carrier()
    seq, times, _ = _packet(2, 4)
    with pytest.raises(ContractError, match="one true length per row"):
        carrier.rebuild_hidden_state(seq, times, [4])
    with pytest.raises(ContractError, match="exceeds the provided packets"):
        carrier.rebuild_hidden_state(seq, times, [5, 1])


# --------------------------------------------------------------------------
# Rollout contract and resets
# --------------------------------------------------------------------------


def test_reset_zeroes_only_the_selected_rows() -> None:
    carrier = _carrier()
    seq, times, _ = _packet(3, 3)
    _, hidden = _rollout(carrier, seq, times)
    survivor = hidden[:, 1].clone()
    returned = carrier.reset_hidden_state(hidden, np.array([True, False, True]))
    assert returned is hidden
    assert not hidden[:, 0].any() and not hidden[:, 2].any()
    torch.testing.assert_close(hidden[:, 1], survivor, rtol=0, atol=0)
    carrier.reset_hidden_state(hidden, [1])
    assert not hidden.any()
    with pytest.raises(ContractError, match="always carries a hidden state"):
        carrier.reset_hidden_state(None, np.array([True]))


def test_reset_accepts_a_state_produced_under_inference_mode() -> None:
    # The evaluator's boundary reset runs outside the ``inference_mode`` block
    # that produced the state (the attempt-cleared intervention on Key-to-Door).
    carrier = _carrier()
    seq, times, _ = _packet(3, 3)
    hidden = carrier.init_hidden_state(3, torch.device("cpu"))
    with torch.inference_mode():
        for step in range(seq.shape[1]):
            _, hidden = carrier(
                seq[:, step : step + 1], times[:, step : step + 1], hidden
            )
    assert hidden.is_inference()
    survivor = hidden[:, 1].clone()
    returned = carrier.reset_hidden_state(hidden, np.array([True, False, True]))
    assert returned is hidden
    assert not hidden[:, 0].any() and not hidden[:, 2].any()
    torch.testing.assert_close(hidden[:, 1], survivor, rtol=0, atol=0)


def test_online_steps_accept_one_valid_decision_per_row() -> None:
    carrier = _carrier()
    seq, times, _ = _packet(2, 3)
    hidden = carrier.init_hidden_state(2, torch.device("cpu"))
    with pytest.raises(ContractError, match="one valid decision"):
        carrier(seq[:, :2], times[:, :2], hidden)
    invalid = seq[:, :1].clone()
    invalid[0, 0, -1] = 0.0
    with pytest.raises(ContractError, match="one valid decision"):
        carrier(invalid, times[:, :1], hidden)


def test_gradients_flow_through_the_whole_prefix() -> None:
    carrier = _carrier().train()
    seq, times, _ = _packet(2, 5)
    seq.requires_grad_(True)
    out, _ = carrier(seq, times)
    out[:, -1].sum().backward()
    assert seq.grad is not None
    assert seq.grad[:, :, :TOKEN].abs().sum(-1).gt(0).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in carrier.parameters()
    )


# --------------------------------------------------------------------------
# Serializer
# --------------------------------------------------------------------------


def test_the_hidden_state_round_trips_through_a_checkpoint() -> None:
    carrier = _carrier()
    seq, times, _ = _packet(2, 4)
    _, hidden = _rollout(carrier, seq, times)
    payload = _hidden_state(hidden)
    assert payload["schema"] == GRU_HIDDEN_STATE_SCHEMA == "amago-gru-hidden-state.v1"
    assert (payload["layers"], payload["batch_size"], payload["d_hidden"]) == (
        LAYERS,
        2,
        WIDTH,
    )
    restored = _restore_hidden_state(_experiment(carrier), payload)
    assert isinstance(restored, torch.Tensor)
    torch.testing.assert_close(restored, hidden, rtol=0, atol=0)
    assert restored is not hidden


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda p: p.pop("d_hidden"), "GRU hidden state"),
        (lambda p: p.update(layers=LAYERS + 1), "disagrees with its record"),
        (
            lambda p: p.update(hidden=torch.zeros(LAYERS + 1, 2, WIDTH)),
            "disagrees with its record",
        ),
        (lambda p: p["hidden"].__setitem__((0, 0, 0), float("nan")), "not finite"),
        (lambda p: p.update(hidden=p["hidden"].double()), "float32"),
    ],
)
def test_invalid_hidden_states_are_rejected(mutate, message: str) -> None:
    carrier = _carrier()
    seq, times, _ = _packet(2, 3)
    _, hidden = _rollout(carrier, seq, times)
    payload = _hidden_state(hidden)
    mutate(payload)
    with pytest.raises(ContractError, match=message):
        _restore_hidden_state(_experiment(carrier), payload)


def test_a_state_from_a_different_carrier_geometry_is_rejected() -> None:
    carrier = _carrier()
    seq, times, _ = _packet(1, 3)
    payload = _hidden_state(_rollout(carrier, seq, times)[1])
    deeper = _carrier(n_layers=LAYERS + 1)
    with pytest.raises(ContractError, match="does not match the carrier"):
        _restore_hidden_state(_experiment(deeper), payload)
    payload = _hidden_state(_rollout(carrier, seq, times)[1])
    payload["layers"] = LAYERS + 1
    payload["hidden"] = torch.zeros(LAYERS + 1, 1, WIDTH)
    with pytest.raises(ContractError, match="does not match the carrier"):
        _restore_hidden_state(_experiment(carrier), payload)


def test_a_gru_state_never_restores_onto_another_carrier() -> None:
    gru = _carrier()
    seq, times, _ = _packet(1, 2)
    payload = _hidden_state(_rollout(gru, seq, times)[1])
    gin.clear_config()
    target = "amago.nets.traj_encoders.TformerTrajEncoder"
    settings = {
        "d_model": WIDTH,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 32,
        "attention_type": VanillaAttention,
    }
    for name, value in settings.items():
        gin.bind_parameter(f"{target}.{name}", value)
    other = HistoryTrajEncoder(
        TOKEN + 1, 8, bypass=False, token_dim=TOKEN, d_model=WIDTH
    )
    with pytest.raises(ContractError, match="does not match the policy"):
        _restore_hidden_state(_experiment(other), payload)
    gin.clear_config()


def test_cache_measurements_count_the_state_tensor() -> None:
    carrier = _carrier()
    hidden = carrier.init_hidden_state(5, torch.device("cpu"))
    measured = cache_measurements(hidden)
    assert measured["cache_bytes"] == LAYERS * 5 * WIDTH * 4
    assert measured["cache_float_bytes"] == measured["cache_bytes"]
    assert measured["cache_dtypes"] == ["torch.float32"]


# --------------------------------------------------------------------------
# Study configuration
# --------------------------------------------------------------------------


def test_the_study_binds_the_gru_carrier_explicitly() -> None:
    study = load_retired_summary_memory_study()
    config = experiment_config(
        study.contract("dark_key_to_door"),
        study,
        condition="raw_gru",
        seed=0,
        repository=ROOT,
        device="cpu",
    )
    assert config.model.architecture_id == GRU_HISTORY_ARCHITECTURE_ID
    assert config.model.dat is None
    mapping = config.as_runtime_mapping()
    assert mapping["model"]["trajectory_encoder"] == "gru"
    assert "dat" not in mapping["model"]
    mapping["model"]["public_contract"] = {"schema": "test-public-contract.v1"}
    components = configure_amago(mapping["model"], mapping["training"])
    assert components.trajectory_encoder is GRUHistoryTrajEncoder
    assert components.trajectory_encoder is not DATTrajEncoder
    target = "reasoned_icrl.model.trajectory_encoder.GRUHistoryTrajEncoder"
    assert gin.query_parameter(f"{target}.d_model") == config.model.width == 256
    assert gin.query_parameter(f"{target}.n_layers") == config.model.layers == 3
    gin.clear_config()


def test_the_resume_contract_names_the_packet_and_no_attention() -> None:
    carrier = _carrier()
    experiment = SimpleNamespace(
        encoder_architecture_id=GRU_HISTORY_ARCHITECTURE_ID,
        policy_condition="raw_gru",
        policy=SimpleNamespace(
            traj_encoder=carrier,
            tstep_encoder=SimpleNamespace(spec=SimpleNamespace(sha256="packet")),
        ),
        learner_contract="amago-optimizer-ownership.v1",
        reasoned_training_settings={},
    )
    contract = _runtime_contract(experiment)
    assert contract["packet_sha256"] == "packet"
    assert contract["condition"] == "raw_gru"
    assert "attention_sha256" not in contract
