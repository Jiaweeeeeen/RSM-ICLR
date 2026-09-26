"""The sliding-window carrier: spec §4.3 and §11 test 12.

Every check is CPU, FP32 and deterministic. The banded dense forward and the
rolling cached rollout must agree to float32 rounding across evictions; the
one-pass rebuild after a learner update must equal the live state slot for
slot; up to ``W`` records the carrier is the donor Transformer; a record beyond
the depth-scaled band has no influence at all; the state round-trips through a
checkpoint; and the spec's and the carrier's refusals are pinned.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gin
import numpy as np
import pytest
import torch
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.experiments.contracts import ContractError, WindowSpec
from reasoned_icrl.model.dat_transformer import EMPTY_TIME, DATHiddenState
from reasoned_icrl.model.trajectory_encoder import (
    HistoryTrajEncoder,
    WindowTrajEncoder,
    _copy_row,
)
from reasoned_icrl.model.window_transformer import WINDOW_HIDDEN_STATE_SCHEMA
from reasoned_icrl.runtime.checkpointing import (
    _hidden_state,
    _restore_hidden_state,
)
from reasoned_icrl.runtime.experiment import cache_measurements

TOKEN, WIDTH = 16, 32
W = 4
PARITY_ATOL = 2e-6


def _configure(layers: int, *, width: int = WIDTH, heads: int = 2) -> None:
    torch.compiler.set_stance("force_eager")
    torch.set_num_threads(1)
    gin.clear_config()
    target = "amago.nets.traj_encoders.TformerTrajEncoder"
    for name, value in {
        "d_model": width,
        "n_heads": heads,
        "n_layers": layers,
        "d_ff": 2 * width,
        "attention_type": VanillaAttention,
        "dropout_ff": 0.0,
        "dropout_emb": 0.0,
        "dropout_attn": 0.0,
        "dropout_qkv": 0.0,
    }.items():
        gin.bind_parameter(f"{target}.{name}", value)


def _carrier(
    layers: int = 2, *, window: int = W, seed: int = 0, max_seq_len: int = 16
) -> WindowTrajEncoder:
    _configure(layers)
    torch.manual_seed(seed)
    return WindowTrajEncoder(
        TOKEN + 1,
        max_seq_len,
        spec=WindowSpec(segment_length=window),
        token_dim=TOKEN,
        d_model=WIDTH,
    ).eval()


def _packet(
    batch: int, length: int, lengths: list[int] | None = None, *, seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    seq = torch.randn(batch, length, TOKEN + 1)
    valid = torch.zeros(batch, length, 1)
    for row in range(batch):
        valid[row, : length if lengths is None else lengths[row]] = 1.0
    seq[..., -1:] = valid
    times = torch.arange(length).view(1, length, 1).expand(batch, -1, -1).contiguous()
    return seq, times


def _rollout(
    carrier: WindowTrajEncoder,
    seq: torch.Tensor,
    times: torch.Tensor,
    hidden: DATHiddenState | None = None,
) -> tuple[torch.Tensor, DATHiddenState]:
    """Cached decisions on every row (every record marked valid)."""
    if hidden is None:
        hidden = carrier.init_hidden_state(seq.shape[0], torch.device("cpu"))
    outputs = []
    with torch.no_grad():
        for step in range(seq.shape[1]):
            record = seq[:, step : step + 1].clone()
            record[..., -1] = 1.0
            out, hidden = carrier(record, times[:, step : step + 1], hidden)
            outputs.append(out)
    return torch.cat(outputs, 1), hidden


def _experiment(carrier: torch.nn.Module) -> Any:
    return SimpleNamespace(
        policy=SimpleNamespace(traj_encoder=carrier), DEVICE=torch.device("cpu")
    )


def _select_row(
    carrier: WindowTrajEncoder, state: DATHiddenState, row: int
) -> DATHiddenState:
    single = carrier.init_hidden_state(1, torch.device("cpu"))
    _copy_row(state, row, single, 0)
    return single


def _assert_row_parity(
    actual: DATHiddenState, row: int, expected: DATHiddenState, expected_row: int
) -> None:
    """Equal counters and times, every retained slot equal, the rest NaN."""
    filled = int(expected.lengths[expected_row])
    assert int(actual.lengths[row]) == filled
    assert actual.times[row].tolist() == expected.times[expected_row].tolist()
    for ours, theirs in zip(actual.layers, expected.layers, strict=True):
        for name, tensor in ours.tensors().items():
            other = theirs.tensors()[name]
            torch.testing.assert_close(
                tensor[row, :filled],
                other[expected_row, :filled],
                rtol=1e-5,
                atol=PARITY_ATOL,
                msg=f"{name} row {row}",
            )
            assert torch.isnan(tensor[row, filled:]).all(), (name, row)
            assert torch.isnan(other[expected_row, filled:]).all(), (name, row)


# --------------------------------------------------------------------------
# Dense equals cached; rebuild equals live
# --------------------------------------------------------------------------


@pytest.mark.parametrize("layers", [1, 2])
def test_banded_dense_equals_the_rolling_cached_rollout(layers: int) -> None:
    carrier = _carrier(layers)
    lengths = [3 * W + 2, 2 * W + 1, W]
    seq, times = _packet(3, max(lengths), lengths)
    with torch.no_grad():
        dense, none = carrier(seq, times)
    assert none is None
    cached, hidden = _rollout(carrier, seq, times)
    for row, length in enumerate(lengths):
        assert not dense[row, length:].any()
        error = (dense[row, :length] - cached[row, :length]).abs().amax(-1)
        assert float(error.max()) <= PARITY_ATOL, (row, error.tolist())
    total = max(lengths)
    # AMAGO's post-step convention: a full row keeps W - 1 sources.
    assert hidden.lengths.tolist() == [W - 1] * 3
    assert hidden.times[:, : W - 1].tolist() == [list(range(total - W + 1, total))] * 3
    assert bool((hidden.times[:, W - 1 :] == EMPTY_TIME).all())


def test_rebuild_equals_an_uninterrupted_rollout() -> None:
    carrier = _carrier(2)
    lengths = [3 * W + 2, 2 * W, W - 1, 0, 1]  # evicting, evicting, full, empty, one
    seq, times = _packet(5, max(lengths), lengths)
    rebuilt = carrier.rebuild_hidden_state(seq, times, lengths)
    assert rebuilt.lengths.tolist() == [W - 1, W - 1, W - 1, 0, 1]
    torch.manual_seed(9)
    following = torch.randn(5, 1, TOKEN + 1)
    following[..., -1] = 1.0
    for row, length in enumerate(lengths):
        live = carrier.init_hidden_state(1, torch.device("cpu"))
        if length:
            _rollout(
                carrier, seq[row : row + 1, :length], times[:1, :length], hidden=live
            )
        _assert_row_parity(rebuilt, row, live, 0)
        next_time = torch.full((1, 1, 1), length, dtype=torch.long)
        sub = _select_row(carrier, rebuilt, row)
        with torch.no_grad():
            expected, _ = carrier(following[row : row + 1], next_time, live)
            actual, _ = carrier(following[row : row + 1], next_time, sub)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=PARITY_ATOL)
        _assert_row_parity(sub, 0, live, 0)


def test_rebuild_ignores_padding_and_refuses_bad_lengths() -> None:
    carrier = _carrier(2)
    lengths = [2 * W + 1, 3]
    seq, times = _packet(2, 2 * W + 1, lengths)
    first = carrier.rebuild_hidden_state(seq, times, lengths)
    noisy = seq.clone()
    noisy[1, 3:, :TOKEN] += 5.0  # beyond row 1's true prefix
    again = carrier.rebuild_hidden_state(noisy, times, lengths)
    for row in range(2):
        _assert_row_parity(again, row, first, row)
    with pytest.raises(ContractError, match="one true length"):
        carrier.rebuild_hidden_state(seq, times, [1])
    with pytest.raises(ContractError, match="exceeds"):
        carrier.rebuild_hidden_state(seq, times, [2 * W + 2, 3])
    with pytest.raises(ContractError, match="nonnegative"):
        carrier.rebuild_hidden_state(seq, times, [-1, 3])


# --------------------------------------------------------------------------
# The window is the donor up to W, and blind beyond the depth-scaled band
# --------------------------------------------------------------------------


def test_up_to_w_records_the_window_is_the_donor_transformer() -> None:
    window = _carrier(2, window=8)
    _configure(2)
    torch.manual_seed(0)
    donor = HistoryTrajEncoder(
        TOKEN + 1, 16, bypass=False, token_dim=TOKEN, d_model=WIDTH
    ).eval()
    for index, block in enumerate(window.backbone.layers):
        theirs = donor.backbone.tformer.layers[index].state_dict()
        for name, value in block.state_dict().items():
            assert torch.equal(value, theirs[name]), (index, name)
    seq, times = _packet(2, 8)
    with torch.no_grad():
        ours, _ = window(seq, times)
        reference, _ = donor(seq, times)
    torch.testing.assert_close(ours, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("layers", [1, 2])
def test_records_beyond_the_depth_scaled_band_have_no_influence(layers: int) -> None:
    """A band of W per layer reaches ``layers * (W - 1)`` positions back: the
    cache holds contextualised keys, so the top layer sees further than W raw
    records, and nothing beyond that reach at all."""
    carrier = _carrier(layers)
    length = 3 * W
    seq, times = _packet(1, length)
    other = seq.clone()
    other[0, 0, :TOKEN] += 1.0
    with torch.no_grad():
        base, _ = carrier(seq, times)
        moved, _ = carrier(other, times)
    reach = layers * (W - 1)
    assert not torch.allclose(base[0, reach], moved[0, reach])
    torch.testing.assert_close(
        base[0, reach + 1 :], moved[0, reach + 1 :], rtol=0, atol=0
    )


# --------------------------------------------------------------------------
# State: reset, serializer, bytes, refusals
# --------------------------------------------------------------------------


def test_reset_clears_only_the_selected_rows() -> None:
    carrier = _carrier(1)
    seq, times = _packet(2, W + 1)
    _, hidden = _rollout(carrier, seq, times)
    kept = {
        f"{index}.{name}": tensor[1].clone()
        for index, cache in enumerate(hidden.layers)
        for name, tensor in cache.tensors().items()
    }
    kept_times = hidden.times[1].clone()
    assert carrier.reset_hidden_state(hidden, np.array([True, False])) is hidden
    assert hidden.lengths.tolist() == [0, W - 1]
    assert bool((hidden.times[0] == EMPTY_TIME).all())
    assert hidden.times[1].tolist() == kept_times.tolist()
    for index, cache in enumerate(hidden.layers):
        for name, tensor in cache.tensors().items():
            assert torch.isnan(tensor[0]).all()
            torch.testing.assert_close(
                tensor[1], kept[f"{index}.{name}"], rtol=0, atol=0, equal_nan=True
            )


def test_the_hidden_state_round_trips_through_a_checkpoint() -> None:
    carrier = _carrier(2)
    seq, times = _packet(2, W + 2)
    _, hidden = _rollout(carrier, seq, times)
    payload = _hidden_state(hidden)
    assert payload["schema"] == WINDOW_HIDDEN_STATE_SCHEMA
    assert payload["window_sha256"] == carrier.spec.sha256
    restored = _restore_hidden_state(_experiment(carrier), payload)
    assert isinstance(restored, DATHiddenState) and restored is not hidden
    assert restored.schema == WINDOW_HIDDEN_STATE_SCHEMA
    for row in range(2):
        _assert_row_parity(restored, row, hidden, row)
    next_time = torch.full((2, 1, 1), W + 2, dtype=torch.long)
    with torch.no_grad():
        a, _ = carrier(seq[:, :1], next_time, hidden)
        b, _ = carrier(seq[:, :1], next_time, restored)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    other = _carrier(2, window=W + 1)
    with pytest.raises(ContractError, match="window identity does not match"):
        _restore_hidden_state(_experiment(other), payload)
    relabeled = dict(payload)
    relabeled["schema"] = "amago-dat-hidden-state.v1"
    relabeled["attention_sha256"] = relabeled.pop("window_sha256")
    with pytest.raises(ContractError, match="does not match the policy"):
        _restore_hidden_state(_experiment(carrier), relabeled)


def test_the_study_window_allocates_the_documented_bytes() -> None:
    """Plan §5.2: 3 layers x (K, V) x 40 slots x 256 x 4 B of cache per actor."""
    _configure(3, width=256, heads=8)
    torch.manual_seed(0)
    carrier = WindowTrajEncoder(
        TOKEN + 1, 16, spec=WindowSpec(segment_length=40), token_dim=TOKEN, d_model=256
    ).eval()
    measured = cache_measurements(carrier.init_hidden_state(1, torch.device("cpu")))
    assert measured["cache_float_bytes"] == 3 * 2 * 40 * 256 * 4 == 245_760
    assert measured["cache_bytes"] == 245_760 + 40 * 8 + 4  # times and the length
    assert sum(p.numel() for p in carrier.parameters()) == sum(
        p.numel() for p in carrier.backbone.parameters()
    )


def test_the_spec_and_the_carrier_refuse_unqualified_settings() -> None:
    for kwargs, message in [
        ({"segment_length": 3}, "segment_length"),
        ({"segment_length": 4, "cache_dtype": "float64"}, "cache dtype"),
        ({"segment_length": 4, "schema": "window-memory.v0"}, "schema"),
    ]:
        with pytest.raises(ContractError, match=message):
            WindowSpec(**kwargs)  # type: ignore[arg-type]
    spec = WindowSpec(segment_length=4)
    assert spec.capacity == 4 and spec.to_dict()["sha256"] == spec.sha256
    assert spec.sha256 != WindowSpec(segment_length=5).sha256
    _configure(1)
    with pytest.raises(ContractError, match="resolved window spec"):
        WindowTrajEncoder(TOKEN + 1, 8, token_dim=TOKEN, d_model=WIDTH)
    with pytest.raises(ContractError, match="without the state bypass"):
        WindowTrajEncoder(2 * TOKEN + 1, 8, spec=spec, token_dim=TOKEN, d_model=WIDTH)
    carrier = _carrier(1)
    seq, times = _packet(2, 3)
    hidden = carrier.init_hidden_state(2, torch.device("cpu"))
    with pytest.raises(ContractError, match="one valid decision"):
        carrier(seq[:, :2], times[:, :2], hidden)
    invalid = seq[:, :1].clone()
    invalid[0, 0, -1] = 0.0
    with pytest.raises(ContractError, match="one valid decision"):
        carrier(invalid, times[:, :1], hidden)
    carrier.train()
    with pytest.raises(ContractError, match="evaluation-only"):
        carrier(seq[:, :1], times[:, :1], hidden)
    carrier.eval()
    other = _carrier(1, window=W + 1)
    with pytest.raises(ContractError, match="different carrier"):
        other(seq[:, :1], times[:, :1], hidden)
    holes = seq.clone()
    holes[0, 1, -1] = 0.0
    with pytest.raises(ContractError, match="right-padded"):
        carrier(holes, times)
