"""The Memo comparator carrier (ME1 acceptance): dense/cached/rebuild parity
over several accumulated summaries, unequal actor lengths, exact boundaries
and asynchronous outer resets; causality; full gradients through every
accumulated summary; the cleared intervention; the training segmentation;
independent resets; the growing live state and the spec's refusals.

Every check is CPU, FP32 and deterministic. The ordinary blocks are the
donor's own, so the masked-block parity in ``test_summary_transformer.py``
covers them; what is pinned here is the Memo lifecycle.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import gin
import numpy as np
import pytest
import torch
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.experiments.contracts import (
    ContractError,
    MemoSpec,
    memo_live_slots,
    memo_state_bytes,
)
from reasoned_icrl.model.memo_transformer import MemoHiddenState, MemoTransformer
from reasoned_icrl.model.trajectory_encoder import MemoTrajEncoder
from reasoned_icrl.runtime.experiment import cache_measurements, state_tensor_bytes

TOKEN, WIDTH, LAYERS, HEADS = 16, 32, 2, 2
L, S = 4, 2
MAX_INDEX = 19  # tasks of at most 20 records: up to four boundaries
PARITY_ATOL = 2e-6


def _configure() -> None:
    torch.compiler.set_stance("force_eager")
    torch.set_num_threads(1)
    gin.clear_config()
    target = "amago.nets.traj_encoders.TformerTrajEncoder"
    for name, value in {
        "d_model": WIDTH,
        "n_heads": HEADS,
        "n_layers": LAYERS,
        "d_ff": 64,
        "attention_type": VanillaAttention,
        "dropout_ff": 0.0,
        "dropout_emb": 0.0,
        "dropout_attn": 0.0,
        "dropout_qkv": 0.0,
    }.items():
        gin.bind_parameter(f"{target}.{name}", value)


def _spec(**kw: Any) -> MemoSpec:
    settings: dict[str, Any] = {
        "segment_length": L,
        "summary_tokens": S,
        "d_model": WIDTH,
    }
    settings.update(kw)
    return MemoSpec(**settings)


def _carrier(
    *, seed: int = 0, max_seq_len: int = MAX_INDEX, **kw: Any
) -> MemoTrajEncoder:
    _configure()
    torch.manual_seed(seed)
    return MemoTrajEncoder(
        TOKEN + 1,
        max_seq_len,
        spec=_spec(**kw),
        token_dim=TOKEN,
        d_model=WIDTH,
        initialization_seed=seed,
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
    carrier: MemoTrajEncoder,
    seq: torch.Tensor,
    times: torch.Tensor,
    steps: int | None = None,
    hidden: MemoHiddenState | None = None,
) -> tuple[torch.Tensor, MemoHiddenState]:
    """Cached decisions on every row (every record marked valid)."""
    if hidden is None:
        hidden = carrier.init_hidden_state(seq.shape[0], torch.device("cpu"))
    outputs = []
    with torch.no_grad():
        for step in range(seq.shape[1] if steps is None else steps):
            record = seq[:, step : step + 1].clone()
            record[..., -1] = 1.0
            out, hidden = carrier(record, times[:, step : step + 1], hidden)
            outputs.append(out)
    return torch.cat(outputs, 1), hidden


def _assert_state_parity(actual: MemoHiddenState, expected: MemoHiddenState) -> None:
    """Equal counters, every filled cache slot equal, the rest NaN."""
    assert actual.lengths.tolist() == expected.lengths.tolist()
    assert actual.segment.tolist() == expected.segment.tolist()
    for ours, theirs in zip(actual.layers, expected.layers, strict=True):
        for name, tensor in ours.tensors().items():
            other = theirs.tensors()[name]
            for row, filled in enumerate(expected.lengths.tolist()):
                torch.testing.assert_close(
                    tensor[row, :filled],
                    other[row, :filled],
                    rtol=1e-5,
                    atol=PARITY_ATOL,
                    msg=f"{name} row {row}",
                )
                assert torch.isnan(tensor[row, filled:]).all(), (name, row)
                assert torch.isnan(other[row, filled:]).all(), (name, row)


# --------------------------------------------------------------------------
# 1-2. Dense equals cached over several accumulated summaries; rebuild equals
#      an uninterrupted rollout
# --------------------------------------------------------------------------


def test_training_forward_equals_the_cached_rollout_across_boundaries() -> None:
    carrier = _carrier()
    lengths = [
        4 * L + 3,
        3 * L + 2,
        2 * L + 1,
        L,
        4 * L,
    ]  # four, three, two, zero, three
    seq, times = _packet(5, max(lengths), lengths)
    with torch.no_grad():
        dense, none = carrier(seq, times)
    assert none is None
    cached, hidden = _rollout(carrier, seq, times)
    for row, length in enumerate(lengths):
        assert not dense[row, length:].any()
        error = (dense[row, :length] - cached[row, :length]).abs().amax(-1)
        assert float(error.max()) <= PARITY_ATOL, (row, error.tolist())
    # Every row ran the same 19 records: four boundaries crossed, three records
    # of the fifth segment open behind 4 * S accumulated summary slots.
    assert hidden.segment.tolist() == [4] * 5
    assert hidden.lengths.tolist() == [memo_live_slots(carrier.spec, 4 * L + 3)] * 5
    assert hidden.lengths.tolist() == [4 * S + 3] * 5


def test_rebuild_equals_an_uninterrupted_rollout() -> None:
    carrier = _carrier()
    lengths = [
        3 * L + 2,
        2 * L,
        L,
        0,
        L + 1,
        4 * L,
    ]  # inside, at, at, empty, inside, at
    seq, times = _packet(6, max(lengths), lengths)
    rebuilt = carrier.rebuild_hidden_state(seq, times, lengths)
    # The cached rollout crosses a boundary lazily, so a prefix ending exactly
    # at one keeps its full segment with the summary pending.
    assert rebuilt.segment.tolist() == [3, 1, 0, 0, 1, 3]
    assert rebuilt.lengths.tolist() == [
        memo_live_slots(carrier.spec, length) for length in lengths
    ]
    assert rebuilt.lengths.tolist() == [3 * S + 2, S + L, L, 0, S + 1, 3 * S + L]
    torch.manual_seed(9)
    following = torch.randn(6, 2, TOKEN + 1)
    following[..., -1] = 1.0
    for row, length in enumerate(lengths):
        live = carrier.init_hidden_state(1, torch.device("cpu"))
        if length:
            _rollout(
                carrier, seq[row : row + 1, :length], times[:1, :length], hidden=live
            )
        sub = rebuilt.select(torch.tensor([row]))
        _assert_state_parity(sub, live)
        expected, _ = _rollout(
            carrier, following[row : row + 1], times[:1, :2], hidden=live
        )
        actual, _ = _rollout(
            carrier, following[row : row + 1], times[:1, :2], hidden=sub
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=PARITY_ATOL)
        _assert_state_parity(sub, live)


def test_rebuild_ignores_beyond_the_true_prefix_and_refuses_bad_lengths() -> None:
    carrier = _carrier()
    seq, times = _packet(2, 2 * L + 1)
    lengths = [L + 1, 2]
    rebuilt = carrier.rebuild_hidden_state(seq, times, lengths)
    tampered = seq.clone()
    tampered[0, L + 1 :, :TOKEN] = 99.0
    tampered[1, 2:, :TOKEN] = -99.0
    again = carrier.rebuild_hidden_state(tampered, times, lengths)
    for left, right in zip(rebuilt.layers, again.layers, strict=True):
        for name, tensor in left.tensors().items():
            torch.testing.assert_close(
                tensor, right.tensors()[name], rtol=0, atol=0, equal_nan=True
            )
    with pytest.raises(ContractError, match="one true length per row"):
        carrier.rebuild_hidden_state(seq, times, [1])
    with pytest.raises(ContractError, match="exceeds the provided packets"):
        carrier.rebuild_hidden_state(seq, times, [2 * L + 2, 0])


def test_rebuild_after_a_weight_change_equals_a_live_rollout_under_new_weights() -> (
    None
):
    carrier = _carrier()
    seq, times = _packet(2, 3 * L + 1, [3 * L + 1, 2 * L + 2])
    with torch.no_grad():
        for parameter in carrier.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    rebuilt = carrier.rebuild_hidden_state(seq, times, [3 * L + 1, 2 * L + 2])
    live_outputs, live = _rollout(carrier, seq[:, : 2 * L + 2], times[:, : 2 * L + 2])
    del live_outputs
    _rollout(
        carrier,
        seq[:1, 2 * L + 2 : 3 * L + 1],
        times[:1, 2 * L + 2 : 3 * L + 1],
        hidden=live.select(torch.tensor([0])),
    )
    # Row 1 stopped at 2L + 2 in both; compare it directly.
    _assert_state_parity(
        rebuilt.select(torch.tensor([1])), live.select(torch.tensor([1]))
    )


# --------------------------------------------------------------------------
# 3. Causality: future records and summary queries cannot reach earlier
#    outputs; a boundary's summary influences only later segments
# --------------------------------------------------------------------------


def test_future_records_and_summaries_cannot_reach_earlier_outputs() -> None:
    carrier = _carrier()
    seq, times = _packet(1, 3 * L + 3)
    with torch.no_grad():
        base, _ = carrier(seq, times)
        for position in (1, L - 1, L, L + 2, 2 * L + 1, 3 * L):
            changed = seq.clone()
            changed[:, position, :TOKEN] += 20.0
            out, _ = carrier(changed, times)
            torch.testing.assert_close(
                out[:, :position], base[:, :position], rtol=0, atol=0
            )
            assert not torch.allclose(out[:, position:], base[:, position:])
    # Perturbing the summary embeddings changes nothing in the first segment
    # (no summary has been written yet) and everything after the first boundary.
    with torch.no_grad():
        carrier.backbone.summary_embedding.add_(5.0)
        out, _ = carrier(seq, times)
    torch.testing.assert_close(out[:, :L], base[:, :L], rtol=0, atol=0)
    assert not torch.allclose(out[:, L:], base[:, L:])


# --------------------------------------------------------------------------
# 4. Gradients through every accumulated summary; the cleared rollout
# --------------------------------------------------------------------------


def _weighted_loss(
    carrier: MemoTrajEncoder, seq: torch.Tensor, times: torch.Tensor, position: int
) -> torch.Tensor:
    # A random projection of the output: a LayerNorm output summed over its
    # features has an exactly zero gradient, which would hide everything.
    torch.manual_seed(3)
    weights = torch.randn(WIDTH)
    out, _ = carrier(seq, times)
    return (out[0, position] * weights).sum()


def test_a_later_segment_loss_reaches_every_earlier_segment_and_the_queries() -> None:
    carrier = _carrier()
    carrier.train()
    seq, times = _packet(1, 4 * L)
    seq.requires_grad_(True)
    loss = _weighted_loss(carrier, seq, times, 4 * L - 1)
    grads = torch.autograd.grad(loss, [seq, carrier.backbone.summary_embedding])
    per_record = grads[0][0, :, :TOKEN].norm(dim=-1)
    # Segments 0, 1 and 2 reach the last decision only through their
    # summaries; segment 3 directly. Every record carries gradient.
    assert bool((per_record[: 3 * L] > 0).all()), per_record.tolist()
    assert bool((per_record[3 * L :] > 0).all())
    assert float(grads[1].norm()) > 0
    # Nothing is detached: the summary of segment 0 feeds the summary of
    # segment 1 which feeds segment 3's block, so the record gradients of
    # segment 0 differ from zero even when segment 3 reads only summaries.
    carrier.eval()


def test_summary_cleared_discards_the_summaries_at_every_boundary() -> None:
    carrier = _carrier()
    seq, times = _packet(2, 3 * L + 1)
    hidden = carrier.init_hidden_state(2, torch.device("cpu"))
    hidden.summary_cleared = True
    cached, hidden = _rollout(carrier, seq, times, hidden=hidden)
    assert hidden.segment.tolist() == [3, 3]
    assert hidden.lengths.tolist() == [1, 1]  # no summary slots survive a boundary
    # Each segment is then the first segment of a fresh task: the dense path
    # over each segment alone reproduces it.
    with torch.no_grad():
        for start in range(0, 3 * L + 1, L):
            stop = min(start + L, 3 * L + 1)
            dense, _ = carrier(seq[:, start:stop], times[:, : stop - start])
            error = (dense - cached[:, start:stop]).abs().max()
            assert float(error) <= PARITY_ATOL, start
    # A retained rollout of the same records differs after the first boundary.
    retained, _ = _rollout(carrier, seq, times)
    torch.testing.assert_close(retained[:, :L], cached[:, :L], rtol=0, atol=0)
    assert not torch.allclose(retained[:, L:], cached[:, L:])


# --------------------------------------------------------------------------
# 5. Training segmentation: one uniform draw per forward, rollout fixed
# --------------------------------------------------------------------------


def test_training_draws_one_segment_length_per_forward_from_the_jitter_range() -> None:
    spec = _spec(segment_length=32, summary_tokens=4)
    assert spec.jitter_range == (26, 38)
    assert _spec(training_segment_jitter=0.0).jitter_range == (L, L)
    carrier = _carrier(segment_length=10, summary_tokens=2, max_seq_len=40)
    low, high = carrier.spec.jitter_range
    assert (low, high) == (8, 12)
    carrier.train()
    torch.manual_seed(0)
    draws = {carrier.backbone.dense_segment_length() for _ in range(200)}
    assert draws == set(range(low, high + 1))
    carrier.eval()
    assert {carrier.backbone.dense_segment_length() for _ in range(20)} == {10}
    # Under training the dense pass segments the same task differently, so
    # its outputs differ from the fixed-L rollout; in evaluation they agree.
    seq, times = _packet(1, 25)
    torch.manual_seed(1)
    carrier.train()
    out_a, _ = carrier(seq, times)
    torch.manual_seed(2)
    out_b, _ = carrier(seq, times)
    assert not torch.allclose(out_a, out_b)
    carrier.eval()
    with torch.no_grad():
        dense, _ = carrier(seq, times)
    cached, _ = _rollout(carrier, seq, times)
    assert float((dense - cached).abs().max()) <= PARITY_ATOL


def test_fixed_segmentation_trains_on_the_rollout_segments() -> None:
    """``memo_fixed``: jitter 0 makes the dense training pass segment exactly
    as the rollout does, so with dropout off the training-mode outputs equal
    the cached rollout; the identity differs from the jittered spec's."""
    carrier = _carrier(
        segment_length=10, summary_tokens=2, max_seq_len=40, training_segment_jitter=0.0
    )
    assert carrier.spec.jitter_range == (10, 10)
    assert carrier.spec.sha256 != _spec(segment_length=10, summary_tokens=2).sha256
    seq, times = _packet(2, 33)
    carrier.train()
    torch.manual_seed(1)
    assert {carrier.backbone.dense_segment_length() for _ in range(50)} == {10}
    out_a, _ = carrier(seq, times)
    torch.manual_seed(2)
    out_b, _ = carrier(seq, times)
    # Two training forwards differ only by the sigma-reparam power iteration
    # AMAGO's linear layers run in training mode, not by the segmentation.
    torch.testing.assert_close(out_a, out_b, rtol=1e-4, atol=1e-5)
    carrier.eval()
    cached, hidden = _rollout(carrier, seq, times)
    assert float((out_a.detach() - cached).abs().max()) <= 1e-5
    assert hidden.segment.tolist() == [3, 3]
    # Gradients still reach every earlier segment through the summaries.
    carrier.train()
    seq.requires_grad_(True)
    grads = torch.autograd.grad(
        _weighted_loss(carrier, seq, times, 32),
        [seq, carrier.backbone.summary_embedding],
    )
    assert bool((grads[0][0, :, :TOKEN].norm(dim=-1) > 0).all())
    assert float(grads[1].norm()) > 0


def test_every_valid_record_enters_exactly_one_segment(monkeypatch: Any) -> None:
    carrier = _carrier()
    seen: list[tuple[int, int, int]] = []
    original = carrier.backbone.block_forward

    def spy(x: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        seen.append((x.shape[1], int(key_valid[0].sum()), int(key_valid[1].sum())))
        return original(x, key_valid)

    monkeypatch.setattr(carrier.backbone, "block_forward", spy)
    seq, times = _packet(2, 2 * L + 3, [2 * L + 3, L + 1])
    with torch.no_grad():
        carrier(seq, times)
    # Blocks: [records L | queries S], [summaries S | records L | queries S],
    # [summaries 2S | records 3]; row 1 is valid for L + 1 records only.
    assert seen == [
        (L + S, L + S, L + S),
        (S + L + S, S + L + S, S + 1 + S),
        (2 * S + 3, 2 * S + 3, 2 * S),
    ]


# --------------------------------------------------------------------------
# 6. Resets are independent; the live state grows with the boundaries
# --------------------------------------------------------------------------


def _snapshot(hidden: MemoHiddenState, row: int) -> dict[str, torch.Tensor]:
    out = {
        "lengths": hidden.lengths[row].clone(),
        "segment": hidden.segment[row].clone(),
    }
    for index, cache in enumerate(hidden.layers):
        for name, tensor in cache.tensors().items():
            out[f"{index}.{name}"] = tensor[row].clone()
    return out


def test_reset_leaves_other_rows_byte_identical_and_restarts_the_row() -> None:
    carrier = _carrier()
    seq, times = _packet(3, 2 * L + 1)
    _, hidden = _rollout(carrier, seq, times)
    before = {row: _snapshot(hidden, row) for row in range(3)}
    carrier.reset_hidden_state(hidden, np.array([False, True, False]))
    for row in (0, 2):
        for name, tensor in before[row].items():
            torch.testing.assert_close(
                _snapshot(hidden, row)[name], tensor, rtol=0, atol=0, equal_nan=True
            )
    assert int(hidden.lengths[1]) == 0 and int(hidden.segment[1]) == 0
    for cache in hidden.layers:
        for tensor in cache.tensors().values():
            assert torch.isnan(tensor[1]).all()
    # The reset row restarts as a fresh task while the others continue.
    fresh = carrier.init_hidden_state(1, torch.device("cpu"))
    expected, _ = _rollout(carrier, seq[1:2, :3], times[:1, :3], hidden=fresh)
    actual, _ = _rollout(carrier, seq[:, :3], times[:, :3], hidden=hidden)
    torch.testing.assert_close(actual[1:2], expected, rtol=1e-5, atol=PARITY_ATOL)
    assert hidden.segment.tolist() == [2, 0, 2]


def test_the_live_state_grows_by_s_per_boundary_within_the_allocation() -> None:
    carrier = _carrier()
    spec = carrier.spec
    assert carrier.capacity == spec.capacity(MAX_INDEX) == 4 * S + L + S
    single = carrier.init_hidden_state(1, torch.device("cpu"))
    allocated = cache_measurements(single)["cache_bytes"]
    assert allocated == memo_state_bytes(
        spec, max_index=MAX_INDEX, layers=LAYERS, heads=HEADS, width=WIDTH
    )
    assert sum(state_tensor_bytes(single).values()) == allocated
    # The allocation covers every task the carrier was built for and a little
    # more: the last summary of a 20-record task is never written, so the
    # cache also fits the sixth segment's records; the boundary that would
    # need a seventh summary slot pair is the first thing that overflows.
    overflow = 6 * L + 1
    seq, times = _packet(1, overflow)
    hidden = carrier.init_hidden_state(1, torch.device("cpu"))
    observed = []
    for step in range(overflow - 1):
        _rollout(
            carrier, seq[:, step : step + 1], times[:, step : step + 1], hidden=hidden
        )
        observed.append(int(hidden.lengths[0]))
        assert observed[-1] == memo_live_slots(spec, step + 1)
        assert observed[-1] <= carrier.capacity
    # Filled slots rise through a segment and drop by L - S at each boundary.
    assert observed[:L] == list(range(1, L + 1))
    assert observed[L] == S + 1 and observed[2 * L] == 2 * S + 1
    assert observed[MAX_INDEX] == 4 * S + L  # the longest declared task, full
    assert max(observed) == carrier.capacity
    # A task longer than the allocation is refused rather than evicted.
    with pytest.raises(ContractError, match="longer than the carrier was built for"):
        _rollout(carrier, seq[:, -1:], times[:, -1:], hidden=hidden)


# --------------------------------------------------------------------------
# 7. Spec and construction refusals; the identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"segment_length": 3}, "segment_length"),
        ({"summary_tokens": 0}, "summary_tokens"),
        ({"segment_length": 4, "summary_tokens": 4}, "fewer summary tokens"),
        ({"carry": "overwrite"}, "RMT variant"),
        ({"training_segment_jitter": 1.0}, "jitter"),
        ({"detach": "segment"}, "detach none"),
        ({"position": "segment-local"}, "concatenated"),
        ({"cache_dtype": "int8"}, "cache dtype"),
        ({"schema": "memo-summary.v0"}, "schema"),
    ],
)
def test_the_spec_refuses_unqualified_settings(
    settings: dict[str, Any], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        _spec(**settings)


def test_the_spec_identity_changes_with_every_field() -> None:
    base = _spec()
    variants = {
        "segment_length": 8,
        "summary_tokens": 3,
        "training_segment_jitter": 0.0,
        "d_model": 64,
        "cache_dtype": "bfloat16",
    }
    assert set(variants) <= {field.name for field in fields(MemoSpec)}
    for name, value in variants.items():
        assert replace(base, **{name: value}).sha256 != base.sha256, name
    assert base.to_dict()["sha256"] == base.sha256
    assert base.summaries_before(0) == 0 and base.summaries_before(L) == 1
    assert base.capacity(L) == S + L + S
    with pytest.raises(ContractError):
        base.summaries_before(-1)


def test_the_carrier_refuses_inconsistent_construction_and_use() -> None:
    _configure()
    with pytest.raises(ContractError, match="requires a resolved Memo spec"):
        MemoTrajEncoder(TOKEN + 1, MAX_INDEX, token_dim=TOKEN, d_model=WIDTH)
    with pytest.raises(ContractError, match="without the state bypass"):
        MemoTrajEncoder(
            2 * TOKEN + 1, MAX_INDEX, spec=_spec(), token_dim=TOKEN, d_model=WIDTH
        )
    with pytest.raises(ContractError, match="width disagrees"):
        MemoTrajEncoder(
            TOKEN + 1, MAX_INDEX, spec=_spec(d_model=64), token_dim=TOKEN, d_model=WIDTH
        )
    # A task shorter than one segment is legal: no boundary, L + S slots.
    short = MemoTrajEncoder(
        TOKEN + 1, L - 1, spec=_spec(), token_dim=TOKEN, d_model=WIDTH
    )
    assert short.capacity == L + S
    carrier = _carrier()
    assert isinstance(carrier.backbone, MemoTransformer)
    assert carrier.memo_protocol_identity.tolist() == list(
        bytes.fromhex(carrier.spec.sha256)
    )
    seq, times = _packet(2, 3)
    hidden = carrier.init_hidden_state(2, torch.device("cpu"))
    with pytest.raises(ContractError, match="one valid decision per row"):
        carrier(seq, times, hidden)
    other = _carrier(summary_tokens=1)
    with pytest.raises(ContractError, match="different carrier"):
        other(seq[:, :1], times[:, :1], hidden)
    carrier.train()
    with pytest.raises(ContractError, match="evaluation-only"):
        carrier(seq[:, :1], times[:, :1], hidden)
    carrier.eval()
    # Only the summary embeddings are new; every other parameter is the donor's.
    names = {name for name, _ in carrier.named_parameters()}
    assert "backbone.summary_embedding" in names
    assert not any(
        "memory" in name or "write" in name or "role" in name for name in names
    )
