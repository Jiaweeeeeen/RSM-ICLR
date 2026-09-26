"""The segment/summary carrier: spec §11 tests 1-10.

Every check is CPU, FP32 and deterministic. Dense training execution and the
cached rollout must agree to float32 rounding (2e-6, flat across slots) across
boundaries, for both regimes, both writers and both relational routes; the
rebuild after a learner update must equal an uninterrupted rollout; the one
content mask rule, record ownership, gradient routing, boundedness, resets and
the spec's refusals are each pinned on their own. The timestep-record route's
own semantics are in ``test_summary_routing.py``.
"""

from __future__ import annotations

import copy
from dataclasses import fields, replace
from typing import Any

import gin
import numpy as np
import pytest
import torch
from amago.nets.traj_encoders import TformerTrajEncoder
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.experiments.contracts import ContractError, DATSpec, SummarySpec
from reasoned_icrl.model.dat_transformer import DATBlock
from reasoned_icrl.model.summary_transformer import (
    MaskedOrdinaryBlock,
    SummaryHiddenState,
    SummaryTransformer,
)
from reasoned_icrl.model.trajectory_encoder import SummaryTrajEncoder
from reasoned_icrl.runtime.experiment import cache_measurements

TOKEN, WIDTH, LAYERS = 16, 32, 2
C, M = 4, 2
PARITY_ATOL = 2e-6
LEGACY, ROUTED = "causal_prefix", "timestep_records"
REGIMES = [
    ("summary", "same", False, LEGACY),
    ("segment", "same", False, LEGACY),
    ("summary", "same", True, LEGACY),
    ("segment", "same", True, LEGACY),
    ("summary", "relational_off", True, LEGACY),
    ("summary", "same", True, ROUTED),
    ("segment", "same", True, ROUTED),
]


def _configure() -> None:
    torch.compiler.set_stance("force_eager")
    torch.set_num_threads(1)
    gin.clear_config()
    target = "amago.nets.traj_encoders.TformerTrajEncoder"
    for name, value in {
        "d_model": WIDTH,
        "n_heads": 2,
        "n_layers": LAYERS,
        "d_ff": 64,
        "attention_type": VanillaAttention,
        "dropout_ff": 0.0,
        "dropout_emb": 0.0,
        "dropout_attn": 0.0,
        "dropout_qkv": 0.0,
    }.items():
        gin.bind_parameter(f"{target}.{name}", value)


def _spec(
    regime: str = "summary",
    writer: str = "same",
    route: str = LEGACY,
    **kw: Any,
) -> SummarySpec:
    settings: dict[str, Any] = {
        "segment_length": C,
        "memory_tokens": M,
        "regime": regime,
        "writer": writer,
        "relational_sources": route,
        "d_model": WIDTH,
    }
    settings.update(kw)
    return SummarySpec(**settings)


def _dat(capacity: int = M + C + M, **kw: Any) -> DATSpec:
    settings: dict[str, Any] = {
        "layer_indices": (LAYERS - 1,),
        "d_model": WIDTH,
        "total_heads": 2,
        "relational_heads": 1,
        "relation_channels": 4,
        "relation_projection_dim": 4,
        "max_relative_distance": capacity,
    }
    settings.update(kw)
    return DATSpec(**settings)


def _carrier(
    regime: str = "summary",
    writer: str = "same",
    dat: bool = False,
    route: str = LEGACY,
    *,
    seed: int = 0,
    max_seq_len: int = 16,
    **spec_kw: Any,
) -> SummaryTrajEncoder:
    _configure()
    torch.manual_seed(seed)
    return SummaryTrajEncoder(
        TOKEN + 1,
        max_seq_len,
        spec=_spec(regime, writer, route, **spec_kw),
        dat=_dat() if dat else None,
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
    carrier: SummaryTrajEncoder,
    seq: torch.Tensor,
    times: torch.Tensor,
    steps: int | None = None,
    hidden: SummaryHiddenState | None = None,
) -> tuple[torch.Tensor, SummaryHiddenState]:
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


def _assert_state_parity(
    actual: SummaryHiddenState, expected: SummaryHiddenState
) -> None:
    """Equal memory and counters, every filled cache slot equal, the rest NaN."""
    assert actual.lengths.tolist() == expected.lengths.tolist()
    assert actual.segment.tolist() == expected.segment.tolist()
    torch.testing.assert_close(
        actual.memory, expected.memory, rtol=1e-5, atol=PARITY_ATOL
    )
    torch.testing.assert_close(
        actual.initial_memory, expected.initial_memory, rtol=0, atol=0
    )
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
# 1. The masked ordinary block is the donor layer
# --------------------------------------------------------------------------


def test_masked_ordinary_block_equals_the_donor_layer_under_a_causal_mask() -> None:
    _configure()
    torch.manual_seed(0)
    donor = TformerTrajEncoder(tstep_dim=TOKEN, max_seq_len=16).eval()
    layer = donor.tformer.layers[0]
    block = MaskedOrdinaryBlock(layer).eval()
    assert block.state_dict().keys() == layer.state_dict().keys()
    donor_parameters = dict(layer.named_parameters())
    assert all(
        parameter is donor_parameters[name]
        for name, parameter in block.named_parameters()
    )
    torch.manual_seed(2)
    x = torch.randn(2, 6, WIDTH)
    allowed = torch.tril(torch.ones(6, 6, dtype=torch.bool)).view(1, 1, 6, 6)
    with torch.no_grad():
        torch.testing.assert_close(
            block(x, allowed.expand(2, 1, 6, 6)), layer(x), rtol=1e-5, atol=1e-5
        )


# --------------------------------------------------------------------------
# 2-3. Dense equals cached, rebuild equals live rollout
# --------------------------------------------------------------------------


@pytest.mark.parametrize("regime,writer,dat,route", REGIMES)
def test_training_forward_equals_the_cached_rollout_across_boundaries(
    regime: str, writer: str, dat: bool, route: str
) -> None:
    carrier = _carrier(regime, writer, dat, route)
    lengths = [3 * C + 2, 2 * C + 1, C]  # three, two and one boundary crossed
    seq, times = _packet(3, max(lengths), lengths)
    with torch.no_grad():
        dense, none = carrier(seq, times)
    assert none is None
    cached, hidden = _rollout(carrier, seq, times)
    for row, length in enumerate(lengths):
        assert not dense[row, length:].any()
        error = (dense[row, :length] - cached[row, :length]).abs().amax(-1)
        assert float(error.max()) <= PARITY_ATOL, (row, error.tolist())
    assert hidden.segment.tolist() == [3, 3, 3]
    assert hidden.lengths.tolist() == [M + 2, M + 2, M + 2]


@pytest.mark.parametrize("regime,writer,dat,route", REGIMES)
def test_rebuild_equals_an_uninterrupted_rollout(
    regime: str, writer: str, dat: bool, route: str
) -> None:
    carrier = _carrier(regime, writer, dat, route)
    lengths = [3 * C + 2, 2 * C, C, 0, C + 1]  # inside, at, at, empty, inside
    seq, times = _packet(5, max(lengths), lengths)
    rebuilt = carrier.rebuild_hidden_state(seq, times, lengths)
    # The cached rollout crosses a boundary lazily, so a prefix ending exactly
    # at one keeps its full segment with the write pending: q = (L - 1) // C
    # segments written, r = L - qC records (1 <= r <= C) in the open segment.
    assert rebuilt.segment.tolist() == [3, 1, 0, 0, 1]
    assert rebuilt.lengths.tolist() == [M + 2, M + C, M + C, 0, M + 1]
    torch.manual_seed(9)
    following = torch.randn(5, 2, TOKEN + 1)
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
    seq, times = _packet(2, 2 * C + 1)
    lengths = [C + 1, 2]
    rebuilt = carrier.rebuild_hidden_state(seq, times, lengths)
    tampered = seq.clone()
    tampered[0, C + 1 :, :TOKEN] = 99.0
    tampered[1, 2:, :TOKEN] = -99.0
    again = carrier.rebuild_hidden_state(tampered, times, lengths)
    torch.testing.assert_close(rebuilt.memory, again.memory, rtol=0, atol=0)
    for left, right in zip(rebuilt.layers, again.layers, strict=True):
        for name, tensor in left.tensors().items():
            torch.testing.assert_close(
                tensor, right.tensors()[name], rtol=0, atol=0, equal_nan=True
            )
    with pytest.raises(ContractError, match="one true length per row"):
        carrier.rebuild_hidden_state(seq, times, [1])
    with pytest.raises(ContractError, match="exceeds the provided packets"):
        carrier.rebuild_hidden_state(seq, times, [2 * C + 2, 0])


# --------------------------------------------------------------------------
# 4. Causality and the one mask rule
# --------------------------------------------------------------------------


def test_future_records_and_writes_cannot_reach_earlier_outputs() -> None:
    carrier = _carrier("summary", dat=True)
    seq, times = _packet(1, 2 * C + 3)
    with torch.no_grad():
        base, _ = carrier(seq, times)
        for position in (1, C - 1, C, C + 2, 2 * C + 1):
            changed = seq.clone()
            changed[:, position, :TOKEN] += 20.0
            later, _ = carrier(changed, times)
            torch.testing.assert_close(
                later[:, :position], base[:, :position], rtol=0, atol=0
            )
            assert not torch.equal(later[:, position], base[:, position])
        # Write outputs of segment j reach segment j+1, never segment j.
        backbone = carrier.backbone
        backbone.write_queries.add_(1.0)
        moved, _ = carrier(seq, times)
        torch.testing.assert_close(moved[:, :C], base[:, :C], rtol=0, atol=0)
        assert not torch.equal(moved[:, C : 2 * C], base[:, C : 2 * C])


def test_the_one_mask_rule_by_inspection_and_by_finite_difference() -> None:
    carrier = _carrier()
    backbone = carrier.backbone
    key_valid = torch.ones(1, backbone.capacity, dtype=torch.bool)
    key_valid[0, M + C - 1] = False  # one padded record slot
    allowed = backbone.allowed_mask(key_valid)[0]
    read, record, write = (
        range(0, M),
        range(M, M + C),
        range(M + C, backbone.capacity),
    )
    for q in read:
        assert set(torch.where(allowed[q])[0].tolist()) == {k for k in read if k <= q}
    for q in record:
        expected = set(read) | {k for k in record if k <= q and key_valid[0, k]}
        assert set(torch.where(allowed[q])[0].tolist()) == expected
    for q in write:
        expected = (
            set(read)
            | {k for k in record if key_valid[0, k]}
            | {k for k in write if k <= q}
        )
        assert set(torch.where(allowed[q])[0].tolist()) == expected
    # Read rows attend to no record: perturbing every record leaves them fixed.
    torch.manual_seed(3)
    records = torch.randn(1, C, TOKEN)
    memory = backbone.memory_init.unsqueeze(0)
    with torch.no_grad():
        first = backbone.segment_forward(
            backbone.embed_segment(records, memory), key_valid
        )
        second = backbone.segment_forward(
            backbone.embed_segment(records + 5.0, memory), key_valid
        )
    torch.testing.assert_close(first[:, :M], second[:, :M], rtol=0, atol=0)
    assert not torch.equal(first[:, M:], second[:, M:])


# --------------------------------------------------------------------------
# 5. Ownership: each valid record enters exactly one segment
# --------------------------------------------------------------------------


def test_every_valid_record_enters_exactly_one_segment(monkeypatch: Any) -> None:
    carrier = _carrier()
    backbone = carrier.backbone
    seen: list[torch.Tensor] = []
    original = backbone.embed_segment

    def counting(records: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        seen.append(records.clone())
        return original(records, memory)

    monkeypatch.setattr(backbone, "embed_segment", counting)
    length = 2 * C + 3
    tokens = torch.zeros(1, length, TOKEN)
    tokens[0, :, 0] = torch.arange(1, length + 1).float()  # unique marker per record
    valid = torch.ones(1, length, dtype=torch.bool)
    with torch.no_grad():
        backbone.training_forward(tokens, valid)
    assert len(seen) == 3
    markers = torch.cat([segment[0, :, 0] for segment in seen])
    counts = {
        int(marker): int((markers == marker).sum()) for marker in range(1, length + 1)
    }
    assert all(count == 1 for count in counts.values()), counts
    assert int((markers == 0).sum()) == 3 * C - length  # padded slots carry zeros


# --------------------------------------------------------------------------
# 6. Gradients through the writes
# --------------------------------------------------------------------------


def _segment_loss_gradients(
    regime: str, **spec_kw: Any
) -> dict[str, torch.Tensor | None]:
    carrier = _carrier(regime, **spec_kw).train()
    backbone = carrier.backbone
    tokens = torch.randn(1, 3 * C, TOKEN, requires_grad=True)
    valid = torch.ones(1, 3 * C, dtype=torch.bool)
    out, _ = backbone.training_forward(tokens, valid)
    # A random projection of the third segment's outputs: the plain sum of a
    # LayerNorm output over its features is nearly input-invariant, so it
    # would measure rounding noise rather than the write path.
    torch.manual_seed(11)
    (out[:, 2 * C :] * torch.randn_like(out[:, 2 * C :])).sum().backward()
    assert tokens.grad is not None
    return {
        "write_queries": backbone.write_queries.grad,
        "memory_projection": backbone.memory_projection.weight.grad,
        "segment_0_records": tokens.grad[:, :C],
        "segment_1_records": tokens.grad[:, C : 2 * C],
    }


def test_a_later_segment_loss_reaches_the_writes_under_the_summary_regime() -> None:
    grads = _segment_loss_gradients("summary")
    for name in ("write_queries", "memory_projection", "segment_0_records"):
        grad = grads[name]
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_the_segment_regime_carries_nothing_across_a_boundary() -> None:
    grads = _segment_loss_gradients("segment")
    for name in ("write_queries", "memory_projection", "segment_0_records"):
        grad = grads[name]
        assert grad is None or not grad.abs().sum()
    assert grads["segment_1_records"] is not None
    assert not grads["segment_1_records"].abs().sum()


def test_boundary_detach_stops_a_later_segment_loss_at_the_boundary() -> None:
    """The truncated-gradient ablation: a later segment's
    loss no longer reaches earlier writes or records, exactly as under the
    segment regime, although the carried memory is still read."""
    grads = _segment_loss_gradients("summary", detach="boundary")
    for name in ("write_queries", "memory_projection", "segment_0_records"):
        grad = grads[name]
        assert grad is None or not grad.abs().sum(), name
    assert grads["segment_1_records"] is not None
    assert not grads["segment_1_records"].abs().sum()


def test_boundary_detach_leaves_every_forward_value_unchanged() -> None:
    full = _carrier("summary", seed=3).train()
    truncated = _carrier("summary", seed=3, detach="boundary").train()
    truncated.load_state_dict(full.state_dict())
    assert truncated.spec.sha256 != full.spec.sha256
    torch.manual_seed(5)
    tokens = torch.randn(2, 3 * C, TOKEN)
    valid = torch.ones(2, 3 * C, dtype=torch.bool)
    valid[1, 2 * C + 1 :] = False
    out_full, memory_full = full.backbone.training_forward(tokens, valid)
    out_cut, memory_cut = truncated.backbone.training_forward(tokens, valid)
    torch.testing.assert_close(out_cut, out_full, rtol=0, atol=0)
    torch.testing.assert_close(memory_cut, memory_full, rtol=0, atol=0)
    assert memory_full.requires_grad and not memory_cut.requires_grad


def test_the_residual_rewrite_adds_the_write_to_the_memory_it_read() -> None:
    """The residual-rewrite ablation: with the same weights
    the first boundary writes ``memory_init + projection`` where the replacing
    carrier writes ``projection``; the segment outputs are identical up to that
    boundary; a later-segment loss still reaches every earlier write and record;
    and the replacing write is unmarked in the identity hash."""
    replacing = _carrier("summary", seed=3).train()
    residual = _carrier("summary", seed=3, rewrite="residual").train()
    residual.load_state_dict(replacing.state_dict())
    assert residual.spec.sha256 != replacing.spec.sha256
    assert "rewrite" not in replacing.spec._hashed()
    assert replacing.spec._hashed()["detach"] == "none"  # detach stays hashed
    torch.manual_seed(5)
    tokens = torch.randn(2, C, TOKEN)
    valid = torch.ones(2, C, dtype=torch.bool)
    with torch.no_grad():
        out_replace, memory_replace = replacing.backbone.training_forward(tokens, valid)
        out_residual, memory_residual = residual.backbone.training_forward(
            tokens, valid
        )
    torch.testing.assert_close(out_residual, out_replace, rtol=0, atol=0)
    torch.testing.assert_close(
        memory_residual,
        memory_replace + replacing.backbone.memory_init.unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    grads = _segment_loss_gradients("summary", rewrite="residual")
    for name in ("write_queries", "memory_projection", "segment_0_records"):
        grad = grads[name]
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_the_gated_rewrite_scales_the_added_write_by_a_near_closed_gate() -> None:
    """T3: with the residual carrier's weights the gated
    carrier's first boundary adds sigmoid(WRITE_GATE_INIT) times the residual's
    write (the gate's weights start at zero, its bias near closed), the segment
    outputs are identical up to that boundary, a later-segment loss reaches the
    gate as well as every earlier write and record, and the rule is hashed."""
    from reasoned_icrl.model.summary_transformer import WRITE_GATE_INIT

    residual = _carrier("summary", seed=3, rewrite="residual").train()
    gated = _carrier("summary", seed=3, rewrite="gated").train()
    assert residual.backbone.write_gate is None
    assert gated.backbone.write_gate is not None
    gated.load_state_dict(
        {
            key: value
            for key, value in residual.state_dict().items()
            if key != "summary_protocol_identity"
        },
        strict=False,
    )
    assert gated.spec.sha256 != residual.spec.sha256
    assert gated.spec._hashed()["rewrite"] == "gated"
    torch.manual_seed(5)
    tokens = torch.randn(2, C, TOKEN)
    valid = torch.ones(2, C, dtype=torch.bool)
    with torch.no_grad():
        out_residual, memory_residual = residual.backbone.training_forward(
            tokens, valid
        )
        out_gated, memory_gated = gated.backbone.training_forward(tokens, valid)
    torch.testing.assert_close(out_gated, out_residual, rtol=0, atol=0)
    init = residual.backbone.memory_init.unsqueeze(0)
    gate = torch.sigmoid(torch.tensor(WRITE_GATE_INIT))
    torch.testing.assert_close(
        memory_gated - init, gate * (memory_residual - init), rtol=1e-5, atol=1e-6
    )
    grads = _segment_loss_gradients("summary", rewrite="gated")
    for name in ("write_queries", "memory_projection", "segment_0_records"):
        grad = grads[name]
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    carrier = _carrier("summary", rewrite="gated").train()
    backbone = carrier.backbone
    assert backbone.write_gate is not None
    tokens = torch.randn(1, 3 * C, TOKEN)
    out, _ = backbone.training_forward(tokens, torch.ones(1, 3 * C, dtype=torch.bool))
    torch.manual_seed(11)
    (out[:, 2 * C :] * torch.randn_like(out[:, 2 * C :])).sum().backward()
    for parameter in (backbone.write_gate.weight, backbone.write_gate.bias):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("dat,route", [(False, LEGACY), (True, ROUTED)])
def test_the_gated_rewrite_keeps_dense_cached_and_rebuild_parity(
    dat: bool, route: str
) -> None:
    carrier = _carrier("summary", "same", dat, route, rewrite="gated")
    assert carrier.backbone.write_gate is not None
    with torch.no_grad():  # an input-dependent gate, not the near-closed constant
        carrier.backbone.write_gate.weight.normal_(std=0.5)
        carrier.backbone.write_gate.bias.zero_()
    residual = _carrier("summary", "same", dat, route, rewrite="residual")
    residual.load_state_dict(
        {
            key: value
            for key, value in carrier.state_dict().items()
            if key != "summary_protocol_identity" and "write_gate" not in key
        },
        strict=False,
    )
    lengths = [3 * C + 2, 2 * C + 1, C]  # three, two and one boundary crossed
    seq, times = _packet(3, max(lengths), lengths)
    with torch.no_grad():
        dense, _ = carrier(seq, times)
    cached, hidden = _rollout(carrier, seq, times)
    for row, length in enumerate(lengths):
        error = (dense[row, :length] - cached[row, :length]).abs().amax(-1)
        assert float(error.max()) <= PARITY_ATOL, (row, error.tolist())
    assert hidden.segment.tolist() == [3, 3, 3]
    _, other = _rollout(residual, seq, times)
    assert not torch.allclose(hidden.memory, other.memory, atol=1e-3)
    rebuilt = carrier.rebuild_hidden_state(seq, times, lengths)
    for row, length in enumerate(lengths):
        live = carrier.init_hidden_state(1, torch.device("cpu"))
        _rollout(carrier, seq[row : row + 1, :length], times[:1, :length], hidden=live)
        _assert_state_parity(rebuilt.select(torch.tensor([row])), live)


@pytest.mark.parametrize("dat,route", [(False, LEGACY), (True, ROUTED)])
def test_the_residual_rewrite_keeps_dense_cached_and_rebuild_parity(
    dat: bool, route: str
) -> None:
    carrier = _carrier("summary", "same", dat, route, rewrite="residual")
    replacing = _carrier("summary", "same", dat, route)
    replacing.load_state_dict(
        {
            key: value
            for key, value in carrier.state_dict().items()
            if key != "summary_protocol_identity"
        },
        strict=False,
    )
    lengths = [3 * C + 2, 2 * C + 1, C]  # three, two and one boundary crossed
    seq, times = _packet(3, max(lengths), lengths)
    with torch.no_grad():
        dense, _ = carrier(seq, times)
    cached, hidden = _rollout(carrier, seq, times)
    for row, length in enumerate(lengths):
        error = (dense[row, :length] - cached[row, :length]).abs().amax(-1)
        assert float(error.max()) <= PARITY_ATOL, (row, error.tolist())
    assert hidden.segment.tolist() == [3, 3, 3]
    # The rule is live on the cached path: the carried memory differs from the
    # replacing carrier's after the same rollout with the same weights.
    _, other = _rollout(replacing, seq, times)
    assert not torch.allclose(hidden.memory, other.memory, atol=1e-3)
    rebuilt = carrier.rebuild_hidden_state(seq, times, lengths)
    for row, length in enumerate(lengths):
        live = carrier.init_hidden_state(1, torch.device("cpu"))
        _rollout(carrier, seq[row : row + 1, :length], times[:1, :length], hidden=live)
        _assert_state_parity(rebuilt.select(torch.tensor([row])), live)


# --------------------------------------------------------------------------
# 7. The relational-write-off writer (one segment, identical incoming memory)
# --------------------------------------------------------------------------


def test_the_write_off_writer_drops_the_relational_branch_on_write_rows_only() -> None:
    """Within one segment fed the same memory: READ and RECORD rows are
    byte-identical to the ``same`` writer and WRITE rows equal a model whose
    relational output is zeroed. Later segments are expected to differ, since
    the written memory differs; that is the point of the control."""
    same = _carrier("summary", "same", dat=True)
    write_off = _carrier("summary", "relational_off", dat=True)
    write_off.load_state_dict(
        {
            key: value
            for key, value in same.state_dict().items()
            if key != "summary_protocol_identity"
        },
        strict=False,
    )
    assert not torch.equal(
        same.summary_protocol_identity, write_off.summary_protocol_identity
    )
    torch.manual_seed(4)
    records = torch.randn(2, C, TOKEN)
    key_valid = torch.ones(2, M + C + M, dtype=torch.bool)
    memory = same.backbone.memory_init.unsqueeze(0).expand(2, -1, -1)
    with torch.no_grad():
        with_writer = same.backbone.segment_forward(
            same.backbone.embed_segment(records, memory), key_valid
        )
        without = write_off.backbone.segment_forward(
            write_off.backbone.embed_segment(records, memory), key_valid
        )
        block = same.backbone.layers[LAYERS - 1]
        assert isinstance(block, DATBlock)
        handle = block.attention.second_out.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output)
        )
        try:
            removed = same.backbone.segment_forward(
                same.backbone.embed_segment(records, memory), key_valid
            )
        finally:
            handle.remove()
    torch.testing.assert_close(
        without[:, : M + C], with_writer[:, : M + C], rtol=0, atol=0
    )
    assert not torch.equal(without[:, M + C :], with_writer[:, M + C :])
    torch.testing.assert_close(
        without[:, M + C :], removed[:, M + C :], rtol=1e-5, atol=1e-6
    )


# --------------------------------------------------------------------------
# 8. Boundedness
# --------------------------------------------------------------------------


def test_rollout_state_is_independent_of_task_length_and_lengths_stay_bounded() -> None:
    short = _carrier(max_seq_len=8)
    long = _carrier(max_seq_len=512)
    short_bytes = cache_measurements(short.init_hidden_state(3, torch.device("cpu")))
    long_bytes = cache_measurements(long.init_hidden_state(3, torch.device("cpu")))
    assert short_bytes == long_bytes and short_bytes["cache_bytes"] > 0
    seq, times = _packet(2, 3 * C + 1)
    hidden = long.init_hidden_state(2, torch.device("cpu"))
    with torch.no_grad():
        for step in range(seq.shape[1]):
            record = seq[:, step : step + 1].clone()
            record[..., -1] = 1.0
            long(record, times[:, step : step + 1], hidden)
            assert int(hidden.lengths.max()) <= M + C <= hidden.capacity
            assert int(hidden.lengths.min()) >= M
    assert hidden.segment.tolist() == [3, 3]


# --------------------------------------------------------------------------
# 9. Resets and the summary-cleared intervention
# --------------------------------------------------------------------------


def _snapshot(hidden: SummaryHiddenState, row: int) -> dict[str, torch.Tensor]:
    tensors = {"memory": hidden.memory[row].clone()}
    for index, cache in enumerate(hidden.layers):
        for name, tensor in cache.tensors().items():
            tensors[f"{index}.{name}"] = tensor[row].clone()
    return tensors


def test_reset_leaves_other_rows_byte_identical() -> None:
    carrier = _carrier()
    seq, times = _packet(2, C + 2)
    _, hidden = _rollout(carrier, seq, times)
    kept = _snapshot(hidden, 1)
    assert carrier.reset_hidden_state(hidden, np.array([True, False])) is hidden
    for name, value in _snapshot(hidden, 1).items():
        torch.testing.assert_close(value, kept[name], rtol=0, atol=0, equal_nan=True)
    assert hidden.lengths.tolist() == [0, M + 2] and hidden.segment.tolist() == [0, 1]
    torch.testing.assert_close(hidden.memory[0], hidden.initial_memory, rtol=0, atol=0)
    for cache in hidden.layers:
        for tensor in cache.tensors().values():
            assert torch.isnan(tensor[0]).all()
    fresh = carrier.init_hidden_state(1, torch.device("cpu"))
    after, _ = _rollout(
        carrier, seq[:1, :1], times[:1, :1], hidden=hidden.select(torch.tensor([0]))
    )
    again, _ = _rollout(carrier, seq[:1, :1], times[:1, :1], hidden=fresh)
    torch.testing.assert_close(after, again, rtol=1e-5, atol=PARITY_ATOL)


def test_summary_cleared_writes_the_initial_memory_at_every_boundary() -> None:
    carrier = _carrier()
    seq, times = _packet(1, 2 * C + 1)
    retained = carrier.init_hidden_state(1, torch.device("cpu"))
    cleared = carrier.init_hidden_state(1, torch.device("cpu"))
    cleared.summary_cleared = True
    kept_out, _ = _rollout(carrier, seq, times, hidden=retained)
    cleared_out, _ = _rollout(carrier, seq, times, hidden=cleared)
    assert cleared.segment.tolist() == retained.segment.tolist() == [2]
    torch.testing.assert_close(
        cleared.memory[0], cleared.initial_memory, rtol=0, atol=0
    )
    assert not torch.equal(retained.memory[0], retained.initial_memory)
    torch.testing.assert_close(cleared_out[:, :C], kept_out[:, :C], rtol=0, atol=0)
    assert not torch.allclose(cleared_out[:, C:], kept_out[:, C:])
    segment = _carrier("segment")
    kept, _ = _rollout(segment, seq, times)
    flagged = segment.init_hidden_state(1, torch.device("cpu"))
    flagged.summary_cleared = True
    same, _ = _rollout(segment, seq, times, hidden=flagged)
    torch.testing.assert_close(kept, same, rtol=0, atol=0)


@pytest.mark.parametrize(
    "writer,dat,route",
    [
        ("same", False, LEGACY),
        ("same", True, LEGACY),
        ("relational_off", True, LEGACY),
        ("same", True, ROUTED),
    ],
)
def test_a_cleared_summary_rollout_equals_the_segment_regime(
    writer: str, dat: bool, route: str
) -> None:
    """The intervention's reference execution: with the same weights, a cleared
    summary rollout is the segment regime (both reset the memory to
    ``memory_init`` at every boundary), so it must equal the segment carrier's
    dense forward to rollout parity. No training is involved. Under the
    revised route this is the matched segment control of the 8M study."""
    summary = _carrier("summary", writer, dat, route)
    segment = _carrier("segment", "same", dat, route)
    segment.load_state_dict(
        {
            key: value
            for key, value in summary.state_dict().items()
            if key != "summary_protocol_identity"
        },
        strict=False,
    )
    lengths = [3 * C + 1, 2 * C + 2, C + 1]
    seq, times = _packet(3, max(lengths), lengths)
    with torch.no_grad():
        dense, none = segment(seq, times)
    assert none is None
    cleared = summary.init_hidden_state(3, torch.device("cpu"))
    cleared.summary_cleared = True
    cleared_out, cleared = _rollout(summary, seq, times, hidden=cleared)
    retained_out, retained = _rollout(summary, seq, times)
    for row, length in enumerate(lengths):
        error = (dense[row, :length] - cleared_out[row, :length]).abs().amax(-1)
        assert float(error.max()) <= PARITY_ATOL, (row, error.tolist())
    assert cleared.segment.tolist() == retained.segment.tolist() == [3, 3, 3]
    torch.testing.assert_close(
        cleared.memory,
        cleared.initial_memory.unsqueeze(0).expand(3, -1, -1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(cleared_out[:, :C], retained_out[:, :C], rtol=0, atol=0)
    assert not torch.allclose(cleared_out[:, C:], retained_out[:, C:])


# --------------------------------------------------------------------------
# 10. The spec and the carrier's refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override,message",
    [
        ({"segment_length": 3}, "segment_length"),
        ({"memory_tokens": 0}, "memory_tokens"),
        ({"regime": "window"}, "regime"),
        ({"writer": "content"}, "writer"),
        ({"regime": "segment", "writer": "relational_off"}, "relational-write-off"),
        ({"route": "records"}, "relational_sources"),
        ({"writer": "relational_off", "route": ROUTED}, "legacy writer ablation"),
        ({"detach": "never"}, "detach"),
        ({"regime": "segment", "detach": "boundary"}, "summary regime"),
        ({"rewrite": "sum"}, "rewrite"),
        ({"regime": "segment", "rewrite": "residual"}, "summary regime"),
        ({"regime": "segment", "rewrite": "gated"}, "summary regime"),
        ({"position": "global"}, "segment-local"),
        ({"d_model": 0}, "positive"),
        ({"cache_dtype": "float64"}, "cache dtype"),
        ({"schema": "summary-memory.v0"}, "schema"),
    ],
)
def test_the_spec_refuses_unqualified_settings(
    override: dict[str, Any], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        _spec(**override)


def test_the_spec_identity_changes_with_every_field() -> None:
    base = _spec()
    alternatives: dict[str, Any] = {
        "segment_length": C + 1,
        "memory_tokens": M + 1,
        "regime": "segment",
        "writer": "relational_off",
        "relational_sources": ROUTED,
        "detach": "boundary",
        "rewrite": "residual",
        "d_model": WIDTH * 2,
        "cache_dtype": "bfloat16",
    }
    assert base.capacity == M + C + M
    assert (base.read_slots, base.record_slots, base.write_slots) == (
        range(0, M),
        range(M, M + C),
        range(M + C, M + C + M),
    )
    assert base.to_dict()["sha256"] == base.sha256
    for field in fields(SummarySpec):
        if field.name in ("position", "schema"):
            continue  # a single admitted literal each; a change is refused above
        kwargs = {field.name: alternatives[field.name]}
        if field.name == "writer":
            kwargs["regime"] = "summary"
        assert replace(base, **kwargs).sha256 != base.sha256, field.name


def test_the_carrier_refuses_inconsistent_construction_and_use() -> None:
    _configure()
    with pytest.raises(ContractError, match="resolved summary spec"):
        SummaryTrajEncoder(TOKEN + 1, 8, token_dim=TOKEN, d_model=WIDTH)
    with pytest.raises(ContractError, match="without the state bypass"):
        SummaryTrajEncoder(
            2 * TOKEN + 1, 8, spec=_spec(), token_dim=TOKEN, d_model=WIDTH
        )
    with pytest.raises(ContractError, match="segment capacity"):
        SummaryTrajEncoder(
            TOKEN + 1,
            8,
            spec=_spec(),
            dat=_dat(capacity=M + C + M + 1),
            token_dim=TOKEN,
            d_model=WIDTH,
        )
    with pytest.raises(ContractError, match="needs a dual-attention block"):
        SummaryTrajEncoder(
            TOKEN + 1,
            8,
            spec=_spec(writer="relational_off"),
            token_dim=TOKEN,
            d_model=WIDTH,
        )
    with pytest.raises(ContractError, match="ordinary blocks have none"):
        SummaryTrajEncoder(
            TOKEN + 1, 8, spec=_spec(route=ROUTED), token_dim=TOKEN, d_model=WIDTH
        )
    with pytest.raises(ContractError, match="dual-content control has none"):
        SummaryTrajEncoder(
            TOKEN + 1,
            8,
            spec=_spec(route=ROUTED),
            dat=_dat(
                mode="dual_content",
                control_content_head_dim=8,
                control_second_head_dim=8,
            ),
            token_dim=TOKEN,
            d_model=WIDTH,
        )
    carrier = _carrier()
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
    other = _carrier("segment")
    with pytest.raises(ContractError, match="different carrier"):
        other(seq[:, :1], times[:, :1], hidden)
    holes = seq.clone()
    holes[0, 1, -1] = 0.0
    with pytest.raises(ContractError, match="right-padded"):
        carrier(holes, times)
    assert isinstance(copy.deepcopy(carrier).backbone, SummaryTransformer)
