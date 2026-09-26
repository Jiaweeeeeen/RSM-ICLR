"""R2: the timestep-record relational route of ``amago-dat-summary-v2``.

SPEC §3 fixes one routing table for the summary carrier's dual-attention
blocks: content attention keeps the causal key mask over ``[READ | RECORD |
WRITE]``; the relational branch reads valid RECORD slots only, from RECORD and
WRITE receivers, and READ rows, padded rows and all-empty-source rows have an
exactly zero relational output after the projection, bias included. The same
route serves the summary regime and the matched no-carry segment control.

Every check is CPU, FP32 and deterministic. Direct-source exclusion is tested
with fixed layer inputs and by inspecting the branch's score normalization and
symbol aggregation; the indirect path (a RECORD row reading the summary through
content attention, and a later layer relating that contextualized record) is
shown to remain, so the tests distinguish the two rather than claiming total
independence from summary values. Dense/cached/rebuild parity, the segment
control and the spec refusals for the routed carrier are in
``test_summary_transformer.py``, whose ``REGIMES`` include both routes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gin
import numpy as np
import pytest
import torch
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.experiments.contracts import (
    DAT_SUMMARY_ARCHITECTURE_ID,
    DAT_SUMMARY_V2_ARCHITECTURE_ID,
    ContractError,
    DATSpec,
    SummarySpec,
)
from reasoned_icrl.model.dat_transformer import DATBlock, DualAttention
from reasoned_icrl.model.summary_transformer import (
    ROLE_READ,
    ROLE_RECORD,
    ROLE_WRITE,
    SummaryHiddenState,
)
from reasoned_icrl.model.trajectory_encoder import SummaryTrajEncoder
from reasoned_icrl.runtime.checkpointing import (
    _hidden_state,
    _restore_hidden_state,
    _runtime_contract,
    validate_checkpoint_architecture,
)

TOKEN, WIDTH, LAYERS = 16, 32, 2
C, M = 4, 2
CAP = M + C + M
READ, RECORD, WRITE = range(0, M), range(M, M + C), range(M + C, CAP)
LEGACY, ROUTED = "causal_prefix", "timestep_records"
ALL_LAYERS = tuple(range(LAYERS))
LAST_LAYER = (LAYERS - 1,)
PARITY_ATOL = 2e-6


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


def _spec(regime: str = "summary", route: str = ROUTED) -> SummarySpec:
    return SummarySpec(
        segment_length=C,
        memory_tokens=M,
        regime=regime,
        relational_sources=route,
        d_model=WIDTH,
    )


def _dat(mode: str = "dat", layers: tuple[int, ...] = LAST_LAYER) -> DATSpec:
    control = {"control_content_head_dim": 8, "control_second_head_dim": 8}
    return DATSpec(
        layer_indices=layers,
        mode=mode,
        d_model=WIDTH,
        total_heads=2,
        relational_heads=1,
        relation_channels=4,
        relation_projection_dim=4,
        max_relative_distance=CAP,
        **(control if mode == "dual_content" else {}),
    )


def _carrier(
    regime: str = "summary",
    route: str = ROUTED,
    *,
    mode: str = "dat",
    layers: tuple[int, ...] = LAST_LAYER,
    seed: int = 0,
    max_seq_len: int = 16,
) -> SummaryTrajEncoder:
    _configure()
    torch.manual_seed(seed)
    return SummaryTrajEncoder(
        TOKEN + 1,
        max_seq_len,
        spec=_spec(regime, route),
        dat=_dat(mode, layers),
        token_dim=TOKEN,
        d_model=WIDTH,
        initialization_seed=seed,
    ).eval()


def _attention(carrier: SummaryTrajEncoder, layer: int = LAYERS - 1) -> DualAttention:
    block = carrier.backbone.layers[layer]
    assert isinstance(block, DATBlock)
    return block.attention


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
    hidden: SummaryHiddenState | None = None,
) -> tuple[torch.Tensor, SummaryHiddenState]:
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


def _segment_masks(
    carrier: SummaryTrajEncoder, key_valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    backbone = carrier.backbone
    relational = backbone.relational_allowed_mask(key_valid)
    assert relational is not None
    return backbone.allowed_mask(key_valid), relational


def _halves(carrier: SummaryTrajEncoder, out: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Split a block attention output into its content and relational halves."""
    spec = _attention(carrier).spec
    return out[..., : spec.content_width], out[..., spec.content_width :]


def _key_valid(batch: int, padded: int = 0) -> torch.Tensor:
    """Segment validity with the last ``padded`` RECORD slots invalid."""
    key_valid = torch.ones(batch, CAP, dtype=torch.bool)
    if padded:
        key_valid[:, RECORD.stop - padded : RECORD.stop] = False
    return key_valid


# --------------------------------------------------------------------------
# The routing table, by inspection of the masks
# --------------------------------------------------------------------------


def test_the_routing_table_by_inspection() -> None:
    carrier = _carrier()
    key_valid = _key_valid(1, padded=1)
    content, relational = _segment_masks(carrier, key_valid)
    content, relational = content[0], relational[0]
    valid_records = {k for k in RECORD if key_valid[0, k]}
    for q in READ:
        assert set(torch.where(content[q])[0].tolist()) == {k for k in READ if k <= q}
        assert not relational[q].any()
    for q in RECORD:
        assert set(torch.where(content[q])[0].tolist()) == set(READ) | {
            k for k in valid_records if k <= q
        }
        expected = {k for k in valid_records if k <= q} if key_valid[0, q] else set()
        assert set(torch.where(relational[q])[0].tolist()) == expected
    for q in WRITE:
        assert set(torch.where(content[q])[0].tolist()) == (
            set(READ) | valid_records | {k for k in WRITE if k <= q}
        )
        assert set(torch.where(relational[q])[0].tolist()) == valid_records
    # The relational sources are always a subset of the content sources.
    assert not (relational & ~content).any()
    # An all-padding segment leaves the WRITE rows with no relational source.
    _, empty = _segment_masks(carrier, _key_valid(1, padded=C))
    assert not empty[0, list(WRITE)].any() and not empty.any()
    # The content mask is the legacy mask, unchanged by the route.
    legacy = _carrier(route=LEGACY)
    torch.testing.assert_close(
        legacy.backbone.allowed_mask(key_valid), content.unsqueeze(0), rtol=0, atol=0
    )
    assert legacy.backbone.relational_allowed_mask(key_valid) is None
    roles = carrier.backbone.slot_roles(torch.device("cpu")).tolist()
    assert roles == [ROLE_READ] * M + [ROLE_RECORD] * C + [ROLE_WRITE] * M


# --------------------------------------------------------------------------
# Score normalization and symbol aggregation
# --------------------------------------------------------------------------


def _capture_branch(
    carrier: SummaryTrajEncoder, key_valid: torch.Tensor, monkeypatch: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """The dense segment's relational weights and symbol buckets at the DAT layer."""
    attention = _attention(carrier)
    captured: dict[str, torch.Tensor] = {}
    weights, buckets = attention.relational_weights, attention.symbol_buckets

    def record_weights(*args: Any, **kwargs: Any) -> torch.Tensor:
        captured["beta"] = weights(*args, **kwargs)
        return captured["beta"]

    def record_buckets(*args: Any, **kwargs: Any) -> torch.Tensor:
        captured["buckets"] = buckets(*args, **kwargs)
        return captured["buckets"]

    monkeypatch.setattr(attention, "relational_weights", record_weights)
    monkeypatch.setattr(attention, "symbol_buckets", record_buckets)
    torch.manual_seed(3)
    records = torch.randn(key_valid.shape[0], C, TOKEN)
    backbone = carrier.backbone
    memory = backbone.memory_init.unsqueeze(0).expand(key_valid.shape[0], -1, -1)
    with torch.no_grad():
        backbone.segment_forward(backbone.embed_segment(records, memory), key_valid)
    return captured["beta"], captured["buckets"]


def test_score_normalization_and_symbol_aggregation_exclude_read_and_write_sources(
    monkeypatch: Any,
) -> None:
    carrier = _carrier()
    key_valid = _key_valid(2, padded=1)
    beta, buckets = _capture_branch(carrier, key_valid, monkeypatch)
    assert beta.shape == (2, 1, CAP, CAP) and torch.isfinite(beta).all()
    assert buckets.shape == (2, 1, CAP, CAP + 1) and torch.isfinite(buckets).all()
    excluded_keys = list(READ) + list(WRITE)
    assert torch.equal(
        beta[..., excluded_keys], torch.zeros_like(beta[..., excluded_keys])
    )
    # Receivers: READ rows and the padded RECORD row carry no weight at all;
    # every routed row normalizes to one over the valid RECORD sources.
    for row in range(2):
        for q in range(CAP):
            mass = float(beta[row, 0, q].sum())
            routed = q in WRITE or (q in RECORD and bool(key_valid[row, q]))
            assert mass == pytest.approx(1.0 if routed else 0.0, abs=1e-6), q
            assert float(buckets[row, 0, q].sum()) == pytest.approx(mass, abs=1e-6)
            # Offsets only a READ or WRITE source could produce hold nothing:
            # for a RECORD receiver at slot M + i, distances beyond i reach
            # only READ slots; for a WRITE receiver at M + C + j, distances
            # beyond C + j reach only READ slots and distances at most j reach
            # only WRITE slots (its own slot included).
            if q in RECORD:
                foreign = [CAP - d for d in range(q - RECORD.start + 1, q + 1)]
            elif q in WRITE:
                j = q - WRITE.start
                foreign = [CAP - d for d in range(0, j + 1)]
                foreign += [CAP - d for d in range(C + j + 1, q + 1)]
            else:
                foreign = list(range(CAP + 1))
            assert not buckets[row, 0, q, foreign].any(), q
    # The legacy route puts weight on READ sources: the check discriminates.
    legacy_beta, _ = _capture_branch(_carrier(route=LEGACY), key_valid, monkeypatch)
    assert legacy_beta[..., list(RECORD), :][..., list(READ)].abs().sum() > 0


# --------------------------------------------------------------------------
# Direct-source exclusion with fixed layer inputs, and the indirect path
# --------------------------------------------------------------------------


def _fixed_input_outputs(
    carrier: SummaryTrajEncoder, x: torch.Tensor, key_valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    attention = _attention(carrier)
    times = torch.arange(CAP).unsqueeze(0).expand(x.shape[0], -1)
    backbone = carrier.backbone
    with torch.no_grad():
        out = attention(
            x,
            times,
            times,
            backbone.allowed_mask(key_valid),
            relational_allowed=backbone.relational_allowed_mask(key_valid),
        )
    return _halves(carrier, out)


@pytest.mark.parametrize("perturbed", ["READ", "WRITE"])
def test_read_and_write_source_values_do_not_reach_the_relational_output(
    perturbed: str,
) -> None:
    """Fixed layer inputs, SPEC §6.1: perturb the READ rows (or the first
    WRITE row) of the block input. Every receiver those rows serve only as
    *sources* -- the RECORD and WRITE rows for a READ perturbation, the later
    WRITE rows for a WRITE perturbation -- keeps its relational output bit for
    bit, while its content output moves. The perturbed rows are also
    receivers, so their own relational output may change through their
    queries; that is not a source path."""
    carrier = _carrier()
    key_valid = _key_valid(2, padded=1)
    torch.manual_seed(5)
    x = torch.randn(2, CAP, WIDTH)
    moved = x.clone()
    rows = list(READ) if perturbed == "READ" else [WRITE.start]
    moved[:, rows] += 3.0
    content, relational = _fixed_input_outputs(carrier, x, key_valid)
    content_after, relational_after = _fixed_input_outputs(carrier, moved, key_valid)
    receivers = (
        list(RECORD) + list(WRITE)
        if perturbed == "READ"
        else list(range(WRITE.start + 1, WRITE.stop))
    )
    torch.testing.assert_close(
        relational_after[:, receivers], relational[:, receivers], rtol=0, atol=0
    )
    assert not torch.equal(content_after[:, receivers], content[:, receivers])
    # The legacy route lets the same perturbation through: the check discriminates.
    legacy = _carrier(route=LEGACY)
    legacy.load_state_dict(_weights(carrier), strict=False)
    attention = _attention(legacy)
    times = torch.arange(CAP).unsqueeze(0).expand(2, -1)
    allowed = legacy.backbone.allowed_mask(key_valid)
    with torch.no_grad():
        _, before = _halves(legacy, attention(x, times, times, allowed))
        _, after = _halves(legacy, attention(moved, times, times, allowed))
    assert not torch.equal(after[:, receivers], before[:, receivers])


def _weights(carrier: SummaryTrajEncoder) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in carrier.state_dict().items()
        if not key.endswith("protocol_identity")
    }


def test_summary_values_still_reach_records_indirectly_through_content() -> None:
    """The route excludes READ rows as *direct* relational sources only. A
    RECORD row reads the summary through content attention in the first
    block, so the second block's relational computation over that
    contextualized record does depend on the summary. SPEC §3 forbids
    claiming total independence; this pins the indirect path."""
    carrier = _carrier(layers=ALL_LAYERS)
    backbone = carrier.backbone
    key_valid = _key_valid(1)
    torch.manual_seed(6)
    records = torch.randn(1, C, TOKEN)
    memory = backbone.memory_init.unsqueeze(0)
    captured: list[torch.Tensor] = []
    attention = _attention(carrier, LAYERS - 1)
    handle = attention.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            first = backbone.segment_forward(
                backbone.embed_segment(records, memory), key_valid
            )
            second = backbone.segment_forward(
                backbone.embed_segment(records, memory + 2.0), key_valid
            )
    finally:
        handle.remove()
    assert not torch.equal(first[:, list(RECORD)], second[:, list(RECORD)])
    _, before = _halves(carrier, captured[0])
    _, after = _halves(carrier, captured[1])
    assert not torch.equal(before[:, list(RECORD)], after[:, list(RECORD)])
    # READ rows never carry a relational output, in either forward.
    assert not before[:, list(READ)].any() and not after[:, list(READ)].any()


# --------------------------------------------------------------------------
# Exact zeros, finiteness and the zero-source write
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["dat", "symbol_only"])
@pytest.mark.parametrize("padded", [0, 1, C])
def test_read_padded_and_empty_source_rows_are_exactly_zero_and_finite(
    mode: str, padded: int
) -> None:
    carrier = _carrier(mode=mode)
    key_valid = _key_valid(2, padded=padded)
    torch.manual_seed(7)
    x = torch.randn(2, CAP, WIDTH, requires_grad=True)
    attention = _attention(carrier)
    times = torch.arange(CAP).unsqueeze(0).expand(2, -1)
    backbone = carrier.backbone
    out = attention(
        x,
        times,
        times,
        backbone.allowed_mask(key_valid),
        relational_allowed=backbone.relational_allowed_mask(key_valid),
    )
    content, relational = _halves(carrier, out)
    assert torch.isfinite(out).all()
    assert attention.second_out.bias.abs().sum() > 0  # the zero is not the bias
    zero_rows = list(READ) + [q for q in RECORD if not key_valid[0, q]]
    if padded == C:
        zero_rows += list(WRITE)
    assert torch.equal(
        relational[:, zero_rows], torch.zeros_like(relational[:, zero_rows])
    )
    live_rows = [q for q in range(CAP) if q not in zero_rows]
    if live_rows:
        assert relational[:, live_rows].abs().sum() > 0
    assert content[:, zero_rows].abs().sum() > 0  # content is untouched
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in attention.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_a_write_over_an_all_padding_segment_follows_the_zero_source_rule() -> None:
    """SPEC §6.2: when a WRITE has no valid record, the memory update is the
    same block computation with the relational projection output zeroed on
    every row; content attention over READ and earlier WRITE rows still runs,
    and the result is finite."""
    carrier = _carrier()
    backbone = carrier.backbone
    key_valid = _key_valid(1, padded=C)
    records = torch.zeros(1, C, TOKEN)
    memory = backbone.memory_init.unsqueeze(0) + 0.5
    with torch.no_grad():
        routed = backbone.segment_forward(
            backbone.embed_segment(records, memory), key_valid
        )
        attention = _attention(carrier)
        handle = attention.second_out.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output)
        )
        try:
            zeroed = backbone.segment_forward(
                backbone.embed_segment(records, memory), key_valid
            )
        finally:
            handle.remove()
    assert torch.isfinite(routed).all()
    torch.testing.assert_close(routed, zeroed, rtol=0, atol=0)
    written = backbone.memory_projection(routed[:, WRITE.start :])
    assert torch.isfinite(written).all() and written.abs().sum() > 0


def test_a_batch_with_all_padding_segments_trains_finitely() -> None:
    """A short row beside a long one meets whole segments of padding in the
    dense path; outputs and every gradient stay finite, and the short row's
    outputs equal its own solo forward."""
    carrier = _carrier("summary", layers=ALL_LAYERS).train()
    backbone = carrier.backbone
    lengths = [C, 3 * C + 1]
    torch.manual_seed(8)
    tokens = torch.randn(2, 3 * C + 1, TOKEN, requires_grad=True)
    valid = torch.arange(3 * C + 1).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    out, memory = backbone.training_forward(tokens, valid)
    assert torch.isfinite(out).all() and torch.isfinite(memory).all()
    (out.sum() + memory.sum()).backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    for name, parameter in backbone.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name
    with torch.no_grad():
        solo, _ = backbone.training_forward(tokens[:1, :C].detach(), valid[:1, :C])
    torch.testing.assert_close(
        out[0, :C].detach(), solo[0], rtol=1e-5, atol=PARITY_ATOL
    )
    assert not out[0, C:].detach().any()


# --------------------------------------------------------------------------
# Causality, WRITE leakage, RL-visible rows
# --------------------------------------------------------------------------


def test_future_records_and_same_segment_writes_cannot_reach_earlier_outputs() -> None:
    carrier = _carrier(layers=ALL_LAYERS)
    seq, times = _packet(1, 2 * C + 3)
    with torch.no_grad():
        base, none = carrier(seq, times)
        assert none is None and base.shape == (1, 2 * C + 3, WIDTH)
        for position in (1, C - 1, C, C + 2, 2 * C + 1):
            changed = seq.clone()
            changed[:, position, :TOKEN] += 20.0
            later, _ = carrier(changed, times)
            torch.testing.assert_close(
                later[:, :position], base[:, :position], rtol=0, atol=0
            )
            assert not torch.equal(later[:, position], base[:, position])
        carrier.backbone.write_queries.add_(1.0)
        moved, _ = carrier(seq, times)
        torch.testing.assert_close(moved[:, :C], base[:, :C], rtol=0, atol=0)
        assert not torch.equal(moved[:, C : 2 * C], base[:, C : 2 * C])
    # One control vector per record and nothing for READ or WRITE rows: the
    # actor and critic never see a summary slot, so no action or RL loss can
    # come from one.
    hidden = carrier.init_hidden_state(1, torch.device("cpu"))
    with torch.no_grad():
        for step in range(C + 1):
            out, _ = carrier(seq[:, step : step + 1], times[:, step : step + 1], hidden)
            assert out.shape == (1, 1, WIDTH)
    assert hidden.segment.tolist() == [1]


# --------------------------------------------------------------------------
# Gradients through the writes, and the matched segment control
# --------------------------------------------------------------------------


def _segment_loss_gradients(regime: str) -> dict[str, torch.Tensor | None]:
    carrier = _carrier(regime, layers=ALL_LAYERS).train()
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
    attention = _attention(carrier)
    return {
        "write_queries": backbone.write_queries.grad,
        "memory_projection": backbone.memory_projection.weight.grad,
        "relation_out": attention.relation_out.grad,
        "segment_0_records": tokens.grad[:, :C],
        "segment_1_records": tokens.grad[:, C : 2 * C],
    }


def test_a_later_segment_loss_reaches_the_writes_under_the_routed_summary() -> None:
    grads = _segment_loss_gradients("summary")
    for name, grad in grads.items():
        assert (
            grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        ), name


def test_the_routed_segment_control_carries_nothing_across_a_boundary() -> None:
    grads = _segment_loss_gradients("segment")
    for name in ("write_queries", "memory_projection", "segment_0_records"):
        grad = grads[name]
        assert grad is None or not grad.abs().sum(), name
    assert grads["segment_1_records"] is not None
    assert not grads["segment_1_records"].abs().sum()


def test_summary_and_segment_share_the_model_and_differ_only_in_carry() -> None:
    summary = _carrier("summary", layers=ALL_LAYERS)
    segment = _carrier("segment", layers=ALL_LAYERS)
    assert _weights(summary).keys() == _weights(segment).keys()
    segment.load_state_dict(_weights(summary), strict=False)
    seq, times = _packet(1, 2 * C + 1)
    with torch.no_grad():
        kept, _ = summary(seq, times)
        reset, _ = segment(seq, times)
    torch.testing.assert_close(kept[:, :C], reset[:, :C], rtol=0, atol=0)
    assert not torch.allclose(kept[:, C:], reset[:, C:])
    assert summary.spec.sha256 != segment.spec.sha256


# --------------------------------------------------------------------------
# Rollout: independent resets and asynchronous outer boundaries
# --------------------------------------------------------------------------


def test_asynchronous_outer_resets_keep_rows_independent_under_the_route() -> None:
    carrier = _carrier(layers=ALL_LAYERS)
    seq, times = _packet(2, 2 * C + 3)
    hidden = carrier.init_hidden_state(2, torch.device("cpu"))
    outputs = []
    with torch.no_grad():
        for step in range(seq.shape[1]):
            if step == C + 1:  # row 0 starts a new outer task mid-segment
                assert carrier.reset_hidden_state(hidden, np.array([True, False]))
            out, _ = carrier(seq[:, step : step + 1], times[:, step : step + 1], hidden)
            outputs.append(out)
    batched = torch.cat(outputs, 1)
    # Row 1 equals its own uninterrupted rollout; row 0's second task equals a
    # fresh rollout of its post-reset records.
    solo_1, state_1 = _rollout(carrier, seq[1:2], times[1:2])
    torch.testing.assert_close(batched[1], solo_1[0], rtol=1e-5, atol=PARITY_ATOL)
    solo_0, state_0 = _rollout(carrier, seq[:1, C + 1 :], times[:1, : C + 2])
    torch.testing.assert_close(
        batched[0, C + 1 :], solo_0[0], rtol=1e-5, atol=PARITY_ATOL
    )
    assert hidden.segment.tolist() == [1, 2]
    assert state_0.segment.tolist() == [1] and state_1.segment.tolist() == [2]
    torch.testing.assert_close(
        hidden.memory[0], state_0.memory[0], rtol=1e-5, atol=PARITY_ATOL
    )
    torch.testing.assert_close(
        hidden.memory[1], state_1.memory[0], rtol=1e-5, atol=PARITY_ATOL
    )


# --------------------------------------------------------------------------
# Identities: weights-only reload, resume state and the legacy carrier
# --------------------------------------------------------------------------


def test_the_route_enters_the_summary_identity_and_leaves_the_legacy_hash_alone() -> (
    None
):
    legacy, routed = _spec(route=LEGACY), _spec(route=ROUTED)
    assert legacy.sha256 != routed.sha256
    assert routed.to_dict()["relational_sources"] == ROUTED
    assert legacy.to_dict()["relational_sources"] == LEGACY
    # The legacy hash is the hash of the pre-R2 field set, so every identity
    # recorded before the route existed still resolves to the same value.
    assert "relational_sources" not in legacy._hashed()
    assert (
        legacy.sha256
        == SummarySpec(
            segment_length=C, memory_tokens=M, regime="summary", d_model=WIDTH
        ).sha256
    )
    assert routed._hashed()["relational_sources"] == ROUTED


def test_v1_and_v2_carriers_compute_different_functions_on_equal_weights() -> None:
    routed = _carrier(layers=ALL_LAYERS)
    legacy = _carrier(route=LEGACY, layers=ALL_LAYERS)
    legacy.load_state_dict(_weights(routed), strict=False)
    assert not torch.equal(
        routed.summary_protocol_identity, legacy.summary_protocol_identity
    )
    torch.testing.assert_close(
        routed.attention_protocol_identity, legacy.attention_protocol_identity
    )
    seq, times = _packet(1, 2 * C + 1)
    with torch.no_grad():
        a, _ = routed(seq, times)
        b, _ = legacy(seq, times)
    assert not torch.allclose(a, b)


def test_weights_only_reload_and_the_rebuilt_state_reproduce_the_carrier() -> None:
    carrier = _carrier(layers=ALL_LAYERS)
    fresh = _carrier(layers=ALL_LAYERS, seed=3)
    validate_checkpoint_architecture(
        carrier.state_dict(),
        DAT_SUMMARY_V2_ARCHITECTURE_ID,
        expected_state=fresh.state_dict(),
    )
    fresh.load_state_dict(carrier.state_dict())
    seq, times = _packet(3, 2 * C + 3, [2 * C + 3, C + 1, 2 * C])
    with torch.no_grad():
        expected, _ = carrier(seq, times)
        actual, _ = fresh(seq, times)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A learner update moves the weights; the rebuilt state equals a live
    # rollout under the new weights, boundary by boundary.
    with torch.no_grad():
        for parameter in fresh.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    lengths = [2 * C + 3, C + 1, 2 * C]
    rebuilt = fresh.rebuild_hidden_state(seq, times, lengths)
    assert rebuilt.segment.tolist() == [2, 1, 1]
    torch.manual_seed(9)
    following = torch.randn(3, 1, TOKEN + 1)
    following[..., -1] = 1.0
    for row, length in enumerate(lengths):
        live = fresh.init_hidden_state(1, torch.device("cpu"))
        _rollout(fresh, seq[row : row + 1, :length], times[:1, :length], hidden=live)
        expected, _ = _rollout(fresh, following[row : row + 1], times[:1, :1], live)
        actual, _ = _rollout(
            fresh,
            following[row : row + 1],
            times[:1, :1],
            rebuilt.select(torch.tensor([row])),
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=PARITY_ATOL)


def test_identities_load_only_into_matching_operators() -> None:
    routed = _carrier()
    legacy = _carrier(route=LEGACY)
    with pytest.raises(ContractError, match="protocol"):
        validate_checkpoint_architecture(
            legacy.state_dict(),
            DAT_SUMMARY_V2_ARCHITECTURE_ID,
            expected_state=routed.state_dict(),
        )
    with pytest.raises(ContractError, match="protocol"):
        validate_checkpoint_architecture(
            routed.state_dict(),
            DAT_SUMMARY_ARCHITECTURE_ID,
            expected_state=legacy.state_dict(),
        )
    seq, times = _packet(1, C + 1)
    _, hidden = _rollout(routed, seq, times)
    payload = _hidden_state(hidden)
    assert payload["spec_sha256"] == routed.spec.sha256

    def experiment(carrier: SummaryTrajEncoder) -> SimpleNamespace:
        return SimpleNamespace(
            policy=SimpleNamespace(traj_encoder=carrier), DEVICE=torch.device("cpu")
        )

    with pytest.raises(ContractError, match="identity does not match"):
        _restore_hidden_state(experiment(legacy), payload)
    restored = _restore_hidden_state(experiment(routed), payload)
    assert isinstance(restored, SummaryHiddenState)
    with pytest.raises(ContractError, match="different carrier"):
        legacy(seq[:, :1], times[:, :1], hidden)
    with torch.no_grad():
        a, _ = routed(seq[:, :1], times[:, :1], hidden)
        b, _ = routed(seq[:, :1], times[:, :1], restored)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_the_resume_contract_carries_the_route_through_the_summary_identity() -> None:
    def contract(carrier: SummaryTrajEncoder, architecture: str, condition: str) -> Any:
        return _runtime_contract(
            SimpleNamespace(
                encoder_architecture_id=architecture,
                policy_condition=condition,
                policy=SimpleNamespace(
                    traj_encoder=carrier,
                    tstep_encoder=SimpleNamespace(
                        spec=SimpleNamespace(sha256="packet")
                    ),
                ),
                learner_contract="amago-optimizer-ownership.v1",
                reasoned_training_settings={},
            )
        )

    routed = contract(_carrier(), DAT_SUMMARY_V2_ARCHITECTURE_ID, "fixed_summary")
    legacy = contract(
        _carrier(route=LEGACY), DAT_SUMMARY_ARCHITECTURE_ID, "raw_dat_summary"
    )
    assert routed["summary_sha256"] != legacy["summary_sha256"]
    assert routed["attention_sha256"] == legacy["attention_sha256"]
    assert routed["architecture_id"] == DAT_SUMMARY_V2_ARCHITECTURE_ID


# --------------------------------------------------------------------------
# Refusals at the block
# --------------------------------------------------------------------------


def test_the_block_refuses_a_route_it_cannot_honor() -> None:
    _configure()
    torch.manual_seed(0)
    x = torch.randn(1, CAP, WIDTH)
    times = torch.arange(CAP).unsqueeze(0)
    allowed = torch.tril(torch.ones(CAP, CAP, dtype=torch.bool)).unsqueeze(0)
    control = DualAttention(_dat("dual_content"))
    with pytest.raises(ContractError, match="dual-content control has none"):
        control(x, times, times, allowed, relational_allowed=allowed)
    attention = DualAttention(_dat())
    with pytest.raises(ContractError, match="match the content mask shape"):
        attention(x, times, times, allowed, relational_allowed=allowed[:, :1])
    wider = torch.ones(1, CAP, CAP, dtype=torch.bool)
    with pytest.raises(ContractError, match="content branch may not"):
        attention(x, times, times, allowed, relational_allowed=wider)
    out = attention(x, times, times, allowed, relational_allowed=allowed)
    assert out.shape == (1, CAP, WIDTH) and torch.isfinite(out).all()
