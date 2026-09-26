"""Numerical qualification of the dual-attention operator and its caches.

Three levels, all of which the integration plan requires before any pilot is
interpretable:

1. **Source parity** against the pinned upstream operator, with SigmaReparam and
   head scaling disabled so the comparison is of the attention itself.
2. **An independent oracle**: an explicit pairwise sum written from the paper's
   equations, which shares no code with either implementation.
3. **Runtime behaviour**: causality, dense/cached agreement, resets, rollover
   and gradient routing under the AMAGO parameterization actually trained.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import gin
import pytest
import torch
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.experiments.contracts import ContractError, DATSpec
from reasoned_icrl.model.dat_transformer import (
    EMPTY_TIME,
    DualAttention,
    _dense_content_attention,
    _fused_content_attention,
    content_backend,
    require_right_padded,
    symbol_offsets,
)
from reasoned_icrl.model.trajectory_encoder import DATTrajEncoder

ROOT = Path(__file__).resolve().parents[2]

#: Dense and cached execution, and our operator and the pinned one, reduce the
#: same sums in a different order, so they agree to float32 rounding rather than
#: exactly. That rounding scales with the magnitude of the reduction, not with
#: any fixed absolute figure: the compared tensors here run from forward outputs
#: of magnitude ~0.2 to parameter gradients of magnitude ~630, so one hardcoded
#: ``atol`` is either vacuous at the top of that range or spuriously tight at the
#: bottom. Measured across every comparison below, on both the Linux and the Mac
#: backbone, the worst disagreement is 7.4e-7 of the compared tensor's own
#: maximum, i.e. about six times float32 ``eps``, and it does not grow with
#: sequence position. The absolute term is therefore derived from that maximum
#: with roughly five times headroom. ``rtol`` already covers entries that are
#: large individually; this term is what entries driven to near zero by
#: cancellation need, and those are the only ones it decides.
FP32_SCALED_ATOL = 4e-6

#: The fused CUDA kernel's backward recomputes the attention and accumulates
#: dq/dk in a different order again from the dense path's two einsums, so it is
#: a noisier comparison than its forward: measured over twelve seeds on this
#: backbone the worst gradient disagreement is 8.0e-6 of the gradient's own
#: maximum, against 6.7e-7 for the forward. The kernel is bitwise reproducible
#: run to run, so this is reduction order and not atomics, and the figure below
#: keeps roughly three times headroom over the measurement.
FUSED_BACKWARD_SCALED_ATOL = 2.5e-5

#: Floor for the accumulating-drift check, which compares per-position errors of
#: ~2e-6 to each other rather than to the outputs. A magnitude-scaled tolerance
#: would exceed the quantities being compared and make that check vacuous, so
#: this one stays a fixed figure, well under the measured margin.
DRIFT_FLOOR = 2e-7
REFERENCE = ROOT / "tests/model/reference/dual_attention"
D_MODEL, TOTAL_HEADS, RELATIONAL_HEADS = 256, 8, 2
HEAD_DIM = D_MODEL // TOTAL_HEADS


def _grad(param: object) -> torch.Tensor:
    """The gradient of a parameter the backward pass must have reached."""
    grad = getattr(param, "grad", None)
    assert isinstance(grad, torch.Tensor)
    return grad


def _assert_fp32_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    relative: float = FP32_SCALED_ATOL,
) -> None:
    """Compare two float32 reductions to rounding scaled by their magnitude."""
    scale = max(
        float(actual.detach().abs().max()), float(expected.detach().abs().max())
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=relative * scale)


def _spec(**overrides: object) -> DATSpec:
    settings: dict[str, object] = {
        "layer_indices": (2,),
        "max_relative_distance": 6,
    }
    settings.update(overrides)
    return DATSpec(**settings)  # type: ignore[arg-type]


def _configure_backbone() -> None:
    torch.compiler.set_stance("force_eager")
    torch.set_num_threads(1)
    gin.clear_config()
    target = "amago.nets.traj_encoders.TformerTrajEncoder"
    for name, value in {
        "d_model": D_MODEL,
        "n_heads": TOTAL_HEADS,
        "n_layers": 3,
        "d_ff": 1024,
        "attention_type": VanillaAttention,
        "dropout_ff": 0.0,
        "dropout_emb": 0.0,
        "dropout_attn": 0.0,
        "dropout_qkv": 0.0,
    }.items():
        gin.bind_parameter(f"{target}.{name}", value)


def _carrier(spec: DATSpec, max_seq_len: int = 8) -> DATTrajEncoder:
    _configure_backbone()
    torch.manual_seed(0)
    return DATTrajEncoder(65, max_seq_len, spec=spec).eval()


def _packet(batch: int, length: int, *, valid_lengths: list[int] | None = None):
    torch.manual_seed(1)
    seq = torch.randn(batch, length, 65)
    valid = torch.zeros(batch, length, 1)
    for row in range(batch):
        keep = length if valid_lengths is None else valid_lengths[row]
        valid[row, :keep] = 1.0
    seq[..., -1:] = valid
    times = (
        torch.arange(length).view(1, length, 1).expand(batch, length, 1).contiguous()
    )
    return seq, times, valid


def _causal(batch: int, length: int) -> torch.Tensor:
    return (
        torch.tril(torch.ones(length, length, dtype=torch.bool))
        .unsqueeze(0)
        .expand(batch, length, length)
    )


# --------------------------------------------------------------------------
# 1. Parity with the pinned upstream operator
# --------------------------------------------------------------------------


def test_vendored_reference_matches_its_recorded_hashes() -> None:
    """A silent upstream substitution must not be able to pass as parity."""
    manifest = json.loads((REFERENCE / "SOURCES.json").read_text())
    recorded = manifest["vendored_test_fixtures"]
    assert recorded["commit"] == "dce218cbf5ec9aa7f90687c1323050a1fba17966"
    assert (REFERENCE / "LICENSE").is_file()
    for entry in recorded["files"]:
        digest = hashlib.sha256((REFERENCE / entry["path"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"], entry["path"]


def _reference_module(spec: DATSpec):
    from tests.model.reference.dual_attention.relational_attention import (
        RelationalAttention,
    )

    return RelationalAttention(
        d_model=spec.d_model,
        n_heads=spec.relational_heads,
        n_relations=spec.relation_channels,
        dropout=0.0,
        rel_activation=spec.relation_activation,
        rel_proj_dim=spec.relation_projection_dim,
        add_bias_kv=False,
        add_bias_out=True,
        total_n_heads=spec.total_heads,
        symmetric_rels=spec.symmetric_relations,
        use_relative_positional_symbols=True,
    )


def _copy_into_reference(module: DualAttention, reference) -> None:
    """Move our declared weights into the pinned operator's parameters."""
    assert module.selection_q is not None and module.selection_k is not None
    reference.wq_attn.weight.data.copy_(module.selection_q.weight.data)
    reference.wk_attn.weight.data.copy_(module.selection_k.weight.data)
    if module.relation_q is not None:
        reference.wq_rel.weight.data.copy_(module.relation_q.weight.data)
    if module.relation_k is not None:
        reference.wk_rel.weight.data.copy_(module.relation_k.weight.data)
    if module.relation_out is not None:
        reference.wr.data.copy_(module.relation_out.data)
    assert module.symbol_projection is not None
    reference.wv.weight.data.copy_(module.symbol_projection.weight.data)
    reference.wo.weight.data.copy_(module.second_out.weight.data)
    reference.wo.bias.data.copy_(module.second_out.bias.data)


def _dense_symbols(module: DualAttention, length: int) -> torch.Tensor:
    """Materialize the reference's ``[len, len, d_model]`` relative symbols."""
    assert module.symbol_table is not None
    times = torch.arange(length)
    offsets = symbol_offsets(
        times.unsqueeze(0), times.unsqueeze(0), module.spec.max_relative_distance
    )[0]
    return module.symbol_table[offsets]


@pytest.mark.parametrize("symmetric", [False, True])
def test_relational_branch_matches_the_pinned_official_operator(
    symmetric: bool,
) -> None:
    torch.manual_seed(0)
    spec = _spec(symmetric_relations=symmetric)
    module = DualAttention(spec, sigma_reparam=False).eval()
    reference = _reference_module(spec).eval()
    _copy_into_reference(module, reference)

    batch, length = 2, 5
    x = torch.randn(batch, length, spec.d_model, requires_grad=True)
    mirror = x.detach().clone().requires_grad_(True)
    times = torch.arange(length).unsqueeze(0).expand(batch, length)
    allowed = _causal(batch, length).unsqueeze(1)

    ours = module.second_out(
        module.relational_branch(
            module.project_selection(x, query=True),
            module.project_selection(x, query=False),
            module.project_relation(x, query=True),
            module.project_relation(x, query=False),
            allowed,
            symbol_offsets(times, times, spec.max_relative_distance),
        )
    )
    theirs, _, _ = reference(mirror, _dense_symbols(module, length), is_causal=True)
    _assert_fp32_close(ours, theirs)

    # Gradients must agree for the input and for every mapped parameter.
    weights = torch.arange(ours.shape[-1]).float()
    (ours * weights).sum().backward()
    (theirs * weights).sum().backward()
    assert x.grad is not None and mirror.grad is not None
    _assert_fp32_close(x.grad, mirror.grad)
    assert module.selection_q is not None and module.relation_out is not None
    assert module.symbol_projection is not None
    for ours_param, theirs_param in (
        (module.selection_q.weight, reference.wq_attn.weight),
        (module.selection_k.weight, reference.wk_attn.weight),
        (module.relation_out, reference.wr),
        (module.symbol_projection.weight, reference.wv.weight),
        (module.second_out.weight, reference.wo.weight),
        (module.second_out.bias, reference.wo.bias),
    ):
        _assert_fp32_close(_grad(ours_param), _grad(theirs_param))
    if not symmetric:
        assert module.relation_q is not None
        _assert_fp32_close(
            _grad(module.relation_q.weight), _grad(reference.wq_rel.weight)
        )


def test_branch_output_widths_follow_the_executable_reference() -> None:
    """DAT's code concatenates branch-sized outputs; the appendix prints full width."""
    spec = _spec()
    module = DualAttention(spec, sigma_reparam=False)
    assert (spec.content_width, spec.relational_width) == (192, 64)
    assert module.content_out.weight.shape == (192, 192)
    assert module.second_out.weight.shape == (64, 64)
    reference = _reference_module(spec)
    assert reference.wo.weight.shape == module.second_out.weight.shape


def test_explicit_pairwise_sum_is_an_independent_oracle() -> None:
    """A literal transcription of the paper's equation, sharing no code."""
    torch.manual_seed(0)
    spec = _spec()
    module = DualAttention(spec, sigma_reparam=False).eval()
    length = 4
    x = torch.randn(1, length, spec.d_model)
    times = torch.arange(length).unsqueeze(0)
    allowed = _causal(1, length).unsqueeze(1)
    ours = module.relational_branch(
        module.project_selection(x, query=True),
        module.project_selection(x, query=False),
        module.project_relation(x, query=True),
        module.project_relation(x, query=False),
        allowed,
        symbol_offsets(times, times, spec.max_relative_distance),
    )

    assert module.selection_q is not None and module.selection_k is not None
    assert module.relation_q is not None and module.relation_k is not None
    assert module.relation_out is not None and module.symbol_projection is not None
    q_sel = module.project_selection(x, query=True)[0]
    k_sel = module.project_selection(x, query=False)[0]
    q_rel = module.project_relation(x, query=True)[0]
    k_rel = module.project_relation(x, query=False)[0]
    table = module.symbol_projection(module.symbol_table).reshape(
        -1, spec.relational_heads, spec.head_dim
    )
    expected = torch.zeros(length, spec.relational_heads, spec.head_dim)
    for head in range(spec.relational_heads):
        for t in range(length):
            scores = [
                float((q_sel[t, head] * k_sel[i, head]).sum())
                / math.sqrt(spec.head_dim)
                for i in range(t + 1)
            ]
            top = max(scores)
            weights = [math.exp(s - top) for s in scores]
            total = sum(weights)
            for i, weight in enumerate(weights):
                beta = weight / total
                rho = torch.tensor(
                    [
                        float((q_rel[t, r] * k_rel[i, r]).sum())
                        / math.sqrt(spec.relation_projection_dim)
                        for r in range(spec.relation_channels)
                    ]
                )
                expected[t, head] += beta * (
                    module.relation_out[head] @ rho
                    + table[
                        max(i - t, -spec.max_relative_distance)
                        + spec.max_relative_distance,
                        head,
                    ]
                )
    torch.testing.assert_close(
        ours[0], expected.reshape(length, -1), rtol=1e-4, atol=1e-5
    )


# --------------------------------------------------------------------------
# 2. Modes, masking and causality
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("dat", {}),
        ("symbol_only", {}),
        (
            "dual_content",
            {"control_content_head_dim": 40, "control_second_head_dim": 48},
        ),
    ],
)
def test_every_mode_returns_the_residual_width(mode: str, extra: dict) -> None:
    torch.manual_seed(0)
    spec = _spec(mode=mode, **extra)
    module = DualAttention(spec, sigma_reparam=False).eval()
    batch, length = 2, 5
    x = torch.randn(batch, length, spec.d_model)
    times = torch.arange(length).unsqueeze(0).expand(batch, length)
    out = module(x, times, times, _causal(batch, length))
    assert out.shape == (batch, length, spec.d_model)
    assert torch.isfinite(out).all()


def test_symbol_only_drops_relation_modules_rather_than_disabling_them() -> None:
    """A disconnected parameter would still be counted as capacity."""
    module = DualAttention(_spec(mode="symbol_only"), sigma_reparam=False)
    assert module.relation_q is None and module.relation_k is None
    assert module.relation_out is None
    assert module.symbol_table is not None
    names = {name for name, _ in module.named_parameters()}
    assert not any("relation" in name for name in names)


def test_dual_content_control_retrieves_features_and_has_no_symbols() -> None:
    spec = _spec(
        mode="dual_content", control_content_head_dim=40, control_second_head_dim=48
    )
    module = DualAttention(spec, sigma_reparam=False)
    assert module.symbol_table is None and module.symbol_projection is None
    assert module.second_qkv is not None
    assert module.second_out.weight.shape == (64, spec.relational_heads * 48)


def test_future_tokens_cannot_change_earlier_outputs() -> None:
    carrier = _carrier(_spec())
    seq, times, _ = _packet(2, 6)
    with torch.no_grad():
        base, _ = carrier(seq, times)
        altered = seq.clone()
        altered[:, 3:, :-1] *= -5.0
        changed, _ = carrier(altered, times)
    torch.testing.assert_close(changed[:, :3], base[:, :3], rtol=0, atol=0)


def test_only_the_causal_prefix_receives_gradient() -> None:
    carrier = _carrier(_spec())
    seq, times, _ = _packet(1, 5)
    seq = seq.clone().requires_grad_(True)
    out, _ = carrier(seq, times)
    out[0, 2].sum().backward()
    assert seq.grad is not None
    assert seq.grad[0, 3:].abs().sum() == 0
    assert seq.grad[0, :3].abs().sum() > 0


def test_padded_rows_are_finite_and_zero() -> None:
    carrier = _carrier(_spec())
    seq, times, _ = _packet(2, 6, valid_lengths=[6, 2])
    with torch.no_grad():
        out, _ = carrier(seq, times)
    assert torch.isfinite(out).all()
    assert out[1, 2:].abs().sum() == 0


def test_an_all_invalid_row_stays_finite_and_differentiable() -> None:
    spec = _spec()
    module = DualAttention(spec, sigma_reparam=False).eval()
    x = torch.randn(1, 3, spec.d_model, requires_grad=True)
    times = torch.arange(3).unsqueeze(0)
    allowed = torch.zeros(1, 3, 3, dtype=torch.bool)
    out = module(x, times, times, allowed)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_holes_and_left_padding_are_rejected_not_silently_accepted() -> None:
    require_right_padded(torch.tensor([[True, True, False, False]]))
    with pytest.raises(ContractError, match="right-padded"):
        require_right_padded(torch.tensor([[True, False, True, False]]))
    with pytest.raises(ContractError, match="right-padded"):
        require_right_padded(torch.tensor([[False, True, True, False]]))


def test_symbol_offsets_clip_and_use_real_times_not_ranks() -> None:
    query = torch.tensor([[10]])
    keys = torch.tensor([[0, 4, 9, 10]])
    offsets = symbol_offsets(query, keys, 6)
    # -10 clips to -6; -6, -1 and 0 map to 0, 5 and 6 after the +D shift.
    assert offsets.tolist() == [[[0, 0, 5, 6]]]


# --------------------------------------------------------------------------
# 3. Cache lifecycle
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("dat", {}),
        ("symbol_only", {}),
        (
            "dual_content",
            {"control_content_head_dim": 40, "control_second_head_dim": 48},
        ),
    ],
)
def test_cached_rollout_matches_dense_execution(mode: str, extra: dict) -> None:
    carrier = _carrier(_spec(mode=mode, **extra))
    seq, times, _ = _packet(2, 7)
    with torch.no_grad():
        dense, _ = carrier(seq, times)
        hidden = carrier.init_hidden_state(2, torch.device("cpu"))
        steps = [
            carrier(seq[:, i : i + 1], times[:, i : i + 1], hidden)[0]
            for i in range(seq.shape[1])
        ]
    cached = torch.cat(steps, 1)
    _assert_fp32_close(cached, dense)
    # A widened tolerance must not be able to hide accumulating drift, so the
    # error is also required to stay flat rather than grow along the sequence.
    error = (cached - dense).abs()
    per_position = [float(error[:, i].max()) for i in range(seq.shape[1])]
    assert max(per_position[4:]) <= 2.0 * max(per_position[1:4]) + DRIFT_FLOOR


def test_outer_reset_clears_only_the_selected_rows() -> None:
    carrier = _carrier(_spec())
    seq, times, _ = _packet(3, 5)
    with torch.no_grad():
        hidden = carrier.init_hidden_state(3, torch.device("cpu"))
        for i in range(3):
            carrier(seq[:, i : i + 1], times[:, i : i + 1], hidden)
        survivor = hidden.times[1].clone()
        carrier.reset_hidden_state(hidden, [0, 2])
    assert hidden.lengths.tolist() == [0, 3, 0]
    torch.testing.assert_close(hidden.times[1], survivor, rtol=0, atol=0)
    assert (hidden.times[0] == EMPTY_TIME).all()
    for cache in hidden.layers:
        for tensor in cache.tensors().values():
            assert torch.isnan(tensor[0]).all()
            assert torch.isfinite(tensor[1, :3]).all()


def test_inner_boundaries_impose_no_attention_barrier() -> None:
    """History survives an attempt boundary; only an outer reset clears it."""
    carrier = _carrier(_spec())
    seq, times, _ = _packet(1, 5)
    with torch.no_grad():
        hidden = carrier.init_hidden_state(1, torch.device("cpu"))
        for i in range(5):
            carrier(seq[:, i : i + 1], times[:, i : i + 1], hidden)
    assert int(hidden.lengths[0]) == 5
    assert hidden.times[0, :5].tolist() == [0, 1, 2, 3, 4]


def test_rollover_matches_a_sliding_window_oracle() -> None:
    """After eviction the reference is a per-layer sliding window, not full attention.

    Recomputing the newest raw suffix would change the earlier representations
    that upper layers already cached, and unrestricted full attention keeps
    sources this cache has dropped. Both tempting comparisons are wrong here.
    """
    capacity = 4
    carrier = _carrier(_spec(), max_seq_len=capacity - 1)
    assert carrier.capacity == capacity
    length = 2 * capacity + 1
    seq, times, _ = _packet(1, length)
    with torch.no_grad():
        hidden = carrier.init_hidden_state(1, torch.device("cpu"))
        cached = [
            carrier(seq[:, i : i + 1], times[:, i : i + 1], hidden)[0]
            for i in range(length)
        ]
        oracle = _sliding_window_reference(carrier, seq, times, capacity)
    for step in range(length):
        _assert_fp32_close(cached[step][:, 0], oracle[step])
    assert int(hidden.lengths[0]) == capacity - 1


def _sliding_window_reference(
    carrier: DATTrajEncoder, seq: torch.Tensor, times: torch.Tensor, capacity: int
) -> list[torch.Tensor]:
    """Dense sliding window over retained rank, built by the same recursion.

    Each layer's stored key for a source is the *layer input* at the step that
    source was processed, and that input was itself produced under the window in
    force at the time. Reading those vectors off a dense full-sequence pass would
    therefore be a different model, which is exactly the trap this oracle exists
    to avoid. So the reference replays step by step, keeping a per-layer list of
    retained inputs and evicting the oldest on AMAGO's post-step schedule.
    """
    backbone = carrier.backbone
    packet = carrier._unpack(seq)
    emb = backbone.preprocess_seq(packet.memory, times)
    length = seq.shape[1]
    retained: list[list[torch.Tensor]] = [[] for _ in backbone.layers]
    retained_times: list[list[torch.Tensor]] = [[] for _ in backbone.layers]
    outputs: list[torch.Tensor] = []
    for step in range(length):
        current = emb[:, step : step + 1]
        current_time = times[:, step : step + 1, 0]
        for index, layer in enumerate(backbone.layers):
            block = torch.cat([*retained[index], current], dim=1)
            block_times = torch.cat([*retained_times[index], current_time], dim=1)
            width = block.shape[1]
            if index in backbone.selected:
                allowed = torch.ones(block.shape[0], width, width, dtype=torch.bool)
                out = layer(block, block_times, block_times, allowed)[:, -1:]
            else:
                out = layer(block)[:, -1:]
            retained[index].append(current)
            retained_times[index].append(current_time)
            if len(retained[index]) == capacity:
                retained[index].pop(0)
                retained_times[index].pop(0)
            current = out
        outputs.append(backbone.norm(current)[:, 0])
    return outputs


def test_nonzero_starting_times_are_carried_through_the_cache() -> None:
    """A rollout that begins mid-trajectory must keep its real positions."""
    carrier = _carrier(_spec())
    seq, _, _ = _packet(2, 5)
    offset = 37
    times = (torch.arange(5) + offset).view(1, 5, 1).expand(2, 5, 1).contiguous()
    with torch.no_grad():
        dense, _ = carrier(seq, times)
        hidden = carrier.init_hidden_state(2, torch.device("cpu"))
        steps = [
            carrier(seq[:, i : i + 1], times[:, i : i + 1], hidden)[0] for i in range(5)
        ]
    _assert_fp32_close(torch.cat(steps, 1), dense)
    assert hidden.times[0, :5].tolist() == [offset + i for i in range(5)]


def test_outer_reset_at_and_after_a_rollover_boundary() -> None:
    """Eviction and reset interact, so both orderings are exercised."""
    capacity = 4
    carrier = _carrier(_spec(), max_seq_len=capacity - 1)
    seq, times, _ = _packet(2, capacity + 2)
    with torch.no_grad():
        hidden = carrier.init_hidden_state(2, torch.device("cpu"))
        for step in range(capacity):  # the last of these evicts
            carrier(seq[:, step : step + 1], times[:, step : step + 1], hidden)
        assert int(hidden.lengths[0]) == capacity - 1
        carrier.reset_hidden_state(hidden, [0])  # reset exactly at the boundary
        assert int(hidden.lengths[0]) == 0
        assert int(hidden.lengths[1]) == capacity - 1
        for step in range(capacity, capacity + 2):
            carrier(seq[:, step : step + 1], times[:, step : step + 1], hidden)
        # Row 0 restarted, so it holds only the two decisions since its reset.
        assert int(hidden.lengths[0]) == 2
        assert hidden.times[0, :2].tolist() == [capacity, capacity + 1]
        carrier.reset_hidden_state(hidden, [1])  # reset just after a rollover
    assert int(hidden.lengths[1]) == 0
    assert (hidden.times[1] == EMPTY_TIME).all()
    for cache in hidden.layers:
        for tensor in cache.tensors().values():
            assert torch.isnan(tensor[1]).all()
            assert torch.isfinite(tensor[0, :2]).all()


def test_cached_execution_refuses_training_mode_and_multi_step_queries() -> None:
    carrier = _carrier(_spec())
    seq, times, _ = _packet(1, 4)
    hidden = carrier.init_hidden_state(1, torch.device("cpu"))
    with torch.no_grad(), pytest.raises(ContractError, match="one valid decision"):
        carrier(seq[:, :2], times[:, :2], hidden)
    carrier.train()
    with torch.no_grad(), pytest.raises(ContractError, match="evaluation-only"):
        carrier(seq[:, :1], times[:, :1], hidden)


def test_gradients_reach_every_enabled_branch() -> None:
    carrier = _carrier(_spec())
    carrier.train()
    seq, times, _ = _packet(2, 5)
    out, _ = carrier(seq, times)
    out.pow(2).mean().backward()
    attention = carrier.backbone.layers[2].attention
    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_the_discarded_ordinary_attention_is_not_retained() -> None:
    """A replaced block must not keep its donor QKV as unused parameters."""
    carrier = _carrier(_spec())
    names = {name for name, _ in carrier.backbone.layers[2].named_parameters()}
    assert not any("attention_layer" in name for name in names)
    assert not any(name.startswith("qkv_projection") for name in names)


# --------------------------------------------------------------------------
# 4. The CUDA content kernel
# --------------------------------------------------------------------------
#
# The selected block runs a fused content kernel on CUDA and the dense one on
# CPU. That is a reduction-order choice, not a scientific one, so it stays out
# of ``DATSpec`` and the identity hash; what has to be qualified is that the two
# really do compute the same operator, including at the masks the cached path
# produces and in the gradients the learner consumes.


@pytest.mark.cuda
def test_the_fused_content_kernel_computes_the_dense_operator(
    cuda_runtime: None,
) -> None:
    """Both content paths, same inputs, forward and backward."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, length, heads, head_dim = 2, 48, 6, 32
    shape = (batch, length, heads, head_dim)
    inputs = [torch.randn(shape, device=device) for _ in range(3)]
    allowed = _causal(batch, length).unsqueeze(1).to(device)
    scale = 1.0 / math.sqrt(head_dim)

    outputs = []
    for attention in (_dense_content_attention, _fused_content_attention):
        tensors = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
        out = attention(*tensors, allowed, scale=scale)
        out.pow(2).sum().backward()
        outputs.append((out, [_grad(tensor) for tensor in tensors]))

    (dense_out, dense_grads), (fused_out, fused_grads) = outputs
    _assert_fp32_close(fused_out, dense_out)
    for fused_grad, dense_grad in zip(fused_grads, dense_grads, strict=True):
        _assert_fp32_close(fused_grad, dense_grad, relative=FUSED_BACKWARD_SCALED_ATOL)


@pytest.mark.cuda
def test_the_fused_content_kernel_zeroes_rows_with_no_permitted_source(
    cuda_runtime: None,
) -> None:
    """A padded row and a freshly reset environment must stay finite, not NaN.

    ``scaled_dot_product_attention`` returns NaN for a fully masked row, and a
    NaN multiplied by zero afterwards is still NaN in the backward pass, so the
    neutralization has to happen before the softmax. This is the case that
    catches it being done in the wrong order.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, length, heads, head_dim = 2, 6, 4, 16
    shape = (batch, length, heads, head_dim)
    tensors = [torch.randn(shape, device=device, requires_grad=True) for _ in range(3)]
    allowed = _causal(batch, length).unsqueeze(1).to(device).clone()
    allowed[1, :, 3:] = False  # row 1 is padded from decision 3 onwards

    out = _fused_content_attention(*tensors, allowed, scale=1.0 / math.sqrt(head_dim))
    assert torch.isfinite(out).all()
    assert (out[1, 3:] == 0).all()
    out.pow(2).sum().backward()
    for tensor in tensors:
        assert torch.isfinite(_grad(tensor)).all()

    mirror = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
    dense = _dense_content_attention(*mirror, allowed, scale=1.0 / math.sqrt(head_dim))
    _assert_fp32_close(out, dense)


@pytest.mark.cuda
def test_the_cached_rollout_agrees_with_dense_execution_on_cuda(
    cuda_runtime: None,
) -> None:
    """The end-to-end check of the dense/cached contract, on the fused kernel.

    The cached path attends over retained slots with a per-environment length
    rather than plain causality, which is the mask a causal-only kernel could
    not express, so it is qualified here rather than assumed from the CPU run.
    """
    carrier = _carrier(_spec()).to("cuda")
    seq, times, _ = _packet(2, 7)
    seq, times = seq.to("cuda"), times.to("cuda")
    with torch.no_grad():
        dense, _ = carrier(seq, times)
        hidden = carrier.init_hidden_state(2, torch.device("cuda"))
        steps = [
            carrier(seq[:, i : i + 1], times[:, i : i + 1], hidden)[0]
            for i in range(seq.shape[1])
        ]
    _assert_fp32_close(torch.cat(steps, 1), dense)


@pytest.mark.cuda
def test_the_fused_kernel_accepts_the_degenerate_cached_layouts(
    cuda_runtime: None,
) -> None:
    """The extent-one strides the CUDA kernel refuses if they are passed through.

    Both layouts below were taken from a native run that failed with "cutlassF:
    no kernel found to launch": one selected block's query at a single cached
    decision, and the same block's query over a partly filled cache on a
    one-head backbone. In each the extent-one dimension carries the fused QKV
    projection's stride rather than its own, which PyTorch considers contiguous
    and ``Tensor.contiguous`` will not change, so they are built here by their
    strides rather than by a shape that would hide the point.
    """
    device = torch.device("cuda")
    head_dim = 16
    scale = 1.0 / math.sqrt(head_dim)
    # (shape, strides) as [B, T, H, D], whose transpose is what reaches the kernel.
    layouts = (
        ((1, 1, 1, head_dim), (48, 48, 1, 1)),
        ((1, 11, 1, head_dim), (528, 48, 1, 1)),
    )
    for shape, strides in layouts:
        torch.manual_seed(0)
        length = shape[1]
        tensors = [
            torch.randn(4096, device=device).as_strided(shape, strides)
            for _ in range(3)
        ]
        assert tensors[0].transpose(1, 2).stride()[1] == 1
        allowed = (
            torch.ones(shape[0], length, length, dtype=torch.bool, device=device)
            .tril()
            .unsqueeze(1)
        )
        out = _fused_content_attention(*tensors, allowed, scale=scale)
        _assert_fp32_close(
            out, _dense_content_attention(*tensors, allowed, scale=scale)
        )


@pytest.mark.cuda
def test_the_recorded_content_backend_follows_the_device(
    cuda_runtime: None,
) -> None:
    """The audit must report the kernel that ran, not the configured setting."""
    assert content_backend("cpu") == "vanilla"
    assert content_backend("cuda") == "fused"
    assert content_backend(torch.device("cuda", 0)) == "fused"
