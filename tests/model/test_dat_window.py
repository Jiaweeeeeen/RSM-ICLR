"""The dual-attention window (R3, SPEC §5): the revised band behind
``amago-dat-window-v1`` and the byte accounting that chooses its width.

Every check is CPU, FP32 and deterministic. Dense banded training, the rolling
cached rollout and the one-pass rebuild must agree through wraparound with
dual attention in every layer; symbols follow the sources' stored trajectory
times rather than their ranks; the raw receptive field is ``1 + d (W - 1)`` and
not one record more; cached positions survive a checkpoint; the state refuses
every other carrier; and the closed-form state bytes equal the allocation
tensor for tensor, on the test geometry and on the study's, so the window
length the study resolves is a measured match and not slot parity.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import gin
import pytest
import torch
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.experiments.contracts import (
    DAT_WINDOW_ARCHITECTURE_ID,
    ContractError,
    DATSpec,
    SummarySpec,
    WindowSpec,
    cache_slot_floats,
    match_window_to_summary,
    summary_state_bytes,
    window_state_bytes,
)
from reasoned_icrl.model.dat_transformer import EMPTY_TIME, DATBlock, DATHiddenState
from reasoned_icrl.model.summary_transformer import MaskedOrdinaryBlock
from reasoned_icrl.model.trajectory_encoder import (
    SummaryTrajEncoder,
    WindowTrajEncoder,
    _copy_row,
)
from reasoned_icrl.model.window_transformer import (
    DAT_WINDOW_HIDDEN_STATE_SCHEMA,
    WINDOW_HIDDEN_STATE_SCHEMA,
    WindowTransformer,
    receptive_field,
)
from reasoned_icrl.runtime.checkpointing import (
    _hidden_state,
    _restore_hidden_state,
    _runtime_contract,
)
from reasoned_icrl.runtime.experiment import (
    cache_measurements,
    carrier_cost,
    shared_state_bytes,
    state_tensor_bytes,
)

TOKEN, WIDTH, HEADS = 16, 32, 2
W = 4
PARITY_ATOL = 5e-6


def _configure(layers: int, *, width: int = WIDTH, heads: int = HEADS) -> None:
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


def _dat(
    layers: int, *, window: int = W, width: int = WIDTH, heads: int = HEADS, **kw: Any
) -> DATSpec:
    settings: dict[str, Any] = {
        "layer_indices": tuple(range(layers)),
        "d_model": width,
        "total_heads": heads,
        "relational_heads": 1,
        "relation_channels": 4,
        "relation_projection_dim": 4,
        "symbol_dim": 8,
        "max_relative_distance": window,
    }
    settings.update(kw)
    return DATSpec(**settings)


def _carrier(
    layers: int = 2,
    *,
    window: int = W,
    seed: int = 0,
    max_seq_len: int = 16,
    dat: bool = True,
    **dat_overrides: Any,
) -> WindowTrajEncoder:
    _configure(layers)
    torch.manual_seed(seed)
    return WindowTrajEncoder(
        TOKEN + 1,
        max_seq_len,
        spec=WindowSpec(segment_length=window),
        dat=_dat(layers, window=window, **dat_overrides) if dat else None,
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
        assert set(ours.tensors()) == set(theirs.tensors())
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


# --------------------------------------------------------------------------
# Construction: dual attention over the band, the donor everywhere else
# --------------------------------------------------------------------------


def test_the_dat_window_runs_dual_attention_in_every_selected_layer() -> None:
    carrier = _carrier(2)
    backbone = carrier.backbone
    assert isinstance(backbone, WindowTransformer)
    assert backbone.selected == frozenset({0, 1})
    assert all(isinstance(block, DATBlock) for block in backbone.layers)
    assert backbone.hidden_schema == DAT_WINDOW_HIDDEN_STATE_SCHEMA
    assert carrier.dat is not None and carrier.attention_protocol_identity.numel() == 32
    hidden = carrier.init_hidden_state(2, torch.device("cpu"))
    assert hidden.schema == DAT_WINDOW_HIDDEN_STATE_SCHEMA
    assert hidden.window_sha256 == carrier.spec.sha256
    assert hidden.attention_sha256 == carrier.dat.sha256
    for cache in hidden.layers:
        assert cache.variant == "dat"
        assert set(cache.tensors()) == {
            "content_keys",
            "content_values",
            "selection_keys",
            "relation_keys",
        }
    # A partially selected backbone keeps the donor's blocks where unselected.
    mixed = _carrier(2, layer_indices=(1,))
    assert isinstance(mixed.backbone.layers[0], MaskedOrdinaryBlock)
    assert isinstance(mixed.backbone.layers[1], DATBlock)
    mixed_state = mixed.init_hidden_state(1, torch.device("cpu"))
    assert mixed_state.layers[0].variant == "ordinary"
    assert set(mixed_state.layers[0].tensors()) == {"content_keys", "content_values"}
    # The ordinary window is the same class without a spec, on its own schema.
    ordinary = _carrier(2, dat=False)
    assert ordinary.dat is None
    assert ordinary.backbone.hidden_schema == WINDOW_HIDDEN_STATE_SCHEMA
    assert not hasattr(ordinary, "attention_protocol_identity")


def test_the_dat_window_refuses_unqualified_construction() -> None:
    _configure(2)
    donor_kwargs = {"token_dim": TOKEN, "d_model": WIDTH}
    with pytest.raises(ContractError, match="window length"):
        WindowTrajEncoder(
            TOKEN + 1,
            8,
            spec=WindowSpec(segment_length=W),
            dat=_dat(2, window=W + 1),
            **donor_kwargs,
        )
    with pytest.raises(ContractError, match="dual-content control"):
        WindowTrajEncoder(
            TOKEN + 1,
            8,
            spec=WindowSpec(segment_length=W),
            dat=_dat(
                2,
                mode="dual_content",
                control_content_head_dim=8,
                control_second_head_dim=8,
            ),
            **donor_kwargs,
        )
    with pytest.raises(ContractError, match="width disagrees"):
        WindowTrajEncoder(
            TOKEN + 1,
            8,
            spec=WindowSpec(segment_length=W),
            dat=_dat(2, width=64, heads=4),
            **donor_kwargs,
        )
    with pytest.raises(ContractError, match="outside a 1-layer"):
        _configure(1)
        WindowTrajEncoder(
            TOKEN + 1, 8, spec=WindowSpec(segment_length=W), dat=_dat(2), **donor_kwargs
        )
    with pytest.raises(ContractError, match="positive depth"):
        receptive_field(0, W)


# --------------------------------------------------------------------------
# Dense equals cached equals rebuilt, through wraparound
# --------------------------------------------------------------------------


@pytest.mark.parametrize("layers", [1, 2])
def test_banded_dense_equals_the_rolling_cached_rollout(layers: int) -> None:
    carrier = _carrier(layers)
    lengths = [3 * W + 2, 2 * W + 1, W]  # two evicting rows and one exactly full
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
    assert hidden.lengths.tolist() == [W - 1] * 3
    assert hidden.times[:, : W - 1].tolist() == [list(range(total - W + 1, total))] * 3
    assert bool((hidden.times[:, W - 1 :] == EMPTY_TIME).all())


def test_rebuild_equals_an_uninterrupted_rollout_and_survives_a_weight_change() -> None:
    carrier = _carrier(2)
    lengths = [3 * W + 2, 2 * W, W - 1, 0, 1]  # evicting, evicting, full, empty, one
    seq, times = _packet(5, max(lengths), lengths)
    before = carrier.rebuild_hidden_state(seq, times, lengths)
    with torch.no_grad():
        for parameter in carrier.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    rebuilt = carrier.rebuild_hidden_state(seq, times, lengths)
    assert rebuilt.lengths.tolist() == [W - 1, W - 1, W - 1, 0, 1]
    assert not torch.equal(
        torch.nan_to_num(before.layers[1].relation_keys[0]),
        torch.nan_to_num(rebuilt.layers[1].relation_keys[0]),
    )
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
# Symbols read stored times; the receptive field is 1 + d (W - 1)
# --------------------------------------------------------------------------


def test_symbols_follow_the_stored_source_times_after_eviction() -> None:
    """After a wraparound the retained ranks are 0..W-2 but the sources' real
    positions are older; the relational symbols must read the latter."""
    carrier = _carrier(1)
    seq, times = _packet(1, 3 * W)
    _, hidden = _rollout(carrier, seq, times)
    assert hidden.times[0, : W - 1].tolist() == list(range(2 * W + 1, 3 * W))
    assert hidden.times[0, : W - 1].tolist() != list(range(W - 1))
    tampered = _select_row(carrier, hidden, 0)
    tampered.times[0, : W - 1] = torch.arange(W - 1)  # ranks instead of times
    following = seq[:, :1].clone()
    following[..., -1] = 1.0
    next_time = torch.full((1, 1, 1), 3 * W, dtype=torch.long)
    with torch.no_grad():
        honest, _ = carrier(following, next_time, _select_row(carrier, hidden, 0))
        relabeled, _ = carrier(following, next_time, tampered)
    # Only the symbol offsets changed: the cached keys, values and the query's
    # own position are identical, so a difference proves the symbols consumed
    # the stored times.
    assert (honest - relabeled).abs().max() > 1e-5


@pytest.mark.parametrize("layers", [1, 2])
def test_records_beyond_the_depth_scaled_band_have_no_influence(layers: int) -> None:
    """The bound is ``1 + layers (W - 1)`` records, not W: the record at
    exactly that distance still reaches the output through the layers below,
    and the record one step further has exactly zero effect."""
    carrier = _carrier(layers)
    reach = receptive_field(layers, W)
    assert reach == carrier.backbone.raw_receptive_field == 1 + layers * (W - 1)
    length = reach + 3
    seq, times = _packet(1, length)
    other = seq.clone()
    other[0, 0, :TOKEN] += 1.0
    with torch.no_grad():
        base, _ = carrier(seq, times)
        moved, _ = carrier(other, times)
    # Position reach - 1 is the last one whose band chain reaches record 0.
    assert not torch.allclose(base[0, reach - 1], moved[0, reach - 1])
    torch.testing.assert_close(base[0, reach:], moved[0, reach:], rtol=0, atol=0)


# --------------------------------------------------------------------------
# Checkpoint: cached positions survive, other carriers are refused
# --------------------------------------------------------------------------


def test_cached_positions_survive_a_checkpoint_round_trip() -> None:
    carrier = _carrier(2)
    seq, times = _packet(2, 2 * W + 1)
    _, hidden = _rollout(carrier, seq, times)
    assert hidden.times[0, 0].item() > 0  # past the wraparound: rank 0 is not time 0
    payload = _hidden_state(hidden)
    assert payload["schema"] == DAT_WINDOW_HIDDEN_STATE_SCHEMA
    assert payload["window_sha256"] == carrier.spec.sha256
    assert carrier.dat is not None
    assert payload["attention_sha256"] == carrier.dat.sha256
    restored = _restore_hidden_state(_experiment(carrier), payload)
    assert isinstance(restored, DATHiddenState) and restored is not hidden
    assert restored.schema == DAT_WINDOW_HIDDEN_STATE_SCHEMA
    assert restored.window_sha256 == carrier.spec.sha256
    for row in range(2):
        _assert_row_parity(restored, row, hidden, row)
    next_time = torch.full((2, 1, 1), 2 * W + 1, dtype=torch.long)
    with torch.no_grad():
        a, _ = carrier(seq[:, :1], next_time, hidden)
        b, _ = carrier(seq[:, :1], next_time, restored)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_the_state_refuses_every_other_carrier() -> None:
    carrier = _carrier(2)
    seq, times = _packet(2, W + 2)
    _, hidden = _rollout(carrier, seq, times)
    payload = _hidden_state(hidden)
    with pytest.raises(ContractError, match="window identity does not match"):
        _restore_hidden_state(_experiment(_carrier(2, window=W + 1)), payload)
    with pytest.raises(ContractError, match="attention identity does not match"):
        _restore_hidden_state(
            _experiment(_carrier(2, symmetric_relations=True)), payload
        )
    with pytest.raises(ContractError, match="does not match the policy"):
        _restore_hidden_state(_experiment(_carrier(2, dat=False)), payload)
    ordinary = _carrier(2, dat=False)
    _, ordinary_hidden = _rollout(ordinary, seq, times)
    with pytest.raises(ContractError, match="does not match the policy"):
        _restore_hidden_state(_experiment(carrier), _hidden_state(ordinary_hidden))
    relabeled = dict(payload)
    relabeled["schema"] = WINDOW_HIDDEN_STATE_SCHEMA
    relabeled.pop("attention_sha256")
    with pytest.raises(ContractError, match="does not match the policy"):
        _restore_hidden_state(_experiment(carrier), relabeled)
    with pytest.raises(ContractError, match="different carrier"):
        _carrier(2, dat=False)(seq[:, :1], times[:, :1], hidden)
    with pytest.raises(ContractError, match="different carrier"):
        carrier(seq[:, :1], times[:, :1], ordinary_hidden)


def test_the_resume_contract_names_the_window_and_the_attention() -> None:
    carrier = _carrier(2)
    experiment = SimpleNamespace(
        encoder_architecture_id=DAT_WINDOW_ARCHITECTURE_ID,
        policy_condition="fixed_window",
        policy=SimpleNamespace(
            traj_encoder=carrier,
            tstep_encoder=SimpleNamespace(spec=SimpleNamespace(sha256="packet")),
        ),
        learner_contract="amago-optimizer-ownership.v1",
        reasoned_training_settings={},
    )
    contract = _runtime_contract(experiment)
    assert contract["window_sha256"] == carrier.spec.sha256
    assert carrier.dat is not None
    assert contract["attention_sha256"] == carrier.dat.sha256
    assert "summary_sha256" not in contract


# --------------------------------------------------------------------------
# Bytes: the closed form equals the allocation, so W is a measured match
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dat", [False, True])
def test_the_closed_form_state_bytes_equal_the_allocation(dat: bool) -> None:
    layers = 2
    carrier = _carrier(layers, dat=dat)
    single = carrier.init_hidden_state(1, torch.device("cpu"))
    measured = cache_measurements(single)["cache_bytes"]
    assert (
        window_state_bytes(W, carrier.dat, layers=layers, heads=HEADS, width=WIDTH)
        == measured
    )
    assert sum(state_tensor_bytes(single).values()) == measured
    assert shared_state_bytes(single) == 0
    # The summary carrier of the same geometry, both routes of the same table.
    summary = SummarySpec(
        segment_length=W, memory_tokens=2, regime="summary", d_model=WIDTH
    )
    _configure(layers)
    torch.manual_seed(0)
    reference = SummaryTrajEncoder(
        TOKEN + 1,
        16,
        spec=summary,
        dat=_dat(layers, window=summary.capacity) if dat else None,
        token_dim=TOKEN,
        d_model=WIDTH,
    ).eval()
    state = reference.init_hidden_state(1, torch.device("cpu"))
    assert (
        summary_state_bytes(
            summary, reference.dat, layers=layers, heads=HEADS, width=WIDTH
        )
        == cache_measurements(state)["cache_bytes"]
    )
    assert shared_state_bytes(state) == 2 * WIDTH * 4
    assert "memory" in state_tensor_bytes(state)


def test_the_study_geometry_reproduces_the_recorded_bytes() -> None:
    """The 4M study recorded 249,868 B for the DAT summary at 40 slots and
    246,084 B for the ordinary window at W=40; the closed form reproduces both,
    and the DAT window at W=40 allocates exactly the ordinary window's bytes
    because a selected layer's slot is as wide as an ordinary layer's."""
    study = DATSpec(
        layer_indices=(0, 1, 2),
        relational_heads=4,
        relation_channels=8,
        relation_projection_dim=16,
        symmetric_relations=True,
        max_relative_distance=40,
    )
    assert cache_slot_floats(study, layers=3, heads=8, width=256) == (512, 512, 512)
    assert cache_slot_floats(None, layers=3, heads=8, width=256) == (512, 512, 512)
    summary = SummarySpec(segment_length=32, memory_tokens=4, regime="summary")
    assert summary_state_bytes(summary, study, layers=3, heads=8, width=256) == 249_868
    assert window_state_bytes(40, None, layers=3, heads=8, width=256) == 246_084
    assert window_state_bytes(40, study, layers=3, heads=8, width=256) == 246_084


def test_the_matched_window_follows_bytes_not_slot_parity() -> None:
    study = DATSpec(
        layer_indices=(0, 1, 2),
        relational_heads=4,
        relation_channels=8,
        relation_projection_dim=16,
        symmetric_relations=True,
        max_relative_distance=500,
    )

    def match(segment: int, tokens: int) -> Any:
        return match_window_to_summary(
            SummarySpec(segment_length=segment, memory_tokens=tokens, regime="summary"),
            study,
            layers=3,
            heads=8,
            width=256,
        )

    # The three contracts' starting pairs: parity and the byte match coincide,
    # because M=4 memory tokens (4,096 B) are worth less than one 6,152 B slot.
    for (segment, tokens), expected in (((32, 4), 40), ((16, 4), 24), ((64, 4), 72)):
        result = match(segment, tokens)
        assert result.window_length == expected == result.slot_parity_length
        assert 0 < result.residual_bytes < 6_152
        assert result.window_state_bytes <= result.summary_state_bytes
        assert (
            window_state_bytes(expected + 1, study, layers=3, heads=8, width=256)
            > result.summary_state_bytes
        )
    # With M=8 the memory tokens buy a slot: parity would under-allocate.
    eight = match(32, 8)
    assert eight.window_length == 49 and eight.slot_parity_length == 48
    assert match(16, 8).window_length == 33
    assert match(64, 8).window_length == 81
    assert match(128, 4).window_length == 136
    # The smallest admissible summary (C=4, M=1) still matches a window of at
    # least its own slot count: a window slot costs one summary slot plus its
    # int64 source time, and the memory tokens cover the counters' difference.
    smallest = match_window_to_summary(
        SummarySpec(segment_length=4, memory_tokens=1, regime="summary", d_model=8),
        None,
        layers=1,
        heads=1,
        width=8,
    )
    assert smallest.window_length >= 4
    with pytest.raises(ContractError, match="geometry disagrees"):
        cache_slot_floats(study, layers=3, heads=4, width=256)
    with pytest.raises(ContractError, match="disagrees with the model"):
        summary_state_bytes(
            SummarySpec(
                segment_length=4, memory_tokens=1, regime="summary", d_model=64
            ),
            None,
            layers=1,
            heads=2,
            width=32,
        )


# --------------------------------------------------------------------------
# The cost probe covers the wraparound and counts FLOPs
# --------------------------------------------------------------------------


def test_the_cost_probe_covers_the_wraparound_and_counts_flops() -> None:
    carrier = _carrier(2)
    policy = SimpleNamespace(traj_encoder=carrier, eval=carrier.eval)
    cost = carrier_cost(policy, rows=2, device=torch.device("cpu"))
    assert cost["latency_probe"]["steps"] == W + 8
    assert cost["latency_probe"]["decisions_timed"] == W + 7
    assert cost["latency_probe"]["boundaries_timed"] == 0
    assert cost["boundary_latency_seconds"] is None
    assert cost["boundary_flops_counted"] is None
    assert isinstance(cost["decision_flops_counted"], int)
    assert cost["decision_flops_counted"] > 0
    assert cost["persistent_state_bytes"] == sum(cost["state_tensor_bytes"].values())
    assert cost["shared_state_bytes"] == 0
    assert "matrix-multiply" in cost["flops_method"]
    # A wider band reads more sources per decision, so it costs more counted
    # FLOPs at steady state; the probe measures what the carrier actually ran.
    wider = _carrier(2, window=2 * W)
    more = carrier_cost(
        SimpleNamespace(traj_encoder=wider, eval=wider.eval),
        rows=2,
        device=torch.device("cpu"),
    )
    assert more["decision_flops_counted"] > cost["decision_flops_counted"]
    assert more["persistent_state_bytes"] > cost["persistent_state_bytes"]


def test_a_spec_change_that_leaves_shapes_alone_still_changes_the_identity() -> None:
    base = _dat(2)
    assert replace(base, symmetric_relations=True).sha256 != base.sha256
    assert replace(base, max_relative_distance=W + 1).sha256 != base.sha256
    assert (
        WindowSpec(segment_length=W).sha256 != WindowSpec(segment_length=W + 1).sha256
    )
