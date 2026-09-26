"""Measured cost and the mechanism fixtures the comparison depends on.

The integration plan's parameter table is analytical. These tests measure the
constructed modules instead, and freeze the capacity control's branch widths
from counts alone -- before any reward is observed -- so a later result cannot
be explained by an unmatched budget.

They also pin the evaluation fixtures whose definitions are easy to get subtly
wrong: matched prefixes that end at the same public observation, and context
truncation that actually removes evidence rather than merely evicting a key.
"""

from __future__ import annotations

import gin
import pytest
import torch
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.experiments.contracts import DATSpec
from reasoned_icrl.model.dat_transformer import (
    DualAttention,
    dual_content_head_dims,
    selected_parameter_count,
)
from reasoned_icrl.model.trajectory_encoder import DATTrajEncoder

#: Analytical counts from DAT_INTEGRATION_PLAN.md section 9 (a retired document,
#: at commit 99e2b85), which asked to validate them on constructed modules.
PLANNED = {
    160: {"dat": 312_082, "control": 310_540, "content": 40, "second": 48},
    320: {"dat": 353_042, "control": 356_620, "content": 48, "second": 48},
    500: {"dat": 399_122, "control": 402_700, "content": 56, "second": 48},
}


def _spec(distance: int, **overrides: object) -> DATSpec:
    settings: dict[str, object] = {
        "layer_indices": (2,),
        "max_relative_distance": distance,
    }
    settings.update(overrides)
    return DATSpec(**settings)  # type: ignore[arg-type]


def _attention(spec: DATSpec) -> DualAttention:
    torch.manual_seed(0)
    return DualAttention(spec)


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


# --------------------------------------------------------------------------
# Measured capacity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("distance", sorted(PLANNED))
def test_measured_attention_parameters_match_the_planned_counts(
    distance: int,
) -> None:
    """Constructed modules, not arithmetic: the plan asks for this before lock."""
    planned = PLANNED[distance]
    dat = sum(p.numel() for p in _attention(_spec(distance)).parameters())
    control = sum(
        p.numel()
        for p in _attention(
            _spec(
                distance,
                mode="dual_content",
                control_content_head_dim=planned["content"],
                control_second_head_dim=planned["second"],
            )
        ).parameters()
    )
    # The plan's analytical figures are reproduced to within a handful of
    # parameters; the residual is reported rather than assumed away.
    assert abs(dat - planned["dat"]) <= 4, (dat, planned["dat"])
    assert abs(control - planned["control"]) <= 4, (control, planned["control"])


@pytest.mark.parametrize("distance", sorted(PLANNED))
def test_the_capacity_control_stays_within_two_percent(distance: int) -> None:
    planned = PLANNED[distance]
    dat = sum(p.numel() for p in _attention(_spec(distance)).parameters())
    control = sum(
        p.numel()
        for p in _attention(
            _spec(
                distance,
                mode="dual_content",
                control_content_head_dim=planned["content"],
                control_second_head_dim=planned["second"],
            )
        ).parameters()
    )
    assert abs(dat - control) / dat < 0.02


def test_control_widths_can_be_frozen_from_counts_before_any_reward() -> None:
    """The search that picks the control's widths sees only parameter counts."""
    spec = _spec(160)
    target = sum(p.numel() for p in _attention(spec).parameters())
    chosen = dual_content_head_dims(spec, target)
    assert chosen.mode == "dual_content"
    assert chosen.control_content_head_dim is not None
    control = sum(p.numel() for p in _attention(chosen).parameters())
    assert abs(control - target) / target < 0.02


def test_symbol_only_reports_a_smaller_active_capacity() -> None:
    """Its relation modules are absent, not disconnected, so the count is honest."""
    full = _attention(_spec(160))
    symbols = _attention(_spec(160, mode="symbol_only"))
    full_count = sum(p.numel() for p in full.parameters())
    symbol_count = sum(p.numel() for p in symbols.parameters())
    assert symbol_count < full_count
    relation = sum(
        p.numel()
        for name, p in full.named_parameters()
        if name.startswith(("relation_q", "relation_k", "relation_out"))
    )
    assert full_count - symbol_count == relation


def test_the_selected_layer_is_the_only_replaced_attention() -> None:
    carrier = _carrier(_spec(6))
    backbone = carrier.backbone
    assert backbone.selected == frozenset({2})
    active = selected_parameter_count(backbone)
    assert active == sum(p.numel() for p in backbone.layers[2].attention.parameters())
    for index in (0, 1):
        assert hasattr(backbone.layers[index], "attention_layer")


def test_the_cached_floating_point_width_is_unchanged_in_the_selected_layer() -> None:
    """DAT redistributes the cache; it does not compress it. No KV-saving claim."""
    spec = _spec(160)
    ordinary = 2 * spec.total_heads * spec.head_dim
    dat = (
        2 * spec.content_heads * spec.head_dim
        + spec.relational_heads * spec.head_dim
        + spec.relation_channels * spec.relation_projection_dim
    )
    assert ordinary == dat == 512


def test_the_cache_records_a_backend_map_not_one_label() -> None:
    carrier = _carrier(_spec(6))
    hidden = carrier.init_hidden_state(2, torch.device("cpu"))
    variants = [cache.variant for cache in hidden.layers]
    assert variants == ["ordinary", "ordinary", "dat"]
    assert hidden.layers[2].relation_keys is not None
    assert hidden.layers[0].relation_keys is None
    # The relational branch is dense even when the rest of the model is not.
    assert carrier.spec.relational_backend == "dense"


def test_symbol_only_and_dual_content_allocate_only_what_they_use() -> None:
    symbols = _carrier(_spec(6, mode="symbol_only")).init_hidden_state(
        1, torch.device("cpu")
    )
    assert symbols.layers[2].relation_keys is None
    assert symbols.layers[2].selection_keys is not None
    assert symbols.layers[2].selection_values is None

    control = _carrier(
        _spec(
            6,
            mode="dual_content",
            control_content_head_dim=40,
            control_second_head_dim=48,
        )
    ).init_hidden_state(1, torch.device("cpu"))
    assert control.layers[2].relation_keys is None
    assert control.layers[2].selection_values is not None


# --------------------------------------------------------------------------
# Mechanism fixtures
# --------------------------------------------------------------------------


def _prefix(rows: list[list[float]], length: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq = torch.zeros(1, length, 65)
    for step, values in enumerate(rows):
        seq[0, step, : len(values)] = torch.tensor(values)
    seq[..., -1] = 1.0
    times = torch.arange(length).view(1, length, 1)
    return seq, times


def test_matched_prefixes_end_at_the_same_observation_but_differ_earlier() -> None:
    """Matching only the final observation would let the last step explain it."""
    carrier = _carrier(_spec(6))
    ending = [1.0, 0.0, 0.5]
    left, times = _prefix([[0.0, 1.0], [1.0, 0.0], ending], 3)
    right, _ = _prefix([[1.0, 0.0], [0.0, 1.0], ending], 3)
    torch.testing.assert_close(left[:, -1], right[:, -1], rtol=0, atol=0)
    assert not torch.equal(left[:, :-1], right[:, :-1])
    with torch.no_grad():
        a, _ = carrier(left, times)
        b, _ = carrier(right, times)
    # Same final observation, different earlier evidence: the history-conditioned
    # representation must be able to tell them apart.
    assert (a[:, -1] - b[:, -1]).abs().max() > 1e-4


def test_truncation_must_reset_and_rebuild_not_merely_evict() -> None:
    """Evicting a key still leaves older evidence inside contextualized keys.

    So a context-truncation diagnostic that only drops cache slots is measuring
    something weaker than "this evidence was removed". Removing evidence means
    resetting and replaying the suffix, which is what this asserts differs.
    """
    carrier = _carrier(_spec(6), max_seq_len=8)
    seq, times = _prefix([[float(i)] for i in range(6)], 6)
    with torch.no_grad():
        evicted = carrier.init_hidden_state(1, torch.device("cpu"))
        for step in range(6):
            out_evicted, _ = carrier(
                seq[:, step : step + 1], times[:, step : step + 1], evicted
            )
        # Rebuild from the suffix alone, which genuinely removes the prefix.
        rebuilt = carrier.init_hidden_state(1, torch.device("cpu"))
        for step in range(3, 6):
            out_rebuilt, _ = carrier(
                seq[:, step : step + 1], times[:, step : step + 1], rebuilt
            )
    assert int(evicted.lengths[0]) == 6
    assert int(rebuilt.lengths[0]) == 3
    assert (out_evicted - out_rebuilt).abs().max() > 1e-4


def test_nonconsecutive_times_produce_nonconsecutive_symbols() -> None:
    """Retained rank and real position must not be collapsed into one index."""
    carrier = _carrier(_spec(6))
    seq = torch.randn(1, 3, 65)
    seq[..., -1] = 1.0
    dense = torch.tensor([[0], [1], [2]]).view(1, 3, 1)
    sparse = torch.tensor([[0], [4], [9]]).view(1, 3, 1)
    with torch.no_grad():
        close, _ = carrier(seq, dense)
        far, _ = carrier(seq, sparse)
    assert (close[:, -1] - far[:, -1]).abs().max() > 1e-4


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"symmetric_relations": True},
        {"mode": "symbol_only"},
        {
            "mode": "dual_content",
            "control_content_head_dim": 12,
            "control_second_head_dim": 8,
        },
        {"max_relative_distance": 40, "symbol_dim": 8},
    ],
)
@pytest.mark.parametrize("sigma_reparam", [True, False])
def test_the_closed_form_attention_parameter_count_equals_the_module(
    overrides: dict[str, object], sigma_reparam: bool
) -> None:
    from reasoned_icrl.experiments.contracts import (
        attention_parameter_count,
        capacity_matched_control_dims,
    )

    settings: dict[str, object] = {
        "layer_indices": (0,),
        "d_model": 32,
        "total_heads": 2,
        "relational_heads": 1,
        "relation_channels": 4,
        "relation_projection_dim": 4,
        "max_relative_distance": 9,
        **overrides,
    }
    spec = DATSpec(**settings)  # type: ignore[arg-type]
    module = DualAttention(spec, sigma_reparam=sigma_reparam)
    counted = sum(parameter.numel() for parameter in module.parameters())
    assert attention_parameter_count(spec, sigma_reparam=sigma_reparam) == counted
    if spec.mode == "dat" and sigma_reparam:
        # The torch-free search reproduces the module-building search.
        matched = dual_content_head_dims(spec, counted)
        assert capacity_matched_control_dims(spec) == (
            matched.control_content_head_dim,
            matched.control_second_head_dim,
        )
