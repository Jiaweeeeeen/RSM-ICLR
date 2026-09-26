"""The matched conditions: identical packets, different reads, one backbone.

These checks run on every environment. They establish the matched-comparison
invariant at the encoder level: all conditions receive the same tensors and
differ only in what the timestep encoder reads; carriers are causal, gradients
are exact under incremental decoding, and cache resets are per actor. The
opt-in rows of ``DAT_CONDITIONS`` and ``SUMMARY_CONDITIONS`` share their
full-prefix baseline's packet exactly and, where a carrier is bound, satisfy
the same carrier checks.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import gin
import gymnasium as gym
import numpy as np
import pytest
import torch
from amago.envs.amago_env import SequenceWrapper
from amago.envs.env_utils import DummyAsyncVectorEnv
from amago.nets.traj_encoders import TrajEncoder
from amago.nets.transformer import FlashAttention, VanillaAttention

from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import (
    ALL_CONDITIONS,
    CONDITIONS,
    HISTORY_CONDITIONS,
    ConditionSpec,
    DATSpec,
    SummarySpec,
    WindowSpec,
)
from reasoned_icrl.model.step_encoder import TransitionTstepEncoder
from reasoned_icrl.model.trajectory_encoder import (
    DATTrajEncoder,
    GRUHistoryTrajEncoder,
    HistoryTrajEncoder,
    SummaryTrajEncoder,
    WindowTrajEncoder,
)
from reasoned_icrl.runtime.environments import environment_builders
from tests.experiments.fixtures import load_fixture_study

ROOT = Path(__file__).resolve().parents[2]
STAGE1_ENVIRONMENTS = tuple(
    contract.name for contract in load_fixture_study("stage1").contracts
)
"""The frozen Stage-1 roster, independent of subsequently registered tasks.
XLand and match-pattern have their own public-input and carrier parity tests;
the legacy transition/bypass factorial does not define their study rosters."""
OPT_IN_CONDITIONS = tuple(name for name in ALL_CONDITIONS if name not in CONDITIONS)
"""Every row outside the Stage-1 factorial, whether or not its carrier exists."""
CARRIER_CONDITIONS = (
    "raw_dat",
    "raw_dual_content",
    "raw_gru",
    "raw_segment",
    "raw_summary",
    "raw_summary_detach",
    "raw_summary_residual",
    "raw_summary_gated",
    "raw_dat_segment",
    "raw_dat_summary",
    "raw_dat_summary_relational_write_off",
    "raw_dual_content_summary",
    "raw_window",
)
"""Every opt-in row: all of their carriers are bound and checked here."""
SEGMENT_LENGTH, MEMORY_TOKENS = 4, 2
"""A four-record segment: the carrier checks below never cross a boundary, the
summary-transformer tests do."""


def baseline_of(condition: str) -> str:
    """The Stage-1 row an opt-in condition shares its packet with."""
    spec = ALL_CONDITIONS[condition]
    return next(
        name
        for name, base in CONDITIONS.items()
        if (base.evidence, base.bypass) == (spec.evidence, spec.bypass)
    )


def small_dat_spec(spec: ConditionSpec) -> DATSpec:
    """A one-layer dual-attention identity at the 32-wide test backbone.

    A bounded cell clips symbols at its segment capacity, as the study does.
    """
    mode = spec.dat_mode
    assert mode is not None
    control = (
        {"control_content_head_dim": 12, "control_second_head_dim": 8}
        if mode == "dual_content"
        else {}
    )
    distance = (
        small_summary_spec(spec).capacity
        if spec.memory in ("segment", "summary")
        else 9
    )
    return DATSpec(
        layer_indices=(0,),
        mode=mode,
        d_model=32,
        total_heads=2,
        relational_heads=1,
        relation_channels=4,
        relation_projection_dim=4,
        max_relative_distance=distance,
        **control,  # type: ignore[arg-type]
    )


def small_summary_spec(spec: ConditionSpec) -> SummarySpec:
    """The condition's memory regime and writer at the test width."""
    assert spec.memory in ("segment", "summary")
    return SummarySpec(
        segment_length=SEGMENT_LENGTH,
        memory_tokens=MEMORY_TOKENS,
        regime=spec.memory,  # type: ignore[arg-type]
        writer=spec.writer,
        d_model=32,
    )


def build_carrier(condition: str, tstep_dim: int) -> TrajEncoder:
    """The carrier a condition's architecture binds, at the test width."""
    spec = ALL_CONDITIONS[condition]
    if spec.trajectory_encoder == "gru":
        return GRUHistoryTrajEncoder(tstep_dim, 8, d_model=32, n_layers=1)
    if spec.memory in ("segment", "summary"):
        return SummaryTrajEncoder(
            tstep_dim,
            8,
            spec=small_summary_spec(spec),
            dat=small_dat_spec(spec) if spec.dat_mode is not None else None,
            d_model=32,
        )
    if spec.memory == "window":
        return WindowTrajEncoder(
            tstep_dim, 8, spec=WindowSpec(segment_length=SEGMENT_LENGTH), d_model=32
        )
    if spec.dat_mode is not None:
        return DATTrajEncoder(tstep_dim, 8, spec=small_dat_spec(spec), d_model=32)
    return HistoryTrajEncoder(tstep_dim, 8, bypass=spec.bypass, d_model=32)


def config(environment: str = "darkroom", condition: str = "transition_bypass", **kw):
    study = load_fixture_study("stage1")
    return experiment_config(
        study.contract(environment),
        study,
        condition=condition,
        seed=0,
        repository=ROOT,
        **kw,
    )


def adapter(environment: str = "darkroom", *, horizon: int | None = 2) -> Any:
    cfg = config(environment)
    settings = {"parallel_envs": 1}
    if horizon is not None and not cfg.environment.has_fixed_native_horizon:
        settings["horizon"] = horizon
        if cfg.environment.meta_horizon is not None:
            settings["meta_horizon"] = cfg.environment.attempts * (horizon + 1)
    cfg = replace(cfg, environment=replace(cfg.environment, **settings))
    builders, _ = environment_builders(cfg.as_runtime_mapping())
    return builders[0]()


@pytest.mark.parametrize("environment", STAGE1_ENVIRONMENTS)
def test_public_packets_are_exact_and_restorable(environment: str) -> None:
    amago = adapter(environment)
    env = amago.env
    packet, _ = env.reset(seed=7)
    assert tuple(packet["event"]) == (0, 1, 0)
    assert not packet["previous"].any()
    previous = packet["current"].copy()
    for _ in range(10):
        saved = env.state_dict()
        packet, _reward, terminated, truncated, _ = env.step(0)
        env.load_state_dict(saved)
        repeated = env.step(0)
        for key, value in packet.items():
            np.testing.assert_array_equal(value, repeated[0][key])
        available = packet["event"][0]
        if available:
            np.testing.assert_array_equal(packet["previous"], previous)
        previous = packet["current"].copy()
        if terminated or truncated:
            break
    packet, _ = env.reset()
    assert tuple(packet["event"]) == (0, 1, 0)
    assert not packet["previous"].any() and not packet["outcome"].any()
    amago.close()


def encoder_and_inputs(condition: str, environment: str = "darkroom", device="cpu"):
    torch.compiler.set_stance("force_eager")
    torch.set_num_threads(1)
    gin.clear_config()
    amago = adapter(environment, horizon=None)
    env = amago.env
    rows = [env.reset(seed=2)[0]]
    for action in (0, 1, 2):
        rows.append(env.step(action)[0])
    observations = {
        k: torch.tensor(np.stack([r[k] for r in rows]), device=device).unsqueeze(0)
        for k in rows[0]
    }
    actions = env.action_count
    rl2 = torch.zeros(1, 4, actions + 1, device=device)
    rl2[:, 1:, 1] = 1
    spec = ALL_CONDITIONS[condition]
    assert spec.evidence is not None
    step = TransitionTstepEncoder(
        env.observation_space,
        gym.spaces.Box(-np.inf, np.inf, (actions + 1,), np.float32),
        evidence=spec.evidence,
        bypass=spec.bypass,
        public_contract=env.contract,
    ).to(device)
    # AMAGO trunks carry InputNorm running statistics, which are learner state.
    # Evaluation mode freezes them so per-timestep behavior is a pure function.
    step.eval()
    target = "amago.nets.traj_encoders.TformerTrajEncoder"
    gin.bind_parameter(f"{target}.d_model", 32)
    gin.bind_parameter(f"{target}.n_heads", 2)
    gin.bind_parameter(f"{target}.n_layers", 1)
    gin.bind_parameter(f"{target}.d_ff", 64)
    for name in ("dropout_ff", "dropout_emb", "dropout_attn", "dropout_qkv"):
        gin.bind_parameter(f"{target}.{name}", 0.0)
    gin.bind_parameter(
        f"{target}.attention_type",
        FlashAttention if device == "cuda" else VanillaAttention,
    )
    carrier = build_carrier(condition, step.emb_dim).to(device).eval()
    amago.close()
    return step, carrier, observations, rl2


@pytest.mark.parametrize("environment", STAGE1_ENVIRONMENTS)
@pytest.mark.parametrize("condition", HISTORY_CONDITIONS)
def test_packet_invariance_statelessness_and_information(
    condition: str, environment: str
) -> None:
    step, _, obs, rl2 = encoder_and_inputs(condition, environment)
    clean = step(obs, rl2)
    altered = {k: v.clone() for k, v in obs.items()}
    altered["previous"][:, 0] = torch.nan
    altered["outcome"][:, 0] = torch.inf
    torch.testing.assert_close(clean, step(altered, rl2), rtol=0, atol=0)
    incremental = torch.cat(
        [
            step({k: v[:, i : i + 1] for k, v in obs.items()}, rl2[:, i : i + 1])
            for i in range(4)
        ],
        1,
    )
    torch.testing.assert_close(clean, incremental, rtol=1e-5, atol=1e-6)
    assert clean.shape == (1, 4, step.emb_dim)
    for value in altered.values():
        value[:, -1] = 4
    padded = step(altered, rl2)
    assert not padded[:, -1].any()
    torch.testing.assert_close(padded[:, :-1], clean[:, :-1])


@pytest.mark.parametrize("condition", ("raw", "raw_bypass"))
def test_raw_tokens_ignore_endpoint_and_event_payloads(condition: str) -> None:
    step, _, obs, rl2 = encoder_and_inputs(condition)
    expected = step(obs, rl2)
    altered = {key: value.clone() for key, value in obs.items()}
    altered["previous"] += 17
    altered["outcome"] -= 23
    altered["event"] = 1 - altered["event"]
    torch.testing.assert_close(step(altered, rl2), expected, rtol=0, atol=0)


@pytest.mark.parametrize("environment", STAGE1_ENVIRONMENTS)
def test_factorial_conditions_share_public_inputs_backbone_and_budgets(
    environment: str,
) -> None:
    bundles = {
        condition: encoder_and_inputs(condition, environment)
        for condition in HISTORY_CONDITIONS
    }
    reference_config = config(environment, HISTORY_CONDITIONS[0])
    reference_step, reference_carrier, reference_obs, reference_rl2 = bundles[
        HISTORY_CONDITIONS[0]
    ]
    reference_backbone = {
        name: parameter.shape
        for name, parameter in reference_carrier.backbone.named_parameters()
    }
    for condition, (step, carrier, observations, rl2) in bundles.items():
        candidate = config(environment, condition)
        assert candidate.environment == reference_config.environment
        assert candidate.model == reference_config.model
        assert candidate.training == reference_config.training
        assert step.spec.public_contract == reference_step.spec.public_contract
        assert step.spec.observation_dim == reference_step.spec.observation_dim
        assert step.spec.action_dim == reference_step.spec.action_dim
        assert step.spec.token_dim == 64
        assert step.spec.state_dim == (64 if CONDITIONS[condition].bypass else 0)
        assert step(observations, rl2).shape == (1, 4, step.emb_dim)
        assert {
            name: parameter.shape
            for name, parameter in carrier.backbone.named_parameters()
        } == reference_backbone
        for key in reference_obs:
            torch.testing.assert_close(
                observations[key], reference_obs[key], rtol=0, atol=0
            )
        torch.testing.assert_close(rl2, reference_rl2, rtol=0, atol=0)
    for raw_name, transition_name in (
        ("raw", "transition"),
        ("raw_bypass", "transition_bypass"),
    ):
        raw_step, raw_carrier, _, _ = bundles[raw_name]
        transition_step, transition_carrier, _, _ = bundles[transition_name]
        assert raw_step.spec.bypass == transition_step.spec.bypass
        assert type(raw_carrier.fusion) is type(transition_carrier.fusion)


@pytest.mark.parametrize("condition", ("transition", "transition_bypass"))
def test_transition_tokens_use_available_endpoints_but_ignore_masked_payload(
    condition: str,
) -> None:
    step, _, obs, rl2 = encoder_and_inputs(condition)
    expected = step(obs, rl2)
    altered = {key: value.clone() for key, value in obs.items()}
    altered["outcome"][:, 1] += 1
    assert not torch.equal(step(altered, rl2)[:, 1], expected[:, 1])
    masked = {key: value.clone() for key, value in obs.items()}
    masked["previous"][:, 0] = torch.nan
    masked["outcome"][:, 0] = torch.inf
    torch.testing.assert_close(step(masked, rl2), expected, rtol=0, atol=0)


@pytest.mark.parametrize("condition", HISTORY_CONDITIONS)
def test_carrier_causality_incremental_gradients_and_resets(condition: str) -> None:
    torch.manual_seed(19)
    step, carrier, obs, rl2 = encoder_and_inputs(condition)
    times = torch.arange(4).reshape(1, 4, 1)
    seq = step(obs, rl2)
    full, _ = carrier(seq, times)
    hidden = carrier.init_hidden_state(1, torch.device("cpu"))
    online = []
    for index in range(4):
        value, hidden = carrier(
            seq[:, index : index + 1], times[:, index : index + 1], hidden
        )
        online.append(value)
    torch.testing.assert_close(full, torch.cat(online, 1), rtol=1e-5, atol=1e-6)
    changed = seq.detach().clone()
    changed[:, -1, :-1] += 20
    torch.testing.assert_close(
        full[:, :-1], carrier(changed, times)[0][:, :-1], rtol=0, atol=0
    )
    other_step, other_carrier = copy.deepcopy(step), copy.deepcopy(carrier)
    weights = torch.randn_like(full)
    (full * weights).mean().backward()
    other_seq = other_step(obs, rl2)
    prefixes = torch.cat(
        [
            other_carrier(other_seq[:, :end], times[:, :end])[0][:, -1:]
            for end in range(1, 5)
        ],
        1,
    )
    (prefixes * weights).mean().backward()
    for module, other in ((step, other_step), (carrier, other_carrier)):
        for (name, p), (_, q) in zip(
            module.named_parameters(), other.named_parameters(), strict=True
        ):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)
    state = carrier.init_hidden_state(2, torch.device("cpu"))
    _, state = carrier(
        seq[:, :1].expand(2, -1, -1), times[:, :1].expand(2, -1, -1), state
    )
    kept = state.key_cache.data[:, 1].clone()
    carrier.reset_hidden_state(state, np.array([True, False]))
    assert state.seq_lens.tolist() == [0, 1]
    torch.testing.assert_close(state.key_cache.data[:, 1], kept, equal_nan=True)
    assert torch.isnan(state.key_cache.data[:, 0]).all()


@pytest.mark.parametrize("environment", STAGE1_ENVIRONMENTS)
@pytest.mark.parametrize("condition", OPT_IN_CONDITIONS)
def test_opt_in_rows_share_their_baseline_packet_exactly(
    condition: str, environment: str
) -> None:
    """An attention or memory variant changes nothing before the carrier."""
    spec = ALL_CONDITIONS[condition]
    assert spec.uses_history_packet and not spec.bypass
    step, _, obs, rl2 = encoder_and_inputs(condition, environment)
    baseline_step, _, baseline_obs, baseline_rl2 = encoder_and_inputs(
        baseline_of(condition), environment
    )
    assert step.spec == baseline_step.spec
    assert step.spec.sha256 == baseline_step.spec.sha256
    for key in obs:
        torch.testing.assert_close(obs[key], baseline_obs[key], rtol=0, atol=0)
    torch.testing.assert_close(rl2, baseline_rl2, rtol=0, atol=0)
    torch.testing.assert_close(
        step(obs, rl2), baseline_step(baseline_obs, baseline_rl2), rtol=0, atol=0
    )
    altered = {key: value.clone() for key, value in obs.items()}
    altered["previous"][:, 0] = torch.nan
    altered["outcome"][:, 0] = torch.inf
    torch.testing.assert_close(step(obs, rl2), step(altered, rl2), rtol=0, atol=0)


@pytest.mark.parametrize("condition", CARRIER_CONDITIONS)
def test_opt_in_carriers_are_causal_incremental_and_reset_per_row(
    condition: str,
) -> None:
    """The carrier checks above, stated without reference to a cache layout."""
    torch.manual_seed(23)
    step, carrier, obs, rl2 = encoder_and_inputs(condition)
    times = torch.arange(4).reshape(1, 4, 1)
    seq = step(obs, rl2)
    full, _ = carrier(seq, times)
    assert full.shape == (1, 4, carrier.emb_dim)
    hidden = carrier.init_hidden_state(1, torch.device("cpu"))
    online = []
    for index in range(4):
        value, hidden = carrier(
            seq[:, index : index + 1], times[:, index : index + 1], hidden
        )
        online.append(value)
    torch.testing.assert_close(full, torch.cat(online, 1), rtol=1e-5, atol=2e-6)
    changed = seq.detach().clone()
    changed[:, -1, :-1] += 20
    torch.testing.assert_close(
        full[:, :-1], carrier(changed, times)[0][:, :-1], rtol=0, atol=0
    )
    other_step, other_carrier = copy.deepcopy(step), copy.deepcopy(carrier)
    weights = torch.randn_like(full)
    (full * weights).mean().backward()
    other_seq = other_step(obs, rl2)
    prefixes = torch.cat(
        [
            other_carrier(other_seq[:, :end], times[:, :end])[0][:, -1:]
            for end in range(1, 5)
        ],
        1,
    )
    (prefixes * weights).mean().backward()
    for module, other in ((step, other_step), (carrier, other_carrier)):
        for (name, p), (_, q) in zip(
            module.named_parameters(), other.named_parameters(), strict=True
        ):
            if p.grad is None:
                # Copy networks and unused branches carry no gradient anywhere.
                assert q.grad is None, name
                continue
            assert torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)
    # Resetting one actor's row restarts that row alone: after the reset the
    # cleared row behaves like a fresh state and the other row continues.
    with torch.no_grad():
        pair = seq[:, :2].expand(2, -1, -1).contiguous()
        pair_times = times[:, :2].expand(2, -1, -1).contiguous()
        state = carrier.init_hidden_state(2, torch.device("cpu"))
        _, state = carrier(pair[:, :1], pair_times[:, :1], state)
        state = carrier.reset_hidden_state(state, np.array([True, False]))
        after, _ = carrier(pair[:, 1:2], pair_times[:, 1:2], state)
        fresh = carrier.init_hidden_state(1, torch.device("cpu"))
        fresh_out, _ = carrier(seq[:, 1:2], times[:, 1:2], fresh)
        continued = online[1]
    torch.testing.assert_close(after[0:1], fresh_out, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(after[1:2], continued, rtol=1e-5, atol=2e-6)
    assert not torch.allclose(after[0:1], after[1:2])


def test_vector_actor_resets_do_not_clear_other_tasks() -> None:
    env = DummyAsyncVectorEnv(
        [
            lambda: SequenceWrapper(adapter(horizon=1), save_trajs_to=None),
            lambda: SequenceWrapper(adapter(horizon=2), save_trajs_to=None),
        ]
    )
    env.reset()
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(np.zeros((2, 1), dtype=np.int64))
    assert (terminated | truncated).reshape(-1).tolist() == [True, False]
    first, second = (row.current_timestep for row in env.envs)
    assert first[0]["event"][0, 0] == 0 and not first[1].any()
    assert second[0]["event"][0, 0] == 1 and second[2][0, 0] == 5
    env.close()


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_runtime")
@pytest.mark.parametrize("condition", HISTORY_CONDITIONS)
def test_history_cuda_bf16_parity_backward_and_cache(condition: str) -> None:
    torch.manual_seed(31)
    step, carrier, obs, rl2 = encoder_and_inputs(condition, device="cuda")
    times = torch.arange(4, device="cuda").reshape(1, 4, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        seq = step(obs, rl2)
        full, _ = carrier(seq, times)
        hidden = carrier.init_hidden_state(1, torch.device("cuda"))
        online = []
        for index in range(4):
            value, hidden = carrier(
                seq[:, index : index + 1], times[:, index : index + 1], hidden
            )
            online.append(value)
        torch.testing.assert_close(full, torch.cat(online, 1), rtol=0.03, atol=0.01)
        loss = full.float().square().mean()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for module in (step, carrier)
        for p in module.parameters()
    )


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_runtime")
def test_history_rollout_autocast_honors_bf16() -> None:
    from types import SimpleNamespace

    from reasoned_icrl.runtime.experiment import ReasonedExperiment

    settings = SimpleNamespace(
        policy_condition="transition_bypass",
        encoder_architecture_id="amago-history-v1",
        mixed_precision="bf16",
    )
    with ReasonedExperiment.caster(settings):
        assert torch.get_autocast_dtype("cuda") == torch.bfloat16
        value = torch.ones(2, 2, device="cuda")
        assert (value @ value).dtype == torch.bfloat16
