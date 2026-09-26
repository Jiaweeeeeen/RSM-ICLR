"""Configuration, checkpoint, refresh and evaluator integration of the summary carrier.

The seams where a bounded cell has to behave like any other run: it resolves
from the study YAML with a derived DAT clipping distance, binds its carrier
explicitly, keeps its ordinary blocks byte-identical to the full-prefix
reference at step zero, survives a hidden-state checkpoint round trip and
refuses every disagreement, rebuilds its rollout state after a weight change,
writes a different summary for a relabeled trajectory, and records the
summary carrier's write bookkeeping on every event.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gin
import numpy as np
import pytest
import torch
import yaml
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.environments.count_recall import CountRecallQuery
from reasoned_icrl.environments.dark_key_to_door import KeyToDoorAttempt
from reasoned_icrl.experiments.benchmarks import (
    experiment_config,
    load_contract,
    load_study,
)
from reasoned_icrl.experiments.config import dump_config, load_resolved_config
from reasoned_icrl.experiments.contracts import (
    ALL_CONDITIONS,
    DAT_SUMMARY_ARCHITECTURE_ID,
    DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID,
    SUMMARY_ARCHITECTURE_ID,
    WINDOW_ARCHITECTURE_ID,
    ContractError,
    DATSpec,
    SummarySpec,
    WindowSpec,
    attention_parameter_count,
    capacity_matched_control_dims,
)
from reasoned_icrl.experiments.evaluation import (
    HISTORY_MODES,
    SUMMARY_CLEARED,
    AttemptTaskResult,
    CountRecallStreamResult,
    attempt_write_fields,
    query_write_fields,
    task_intervention,
)
from reasoned_icrl.experiments.records import (
    BenchmarkEvent,
    BenchmarkRun,
    ResultValidationError,
    read_benchmark_results,
    validate_benchmark_results,
    write_benchmark_results,
)
from reasoned_icrl.experiments.summary_memory.configs import (
    RETIRED_STUDY,
    load_retired_summary_memory_study,
)
from reasoned_icrl.model.summary_transformer import (
    SUMMARY_HIDDEN_STATE_SCHEMA,
    SummaryHiddenState,
)
from reasoned_icrl.model.trajectory_encoder import (
    HistoryTrajEncoder,
    SummaryTrajEncoder,
    WindowTrajEncoder,
)
from reasoned_icrl.runtime.amago import BOUND_CARRIERS, configure_amago
from reasoned_icrl.runtime.checkpointing import (
    _hidden_state,
    _restore_hidden_state,
    _runtime_contract,
    policy_checkpoint,
    read_policy_checkpoint,
    validate_checkpoint_architecture,
)

ROOT = Path(__file__).resolve().parents[2]
# The bounded carriers exercised here are the legacy raw_* cells, so this
# module reads the retired roster rather than the active study.
STUDY = ROOT / RETIRED_STUDY
BOUNDED = (
    "raw_segment",
    "raw_summary",
    "raw_dat_segment",
    "raw_dat_summary",
    "raw_dat_summary_relational_write_off",
    "raw_dual_content_summary",
)
TOKEN, WIDTH, C, M = 16, 32, 4, 2


def _configure() -> None:
    torch.compiler.set_stance("force_eager")
    torch.set_num_threads(1)
    gin.clear_config()
    target = "amago.nets.traj_encoders.TformerTrajEncoder"
    for name, value in {
        "d_model": WIDTH,
        "n_heads": 2,
        "n_layers": 2,
        "d_ff": 64,
        "attention_type": VanillaAttention,
        "dropout_ff": 0.0,
        "dropout_emb": 0.0,
        "dropout_attn": 0.0,
        "dropout_qkv": 0.0,
    }.items():
        gin.bind_parameter(f"{target}.{name}", value)


def _spec(regime: str = "summary", **kw: Any) -> SummarySpec:
    return SummarySpec(
        segment_length=C, memory_tokens=M, regime=regime, d_model=WIDTH, **kw
    )


def _carrier(
    regime: str = "summary", *, dat: bool = False, token_dim: int = TOKEN
) -> SummaryTrajEncoder:
    _configure()
    torch.manual_seed(0)
    attention = (
        DATSpec(
            layer_indices=(1,),
            d_model=WIDTH,
            total_heads=2,
            relational_heads=1,
            relation_channels=4,
            relation_projection_dim=4,
            max_relative_distance=M + C + M,
        )
        if dat
        else None
    )
    return SummaryTrajEncoder(
        token_dim + 1,
        16,
        spec=_spec(regime),
        dat=attention,
        token_dim=token_dim,
        d_model=WIDTH,
    ).eval()


def _packet(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    seq = torch.randn(batch, length, TOKEN + 1)
    seq[..., -1] = 1.0
    times = torch.arange(length).view(1, length, 1).expand(batch, -1, -1).contiguous()
    return seq, times


def _rollout(
    carrier: SummaryTrajEncoder, seq: torch.Tensor, times: torch.Tensor
) -> tuple[torch.Tensor, SummaryHiddenState]:
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


def _config(condition: str, study_path: Path = STUDY, **overrides: object) -> Any:
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


def _study_copy(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    raw = yaml.safe_load(STUDY.read_text())
    raw["contracts"] = [str(ROOT / "configs" / entry) for entry in raw["contracts"]]
    return raw, tmp_path / "study.yaml"


# --------------------------------------------------------------------------
# Configuration and binding
# --------------------------------------------------------------------------


def test_the_bounded_rows_resolve_with_a_derived_clipping_distance() -> None:
    reference = _config("raw")
    for condition in BOUNDED:
        config = _config(condition)
        summary = config.model.summary
        assert summary is not None
        assert (summary.segment_length, summary.memory_tokens) == (32, 4)
        assert summary.regime == ("segment" if "segment" in condition else "summary")
        assert summary.writer == (
            "relational_off" if condition.endswith("relational_write_off") else "same"
        )
        assert summary.d_model == config.model.width == 256
        assert config.environment == reference.environment
        assert config.training == reference.training
        mode = ALL_CONDITIONS[condition].dat_mode
        if mode is not None:
            assert config.model.architecture_id == (
                DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID
                if mode == "dual_content"
                else DAT_SUMMARY_ARCHITECTURE_ID
            )
            assert config.model.dat is not None
            assert config.model.dat.mode == mode
            assert config.model.dat.max_relative_distance == summary.capacity == 40
            assert config.model.dat.sha256 != _config("raw_dat").model.dat.sha256
        else:
            assert config.model.architecture_id == SUMMARY_ARCHITECTURE_ID
            assert config.model.dat is None
    assert reference.model.summary is None
    assert "summary" not in reference.as_runtime_mapping()["model"]


def test_the_bounded_capacity_control_is_matched_at_the_clipped_distance() -> None:
    """The study block's widths match the full-prefix DAT block; the bounded
    control derives its own at M + C + M, where the symbol table is smaller."""
    from reasoned_icrl.model.dat_transformer import DualAttention

    block = load_retired_summary_memory_study().model["dat"]
    declared = (block["control_content_head_dim"], block["control_second_head_dim"])
    full = _config("raw_dat").model.dat
    assert full is not None
    # The YAML's own widths are the match at the full-prefix distance.
    assert declared == capacity_matched_control_dims(full)
    dat = _config("raw_dat_summary").model.dat
    control = _config("raw_dual_content_summary").model.dat
    assert dat is not None and control is not None
    derived = (control.control_content_head_dim, control.control_second_head_dim)
    assert derived == capacity_matched_control_dims(dat)
    assert derived != declared
    assert control.max_relative_distance == dat.max_relative_distance == 40
    assert control.sha256 != dat.sha256
    counts = {
        name: sum(p.numel() for p in DualAttention(spec).parameters())
        for name, spec in (("dat", dat), ("control", control))
    }
    assert counts["dat"] == attention_parameter_count(dat)
    assert counts["control"] == attention_parameter_count(control)
    # The control's widths come from a grid whose step is one head width, so the
    # attainable match is coarser than the block: with the study's four
    # relational heads the nearest grid point is +2.6 % at the full-prefix
    # distance and +4.8 % at the clipped one. Both over-parameterise the
    # control, which is the conservative direction for the relational claim (a
    # control that still loses had at least as much attention capacity), so the
    # test pins the direction and a bound, not exactness.
    assert counts["control"] >= counts["dat"]
    assert (counts["control"] - counts["dat"]) / counts["dat"] < 0.06
    over = replace(
        dat,
        mode="dual_content",
        control_content_head_dim=56,
        control_second_head_dim=48,
    )
    assert attention_parameter_count(over) / counts["dat"] > 1.3


@pytest.mark.parametrize("condition", BOUNDED)
def test_resolved_bounded_configs_round_trip_and_bind_their_carrier(
    tmp_path: Path, condition: str
) -> None:
    config = _config(condition, output_root=tmp_path)
    path = dump_config(config, tmp_path / condition / "config.yaml")
    recorded = yaml.safe_load(path.read_text())["model"]
    assert recorded["summary"]["sha256"] == config.model.summary.sha256
    if config.model.dat is not None:
        assert recorded["dat"]["max_relative_distance"] == 40
    assert load_resolved_config(path, repository=ROOT) == config
    mapping = config.as_runtime_mapping()
    assert mapping["model"]["summary"]["regime"] == config.model.summary.regime
    mapping["model"]["public_contract"] = {"schema": "test-public-contract.v1"}
    components = configure_amago(mapping["model"], mapping["training"])
    assert components.trajectory_encoder is SummaryTrajEncoder
    target = "reasoned_icrl.model.trajectory_encoder.SummaryTrajEncoder"
    assert gin.query_parameter(f"{target}.spec").sha256 == config.model.summary.sha256
    bound_dat = gin.query_parameter(f"{target}.dat")
    assert (bound_dat is None) == (config.model.dat is None)
    if bound_dat is not None:
        assert bound_dat.max_relative_distance == 40
    assert gin.query_parameter(f"{target}.d_model") == 256
    gin.clear_config()


def test_every_bounded_carrier_is_bound() -> None:
    assert SUMMARY_ARCHITECTURE_ID in BOUND_CARRIERS
    assert DAT_SUMMARY_ARCHITECTURE_ID in BOUND_CARRIERS
    assert DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID in BOUND_CARRIERS
    assert WINDOW_ARCHITECTURE_ID in BOUND_CARRIERS


def test_the_window_row_resolves_binds_and_round_trips(tmp_path: Path) -> None:
    """The window cell carries a `WindowSpec` (no summary, no DAT block) at the
    study's 40 slots, records it with its identity and binds its own carrier."""
    config = _config("raw_window", output_root=tmp_path)
    window = config.model.window
    assert window is not None and window == WindowSpec(segment_length=40)
    assert config.model.summary is None and config.model.dat is None
    assert config.model.architecture_id == WINDOW_ARCHITECTURE_ID
    assert _config("raw").model.window is None
    mapping = config.as_runtime_mapping()
    assert mapping["model"]["window"]["sha256"] == window.sha256
    assert "window" not in _config("raw_summary").as_runtime_mapping()["model"]
    path = dump_config(config, tmp_path / "raw_window" / "config.yaml")
    assert yaml.safe_load(path.read_text())["model"]["window"]["segment_length"] == 40
    assert load_resolved_config(path, repository=ROOT) == config
    mapping["model"]["public_contract"] = {"schema": "test-public-contract.v1"}
    components = configure_amago(mapping["model"], mapping["training"])
    assert components.trajectory_encoder is WindowTrajEncoder
    target = "reasoned_icrl.model.trajectory_encoder.WindowTrajEncoder"
    assert gin.query_parameter(f"{target}.spec") == window
    assert gin.query_parameter(f"{target}.d_model") == 256
    gin.clear_config()


def _contract_with_memory(tmp_path: Path, memory: object) -> Any:
    raw = yaml.safe_load(
        (ROOT / "configs/environments/dark_key_to_door.yaml").read_text()
    )
    raw["memory"] = memory
    path = tmp_path / "dark_key_to_door.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_contract(path)


def test_a_contract_memory_block_overrides_the_study_blocks(tmp_path: Path) -> None:
    """A protocol may set its own segment, memory and window sizes (decision 14:
    Concentration at C=16, M=4, W=24); the study block is the default, the
    contract's values are the ones resolved, recorded and hashed."""
    study = load_retired_summary_memory_study()
    contract = _contract_with_memory(
        tmp_path,
        {
            "summary": {"segment_length": 16, "memory_tokens": 4},
            "window": {"segment_length": 24},
        },
    )
    assert contract.memory == {
        "summary": {"segment_length": 16, "memory_tokens": 4},
        "window": {"segment_length": 24},
    }

    def resolved(condition: str) -> Any:
        return experiment_config(
            contract, study, condition=condition, seed=0, repository=ROOT, device="cpu"
        )

    summary = resolved("raw_summary").model.summary
    assert summary is not None and (summary.segment_length, summary.capacity) == (
        16,
        24,
    )
    assert summary.detach == "none" and summary.position == "segment-local"
    dat = resolved("raw_dat_summary").model.dat
    assert dat is not None and dat.max_relative_distance == 24
    control = resolved("raw_dual_content_summary").model.dat
    assert control is not None and control.max_relative_distance == 24
    window = resolved("raw_window").model.window
    assert window is not None and window.segment_length == 24
    assert resolved("raw").model.summary is None
    default = _config("raw_summary").model.summary
    assert default is not None and default.capacity == 40
    assert summary.sha256 != default.sha256
    config = resolved("raw_summary")
    assert config.as_runtime_mapping()["model"]["summary"]["segment_length"] == 16
    path = dump_config(config, tmp_path / "raw_summary" / "config.yaml")
    assert load_resolved_config(path, repository=ROOT) == config


@pytest.mark.parametrize(
    "memory,message",
    [
        ({"invented": {"x": 1}}, "Unknown contract memory blocks"),
        ({"summary": {}}, "non-empty mapping"),
        ("summary", "mapping of blocks"),
    ],
)
def test_malformed_contract_memory_blocks_are_refused(
    tmp_path: Path, memory: object, message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        _contract_with_memory(tmp_path, memory)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda m: m.pop("window"), r"requires a model\.window"),
        (lambda m: m["window"].update(segment_length=3), "segment_length"),
        (lambda m: m["window"].update(invented=1), "Unknown model.window"),
        (lambda m: m["window"].pop("segment_length"), "requires segment_length"),
    ],
)
def test_window_blocks_outside_the_contract_are_refused(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    raw, path = _study_copy(tmp_path)
    mutation(raw["model"])
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match=message):
        _config("raw_window", path)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda m: m.pop("summary"), r"requires a model\.summary"),
        (lambda m: m["summary"].update(segment_length=3), "segment_length"),
        (
            lambda m: m["summary"].update(regime="segment"),
            "disagrees with the condition",
        ),
        (
            lambda m: m["summary"].update(writer="relational_off"),
            "disagrees with the condition",
        ),
        (lambda m: m["summary"].update(invented=1), "Unknown model.summary"),
        (lambda m: m["summary"].pop("memory_tokens"), "requires memory_tokens"),
    ],
)
def test_summary_blocks_outside_the_contract_are_refused(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    raw, path = _study_copy(tmp_path)
    mutation(raw["model"])
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match=message):
        _config("raw_summary", path)


def test_a_tampered_summary_identity_is_rejected(tmp_path: Path) -> None:
    config = _config("raw_summary", output_root=tmp_path)
    path = dump_config(config, tmp_path / "run" / "config.yaml")
    raw = yaml.safe_load(path.read_text())
    raw["model"]["summary"]["sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="summary identity does not match"):
        load_resolved_config(path, repository=ROOT)


def test_preflight_shows_byte_identical_ordinary_modules_across_regimes() -> None:
    from reasoned_icrl.experiments.summary_memory import experiments as summary_memory

    study = load_retired_summary_memory_study()
    audits = {
        condition: summary_memory.train(
            study,
            benchmark="dark_key_to_door",
            condition=condition,
            seed=0,
            device="cpu",
            preflight=True,
        )
        for condition in (
            "raw",
            "raw_segment",
            "raw_summary",
            "raw_dat_summary",
            "raw_dual_content_summary",
            "raw_window",
        )
    }
    hashes = {
        condition: dict(audit["initial_state_sha256_by_module"])  # type: ignore[index]
        for condition, audit in audits.items()
    }
    blocks = [f"block_{index}" for index in range(3)]
    shared = [
        "timestep",
        "token",
        "actor",
        "critics",
        "input_projection",
        "final_norm",
        *blocks,
    ]
    for condition in ("raw_segment", "raw_summary", "raw_window"):
        for name in shared:
            assert hashes[condition][name] == hashes["raw"][name], (condition, name)
        assert hashes[condition]["trajectory"] != hashes["raw"]["trajectory"]
    # Every block the study selects runs dual attention and so differs from the
    # ordinary backbone's; every unselected block stays byte-identical to it.
    selected = _config("raw_dat_summary").model.dat
    assert selected is not None
    chosen = {f"block_{index}" for index in selected.layer_indices}
    assert chosen, "the study selects at least one dual-attention block"
    control = hashes["raw_dual_content_summary"]
    for name in shared:
        if name in chosen:
            continue
        assert hashes["raw_dat_summary"][name] == hashes["raw"][name], name
        assert control[name] == hashes["raw"][name], name
    for name in chosen:
        index = name.rsplit("_", 1)[1]
        assert hashes["raw_dat_summary"][name] != hashes["raw"][name], name
        assert control[name] != hashes["raw"][name], name
        assert f"dat_attention_{index}" in hashes["raw_dat_summary"]
        assert (
            control[f"dat_attention_{index}"]
            != hashes["raw_dat_summary"][f"dat_attention_{index}"]
        )
    assert audits["raw_summary"]["constructed_cache"]["cache_bytes"] > 0  # type: ignore[index]
    gin.clear_config()


# --------------------------------------------------------------------------
# Hidden-state serializer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dat", (False, True))
def test_the_hidden_state_round_trips_through_a_checkpoint(dat: bool) -> None:
    carrier = _carrier(dat=dat)
    seq, times = _packet(2, C + 2)
    _, hidden = _rollout(carrier, seq, times)
    hidden.summary_cleared = True
    payload = _hidden_state(hidden)
    assert payload["schema"] == SUMMARY_HIDDEN_STATE_SCHEMA
    assert payload["spec_sha256"] == carrier.spec.sha256
    restored = _restore_hidden_state(_experiment(carrier), payload)
    assert isinstance(restored, SummaryHiddenState)
    assert restored.summary_cleared and restored is not hidden
    torch.testing.assert_close(restored.memory, hidden.memory, rtol=0, atol=0)
    assert restored.lengths.tolist() == hidden.lengths.tolist()
    assert restored.segment.tolist() == hidden.segment.tolist()
    for source, target in zip(hidden.layers, restored.layers, strict=True):
        assert source.variant == target.variant
        for name, tensor in source.tensors().items():
            torch.testing.assert_close(
                tensor, target.tensors()[name], rtol=0, atol=0, equal_nan=True
            )
    following, _ = _rollout(carrier, seq[:, :1], times[:, :1])
    del following
    a, _ = carrier(seq[:, :1], times[:, :1], hidden)
    b, _ = carrier(seq[:, :1], times[:, :1], restored)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda p: p.pop("segment"), "Summary hidden state"),
        (lambda p: p.update(spec_sha256="0" * 64), "identity does not match"),
        (lambda p: p.update(capacity=99), "capacity does not match"),
        (lambda p: p.update(batch_size=3), "batch disagrees"),
        (lambda p: p.update(summary_cleared=1), "must be boolean"),
        (
            lambda p: p.update(lengths=torch.full((2,), 99, dtype=torch.int32)),
            "outside the segment capacity",
        ),
        (
            lambda p: p.update(segment=torch.full((2,), -1, dtype=torch.int64)),
            "nonnegative",
        ),
        (lambda p: p["memory"].__setitem__((0, 0, 0), float("nan")), "not finite"),
        (
            lambda p: p["layers"][0]["tensors"]["content_keys"].__setitem__(
                (0, 0, 0, 0), float("nan")
            ),
            "not finite",
        ),
        (lambda p: p["layers"].pop(), "wrong number of layers"),
        (lambda p: p["layers"][0].update(variant="dat"), "variant does not match"),
    ],
)
def test_invalid_hidden_states_are_rejected(mutate: Any, message: str) -> None:
    carrier = _carrier()
    seq, times = _packet(2, 3)
    _, hidden = _rollout(carrier, seq, times)
    payload = _hidden_state(hidden)
    mutate(payload)
    with pytest.raises(ContractError, match=message):
        _restore_hidden_state(_experiment(carrier), payload)


def test_a_summary_state_never_restores_onto_another_carrier() -> None:
    carrier = _carrier()
    seq, times = _packet(1, 2)
    payload = _hidden_state(_rollout(carrier, seq, times)[1])
    with pytest.raises(ContractError, match=r"different carrier|identity does not"):
        _restore_hidden_state(_experiment(_carrier("segment")), payload)
    _configure()
    other = HistoryTrajEncoder(
        TOKEN + 1, 8, bypass=False, token_dim=TOKEN, d_model=WIDTH
    )
    with pytest.raises(ContractError, match="does not match the policy"):
        _restore_hidden_state(_experiment(other), payload)
    gin.clear_config()


def test_the_identities_travel_with_the_weights_and_the_resume_contract() -> None:
    carrier = _carrier(dat=True)
    wrapped = policy_checkpoint(
        carrier.state_dict(),
        condition="raw_dat_summary",
        architecture_id=DAT_SUMMARY_ARCHITECTURE_ID,
    )
    restored = read_policy_checkpoint(
        wrapped,
        condition="raw_dat_summary",
        architecture_id=DAT_SUMMARY_ARCHITECTURE_ID,
    )
    validate_checkpoint_architecture(
        restored, DAT_SUMMARY_ARCHITECTURE_ID, expected_state=carrier.state_dict()
    )
    other = _carrier("segment", dat=True)
    with pytest.raises(ContractError, match="protocol"):
        validate_checkpoint_architecture(
            restored, DAT_SUMMARY_ARCHITECTURE_ID, expected_state=other.state_dict()
        )
    experiment = SimpleNamespace(
        encoder_architecture_id=DAT_SUMMARY_ARCHITECTURE_ID,
        policy_condition="raw_dat_summary",
        policy=SimpleNamespace(
            traj_encoder=carrier,
            tstep_encoder=SimpleNamespace(spec=SimpleNamespace(sha256="packet")),
        ),
        learner_contract="amago-optimizer-ownership.v1",
        reasoned_training_settings={},
    )
    contract = _runtime_contract(experiment)
    assert contract["summary_sha256"] == carrier.spec.sha256
    assert carrier.dat is not None
    assert contract["attention_sha256"] == carrier.dat.sha256
    assert contract["packet_sha256"] == "packet"


# --------------------------------------------------------------------------
# Refresh after a weight change, and relabeled trajectories
# --------------------------------------------------------------------------


def test_rebuild_after_a_weight_change_equals_a_live_rollout_under_new_weights() -> (
    None
):
    carrier = _carrier()
    seq, times = _packet(4, 2 * C + 3)
    lengths = [2 * C + 3, C + 1, 2, 2 * C]  # the last ends exactly at a boundary
    before = carrier.rebuild_hidden_state(seq, times, lengths)
    with torch.no_grad():
        for parameter in carrier.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    after = carrier.rebuild_hidden_state(seq, times, lengths)
    assert not torch.equal(before.memory[0], after.memory[0])
    assert not torch.equal(before.initial_memory, after.initial_memory)
    torch.testing.assert_close(
        after.initial_memory, carrier.backbone.memory_init.detach(), rtol=0, atol=0
    )
    assert after.lengths.tolist() == [M + 3, M + 1, M + 2, M + C]
    assert after.segment.tolist() == [2, 1, 0, 1]
    torch.manual_seed(5)
    following = torch.randn(4, 1, TOKEN + 1)
    following[..., -1] = 1.0
    for row, length in enumerate(lengths):
        live = carrier.init_hidden_state(1, torch.device("cpu"))
        with torch.no_grad():
            for step in range(length):
                carrier(seq[row : row + 1, step : step + 1], times[:1, :1], live)
            expected, _ = carrier(following[row : row + 1], times[:1, :1], live)
            actual, _ = carrier(
                following[row : row + 1],
                times[:1, :1],
                after.select(torch.tensor([row])),
            )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


def test_a_relabeled_mazerunner_trajectory_writes_a_different_summary() -> None:
    from tests.environments.test_mazerunner import (
        TOY_GOALS,
        TOY_PATH,
        TOY_REWARDS,
        relabeler,
        toy_trajectory,
    )

    frozen = toy_trajectory(TOY_PATH, TOY_GOALS, TOY_REWARDS, horizon=8)
    relabeled = relabeler()(frozen)
    rewritten = torch.as_tensor(relabeled.obs["current"], dtype=torch.float32)
    # Hindsight relabeling rewrites the instruction and may cut the trajectory
    # at its last completed goal; compare against the same prefix of the
    # original stream, whose goal fields carry the original instruction.
    original = torch.as_tensor(
        frozen.obs["current"][: rewritten.shape[0]], dtype=torch.float32
    )
    assert original.shape == rewritten.shape and not torch.equal(original, rewritten)
    assert rewritten.shape[0] >= C  # the write rows run on a full segment
    carrier = _carrier(token_dim=int(original.shape[1]))
    valid = torch.ones(1, original.shape[0], dtype=torch.bool)
    with torch.no_grad():
        _, before = carrier.backbone.training_forward(original.unsqueeze(0), valid)
        _, after = carrier.backbone.training_forward(rewritten.unsqueeze(0), valid)
    assert not torch.allclose(before, after)


# --------------------------------------------------------------------------
# Evaluator records: history modes, write counters and evidence ages
# --------------------------------------------------------------------------


def test_environments_declare_only_their_applicable_history_interventions() -> None:
    for name, modes in HISTORY_MODES.items():
        if name == "match_pattern":
            assert modes == ("retained", "current-token")
            assert task_intervention(name) == "current-token"
            continue
        assert modes[0] == "retained" and modes[-1] == SUMMARY_CLEARED
        assert len(modes) == 3 and task_intervention(name) == modes[1]
    assert task_intervention("dark_key_to_door") == "attempt-cleared"
    assert task_intervention("count_recall") == "current-token"


def _attempt(
    index: int, first: int, steps: int, key: int | None, door: int | None = None
) -> KeyToDoorAttempt:
    return KeyToDoorAttempt(
        index=index,
        first_step=first,
        last_step=first + steps - 1,
        steps=steps,
        success=door is not None,
        native_return=float(key is not None) + float(door is not None),
        complete=True,
        key_step=key,
        door_step=door,
    )


def test_attempt_write_fields_count_writes_since_first_and_recent_evidence() -> None:
    # Decisions 1..12; boundaries before decisions 5 and 9 (C = 4).
    writes = tuple(0 if t < 5 else 1 if t < 9 else 2 for t in range(1, 13))
    attempts = (
        _attempt(0, 1, 3, key=2, door=3),  # key at global step 2, door at 3
        _attempt(1, 5, 3, key=1),  # decisions 5..7 (a reset-only step at 4); key at 5
        _attempt(2, 8, 1, key=None),  # decision 8, still in the second segment
        _attempt(3, 9, 4, key=None),  # decisions 9..12, after the second boundary
    )
    result = AttemptTaskResult(7, 0, attempts, None, 3.0, 12, writes)
    assert attempt_write_fields(result, 1) == (0, None, None, None)
    # First evidence: the key at step 2, seen by decision 3 (0 writes). Most
    # recent evidence before attempt 2: the door at step 3 (decision 4, 0
    # writes); before attempts 3 and 4: the key at step 5 (decision 6, 1 write).
    assert attempt_write_fields(result, 2) == (1, 1, 1, False)
    assert attempt_write_fields(result, 3) == (1, 1, 0, True)
    assert attempt_write_fields(result, 4) == (2, 2, 1, False)
    fields = attempt_write_fields(result, 4)
    assert fields.age_recent == 1 and fields.in_current_segment is False
    untracked = AttemptTaskResult(7, 0, attempts, None, 3.0, 12, None)
    assert attempt_write_fields(untracked, 2) == (None, None, None, None)


def test_query_write_fields_count_writes_and_split_the_count_by_segment() -> None:
    values = (1, 0, 1, 1, 0, 1)  # observation 0 is the reset observation
    writes = (0, 0, 0, 0, 1)  # decisions 1..5; one boundary before decision 5
    queries = tuple(
        CountRecallQuery(
            index=i, query=q, true_count=n, answer=n, correct=True, native_reward=0.2
        )
        for i, q, n in ((1, 1, 1), (2, 0, 1), (3, 0, 1), (4, 1, 3), (5, 1, 3))
    )
    result = CountRecallStreamResult(3, 0, queries, 1.0, 5, writes, values)
    # (writes, age, count before the open segment, count inside it)
    assert query_write_fields(result, queries[0]) == (0, 0, 0, 1)  # value 1 at obs 0
    assert query_write_fields(result, queries[1]) == (0, 0, 0, 1)  # value 0 at obs 1
    assert query_write_fields(result, queries[3]) == (0, 0, 0, 3)
    # Decision 5 opens the second segment: all three earlier 1s lie before it.
    assert query_write_fields(result, queries[4]) == (1, 1, 3, 0)
    never = CountRecallQuery(
        index=2, query=1, true_count=0, answer=0, correct=True, native_reward=0.2
    )
    assert query_write_fields(
        CountRecallStreamResult(3, 0, queries, 1.0, 5, writes, (0, 0, 0, 0, 0, 0)),
        never,
    ) == (0, None, 0, 0)
    wrong = CountRecallQuery(
        index=4, query=1, true_count=2, answer=2, correct=True, native_reward=0.2
    )
    with pytest.raises(ResultValidationError, match="true count"):
        query_write_fields(result, wrong)


def _event(**overrides: Any) -> BenchmarkEvent:
    values: dict[str, Any] = {
        "protocol": "native-keydoor-fixed500-first8",
        "benchmark": "dark_key_to_door",
        "condition": "raw_summary",
        "training_seed": 0,
        "checkpoint": "policy_epoch_1",
        "split": "development",
        "history": "retained",
        "task_id": 1_000_000,
        "cluster_id": 1_000_000,
        "rollout_seed": 0,
        "kind": "attempt",
        "event_index": 1,
        "step": 5,
        "numerator": 1,
        "denominator": 1,
        "native_return": 2.0,
    }
    values.update(overrides)
    return BenchmarkEvent(**values)


def test_records_validate_the_write_fields_and_still_read_old_files(
    tmp_path: Path,
) -> None:
    study = load_retired_summary_memory_study()
    contract = study.contract("dark_key_to_door")
    contract = replace(
        contract,
        evaluation=replace(
            contract.evaluation,
            splits={
                name: replace(split, count=1)
                for name, split in contract.evaluation.splits.items()
            },
        ),
    )
    task = contract.roster("development")[0]
    run = BenchmarkRun(
        contract.protocol,
        "dark_key_to_door",
        "raw_summary",
        0,
        "policy_epoch_1",
        "development",
        "retained",
        "completed",
    )
    spans = [(1 + 6 * (i - 1), 6 * i) for i in range(1, 9)]  # six-step attempts
    rows = [
        _event(
            task_id=task,
            cluster_id=task,
            event_index=i,
            step=6,
            writes_before_decision=w,
            evidence_age_writes=a,
            evidence_age_writes_recent=None if a is None else min(a, 1),
            evidence_in_current_segment=None if a is None else min(a, 1) == 0,
            start_step=start,
            end_step=end,
        )
        for (i, (w, a)), (start, end) in zip(
            enumerate(
                [(0, None), (1, None), (2, 1), (3, 0), (4, 3), (5, 4), (6, 5), (7, 6)],
                1,
            ),
            spans,
            strict=True,
        )
    ]
    validate_benchmark_results([contract], [run], rows)
    path = write_benchmark_results(tmp_path / "results.json", [contract], [run], rows)
    _, events = read_benchmark_results(path, [contract])
    assert events[2].writes_before_decision == 2 and events[2].evidence_age_writes == 1
    assert events[3].evidence_age_writes_recent == 0
    assert events[3].evidence_in_current_segment is True
    assert (events[2].start_step, events[2].end_step) == (13, 18)
    for bad, message in (
        ({"writes_before_decision": -1}, "writes_before_decision"),
        ({"evidence_age_writes": 1}, "needs writes_before_decision"),
        ({"writes_before_decision": 1, "evidence_age_writes": 2}, "lies in"),
        (
            {
                "writes_before_decision": 3,
                "evidence_age_writes": 1,
                "evidence_age_writes_recent": 2,
                "evidence_in_current_segment": False,
            },
            "evidence_age_writes_recent needs",
        ),
        (
            {
                "writes_before_decision": 3,
                "evidence_age_writes": 2,
                "evidence_age_writes_recent": 1,
                "evidence_in_current_segment": True,
            },
            "must equal",
        ),
        ({"evidence_in_current_segment": False}, "needs evidence_age_writes_recent"),
        ({"count_before_current_segment": 1}, "both parts"),
        (
            {"count_before_current_segment": 1, "count_in_current_segment": 0},
            "Only query records",
        ),
        ({"start_step": 2, "end_step": 2}, "span must equal"),
        ({"start_step": 0, "end_step": 5}, "1 <= start"),
        ({"end_step": None}, "travel together"),
    ):
        with pytest.raises(ResultValidationError, match=message):
            validate_benchmark_results(
                [contract],
                [run],
                [replace(rows[0], **bad), *rows[1:]],
            )
    # The fixed-final-checkpoint supplement is a second panel under its own
    # rule: accepted beside the selected panel, and a second checkpoint under
    # the same rule is still refused; events must name their run's rule.
    final_run = replace(run, checkpoint="policy_epoch_2", checkpoint_rule="final-epoch")
    final_rows = [
        replace(row, checkpoint="policy_epoch_2", checkpoint_rule="final-epoch")
        for row in rows
    ]
    validate_benchmark_results([contract], [run, final_run], [*rows, *final_rows])
    with pytest.raises(ResultValidationError, match="Select one checkpoint"):
        validate_benchmark_results(
            [contract],
            [run, replace(run, checkpoint="policy_epoch_2")],
            [*rows, *[replace(row, checkpoint="policy_epoch_2") for row in rows]],
        )
    with pytest.raises(ResultValidationError, match="declared evaluation run"):
        validate_benchmark_results(
            [contract],
            [run],
            [replace(rows[0], checkpoint_rule="final-epoch"), *rows[1:]],
        )
    with pytest.raises(ResultValidationError, match="checkpoint rule"):
        validate_benchmark_results(
            [contract],
            [replace(run, checkpoint_rule="latest")],
            rows,  # type: ignore[arg-type]
        )
    # A file written before the fields existed carries none of them.
    import json
    from dataclasses import asdict

    added = (
        "writes_before_decision",
        "evidence_age_writes",
        "evidence_age_writes_recent",
        "evidence_in_current_segment",
        "count_before_current_segment",
        "count_in_current_segment",
        "start_step",
        "end_step",
    )
    legacy = [{k: v for k, v in asdict(row).items() if k not in added} for row in rows]
    (tmp_path / "legacy.json").write_text(
        json.dumps({"runs": [asdict(run)], "events": legacy})
    )
    _, old = read_benchmark_results(tmp_path / "legacy.json", [contract])
    assert all(getattr(e, name) is None for e in old for name in added)
    assert all(e.checkpoint_rule == "selected" for e in old)


def test_the_summary_intervention_is_refused_off_the_summary_regime() -> None:
    from reasoned_icrl.runtime.rollout import rollout

    for carrier in (
        _carrier("segment"),
        HistoryTrajEncoder(TOKEN + 1, 8, bypass=False, token_dim=TOKEN, d_model=WIDTH),
    ):
        experiment = SimpleNamespace(
            policy=SimpleNamespace(traj_encoder=carrier, eval=lambda: None),
            DEVICE=torch.device("cpu"),
            encoder_architecture_id=SUMMARY_ARCHITECTURE_ID,
        )
        environment = SimpleNamespace(unwrapped=_key_to_door(), batched_envs=1)
        with pytest.raises(ContractError, match="memory regime is 'summary'"):
            rollout(
                experiment,
                environment,
                task_ids=(1_000_000,),
                rollout_seed=0,
                history=SUMMARY_CLEARED,
            )
    gin.clear_config()


def _key_to_door() -> Any:
    from reasoned_icrl.environments.dark_key_to_door import DarkKeyToDoorEnv

    return DarkKeyToDoorEnv(split="development", initial_seed=0)


def test_summary_carrier_rows_are_dataclass_fields_with_defaults() -> None:
    result = AttemptTaskResult(1, 0, (), None, 0.0, 0)
    assert result.writes is None
    stream = CountRecallStreamResult(1, 0, (), 0.0, 0)
    assert stream.writes is None and stream.values == ()
    assert np.asarray([]).size == 0
