"""Configuration, roster, checkpoint and runtime-contract seams of the Memo comparator.

The comparator has to behave like any other cell at every seam: its condition
resolves to its own identity and refuses the variants it has none of, its
study loads with the three frozen reference cells and plans exactly three
fits, its configuration round-trips through the resolved YAML, its carrier is
bound explicitly, its hidden state survives a checkpoint round trip and
refuses every disagreement, and its identity travels with the weights and
the resume contract. The carrier's semantics are in
``tests/model/test_memo_transformer.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gin
import pytest
import torch
import yaml
from amago.nets.transformer import VanillaAttention

from reasoned_icrl.environments.base import (
    CONFIRMATION_TASKS,
    DEVELOPMENT_TASKS,
    FINAL_OOD_TASKS,
    FINAL_TASKS,
    REVISED_FINAL_TASKS,
    TRAINING_TASKS,
    benchmark_task_sources,
)
from reasoned_icrl.experiments.benchmarks import (
    MEMORY_BLOCKS,
    TierContrast,
    TierPlan,
    experiment_config,
)
from reasoned_icrl.experiments.config import (
    architecture_id,
    dump_config,
    load_resolved_config,
    model_config,
)
from reasoned_icrl.experiments.contracts import (
    ALL_CONDITIONS,
    MEMO_ARCHITECTURE_ID,
    MEMO_CONDITIONS,
    ConditionSpec,
    ContractError,
    MemoSpec,
    architecture_label,
    architecture_uses_dat,
    architecture_uses_history_packet,
    architecture_uses_memo,
    architecture_uses_summary,
    condition_label,
)
from reasoned_icrl.experiments.summary_memory.configs import (
    MEMO_STUDY,
    MEMO_STUDY_NAME,
    load_memo_study,
    load_summary_memory_study,
    reference_condition,
)
from reasoned_icrl.experiments.summary_memory.jobs import plan_fits, write_jobs_file
from reasoned_icrl.experiments.summary_memory.revised import development_modes
from reasoned_icrl.model.memo_transformer import (
    MEMO_HIDDEN_STATE_SCHEMA,
    MemoHiddenState,
)
from reasoned_icrl.model.trajectory_encoder import (
    HistoryTrajEncoder,
    MemoTrajEncoder,
    SummaryTrajEncoder,
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
from reasoned_icrl.runtime.experiment import carrier_cost, state_tensor_bytes

ROOT = Path(__file__).resolve().parents[2]
TOKEN, WIDTH = 16, 32
L, S = 4, 2


# --------------------------------------------------------------------------
# The condition and its identity
# --------------------------------------------------------------------------


def test_the_memo_condition_resolves_to_its_own_identity() -> None:
    assert set(MEMO_CONDITIONS) == {"memo", "memo_fixed"}
    spec = ALL_CONDITIONS["memo"]
    assert spec == ConditionSpec("transformer", "raw", False, "ordinary", "accumulated")
    assert spec.segmentation == "jittered"
    fixed = ALL_CONDITIONS["memo_fixed"]
    assert fixed == ConditionSpec(
        "transformer", "raw", False, "ordinary", "accumulated", segmentation="fixed"
    )
    assert architecture_id("memo_fixed") == MEMO_ARCHITECTURE_ID
    with pytest.raises(ContractError, match="jitter is a setting of the accumulated"):
        ConditionSpec(
            "transformer", "raw", False, memory="summary", segmentation="fixed"
        )
    assert spec.bounded and spec.uses_history_packet and spec.dat_mode is None
    assert architecture_id("memo") == MEMO_ARCHITECTURE_ID == "amago-memo-v1"
    assert architecture_uses_memo(MEMO_ARCHITECTURE_ID)
    assert architecture_uses_history_packet(MEMO_ARCHITECTURE_ID)
    assert not architecture_uses_summary(MEMO_ARCHITECTURE_ID)
    assert not architecture_uses_dat(MEMO_ARCHITECTURE_ID)
    assert not any(
        architecture_uses_memo(other)
        for other in BOUND_CARRIERS - {MEMO_ARCHITECTURE_ID}
    )
    assert MEMO_ARCHITECTURE_ID in BOUND_CARRIERS
    assert "Memo" in condition_label("memo") and "Memo" in architecture_label(
        MEMO_ARCHITECTURE_ID
    )
    for name, kw in (
        ("dual attention", {"attention": "dat"}),
        ("writer", {"attention": "dat", "writer": "relational_off"}),
        ("gru", {"trajectory_encoder": "gru"}),
    ):
        with pytest.raises(ContractError):
            ConditionSpec(
                kw.pop("trajectory_encoder", "transformer"),
                "raw",
                False,
                memory="accumulated",
                **kw,  # type: ignore[arg-type]
            )
        del name
    # The figure plan's styles: purple for both recipes, solid for the
    # jittered one and dash-dot for fixed segments.
    from reasoned_icrl.analysis.plotting import condition_style

    style = condition_style("memo")
    assert (style.family, style.regime, style.color, style.linestyle) == (
        "memo",
        "accumulated",
        "#CC79A7",
        "-",
    )
    fixed_style = condition_style("memo_fixed")
    assert (fixed_style.color, fixed_style.marker, fixed_style.linestyle) == (
        style.color,
        style.marker,
        "-.",
    )
    assert condition_style("fixed_summary").family != "memo"


# --------------------------------------------------------------------------
# The study, its contract split and the planner
# --------------------------------------------------------------------------


def test_the_confirmation_band_is_disjoint_from_every_other_band() -> None:
    bands = [
        TRAINING_TASKS,
        DEVELOPMENT_TASKS,
        FINAL_TASKS,
        FINAL_OOD_TASKS,
        REVISED_FINAL_TASKS,
        CONFIRMATION_TASKS,
    ]
    assert range(5_000_000, 5_000_256) == CONFIRMATION_TASKS
    assert benchmark_task_sources("confirmation") == CONFIRMATION_TASKS
    for index, band in enumerate(bands):
        for other in bands[index + 1 :]:
            assert band.stop <= other.start or other.stop <= band.start, (band, other)


def test_the_comparator_study_loads_plans_three_fits_and_refuses_reference_fits(
    tmp_path: Path,
) -> None:
    study = load_memo_study()
    assert study.name == MEMO_STUDY_NAME
    assert study.conditions == (
        "memo",
        "memo_fixed",
        "fixed_summary",
        "full_context",
        "full_gru",
        "fixed_window",
        "fixed_segment",
        "full_dual_relational",
        "raw_summary",
        "raw_segment",
        "raw_summary_residual",
    )
    assert study.output_root == "outputs/memo-key-to-door-8m"
    tier = study.tier("dark_key_to_door")
    assert tier.primary == ("memo", "memo_fixed") and tier.supplementary == ()
    assert tier.reference_cells == (
        "fixed_summary",
        "full_context",
        "full_gru",
        "fixed_window",
        "fixed_segment",
        "full_dual_relational",
        "raw_summary",
        "raw_segment",
        "raw_summary_residual",
    )
    assert tier.reference_root == "outputs/summary-memory-8m"
    assert tier.cells == ("memo", "memo_fixed")
    assert tier.compared_cells == study.conditions
    assert tier.qualification_reference == "full_context"
    assert reference_condition(study, "dark_key_to_door") == "full_context"
    assert tier.practical_effect == 1.0
    # The ME4 rows are companions (EXPERIMENTS section 7), declared before
    # any Memo panel was read.
    assert [c.name for c in tier.primary_contrasts] == ["summary - Memo"]
    assert [(c.left, c.right) for c in tier.contrasts] == [
        ("fixed_summary", "memo"),
        ("full_context", "memo"),
        ("full_gru", "memo"),
        ("fixed_summary", "memo_fixed"),
        ("memo", "memo_fixed"),
        ("fixed_summary", "fixed_segment"),
        # The bounded-summary paper's family on RSM and on the
        # residual rewrite, declared before their panels existed.
        ("raw_summary", "raw_segment"),
        ("full_context", "raw_summary"),
        ("raw_summary", "memo"),
        ("raw_summary", "memo_fixed"),
        ("raw_summary", "full_gru"),
        ("raw_summary_residual", "raw_summary"),
        ("raw_summary_residual", "raw_segment"),
        ("full_context", "raw_summary_residual"),
        ("raw_summary_residual", "memo"),
        ("raw_summary_residual", "memo_fixed"),
        ("raw_summary_residual", "full_gru"),
    ]
    assert tier.group_of("memo") == "primary"
    assert tier.group_of("memo_fixed") == "primary"
    with pytest.raises(ContractError, match="frozen reference cell"):
        tier.group_of("fixed_summary")
    contract = study.contract("dark_key_to_door")
    assert contract.roster("confirmation") == tuple(CONFIRMATION_TASKS)
    assert contract.roster("development") == tuple(DEVELOPMENT_TASKS)
    assert set(contract.roster("final")) == set(REVISED_FINAL_TASKS)
    # Exactly the six Memo fits at the study seeds, condition-major.
    plans = plan_fits(study, contract, repository=ROOT, output_root=tmp_path)
    assert [(p.condition, p.seed, p.group, p.status) for p in plans] == [
        (condition, seed, "primary", "missing")
        for condition in ("memo", "memo_fixed")
        for seed in (42, 100, 2026)
    ]
    assert study.tier("dark_key_to_door").fits(study.training_seeds) == 6
    jobs = write_jobs_file(tmp_path / "jobs.txt", plans)
    assert jobs.read_text().count("dark_key_to_door memo ") == 3
    assert jobs.read_text().count("dark_key_to_door memo_fixed ") == 3
    # Either cell alone plans through the generic planner.
    only_fixed = plan_fits(
        study,
        contract,
        repository=ROOT,
        output_root=tmp_path,
        conditions=["memo_fixed"],
    )
    assert [(p.condition, p.seed) for p in only_fixed] == [
        ("memo_fixed", seed) for seed in (42, 100, 2026)
    ]
    with pytest.raises(ContractError, match="frozen reference cell"):
        experiment_config(
            contract, study, condition="fixed_summary", seed=42, repository=ROOT
        )
    with pytest.raises(ContractError, match="not primary cells"):
        plan_fits(study, contract, repository=ROOT, conditions=["full_gru"])
    # The 8M study is untouched: no Memo cell, no reference cells, and its
    # tier-0 contract carries the confirmation split beside its own.
    eight = load_summary_memory_study()
    assert "memo" not in eight.conditions
    assert all(plan.reference_cells == () for plan in eight.tiers.values())
    assert sorted(eight.contract("dark_key_to_door").evaluation.splits) == [
        "confirmation",
        "development",
        "final",
    ]
    assert "memo" in MEMORY_BLOCKS


def test_reference_cells_are_validated_with_their_root_and_kept_out_of_fits() -> None:
    def plan(**kw: Any) -> TierPlan:
        settings: dict[str, Any] = {
            "environment": "dark_key_to_door",
            "tier": 0,
            "purpose": "test",
            "primary": ("memo",),
            "supplementary": (),
            "qualification_reference": "full_context",
            "practical_effect": 1.0,
            "primary_contrasts": (
                TierContrast("summary - Memo", "fixed_summary", "memo"),
            ),
            "reference_cells": ("fixed_summary", "full_context"),
            "reference_root": "outputs/elsewhere",
        }
        settings.update(kw)
        return TierPlan(**settings)

    assert plan().compared_cells == ("memo", "fixed_summary", "full_context")
    with pytest.raises(ContractError, match="declared together"):
        plan(reference_root=None)
    with pytest.raises(ContractError, match="declared together"):
        plan(reference_cells=())
    with pytest.raises(ContractError, match="lists a cell twice"):
        plan(reference_cells=("memo", "fixed_summary"), qualification_reference="memo")
    with pytest.raises(ContractError, match="must be a primary cell"):
        plan(qualification_reference="full_gru")
    with pytest.raises(ContractError, match="outside its groups"):
        plan(primary_contrasts=(TierContrast("x", "full_gru", "memo"),))
    with pytest.raises(
        ContractError, match="Frozen reference cells belong to the Memo"
    ):
        load_study_with_references_under_the_8m_name()


def load_study_with_references_under_the_8m_name() -> None:
    raw = yaml.safe_load((ROOT / MEMO_STUDY).read_text())
    raw["name"] = "summary_memory_8m"
    raw["contracts"] = [str(ROOT / "configs" / entry) for entry in raw["contracts"]]
    path = ROOT / "outputs" / ".test-memo-study.yaml"
    path.parent.mkdir(exist_ok=True)
    try:
        path.write_text(yaml.safe_dump(raw))
        load_summary_memory_study(path)
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Configuration resolution and the resolved round trip
# --------------------------------------------------------------------------


def _model_block(**memo: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "attention_backend": "vanilla",
        "width": 256,
        "layers": 3,
        "heads": 8,
        "feedforward_multiplier": 4,
    }
    if memo is not None:
        block["memo"] = {"segment_length": 32, "summary_tokens": 4, **memo}
    return block


def test_the_memo_block_resolves_and_every_disagreement_is_refused() -> None:
    resolved = model_config(_model_block(), condition="memo")
    assert resolved.architecture_id == MEMO_ARCHITECTURE_ID
    assert resolved.memo == MemoSpec(32, 4, d_model=256)
    assert resolved.summary is None and resolved.dat is None and resolved.window is None
    assert resolved.memo.capacity(500) == 96 and resolved.memo.jitter_range == (26, 38)
    # A full-context cell ignores the block; the Memo cell requires it.
    assert model_config(_model_block(), condition="full_context").memo is None
    block = _model_block()
    del block["memo"]
    with pytest.raises(ContractError, match=r"requires a model\.memo"):
        model_config(block, condition="memo")
    with pytest.raises(ContractError, match=r"Unknown model\.memo settings"):
        model_config(_model_block(memory_tokens=4), condition="memo")
    with pytest.raises(ContractError, match="requires summary_tokens"):
        model_config(
            {**_model_block(), "memo": {"segment_length": 32}}, condition="memo"
        )
    with pytest.raises(ContractError, match="d_model disagrees"):
        model_config(_model_block(d_model=128), condition="memo")
    with pytest.raises(ContractError, match="Recorded Memo identity"):
        model_config(_model_block(sha256="0" * 64), condition="memo")
    recorded = resolved.memo.to_dict()
    assert (
        model_config({**_model_block(), "memo": recorded}, condition="memo").memo
        == resolved.memo
    )
    # The fixed-segment cell derives jitter 0 from its condition row, hashes
    # differently and round-trips its own record; a jittered cell with jitter
    # 0 is refused as the fixed cell under the wrong name.
    fixed = model_config(_model_block(), condition="memo_fixed")
    assert fixed.architecture_id == MEMO_ARCHITECTURE_ID
    assert fixed.memo == MemoSpec(32, 4, training_segment_jitter=0.0, d_model=256)
    assert fixed.memo.sha256 != resolved.memo.sha256
    assert fixed.memo.jitter_range == (32, 32)
    assert (
        model_config(
            {**_model_block(), "memo": fixed.memo.to_dict()}, condition="memo_fixed"
        ).memo
        == fixed.memo
    )
    # A fixed-segment record under the jittered name is refused before its
    # hash is even compared: jitter 0 is the other cell.
    with pytest.raises(ContractError, match="fixed-segment cell, which has its own"):
        model_config({**_model_block(), "memo": fixed.memo.to_dict()}, condition="memo")
    with pytest.raises(ContractError, match="fixed-segment cell, which has its own"):
        model_config(_model_block(training_segment_jitter=0.0), condition="memo")
    # And a jittered record under the fixed name fails its recorded hash.
    with pytest.raises(ContractError, match="Recorded Memo identity"):
        model_config({**_model_block(), "memo": recorded}, condition="memo_fixed")


def test_the_resolved_configuration_round_trips_and_the_smoke_profile_shrinks_it(
    tmp_path: Path,
) -> None:
    study = load_memo_study()
    contract = study.contract("dark_key_to_door")
    config = experiment_config(
        contract,
        study,
        condition="memo",
        seed=42,
        repository=ROOT,
        output_root=tmp_path,
    )
    assert config.model.memo is not None and config.model.memo.d_model == 256
    assert config.run_directory == tmp_path / contract.protocol / "memo" / "seed-42"
    mapping = config.as_runtime_mapping()["model"]
    assert mapping["memo"]["sha256"] == config.model.memo.sha256
    assert "summary" not in mapping and "dat" not in mapping and "window" not in mapping
    path = dump_config(config, tmp_path / "config.yaml")
    loaded = load_resolved_config(path, repository=ROOT)
    assert loaded.model == config.model
    tampered = yaml.safe_load(path.read_text())
    tampered["model"]["memo"]["summary_tokens"] = 8
    (tmp_path / "tampered.yaml").write_text(yaml.safe_dump(tampered))
    with pytest.raises(ContractError, match="Recorded Memo identity"):
        load_resolved_config(tmp_path / "tampered.yaml", repository=ROOT)
    smoke = experiment_config(
        contract,
        study,
        condition="memo",
        seed=42,
        repository=ROOT,
        output_root=tmp_path,
        smoke=True,
        device="cpu",
    )
    assert smoke.model.memo is not None and smoke.model.memo.d_model == 32
    assert smoke.model.width == 32 and smoke.model.memo.segment_length == 32
    assert development_modes(contract, "memo") == (
        "retained",
        "attempt-cleared",
        "summary-cleared",
    )
    assert development_modes(contract, "fixed_segment") == (
        "retained",
        "attempt-cleared",
    )


# --------------------------------------------------------------------------
# The runtime binding
# --------------------------------------------------------------------------


def _runtime_model(**memo: Any) -> dict[str, Any]:
    return {
        "architecture_id": MEMO_ARCHITECTURE_ID,
        "width": WIDTH,
        "layers": 2,
        "heads": 2,
        "feedforward_multiplier": 2,
        "attention_backend": "vanilla",
        "trajectory_encoder": "transformer",
        "evidence": "raw",
        "bypass": False,
        "public_contract": None,
        "initialization_seed": 0,
        "memo": MemoSpec(L, S, d_model=WIDTH).to_dict(),
        **memo,
    }


_TRAINING = {
    "torch_compile": False,
    "reward_multiplier": 1.0,
    "exploration": "epsilon_greedy",
    "epsilon_anneal_steps": 10,
}


def test_configure_amago_binds_the_memo_carrier_and_refuses_a_dat_block() -> None:
    components = configure_amago(_runtime_model(), _TRAINING)
    assert components.trajectory_encoder is MemoTrajEncoder
    assert components.architecture_id == MEMO_ARCHITECTURE_ID
    target = "reasoned_icrl.model.trajectory_encoder.MemoTrajEncoder"
    bound = gin.query_parameter(f"{target}.spec")
    assert bound == MemoSpec(L, S, d_model=WIDTH)
    assert gin.query_parameter(f"{target}.d_model") == WIDTH
    with pytest.raises(ContractError, match=r"needs model\.memo"):
        configure_amago({**_runtime_model(), "memo": None}, _TRAINING)
    with pytest.raises(ContractError, match="no dual-attention block"):
        configure_amago(_runtime_model(dat={"layer_indices": [0]}), _TRAINING)
    with pytest.raises(ContractError, match="without bypass"):
        configure_amago(_runtime_model(bypass=True), _TRAINING)
    with pytest.raises(ContractError, match="Resolved Memo identity"):
        configure_amago(
            _runtime_model(
                memo={**MemoSpec(L, S, d_model=WIDTH).to_dict(), "sha256": "0" * 64}
            ),
            _TRAINING,
        )
    gin.clear_config()


# --------------------------------------------------------------------------
# Hidden-state checkpoints and the identities
# --------------------------------------------------------------------------


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


def _carrier(summary_tokens: int = S) -> MemoTrajEncoder:
    _configure()
    torch.manual_seed(0)
    return MemoTrajEncoder(
        TOKEN + 1,
        16,
        spec=MemoSpec(L, summary_tokens, d_model=WIDTH),
        token_dim=TOKEN,
        d_model=WIDTH,
    ).eval()


def _packet(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    seq = torch.randn(batch, length, TOKEN + 1)
    seq[..., -1] = 1.0
    times = torch.arange(length).view(1, length, 1).expand(batch, -1, -1).contiguous()
    return seq, times


def _rollout(
    carrier: MemoTrajEncoder, seq: torch.Tensor, times: torch.Tensor
) -> MemoHiddenState:
    hidden = carrier.init_hidden_state(seq.shape[0], torch.device("cpu"))
    with torch.no_grad():
        for step in range(seq.shape[1]):
            _, hidden = carrier(
                seq[:, step : step + 1], times[:, step : step + 1], hidden
            )
    return hidden


def _experiment(carrier: torch.nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        policy=SimpleNamespace(traj_encoder=carrier), DEVICE=torch.device("cpu")
    )


def test_the_hidden_state_round_trips_and_refuses_every_disagreement() -> None:
    carrier = _carrier()
    seq, times = _packet(3, 2 * L + 3)
    hidden = _rollout(carrier, seq, times)
    hidden.reset([1])
    assert hidden.segment.tolist() == [2, 0, 2] and hidden.lengths.tolist() == [
        2 * S + 3,
        0,
        2 * S + 3,
    ]
    payload = _hidden_state(hidden)
    assert payload["schema"] == MEMO_HIDDEN_STATE_SCHEMA
    assert payload["spec_sha256"] == carrier.spec.sha256
    restored = _restore_hidden_state(_experiment(carrier), payload)
    assert isinstance(restored, MemoHiddenState)
    assert restored.lengths.tolist() == hidden.lengths.tolist()
    assert restored.segment.tolist() == hidden.segment.tolist()
    for ours, theirs in zip(restored.layers, hidden.layers, strict=True):
        for name, tensor in ours.tensors().items():
            torch.testing.assert_close(
                tensor, theirs.tensors()[name], rtol=0, atol=0, equal_nan=True
            )
    following, _ = _packet(3, 1)
    with torch.no_grad():
        expected, _ = carrier(following, times[:, :1], hidden)
        actual, _ = carrier(following, times[:, :1], restored)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Disagreements: another carrier's identity, a missing key, a length that
    # does not fit its boundary count, a poisoned filled slot, a bad flag.
    with pytest.raises(ContractError, match="identity does not match"):
        _restore_hidden_state(_experiment(_carrier(summary_tokens=1)), payload)
    with pytest.raises(ContractError, match="does not match the policy"):
        _configure()
        _restore_hidden_state(
            _experiment(
                HistoryTrajEncoder(
                    TOKEN + 1, 8, bypass=False, token_dim=TOKEN, d_model=WIDTH
                )
            ),
            payload,
        )
    broken = dict(payload)
    del broken["segment"]
    with pytest.raises(ContractError, match="Memo hidden state"):
        _restore_hidden_state(_experiment(carrier), broken)
    wrong = dict(payload)
    wrong["lengths"] = torch.tensor(
        [1, 0, 1], dtype=torch.int32
    )  # below 2 * S held slots
    with pytest.raises(ContractError, match="disagree with the boundaries"):
        _restore_hidden_state(_experiment(carrier), wrong)
    poisoned = dict(payload)
    poisoned["layers"] = [
        {
            "variant": layer["variant"],
            "tensors": {k: v.clone() for k, v in layer["tensors"].items()},
        }
        for layer in payload["layers"]
    ]
    poisoned["layers"][0]["tensors"]["content_keys"][0, 0] = torch.nan
    with pytest.raises(ContractError, match="not finite"):
        _restore_hidden_state(_experiment(carrier), poisoned)
    flagged = dict(payload)
    flagged["summary_cleared"] = 1
    with pytest.raises(ContractError, match="boolean"):
        _restore_hidden_state(_experiment(carrier), flagged)
    # The state audit lists both counters and every slab, nothing shared.
    named = state_tensor_bytes(hidden)
    assert set(named) == {"lengths", "segment"} | {
        f"layer_{index}.{name}"
        for index in range(2)
        for name in ("content_keys", "content_values")
    }
    gin.clear_config()


def test_the_identity_travels_with_the_weights_and_the_resume_contract() -> None:
    carrier = _carrier()
    wrapped = policy_checkpoint(
        carrier.state_dict(), condition="memo", architecture_id=MEMO_ARCHITECTURE_ID
    )
    restored = read_policy_checkpoint(
        wrapped, condition="memo", architecture_id=MEMO_ARCHITECTURE_ID
    )
    validate_checkpoint_architecture(
        restored, MEMO_ARCHITECTURE_ID, expected_state=carrier.state_dict()
    )
    other = _carrier(summary_tokens=1)
    with pytest.raises(ContractError, match=r"protocol|shape"):
        validate_checkpoint_architecture(
            restored, MEMO_ARCHITECTURE_ID, expected_state=other.state_dict()
        )
    with pytest.raises(ContractError, match="incompatible policy checkpoint"):
        read_policy_checkpoint(
            wrapped, condition="raw_summary", architecture_id=MEMO_ARCHITECTURE_ID
        )
    experiment = SimpleNamespace(
        encoder_architecture_id=MEMO_ARCHITECTURE_ID,
        policy_condition="memo",
        policy=SimpleNamespace(
            traj_encoder=carrier,
            tstep_encoder=SimpleNamespace(spec=SimpleNamespace(sha256="packet")),
        ),
        learner_contract="amago-optimizer-ownership.v1",
        reasoned_training_settings={},
    )
    contract = _runtime_contract(experiment)
    assert contract["memo_sha256"] == carrier.spec.sha256
    assert "summary_sha256" not in contract and "attention_sha256" not in contract
    assert contract["packet_sha256"] == "packet"
    # A summary carrier's weights cannot load into the Memo carrier: the keys differ.
    summary = SummaryTrajEncoder(
        TOKEN + 1,
        16,
        spec=__import__(
            "reasoned_icrl.experiments.contracts", fromlist=["SummarySpec"]
        ).SummarySpec(L, S, "summary", d_model=WIDTH),
        token_dim=TOKEN,
        d_model=WIDTH,
    )
    with pytest.raises(ContractError, match="keys are incompatible"):
        validate_checkpoint_architecture(
            restored, MEMO_ARCHITECTURE_ID, expected_state=summary.state_dict()
        )
    gin.clear_config()


def test_the_cost_probe_times_boundaries_and_reports_the_state_growth() -> None:
    carrier = _carrier()
    policy = SimpleNamespace(traj_encoder=carrier, eval=carrier.eval)
    cost = carrier_cost(policy, rows=2, device=torch.device("cpu"), probes=8)
    # 16 + 1 records fit four boundaries: the probe is capped at the task.
    assert (
        cost["latency_probe"]["probes"] == 4
        and cost["latency_probe"]["boundaries_timed"] == 4
    )
    # The probe never runs past the longest declared task (17 records here),
    # which still holds its four boundaries.
    assert cost["latency_probe"]["steps"] == min(4 * L + 2, 17) == 17
    assert (
        cost["boundary_latency_seconds"] is not None
        and cost["boundary_flops_counted"] > 0
    )
    assert cost["decision_flops_counted"] > 0
    assert cost["persistent_state_bytes"] == sum(cost["state_tensor_bytes"].values())
    assert cost["shared_state_bytes"] == 0
    growth = cost["state_growth"]
    assert growth["capacity_slots"] == carrier.capacity == 4 * S + L + S
    assert growth["longest_task_records"] == 17
    slots = growth["live_slots_by_prefix"]
    assert slots["1"] == 1 and slots[str(L)] == L and slots[str(L + 1)] == S + 1
    assert slots["17"] == 4 * S + 1
    assert (
        growth["live_bytes_by_prefix"]["17"]
        == (4 * S + 1) * growth["bytes_per_slot"] + growth["counter_bytes"]
    )
    assert max(map(int, slots)) == 17 and all(int(k) <= 17 for k in slots)
    gin.clear_config()


# --------------------------------------------------------------------------
# Frozen reference cells in the analysis path (ME3 plumbing)
# --------------------------------------------------------------------------


def _comparator_study_at(tmp_path: Path, reference_root: Path) -> Any:
    raw = yaml.safe_load((ROOT / MEMO_STUDY).read_text())
    raw["contracts"] = [str(ROOT / "configs" / entry) for entry in raw["contracts"]]
    raw["tiers"]["dark_key_to_door"]["reference_root"] = str(reference_root)
    path = tmp_path / "memo-study.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_summary_memory_study(path)


def test_reference_cells_are_read_from_their_own_root_and_labelled(
    tmp_path: Path,
) -> None:
    from reasoned_icrl.analysis.runs import run_directories
    from reasoned_icrl.analysis.tier import completeness_rows
    from reasoned_icrl.experiments.summary_memory.revised import reference_configs

    reference_root = tmp_path / "eight-m"
    study = _comparator_study_at(tmp_path, reference_root)
    contract = study.contract("dark_key_to_door")
    eight = load_summary_memory_study()
    memo_root = tmp_path / "memo"
    protocol = contract.protocol
    # The comparator's own fit and two of the three frozen references exist.
    (memo_root / protocol / "memo" / "seed-42").mkdir(parents=True)
    (memo_root / protocol / "memo" / "seed-42" / "checkpoint.pt").write_bytes(b"x")
    for condition, seed in (("fixed_summary", 42), ("full_context", 100)):
        directory = reference_root / protocol / condition / f"seed-{seed}"
        directory.mkdir(parents=True)
        (directory / "checkpoint.pt").write_bytes(b"x")
        dump_config(
            experiment_config(
                eight.contract("dark_key_to_door"),
                eight,
                condition=condition,
                seed=seed,
                repository=ROOT,
                output_root=reference_root,
            ),
            directory / "config.yaml",
        )
    assert study.cell_root(contract, "memo", memo_root) == memo_root
    assert study.cell_root(contract, "full_gru", memo_root) == reference_root
    assert study.compared_cells(contract) == (
        "memo",
        "memo_fixed",
        "fixed_summary",
        "full_context",
        "full_gru",
        "fixed_window",
        "fixed_segment",
        "full_dual_relational",
        "raw_summary",
        "raw_segment",
        "raw_summary_residual",
    )
    found = run_directories(study, contract, memo_root)
    assert [(c, s) for c, s, _ in found] == [
        ("memo", 42),
        ("fixed_summary", 42),
        ("full_context", 100),
    ]
    assert found[1][2] == reference_root / protocol / "fixed_summary" / "seed-42"
    rows = completeness_rows(study, contract, memo_root, split="confirmation")
    assert [(r["condition"], r["seed"], r["group"], r["status"]) for r in rows][:8] == [
        ("memo", 42, "primary", "complete"),
        ("memo", 100, "primary", "missing"),
        ("memo", 2026, "primary", "missing"),
        ("memo_fixed", 42, "primary", "missing"),
        ("memo_fixed", 100, "primary", "missing"),
        ("memo_fixed", 2026, "primary", "missing"),
        ("fixed_summary", 42, "reference", "complete"),
        ("fixed_summary", 100, "reference", "missing"),
    ]
    assert study.tier("dark_key_to_door").role_of("full_gru") == "reference"
    assert study.tier("dark_key_to_door").role_of("memo") == "primary"
    # Every declared reference must exist at every seed before finalization.
    with pytest.raises(ContractError, match="no saved run"):
        reference_configs(study, contract, repository=ROOT, device="cpu")
    for condition in study.tier("dark_key_to_door").reference_cells:
        for seed in study.training_seeds:
            directory = reference_root / protocol / condition / f"seed-{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            if not (directory / "config.yaml").is_file():
                dump_config(
                    experiment_config(
                        eight.contract("dark_key_to_door"),
                        eight,
                        condition=condition,
                        seed=seed,
                        repository=ROOT,
                        output_root=reference_root,
                    ),
                    directory / "config.yaml",
                )
    configs = reference_configs(study, contract, repository=ROOT, device="cpu")
    assert [(c.condition, c.seed) for c in configs] == [
        (condition, seed)
        for condition in study.tier("dark_key_to_door").reference_cells
        for seed in (42, 100, 2026)
    ]
    assert all(c.output_root == reference_root and c.device == "cpu" for c in configs)
    assert configs[0].model.summary is not None and configs[0].model.dat is not None
    assert (
        configs[0].run_directory
        == reference_root / protocol / "fixed_summary" / "seed-42"
    )
    # The 8M study itself has no reference cells: nothing changes for it.
    assert eight.compared_cells(eight.contract("dark_key_to_door")) == eight.cells(
        eight.contract("dark_key_to_door")
    )
    assert (
        eight.cell_root(eight.contract("dark_key_to_door"), "fixed_summary", memo_root)
        == memo_root
    )


def test_the_count_recall_comparator_study_loads_and_plans_six_fits(
    tmp_path: Path,
) -> None:
    """ME5: the CountRecall Memo pair on the tier-2 contract, the seven frozen
    references from the 8M root (three joined for Figures 2/3),
    the confirmation split and the six declared contrasts at the tier-2 margin."""
    study = load_memo_study("count_recall")
    assert study.name == "memo_count_recall_8m"
    assert study.output_root == "outputs/memo-count-recall-8m"
    contract = study.contract("count_recall")
    tier = study.tier("count_recall")
    assert tier.tier == 2 and tier.primary == ("memo", "memo_fixed")
    assert tier.reference_cells == (
        "fixed_summary",
        "full_dual_relational",
        "full_gru",
        "fixed_window",
        "fixed_segment",
        "raw_summary",
        "raw_segment",
        "full_context",
        "raw_summary_residual",
    )
    assert tier.reference_root == "outputs/summary-memory-8m"
    assert tier.qualification_reference == "full_dual_relational"
    assert reference_condition(study, "count_recall") == "full_dual_relational"
    assert tier.practical_effect == 0.05
    assert [c.name for c in tier.primary_contrasts] == ["summary - Memo"]
    assert [(c.left, c.right) for c in tier.companion_contrasts] == [
        ("fixed_summary", "memo_fixed"),
        ("memo", "memo_fixed"),
        ("full_dual_relational", "memo"),
        ("full_gru", "memo"),
        ("fixed_summary", "fixed_segment"),
        ("raw_summary", "raw_segment"),
        ("full_context", "raw_summary"),
        ("raw_summary", "memo"),
        ("raw_summary", "memo_fixed"),
        ("raw_summary", "full_gru"),
        ("raw_summary_residual", "raw_summary"),
        ("raw_summary_residual", "raw_segment"),
        ("full_context", "raw_summary_residual"),
        ("raw_summary_residual", "memo"),
        ("raw_summary_residual", "memo_fixed"),
        ("raw_summary_residual", "full_gru"),
    ]
    assert contract.roster("confirmation") == tuple(CONFIRMATION_TASKS)
    assert sorted(contract.evaluation.splits) == [
        "confirmation",
        "development",
        "final",
    ]
    for condition, jitter in (("memo", 0.2), ("memo_fixed", 0.0)):
        config = experiment_config(
            contract,
            study,
            condition=condition,
            seed=42,
            repository=ROOT,
            output_root=tmp_path,
        )
        assert config.model.memo is not None
        assert config.model.memo.training_segment_jitter == jitter
        assert config.model.memo.segment_length == 32
        assert config.model.memo.capacity(config.training.max_sequence_length) == 48
        assert (
            config.run_directory == tmp_path / contract.protocol / condition / "seed-42"
        )
        assert development_modes(contract, condition) == (
            "retained",
            "current-token",
            "summary-cleared",
        )
    plans = plan_fits(study, contract, repository=ROOT, output_root=tmp_path)
    assert [(p.condition, p.seed) for p in plans] == [
        (condition, seed)
        for condition in ("memo", "memo_fixed")
        for seed in (42, 100, 2026)
    ]
    with pytest.raises(ContractError, match="frozen reference cell"):
        experiment_config(
            contract, study, condition="fixed_window", seed=42, repository=ROOT
        )
    with pytest.raises(ContractError, match="No Memo comparator study"):
        load_memo_study("concentration")
    # The 8M study's CountRecall contract now carries the split beside its own
    # and still loads unchanged otherwise.
    eight = load_summary_memory_study()
    assert sorted(eight.contract("count_recall").evaluation.splits) == [
        "confirmation",
        "development",
        "final",
    ]
    assert "memo" not in eight.conditions
