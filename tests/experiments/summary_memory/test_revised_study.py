"""R1 acceptance: the 8M study's conditions, budget and split identity.

These tests pin the properties that keep the revision honest: a revised cell
never resolves to a legacy operator's identity, the legacy roster keeps loading
unchanged, the two studies cannot write to the same directory, and the new final
panel is untouched by the 4M study.
"""

from __future__ import annotations

from pathlib import Path

import gin
import pytest
import yaml

from reasoned_icrl.environments.base import (
    DEVELOPMENT_TASKS,
    FINAL_OOD_TASKS,
    FINAL_TASKS,
    REVISED_FINAL_TASKS,
    TRAINING_TASKS,
    benchmark_task_sources,
)
from reasoned_icrl.experiments.benchmarks import (
    experiment_config,
    load_contract,
    load_study,
)
from reasoned_icrl.experiments.config import (
    architecture_id,
    dump_config,
    load_resolved_config,
)
from reasoned_icrl.experiments.contracts import (
    DAT_SUMMARY_ARCHITECTURE_ID,
    DAT_SUMMARY_V2_ARCHITECTURE_ID,
    DAT_WINDOW_ARCHITECTURE_ID,
    REVISED_CONDITIONS,
    WINDOW_ARCHITECTURE_ID,
    ConditionSpec,
    ContractError,
    SummarySpec,
    attention_parameter_count,
    capacity_matched_control_dims,
    condition_label,
    condition_spec,
    control_parameter_match,
    match_window_to_summary,
)
from reasoned_icrl.experiments.summary_memory.configs import (
    DEFAULT_STUDY,
    RETIRED_STUDY,
    RETIRED_STUDY_NAME,
    STUDY_NAME,
    load_retired_summary_memory_study,
    load_summary_memory_study,
    reference_condition,
)
from reasoned_icrl.model.trajectory_encoder import SummaryTrajEncoder, WindowTrajEncoder
from reasoned_icrl.runtime.amago import BOUND_CARRIERS, configure_amago
from reasoned_icrl.utils import repository_root

REQUIRED_CELLS = (
    "full_context",
    "full_dual_relational",
    "full_dual_content",
    "fixed_summary",
    "fixed_segment",
    "fixed_window",
)


@pytest.fixture(scope="module")
def revised_study():
    """The active study: loading without a path must select the 8M roster."""
    return load_summary_memory_study()


def test_the_revised_roster_holds_the_six_required_cells_and_the_optional_gru(
    revised_study,
) -> None:
    assert set(REQUIRED_CELLS) <= set(revised_study.conditions)
    assert "full_gru" in revised_study.conditions
    assert reference_condition(revised_study, "dark_key_to_door") == "full_context"
    for name in REVISED_CONDITIONS:
        assert condition_label(name)


def test_every_revised_condition_resolves_and_preflight_accepts_it(
    revised_study,
) -> None:
    root = repository_root()
    for contract in revised_study.contracts:
        for condition in revised_study.cells(contract):
            config = experiment_config(
                contract, revised_study, condition=condition, seed=42, repository=root
            )
            assert config.model.architecture_id == architecture_id(condition)


def test_a_revised_bounded_cell_never_borrows_a_legacy_carrier_identity() -> None:
    assert architecture_id("fixed_summary") == DAT_SUMMARY_V2_ARCHITECTURE_ID
    assert architecture_id("fixed_segment") == DAT_SUMMARY_V2_ARCHITECTURE_ID
    assert architecture_id("fixed_window") == DAT_WINDOW_ARCHITECTURE_ID
    assert architecture_id("raw_dat_summary") == DAT_SUMMARY_ARCHITECTURE_ID
    assert architecture_id("raw_window") == WINDOW_ARCHITECTURE_ID
    revised = {architecture_id(name) for name in ("fixed_summary", "fixed_window")}
    legacy = {architecture_id(name) for name in ("raw_dat_summary", "raw_window")}
    assert revised.isdisjoint(legacy)


def test_a_full_prefix_cell_keeps_its_legacy_operator_identity() -> None:
    """The study identity changes; the operator does not, so checkpoints load."""
    for revised, legacy in (
        ("full_context", "raw"),
        ("full_dual_relational", "raw_dat"),
        ("full_dual_content", "raw_dual_content"),
        ("full_gru", "raw_gru"),
    ):
        assert architecture_id(revised) == architecture_id(legacy)
        assert condition_spec(revised) == condition_spec(legacy)


def test_the_revised_carriers_bind_only_once_implemented() -> None:
    """R1 registered the identities; R2 bound the summary carrier behind
    ``amago-dat-summary-v2``. The window carrier's binding is R3's check."""
    assert DAT_SUMMARY_V2_ARCHITECTURE_ID in BOUND_CARRIERS
    for name in ("full_context", "full_dual_relational", "full_dual_content"):
        assert architecture_id(name) in BOUND_CARRIERS


# --------------------------------------------------------------------------
# R2: the routed summary and segment cells resolve, round-trip and bind
# --------------------------------------------------------------------------

CAPACITY_BY_PROTOCOL = {
    "native-keydoor-fixed500-first8": 32 + 4 + 4,
    "count-recall-medium": 32 + 4 + 4,
    "concentration-easy": 32 + 4 + 4,  # R4: C=32/M=4 for Easy
    "xland-r1-9x9-small1m-goal-visible-5attempts": 64 + 4 + 4,
}


@pytest.mark.parametrize("condition", ["fixed_summary", "fixed_segment"])
def test_the_routed_bounded_cells_resolve_round_trip_and_bind(
    revised_study, condition: str, tmp_path: Path
) -> None:
    root = repository_root()
    for contract in revised_study.contracts:
        config = experiment_config(
            contract, revised_study, condition=condition, seed=42, repository=root
        )
        assert config.model.architecture_id == DAT_SUMMARY_V2_ARCHITECTURE_ID
        summary = config.model.summary
        assert summary is not None and summary.relational_sources == "timestep_records"
        assert summary.regime == (
            "summary" if condition == "fixed_summary" else "segment"
        )
        assert summary.writer == "same"
        assert summary.capacity == CAPACITY_BY_PROTOCOL[contract.protocol]
        dat = config.model.dat
        assert dat is not None and dat.mode == "dat"
        assert dat.max_relative_distance == summary.capacity
        assert dat.layer_indices == (0, 1, 2)  # every layer, as the study declares
        path = dump_config(
            config, tmp_path / contract.protocol / condition / "config.yaml"
        )
        reloaded = load_resolved_config(path, repository=root)
        assert reloaded == config
        assert yaml.safe_load(path.read_text())["model"]["summary"][
            "relational_sources"
        ] == ("timestep_records")
        mapping = config.as_runtime_mapping()
        mapping["model"]["public_contract"] = {"schema": "test-public-contract.v1"}
        components = configure_amago(mapping["model"], mapping["training"])
        assert components.architecture_id == DAT_SUMMARY_V2_ARCHITECTURE_ID
        assert components.trajectory_encoder is SummaryTrajEncoder
        gin.clear_config()


def test_the_routed_identity_never_equals_the_legacy_one_on_equal_settings() -> None:
    """Key-to-Door keeps C=32, M=4 in both studies, so the legacy
    ``raw_dat_summary`` and the revised ``fixed_summary`` differ only by the
    relational route; their summary identities must still differ, while their
    attention identities agree."""
    root = repository_root()
    revised = load_summary_memory_study()
    retired = load_retired_summary_memory_study()
    new = experiment_config(
        revised.contract("dark_key_to_door"),
        revised,
        condition="fixed_summary",
        seed=42,
        repository=root,
    )
    old = experiment_config(
        retired.contract("dark_key_to_door"),
        retired,
        condition="raw_dat_summary",
        seed=0,
        repository=root,
    )
    assert new.model.summary is not None and old.model.summary is not None
    assert new.model.dat is not None and old.model.dat is not None
    assert (new.model.summary.segment_length, new.model.summary.memory_tokens) == (
        old.model.summary.segment_length,
        old.model.summary.memory_tokens,
    )
    assert new.model.summary.sha256 != old.model.summary.sha256
    assert new.model.summary.relational_sources == "timestep_records"
    assert old.model.summary.relational_sources == "causal_prefix"
    assert new.model.architecture_id != old.model.architecture_id
    assert old.model.architecture_id == DAT_SUMMARY_ARCHITECTURE_ID
    assert new.model.dat.sha256 == old.model.dat.sha256


def test_a_study_block_naming_the_wrong_route_is_refused(tmp_path: Path) -> None:
    root = repository_root()
    raw = yaml.safe_load((root / DEFAULT_STUDY).read_text())
    raw["contracts"] = [str(root / "configs" / entry) for entry in raw["contracts"]]
    raw["model"]["summary"]["relational_sources"] = "causal_prefix"
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="relational_sources disagrees"):
        study = load_study(path)
        experiment_config(
            study.contract("dark_key_to_door"),
            study,
            condition="fixed_summary",
            seed=42,
            repository=root,
        )
    # A block naming the revised route resolves once the roster carries only
    # revised cells; the legacy ordinary-writer pair added to tier 2's
    # supplementary group keeps its own `causal_prefix` route,
    # so the active study names no route in the block and each cell derives its
    # own (both are checked at load).
    raw["model"]["summary"]["relational_sources"] = "timestep_records"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="relational_sources disagrees"):
        load_study(path)
    raw["conditions"] = [c for c in raw["conditions"] if not c.startswith("raw_")]
    # The writer pair is a supplement of tiers 0 and 2.
    for tier in (raw["tiers"]["dark_key_to_door"], raw["tiers"]["count_recall"]):
        tier["supplementary"] = [
            c for c in tier["supplementary"] if not c.startswith("raw_")
        ]
        tier["companion_contrasts"] = [
            c
            for c in tier["companion_contrasts"]
            if not (c["left"].startswith("raw_") or c["right"].startswith("raw_"))
        ]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    study = load_study(path)
    config = experiment_config(
        study.contract("dark_key_to_door"),
        study,
        condition="fixed_summary",
        seed=42,
        repository=root,
    )
    assert config.model.architecture_id == DAT_SUMMARY_V2_ARCHITECTURE_ID


def test_the_timestep_record_route_refuses_combinations_that_compute_nothing_new() -> (
    None
):
    with pytest.raises(ContractError, match="only dual attention"):
        ConditionSpec(
            "transformer",
            "raw",
            memory="summary",
            relational_sources="timestep_records",
        )
    with pytest.raises(ContractError, match="full-prefix block"):
        ConditionSpec(
            "transformer", "raw", attention="dat", relational_sources="timestep_records"
        )
    with pytest.raises(ContractError, match="legacy writer ablation"):
        ConditionSpec(
            "transformer",
            "raw",
            attention="dat",
            memory="summary",
            writer="relational_off",
            relational_sources="timestep_records",
        )
    with pytest.raises(ContractError, match="requires the"):
        ConditionSpec("transformer", "raw", attention="dat", memory="window")


def test_the_retired_roster_still_loads_only_by_explicit_path() -> None:
    legacy = load_summary_memory_study(repository_root() / RETIRED_STUDY)
    assert reference_condition(legacy) == "raw"
    assert legacy.training_seeds == (0, 1, 2)
    assert "raw" in legacy.conditions
    for contract in legacy.contracts:
        nominal = (
            contract.training.epochs
            * contract.training.timesteps_per_epoch
            * contract.environment.parallel_envs
        )
        assert nominal == 4_000_000


def test_the_revised_budget_is_eight_million_on_every_contract(revised_study) -> None:
    assert revised_study.training_seeds == (42, 100, 2026)
    assert revised_study.pilot_seeds == ()
    for contract in revised_study.contracts:
        training = contract.training
        nominal = (
            training.epochs
            * training.timesteps_per_epoch
            * contract.environment.parallel_envs
        )
        assert nominal == 8_000_000
        assert training.timesteps_per_epoch == 500
        assert training.batches_per_epoch == 128
        assert contract.environment.parallel_envs == 16
        # The anneal is declared in absolute actor-local vector calls, so a
        # longer budget does not silently stretch exploration.
        assert training.epsilon_anneal_steps in (50_000, 200_000)


def test_the_saved_checkpoint_labels_reach_the_eight_million_endpoint(
    revised_study,
) -> None:
    """AMAGO labels epochs from zero and saves label multiples of the interval
    plus the final epoch (never a pre-learning epoch), so the saved grid is
    50, 100, ..., 950, 999: label N holds N + 1 epochs. The 8M endpoint (label
    999) is exact; every nominal 0.4M point of EXPERIMENTS sections 3 and 5 has
    a saved label within one
    epoch (8,000 decisions) of it, and the nominal 4M point is label 500 at
    4.008M, one epoch past it (R4 records the measured counts, never the label)."""
    from reasoned_icrl.experiments.artifacts import (
        checkpoint_labels,
        collected_at_label,
    )

    for contract in revised_study.contracts:
        training = contract.training
        actors = contract.environment.parallel_envs
        labels = checkpoint_labels(
            training.epochs,
            training.checkpoint_interval,
            start_learning=training.start_learning_epoch,
        )
        assert labels[0] == 50  # label 0 precedes the first learner update
        collected = {
            collected_at_label(
                label, timesteps_per_epoch=training.timesteps_per_epoch, actors=actors
            )
            for label in labels
        }
        assert 8_000_000 in collected and labels[-1] == 999
        per_epoch = training.timesteps_per_epoch * actors
        for nominal in (
            4_000_000,
            5_200_000,
            5_600_000,
            6_000_000,
            6_400_000,
            6_800_000,
            7_200_000,
            7_600_000,
        ):
            assert min(abs(value - nominal) for value in collected) <= per_epoch
        assert training.epochs % training.checkpoint_interval == 0


def test_the_revised_checkpoint_rule_keeps_endpoint_and_selected_panels_separate(
    revised_study,
) -> None:
    for contract in revised_study.contracts:
        assert contract.evaluation.checkpoint_rule == (
            "collection-endpoint-primary-development-selected-secondary"
        )


def test_the_new_final_panel_is_disjoint_from_every_earlier_split(
    revised_study,
) -> None:
    revised = set(REVISED_FINAL_TASKS)
    assert len(revised) == 256
    for earlier in (TRAINING_TASKS, DEVELOPMENT_TASKS, FINAL_TASKS, FINAL_OOD_TASKS):
        assert revised.isdisjoint(set(earlier))
    assert benchmark_task_sources("final-revised") == REVISED_FINAL_TASKS
    for contract in revised_study.contracts:
        splits = contract.evaluation.splits
        assert splits["final"].source == "final-revised"
        assert splits["final"].count == 256
        assert splits["development"].count == 64


def test_the_two_studies_cannot_write_into_the_same_directory(revised_study) -> None:
    legacy = load_summary_memory_study(repository_root() / RETIRED_STUDY)
    assert revised_study.output_root != legacy.output_root
    root = repository_root()
    legacy_dirs = {
        experiment_config(
            contract, legacy, condition="raw", seed=seed, repository=root
        ).run_directory
        for contract in legacy.contracts
        for seed in legacy.training_seeds
    }
    revised_dirs = {
        experiment_config(
            contract, revised_study, condition=condition, seed=seed, repository=root
        ).run_directory
        for contract in revised_study.contracts
        for condition in revised_study.cells(contract)
        for seed in revised_study.training_seeds
    }
    assert legacy_dirs.isdisjoint(revised_dirs)
    # Tier 1 runs all seven revised cells; tiers 0 and 2 run them and the
    # legacy ordinary-writer pair `raw_summary` / `raw_segment` (later additions); tier
    # 2 also runs the truncated-gradient ablation
    # `raw_summary_detach`; tier 3 runs the six required cells.
    # Tiers 0 and 2 each gained the residual rewrite
    # (commit 798f2df); tier 0 gained the ordinary sliding window `raw_window`
    # , so the counts are 11 / 7 / 11 / 6 cells per tier.
    assert len(revised_dirs) == (11 + 7 + 11 + 6) * 3


def test_an_unknown_split_name_is_refused() -> None:
    with pytest.raises(ContractError, match="Unknown benchmark split"):
        benchmark_task_sources("final-made-up")


# --------------------------------------------------------------------------
# The active study, and the retired 4M roster
# --------------------------------------------------------------------------


def test_loading_without_a_path_selects_the_revised_roster(revised_study) -> None:
    """Nothing in the active code base reaches the 4M roster implicitly."""
    assert revised_study.name == STUDY_NAME == "summary_memory_8m"
    assert reference_condition(revised_study, "dark_key_to_door") == "full_context"
    with pytest.raises(ContractError, match="per environment"):
        reference_condition(revised_study)
    assert Path("configs/summary_memory_8m.yaml") == DEFAULT_STUDY


def test_the_retired_roster_is_bound_to_no_active_contract(revised_study) -> None:
    retired = load_retired_summary_memory_study()
    assert retired.name == RETIRED_STUDY_NAME == "summary_memory"
    active_protocols = {c.protocol for c in revised_study.contracts}
    retired_protocols = {c.protocol for c in retired.contracts}
    # The deferred environments' contracts are kept on disk and belong to no
    # active study; the three active protocols are served by 8m/ contracts.
    assert {"count-recall-hard", "mazerunner-15-randomized-actions"} <= (
        retired_protocols - active_protocols
    )
    for contract in revised_study.contracts:
        assert contract.training.epochs == 1000
    for contract in retired.contracts:
        assert contract.training.epochs == 500


# --------------------------------------------------------------------------
# Fixed bounded-state capacity (R4 retired the development-selection grids)
# --------------------------------------------------------------------------

FIXED_CAPACITY = {
    # protocol: (C, M, W) as docs/METHOD.md fixes them; no sweep
    "native-keydoor-fixed500-first8": (32, 4, 40),
    "concentration-easy": (32, 4, 40),
    "count-recall-medium": (32, 4, 40),
    "xland-r1-9x9-small1m-goal-visible-5attempts": (64, 4, 72),
}


def test_every_contract_fixes_the_agreed_capacity_with_no_grid(revised_study) -> None:
    assert {c.protocol for c in revised_study.contracts} == set(FIXED_CAPACITY)
    for contract in revised_study.contracts:
        segment, tokens, window = FIXED_CAPACITY[contract.protocol]
        summary = contract.memory["summary"]
        assert (summary["segment_length"], summary["memory_tokens"]) == (
            segment,
            tokens,
        )
        assert contract.memory["window"]["segment_length"] == window
        assert not hasattr(contract, "capacity")


def test_a_contract_with_a_capacity_grid_is_refused(tmp_path: Path) -> None:
    source = repository_root() / "configs/environments/8m/dark_key_to_door.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["capacity"] = {"summary_grid": [[32, 4], [64, 4]]}
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="retired in R4"):
        load_contract(path)


def test_easy_resolves_the_new_capacity_and_its_own_state_match(revised_study) -> None:
    """R4 applied the Easy update: C=32/M=4 and the matched W=40 on the
    same carrier geometry as Key-to-Door, so the bytes coincide."""
    root = repository_root()
    easy = revised_study.contract("concentration")
    summary = experiment_config(
        easy, revised_study, condition="fixed_summary", seed=42, repository=root
    )
    assert summary.model.summary is not None
    assert summary.model.summary.capacity == 40
    assert summary.model.dat is not None
    assert summary.model.dat.max_relative_distance == 40
    window = experiment_config(
        easy, revised_study, condition="fixed_window", seed=42, repository=root
    )
    assert window.model.window_match is not None
    assert window.model.window_match.window_length == 40
    assert window.model.window_match.summary_state_bytes == 249_868
    assert window.model.window_match.window_state_bytes == 246_084


# --------------------------------------------------------------------------
# R3: the dual-attention window and the measured state match
# --------------------------------------------------------------------------

EXPECTED_MATCH = {
    # protocol: (W, summary bytes per actor, window bytes per actor)
    "native-keydoor-fixed500-first8": (40, 249_868, 246_084),
    "concentration-easy": (40, 249_868, 246_084),  # R4: C=32/M=4 like Key-to-Door
    "count-recall-medium": (
        40,
        249_868,
        246_084,
    ),  # Supplementary window, C=32/M=4
    "xland-r1-9x9-small1m-goal-visible-5attempts": (72, 446_476, 442_948),
}
"""The measured state match of every active contract. The summary figure for
Key-to-Door is the 249,868 B the 4M study recorded for `raw_dat_summary`, and
the window figure the 246,084 B it recorded for `raw_window`: a selected DAT
layer's slot is as wide as an ordinary layer's, so the DAT window at W=40
allocates exactly what the ordinary window did. Easy's completed R3 measurement
at C=16/M=4 (W=24, 151,564 / 147,652 B) is historical since R4."""
NON_DEFAULT_GEOMETRIES = ((32, 4), (64, 4), (32, 8), (64, 8), (128, 4))
"""Regression geometries for the byte match (the former development grids); they
are model coverage, not experiments to run."""


def _window_config(revised_study, contract, condition: str = "fixed_window", **kw):
    return experiment_config(
        contract,
        revised_study,
        condition=condition,
        seed=42,
        repository=repository_root(),
        **kw,
    )


def test_the_window_length_is_the_measured_state_match(revised_study) -> None:
    """R3 replaced slot parity: W is the largest window whose allocated
    per-actor state does not exceed the summary carrier's, both counted tensor
    by tensor, and the resolved config records the match and its residual."""
    for contract in revised_study.contracts:
        if "fixed_window" not in revised_study.cells(contract):
            continue
        expected_w, summary_bytes, window_bytes = EXPECTED_MATCH[contract.protocol]
        config = _window_config(revised_study, contract)
        match = config.model.window_match
        assert match is not None and config.model.window is not None
        assert config.model.window.segment_length == match.window_length == expected_w
        assert match.summary_state_bytes == summary_bytes
        assert match.window_state_bytes == window_bytes
        assert match.residual_bytes == summary_bytes - window_bytes
        assert 0 < match.residual_bytes < 6_152  # less than one more window slot
        assert match.slot_parity_length == expected_w
        summary = contract.memory["summary"]
        assert (match.reference_segment_length, match.reference_memory_tokens) == (
            summary["segment_length"],
            summary["memory_tokens"],
        )
        # The band's symbols are clipped at W, as a summary cell clips at M+C+M.
        assert config.model.dat is not None and config.model.dat.mode == "dat"
        assert config.model.dat.max_relative_distance == expected_w
        assert config.model.summary is None
        for other in ("full_context", "fixed_summary", "full_dual_relational"):
            assert (
                _window_config(revised_study, contract, other).model.window_match
                is None
            )


def test_the_match_departs_from_slot_parity_on_non_default_geometries(
    revised_study,
) -> None:
    """On M=8 pairs the memory tokens buy a slot, so the byte match exceeds
    C + 2M; the machinery is not a restatement of parity."""
    model = revised_study.model
    for contract in revised_study.contracts:
        if "fixed_window" not in revised_study.cells(contract):
            continue
        dat = _window_config(revised_study, contract).model.dat
        assert dat is not None
        for segment, tokens in NON_DEFAULT_GEOMETRIES:
            match = match_window_to_summary(
                SummarySpec(
                    segment_length=segment, memory_tokens=tokens, regime="summary"
                ),
                dat,
                layers=int(model["layers"]),
                heads=int(model["heads"]),
                width=int(model["width"]),
            )
            assert match.window_length >= match.slot_parity_length
            assert (match.window_length > match.slot_parity_length) == (tokens == 8)
            assert match.window_state_bytes <= match.summary_state_bytes


def test_a_window_that_is_not_the_match_is_refused(tmp_path: Path) -> None:
    source = repository_root() / "configs/environments/8m/dark_key_to_door.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["memory"]["window"]["segment_length"] = 41
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    study = load_summary_memory_study()
    with pytest.raises(ContractError, match=r"measured state match.*W=40"):
        _window_config(study, load_contract(path))
    raw["memory"]["window"]["segment_length"] = 39
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match=r"measured state match.*W=40"):
        _window_config(study, load_contract(path))
    # A study without a summary block cannot resolve the revised window at all:
    # the match needs its reference.
    study_raw = yaml.safe_load(DEFAULT_STUDY.read_text())
    study_raw["contracts"] = [
        str(repository_root() / "configs" / entry) for entry in study_raw["contracts"]
    ]
    del study_raw["model"]["summary"]
    for contract_raw in study_raw["contracts"]:
        contract_yaml = yaml.safe_load(Path(contract_raw).read_text())
        contract_yaml.pop("memory", None)
        (tmp_path / Path(contract_raw).name).write_text(
            yaml.safe_dump(contract_yaml, sort_keys=False)
        )
    study_raw["contracts"] = [
        str(tmp_path / Path(entry).name) for entry in study_raw["contracts"]
    ]
    study_raw["conditions"] = [
        name for name in study_raw["conditions"] if name.startswith("full")
    ] + ["fixed_window"]
    study_raw.pop("tiers")  # the tiers name the summary cells this study drops
    study_path = tmp_path / "study.yaml"
    study_path.write_text(yaml.safe_dump(study_raw, sort_keys=False))
    with pytest.raises(ContractError, match=r"needs a model\.summary block"):
        load_study(study_path)


def test_fixed_window_resolves_round_trips_and_binds(
    revised_study, tmp_path: Path
) -> None:
    assert DAT_WINDOW_ARCHITECTURE_ID in BOUND_CARRIERS
    root = repository_root()
    for contract in revised_study.contracts:
        if "fixed_window" not in revised_study.cells(contract):
            continue
        config = _window_config(
            revised_study, contract, output_root=tmp_path, device="cpu"
        )
        assert config.model.architecture_id == DAT_WINDOW_ARCHITECTURE_ID
        assert config.model.window is not None and config.model.dat is not None
        path = dump_config(config, tmp_path / contract.protocol / "config.yaml")
        recorded = yaml.safe_load(path.read_text())["model"]
        assert recorded["window_match"]["window_length"] == config.model.window.capacity
        assert recorded["dat"]["max_relative_distance"] == config.model.window.capacity
        assert recorded["window"]["sha256"] == config.model.window.sha256
        assert load_resolved_config(path, repository=root) == config
        mapping = config.as_runtime_mapping()
        assert mapping["model"]["window_match"]["residual_bytes"] > 0
        mapping["model"]["public_contract"] = {"schema": "test-public-contract.v1"}
        components = configure_amago(mapping["model"], mapping["training"])
        assert components.trajectory_encoder is WindowTrajEncoder
        assert components.architecture_id == DAT_WINDOW_ARCHITECTURE_ID
        target = "reasoned_icrl.model.trajectory_encoder.WindowTrajEncoder"
        assert gin.query_parameter(f"{target}.spec") == config.model.window
        assert gin.query_parameter(f"{target}.dat") == config.model.dat
        assert gin.query_parameter(f"{target}.initialization_seed") == 42
        gin.clear_config()


def test_a_tampered_window_match_is_refused(tmp_path: Path) -> None:
    study = load_summary_memory_study()
    config = _window_config(
        study, study.contract("dark_key_to_door"), output_root=tmp_path, device="cpu"
    )
    path = dump_config(config, tmp_path / "run" / "config.yaml")
    raw = yaml.safe_load(path.read_text())
    raw["model"]["window_match"]["residual_bytes"] = 0
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="does not reproduce"):
        load_resolved_config(path, repository=repository_root())
    raw = yaml.safe_load(path.read_text())
    raw["model"]["window_match"]["reference_memory_tokens"] = 8  # would give W=49
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match=r"measured state match.*W=49"):
        load_resolved_config(path, repository=repository_root())


def test_the_dual_content_match_is_recalculated_per_environment(revised_study) -> None:
    """The control's widths are derived at each environment's resolved clipping
    distance and the tolerance is recorded. The three contracts share the
    distance 500, so the match is the legacy 96/16 everywhere: +2.5 %, within
    the 5 % requirement and outside the 2 % aim, reported rather than assumed."""
    for contract in revised_study.contracts:
        if "full_dual_content" not in revised_study.cells(contract):
            continue
        dat = _window_config(revised_study, contract, "full_dual_relational").model.dat
        control = _window_config(revised_study, contract, "full_dual_content").model.dat
        assert dat is not None and control is not None
        assert dat.max_relative_distance == control.max_relative_distance == 500
        widths = (control.control_content_head_dim, control.control_second_head_dim)
        assert widths == capacity_matched_control_dims(dat) == (96, 16)
        report = control_parameter_match(control)
        assert report["reference_attention_parameters"] == attention_parameter_count(
            dat
        )
        assert report["control_attention_parameters"] == attention_parameter_count(
            control
        )
        assert report["within_five_percent"] and not report["within_two_percent"]
        assert 0.02 < float(report["relative_difference"]) < 0.03
    with pytest.raises(ContractError, match="dual-content control"):
        control_parameter_match(dat)
