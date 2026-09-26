"""R4 acceptance: per-tier groups, references, contrasts and fit planning.

Tier 1's supplements are not required for its primary completion, tier 2
requires only its three selected cells, an excluded cell is outside scope on an
environment rather than a missing run, and the job planner lists exactly the
declared fits with what their run directories already hold.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from reasoned_icrl.experiments.benchmarks import (
    TierContrast,
    TierPlan,
    experiment_config,
    load_study,
)
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.summary_memory.configs import (
    DEFAULT_STUDY,
    load_retired_summary_memory_study,
    load_summary_memory_study,
    reference_condition,
)
from reasoned_icrl.experiments.summary_memory.jobs import (
    fit_status,
    plan_fits,
    write_jobs_file,
)
from reasoned_icrl.utils import repository_root

ROOT = repository_root()
SIX = (
    "full_context",
    "full_dual_relational",
    "full_dual_content",
    "fixed_summary",
    "fixed_segment",
    "fixed_window",
)


@pytest.fixture(scope="module")
def study():
    return load_summary_memory_study()


def test_the_tiers_match_the_condition_table(study) -> None:
    assert study.tiered and set(study.tiers) == {
        "dark_key_to_door",
        "concentration",
        "count_recall",
        "xland_minigrid",
    }
    tier0 = study.tier("dark_key_to_door")
    assert tier0.tier == 0 and tier0.primary == SIX
    # The ordinary-writer pair joined the GRU as tier-0 supplements with the
    # figure redesign, companion rows only.
    # The residual-rewrite carrier joined,
    # the ordinary sliding window `raw_window`, with its
    # three companion rows after the writer pair's.
    assert tier0.supplementary == (
        "full_gru",
        "raw_summary",
        "raw_segment",
        "raw_summary_residual",
        "raw_window",
    )
    assert [c.name for c in tier0.companion_contrasts][-5:] == [
        "ordinary summary - ordinary segment",
        "DAT summary - ordinary summary",
        "RSM - window",
        "window - RSM no carry",
        "full history - window",
    ]
    assert tier0.qualification_reference == "full_context"
    assert tier0.practical_effect == 1.0  # one door per 500-call task
    assert [c.name for c in tier0.primary_contrasts] == [
        "DAT full - ordinary full",
        "DAT full - dual content",
        "summary - segment",
        "summary - window",
    ]
    assert tier0.fits(study.training_seeds) == 18
    assert tier0.fits(study.training_seeds, "all") == 33
    tier1 = study.tier("concentration")
    assert tier1.tier == 1
    assert tier1.primary == (
        "full_context",
        "full_dual_relational",
        "full_dual_content",
        "full_gru",
    )
    assert tier1.supplementary == ("fixed_summary", "fixed_segment", "fixed_window")
    assert tier1.fits(study.training_seeds) == 12
    assert tier1.fits(study.training_seeds, "supplementary") == 9
    assert [c.name for c in tier1.primary_contrasts] == [
        "DAT full - ordinary full",
        "DAT full - dual content",
        "DAT full - GRU",
    ]
    tier2 = study.tier("count_recall")
    assert tier2.tier == 2 and not tier2.contract_pending
    assert tier2.primary == ("full_dual_relational", "fixed_summary", "fixed_segment")
    # The ICLR-plan additions run as tier-2 supplementary cells.
    assert tier2.supplementary == (
        "full_gru",
        "fixed_window",
        "full_context",
        "full_dual_content",
        "raw_summary",
        "raw_segment",
        "raw_summary_detach",
        "raw_summary_residual",
    )
    assert tier2.fits(study.training_seeds) == 9
    assert tier2.qualification_reference == "full_dual_relational"
    assert [c.name for c in tier2.primary_contrasts] == ["summary - segment"]
    tier3 = study.tier("xland_minigrid")
    assert tier3.tier == 3 and tier3.primary == SIX and tier3.supplementary == ()
    # A protocol name resolves the same tier as its environment name.
    assert study.tier("native-keydoor-fixed500-first8") is tier0


def test_references_resolve_per_tier_and_the_retired_roster_keeps_raw(study) -> None:
    assert reference_condition(study, "dark_key_to_door") == "full_context"
    assert reference_condition(study, "concentration") == "full_context"
    assert reference_condition(study, "count_recall") == "full_dual_relational"
    assert reference_condition(
        study, "xland-r1-9x9-small1m-goal-visible-5attempts"
    ) == ("full_context")
    retired = load_retired_summary_memory_study()
    assert not retired.tiered
    assert reference_condition(retired) == reference_condition(retired, "count_recall")
    assert reference_condition(retired) == "raw"
    with pytest.raises(ContractError, match="no tier"):
        study.tier("darkroom")


def test_primary_completion_never_needs_a_supplementary_or_excluded_cell(study) -> None:
    easy = study.contract("concentration")
    assert "fixed_summary" not in study.primary_cells(easy)
    # Easy runs every revised cell (four primary, three supplementary); the
    # legacy ordinary-writer pair added to the union roster for tier 2's
    # backbone 2x2, the truncated-gradient ablation, the
    # residual-rewrite carrier and the Key-to-Door sliding window
    # are outside its scope.
    assert set(study.cells(easy)) == set(study.conditions) - {
        "raw_summary",
        "raw_segment",
        "raw_summary_detach",
        "raw_summary_residual",
        "raw_window",
    }
    for legacy in ("raw_summary", "raw_segment"):
        with pytest.raises(ContractError, match="outside the concentration tier"):
            study.tier("concentration").group_of(legacy)
        assert study.tier("count_recall").group_of(legacy) == "supplementary"
    xland = study.contract("xland_minigrid")
    assert "full_gru" not in study.cells(xland)
    tier3 = study.tier("xland_minigrid")
    with pytest.raises(ContractError, match="outside the xland_minigrid tier"):
        tier3.group_of("full_gru")
    with pytest.raises(ContractError, match="outside the xland_minigrid tier"):
        experiment_config(xland, study, condition="full_gru", seed=42, repository=ROOT)
    assert study.tier("concentration").group_of("fixed_window") == "supplementary"
    assert study.tier("concentration").group_of("full_gru") == "primary"


def test_tier_plans_refuse_inconsistent_declarations() -> None:
    contrast = TierContrast("a - b", "full_dual_relational", "full_context")
    plan = TierPlan(
        "dark_key_to_door",
        0,
        "test",
        ("full_context", "full_dual_relational"),
        ("full_gru",),
        "full_context",
        1.0,
        (contrast,),
    )
    assert plan.cells == ("full_context", "full_dual_relational", "full_gru")
    with pytest.raises(ContractError, match="must be a primary cell"):
        replace(plan, qualification_reference="full_gru")
    with pytest.raises(ContractError, match="twice"):
        replace(plan, supplementary=("full_context",))
    with pytest.raises(ContractError, match="outside its groups"):
        replace(
            plan,
            primary_contrasts=(TierContrast("x", "fixed_summary", "full_context"),),
        )
    with pytest.raises(ContractError, match="supplementary cell"):
        replace(
            plan, primary_contrasts=(TierContrast("x", "full_gru", "full_context"),)
        )
    with pytest.raises(ContractError, match="positive"):
        replace(plan, practical_effect=0.0)
    with pytest.raises(ContractError, match="two different cells"):
        TierContrast("same", "full_context", "full_context")


def _study_copy(tmp_path: Path) -> tuple[dict, Path]:
    raw = yaml.safe_load((ROOT / DEFAULT_STUDY).read_text())
    raw["contracts"] = [str(ROOT / "configs" / entry) for entry in raw["contracts"]]
    return raw, tmp_path / "study.yaml"


def test_the_loader_checks_tiers_against_the_roster_and_the_contracts(
    tmp_path: Path,
) -> None:
    raw, path = _study_copy(tmp_path)
    raw["tiers"]["dark_key_to_door"]["primary"].append("raw_dat")
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="outside the study roster"):
        load_study(path)
    raw, path = _study_copy(tmp_path)
    del raw["tiers"]["concentration"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="missing \\['concentration'\\]"):
        load_study(path)
    raw, path = _study_copy(tmp_path)
    raw["tiers"]["count_recall"]["contract"] = (
        "pending"  # active contract cannot be pending
    )
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="'pending' exactly when"):
        load_study(path)
    raw, path = _study_copy(tmp_path)
    raw["tiers"]["count_recall"]["qualification_reference"] = "fixed_summary"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match=r"must be a primary cell|full-history"):
        load_summary_memory_study(path)
    raw, path = _study_copy(tmp_path)
    raw["tiers"]["dark_key_to_door"]["extra"] = 1
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match="Unknown tiers"):
        load_study(path)


def test_the_planner_lists_the_declared_fits_with_their_disk_state(
    study, tmp_path: Path
) -> None:
    contract = study.contract("dark_key_to_door")
    plans = plan_fits(study, contract, repository=ROOT, output_root=tmp_path)
    assert len(plans) == 18
    assert [p.condition for p in plans][:6] == ["full_context"] * 3 + [
        "full_dual_relational"
    ] * 3
    assert [p.seed for p in plans][:3] == [42, 100, 2026]
    assert {p.group for p in plans} == {"primary"}
    assert all(p.status == "missing" and p.pending for p in plans)
    assert plans[0].line == "dark_key_to_door full_context 42"
    # Disk states: a completed run, a resumable one and a started one.
    complete = plans[0].run_directory
    complete.mkdir(parents=True)
    (complete / "checkpoint.pt").write_bytes(b"")
    resumable = plans[1].run_directory
    (resumable / "ckpts" / "training_states" / "seed-100_epoch_50").mkdir(parents=True)
    (resumable / "config.yaml").write_text("")
    started = plans[2].run_directory
    started.mkdir(parents=True)
    (started / "config.yaml").write_text("")
    assert [fit_status(p.run_directory) for p in plans[:4]] == [
        "complete",
        "resumable",
        "started",
        "missing",
    ]
    again = plan_fits(study, contract, repository=ROOT, output_root=tmp_path)
    jobs = write_jobs_file(tmp_path / "tier0.jobs", again, header="tier 0")
    lines = jobs.read_text().splitlines()
    assert lines[0] == "# tier 0"
    assert lines[1].startswith("# dark_key_to_door full_context 42")  # complete
    live = [line for line in lines if not line.startswith("#")]
    assert len(live) == 17 and live[0].startswith("dark_key_to_door full_context 100")
    supplements = plan_fits(
        study, contract, repository=ROOT, group="supplementary", output_root=tmp_path
    )
    assert [p.condition for p in supplements] == (
        ["full_gru"] * 3
        + ["raw_summary"] * 3
        + ["raw_segment"] * 3
        + ["raw_summary_residual"] * 3
        + ["raw_window"] * 3
    )
    assert {p.group for p in supplements} == {"supplementary"}
    everything = plan_fits(
        study, contract, repository=ROOT, group="all", output_root=tmp_path
    )
    assert len(everything) == 33
    subset = plan_fits(
        study,
        contract,
        repository=ROOT,
        seeds=[42],
        conditions=["fixed_window", "full_context"],
        output_root=tmp_path,
    )
    assert [(p.condition, p.seed) for p in subset] == [
        ("full_context", 42),
        ("fixed_window", 42),
    ]
    with pytest.raises(ContractError, match="not training seeds"):
        plan_fits(study, contract, repository=ROOT, seeds=[0], output_root=tmp_path)
    with pytest.raises(ContractError, match="not primary cells"):
        plan_fits(
            study,
            contract,
            repository=ROOT,
            conditions=["full_gru"],
            output_root=tmp_path,
        )
    easy = plan_fits(study, study.contract("concentration"), repository=ROOT)
    assert len(easy) == 12 and "fixed_summary" not in {p.condition for p in easy}
    medium = plan_fits(study, study.contract("count_recall"), repository=ROOT)
    assert len(medium) == 9
    assert {p.condition for p in medium} == {
        "full_dual_relational",
        "fixed_summary",
        "fixed_segment",
    }


def test_the_job_script_prints_the_plan_and_writes_the_file(tmp_path: Path) -> None:
    output = tmp_path / "queue" / "tier0.jobs"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_summary_memory_jobs.py"),
            "--benchmark",
            "dark_key_to_door",
            "--output-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "18 primary fits" in result.stdout
    assert "counts: complete=0, resumable=0, started=0, missing=18" in result.stdout
    assert "dependencies:" in result.stdout
    live = [
        line for line in output.read_text().splitlines() if not line.startswith("#")
    ]
    assert len(live) == 18
    assert all(line.split()[:1] == ["dark_key_to_door"] for line in live)
    refused = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_summary_memory_jobs.py"),
            "--benchmark",
            "count_recall",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert refused.returncode == 0 and "9 primary fits" in refused.stdout


def test_the_development_script_refuses_to_finalize_an_incomplete_matrix(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/develop_summary_memory.py"),
            "--benchmark",
            "dark_key_to_door",
            "--output-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--finalize",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert result.returncode == 2
    assert "read only after every primary fit completes" in result.stderr
    assert "missing (not started)" in result.stdout
