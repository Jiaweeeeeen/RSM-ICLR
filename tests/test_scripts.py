"""Repository layout, scripts, notebooks, provenance and retired-artifact guards."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.runtime.checkpointing import (
    AMAGORuntimeState,
    read_policy_checkpoint,
)
from reasoned_icrl.runtime.training import (
    _write_provenance,
    compact_successful_run,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/play.py",
    "scripts/generate_summary_memory_jobs.py",
    "scripts/build_xland_one_rule_manifest.py",
)


def _active_documents() -> tuple[str, ...]:
    """README and every Markdown file under docs/ — all are link-checked."""
    documents = [ROOT / "README.md"]
    documents.extend(sorted((ROOT / "docs").rglob("*.md")))
    return tuple(path.relative_to(ROOT).as_posix() for path in documents)


ACTIVE_DOCUMENTS = _active_documents()
ENVIRONMENTS = (
    "darkroom",
    "dark_key_to_door",
    "count_recall",
    "mazerunner",
    "concentration",
    "xland_minigrid",
    "xland_one_rule",
    "match_pattern",
    "tmaze",
)
CONTRACTS = (
    *(
        name
        for name in ENVIRONMENTS
        if name not in ("xland_one_rule", "match_pattern", "tmaze")
    ),
    "count_recall_hard",
    "mazerunner_15_randomized",
    "concentration_hard",
    "concentration_rank8",
)
"""Contract files under configs/environments, by file name (a protocol may share
an environment module with another). The one-rule XLand, match-pattern and
passive T-Maze contracts exist only under the 8M directory, outside the retired
4M roster."""
STUDIES = ("summary_memory",)
"""Study packages, each with its own CLI subcommand under scripts/train.py and
scripts/evaluate.py."""
ROSTERS = (
    "summary_memory",
    "summary_memory_8m",
    "memo_key_to_door_8m",
    "memo_count_recall_8m",
    "xland_one_rule_8m",
    "match_pattern_8m",
    "keydoor_capacity_8m",
    "tmaze_v3_8m",
    "keydoor_capacity_m1_8m",
    "keydoor_capacity_m8_8m",
    "mazerunner_8m",
)
"""Rosters under configs/: the retired 4M roster (loaded only by explicit path),
the active 8M revision, the two Memo comparators, the XLand one-rule application
and the passive T-Maze benchmark (each its own root, loaded only by explicit
path), all of the summary_memory package."""
NOTEBOOKS = (
    "figure2_in_context_adaptation",
    "countrecall_boundary_segments",
    "figure3_beyond_the_training_horizon",
    "mazerunner_repeated_laps",
    "figure4_learning_matched_experience",
    "figure_attention_summary",
    "figure_summary_length",
    "tables_three_benchmarks",
    "paper_figures",
)
"""The paper notebooks under notebooks/: one per figure of the set, one for
the tables and one that redraws the figures at print size, each reading the
shared ``figures_common`` plumbing."""
NOTEBOOK_FIGURES = {
    "figure2_in_context_adaptation": "figure2_in_context_adaptation",
    "countrecall_boundary_segments": "figure_countrecall_boundary_segments",
    "figure3_beyond_the_training_horizon": "figure3_beyond_the_training_horizon",
    "mazerunner_repeated_laps": "figure_mazerunner_repeated_laps",
    "figure4_learning_matched_experience": "figureA1_learning_matched_experience",
    "figure_attention_summary": "figure_attention_summary",
    "figure_summary_length": "figure_summary_length",
    "paper_figures": "paper_results_at_a_glance",
}
"""The figure each notebook builds first (its file name), when it draws one."""
PROTOCOL_MODULES = (
    "artifacts",
    "benchmarks",
    "config",
    "contracts",
    "count_recall_controls",
    "environments",
    "evaluation",
    "horizon",
    "match_pattern",
    "qualification",
    "records",
    "resumes",
    "xland_one_rule",
)
"""The study protocol: reasoned_icrl/experiments/<module>.py, never importing
the AMAGO runtime, the study package or the analysis."""
RUNTIME_MODULES = (
    "amago",
    "attention",
    "checkpointing",
    "devices",
    "diagnostics",
    "environments",
    "experiment",
    "play",
    "replay",
    "representation",
    "rollout",
    "training",
)
"""The AMAGO integration: reasoned_icrl/runtime/<module>.py."""
REVISED_CONTRACTS = (
    "mazerunner",
    "dark_key_to_door",
    "concentration",
    "count_recall",
    "xland_minigrid",
    "xland_one_rule",
    "match_pattern",
    "dark_key_to_door_m16",
    "tmaze_v3",
    "dark_key_to_door_m1",
    "dark_key_to_door_m8",
)
"""The 8M study's contracts under configs/environments/8m, which restate the
budget and split without mutating the shared 4M contracts; the two Key-to-Door
variants are the in-distribution long-horizon control
(1,000-call budget) and the capacity ablation (M = 16) of the study protocol; the
T-Maze v3 protocol and the ablation's other arms (Key-to-Door M = 1, T-Maze v3 M = 1 and
M = 16) each restate a task with
its own memory block or protocol; Key-to-Door M = 8 joined when the ablation
moved to RSM-O."""


def test_project_and_configuration_version_agree() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    package = next(
        item for item in lock["package"] if item["name"] == project["project"]["name"]
    )
    version = project["project"]["version"]
    assert package["version"] == version == citation["version"]
    assert project["tool"]["uv"]["package"] is True
    assert "jax" not in {
        dep.split("=")[0] for dep in project["project"]["dependencies"]
    }


def test_package_layout_matches_the_documented_ownership() -> None:
    package = ROOT / "reasoned_icrl"
    assert not (ROOT / "src").exists()
    assert {p.name for p in (package / "environments").glob("*.py")} == {
        "__init__.py",
        "base.py",
        "rendering.py",
        "utils.py",
        *(f"{name}.py" for name in ENVIRONMENTS),
    }
    assert {p.name for p in (package / "model").glob("*.py")} == {
        "__init__.py",
        "agent.py",
        "dat_transformer.py",
        "memo_transformer.py",
        "step_encoder.py",
        "summary_transformer.py",
        "trajectory_encoder.py",
        "utils.py",
        "window_transformer.py",
    }
    assert {p.name for p in (package / "experiments").glob("*.py")} == {
        "__init__.py",
        *(f"{name}.py" for name in PROTOCOL_MODULES),
    }
    assert {p.name for p in (package / "runtime").glob("*.py")} == {
        "__init__.py",
        *(f"{name}.py" for name in RUNTIME_MODULES),
    }
    for study in STUDIES:
        files = {p.name for p in (package / "experiments" / study).glob("*.py")}
        assert {"configs.py", "environments.py", "experiments.py"} <= files
    for retired in ("dat_benchmarks", "stage1"):
        assert not (package / "experiments" / retired).exists()
    assert {p.name for p in (ROOT / "configs").glob("*.yaml")} == {
        f"{name}.yaml" for name in ROSTERS
    }
    assert {p.name for p in (ROOT / "configs/environments").glob("*.yaml")} == {
        f"{name}.yaml" for name in CONTRACTS
    }
    assert {p.name for p in (ROOT / "configs/environments/8m").glob("*.yaml")} == {
        f"{name}.yaml" for name in REVISED_CONTRACTS
    }
    assert {p.name for p in (ROOT / "notebooks").glob("*.ipynb")} == {
        f"{name}.ipynb" for name in NOTEBOOKS
    }


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_run_and_name_every_study(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        timeout=120,
    )
    if script.endswith("generate_summary_memory_jobs.py"):
        assert "--benchmark" in result.stdout and "--group" in result.stdout
        return
    if script.endswith("build_xland_one_rule_manifest.py"):
        assert "--write" in result.stdout and "--witnesses" in result.stdout
        return
    assert all(study in result.stdout for study in STUDIES)
    for study in STUDIES:
        result = subprocess.run(
            [sys.executable, str(ROOT / script), study, "--help"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
            timeout=120,
        )
        assert "--seed" in result.stdout


def test_the_queue_launcher_spreads_fits_over_the_allocated_gpus(
    tmp_path: Path,
) -> None:
    jobfile = tmp_path / "jobs.jobs"
    jobfile.write_text(
        "# tier\ndark_key_to_door raw 0\ndark_key_to_door raw 1\n"
        "dark_key_to_door raw_summary 0\ndark_key_to_door raw_summary 1\n"
        "dark_key_to_door raw_window 0\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_summary_memory_queue.sh"), str(jobfile)],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        timeout=120,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "GPUS": "0,1",
            "SLOTS": "2",
            "OUTPUT_ROOT": str(tmp_path / "root"),
        },
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("gpu ")]
    assert [line.split()[1] for line in lines] == ["0", "1", "0", "1", "0"]
    assert "waits for a slot" in lines[-1] and "waits" not in lines[0]
    assert "2 slots on GPUs 0 1 (4 fits at once)" in result.stdout


def test_the_submit_script_shards_packs_in_proportion_to_their_gpus(
    tmp_path: Path,
) -> None:
    jobfile = tmp_path / "shard-test.jobs"
    jobfile.write_text(
        "".join(f"dark_key_to_door raw {i}\n" for i in range(6)), encoding="utf-8"
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/slurm/submit_summary_memory.sh"),
            str(jobfile),
            "2,4",
            "3",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        timeout=60,
        env={**os.environ, "DRY_RUN": "1"},
    )
    shards = sorted(
        (ROOT / "outputs/summary-memory-8m/queue/packs").glob("shard-test-*.jobs")
    )
    try:
        assert len(shards) == 2
        assert [len(s.read_text().splitlines()) for s in shards] == [2, 4]
        commands = [
            line.split() for line in result.stdout.splitlines() if "sbatch" in line
        ]
        assert [c[1:4] for c in commands] == [
            ["--gres=gpu:3090:2", "--cpus-per-task=12", "--mem=18G"],
            ["--gres=gpu:3090:4", "--cpus-per-task=24", "--mem=36G"],
        ]
        assert all(
            c[0] == "sbatch" and any(t.endswith("summary_memory_pack.slurm") for t in c)
            for c in commands
        )
    finally:
        for shard in shards:
            shard.unlink()


def _markdown_sources() -> list[tuple[str, Path, str]]:
    """Every active Markdown document plus the Markdown cells of every notebook."""
    sources = [
        (name, ROOT / name, (ROOT / name).read_text(encoding="utf-8"))
        for name in ACTIVE_DOCUMENTS
    ]
    for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        text = "\n".join(
            "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
        )
        sources.append((notebook.relative_to(ROOT).as_posix(), notebook, text))
    return sources


def test_active_documentation_local_links_resolve() -> None:
    missing: list[str] = []
    for name, document, text in _markdown_sources():
        for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", text):
            target = target.strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            if resolved.is_relative_to(ROOT / "outputs"):
                continue
            if not resolved.exists():
                missing.append(f"{name}: {target}")
    assert not missing, "\n".join(missing)


def test_package_layering_is_enforced() -> None:
    """The protocol never imports the runtime, the study or the analysis; the
    runtime never imports the study or the analysis; the record, contract,
    resume and artifact modules load without amago, torch or the model."""
    program = f"""import importlib, sys

def loaded(*prefixes):
    return sorted(
        name for name in sys.modules
        if any(name == p or name.startswith(p + ".") for p in prefixes)
    )

for name in ("contracts", "records", "resumes", "artifacts"):
    importlib.import_module(f"reasoned_icrl.experiments.{{name}}")
assert not loaded("amago", "torch", "reasoned_icrl.model"), loaded(
    "amago", "torch", "reasoned_icrl.model"
)
for name in {PROTOCOL_MODULES!r}:
    importlib.import_module(f"reasoned_icrl.experiments.{{name}}")
upper = (
    "reasoned_icrl.runtime",
    "reasoned_icrl.experiments.summary_memory",
    "reasoned_icrl.analysis",
)
assert not loaded(*upper), loaded(*upper)
for name in {RUNTIME_MODULES!r}:
    importlib.import_module(f"reasoned_icrl.runtime.{{name}}")
assert not loaded(*upper[1:]), loaded(*upper[1:])
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebooks_are_output_free_and_execute_headlessly(
    notebook: str, tmp_path: Path
) -> None:
    """A paper notebook is committed without outputs, imports the shared
    plumbing script (never the runtime or a subprocess), and executes from a
    fresh kernel against an empty study root: every column then reports no
    records and the figure is still written with its notes."""
    import nbformat
    from nbclient import NotebookClient

    book = nbformat.read(ROOT / "notebooks" / f"{notebook}.ipynb", as_version=4)
    for cell in book.cells:
        if cell.cell_type == "code":
            assert not cell.get("outputs") and cell.get("execution_count") is None
    source = "\n".join(c.source for c in book.cells if c.cell_type == "code")
    assert "from figures_common import" in source
    for forbidden in ("reasoned_icrl.runtime", "subprocess"):
        assert forbidden not in source, forbidden
    (tmp_path / "outputs").mkdir()
    environment_variables = {
        **os.environ,
        "REASONED_ICRL_OUTPUTS": str(tmp_path / "outputs"),
        "FIGURE_OUTPUT_ROOT": str(tmp_path / "figures"),
        "MPLBACKEND": "Agg",
    }
    client = NotebookClient(
        book,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    with pytest.MonkeyPatch.context() as patch:
        for key, value in environment_variables.items():
            patch.setenv(key, value)
        client.execute()
    if notebook.startswith("tables"):
        # The tables notebook draws nothing: every table is written empty with
        # its notes when no report or record exists.
        data = {p.name for p in (tmp_path / "data").iterdir()}
        assert f"{notebook}_table1_endpoints.csv" in data
        assert f"{notebook}_claims_matrix.md" in data
    else:
        written = {p.name for p in (tmp_path / "figures").iterdir()}
        figure = NOTEBOOK_FIGURES[notebook]
        assert {f"{figure}.png", f"{figure}.pdf"} <= written
        assert (tmp_path / "data" / f"{figure}_notes.json").is_file()
        return
    assert (tmp_path / "data" / f"{notebook}_notes.json").is_file()


def test_successful_run_compaction_keeps_research_artifacts(tmp_path: Path) -> None:
    run = tmp_path / "baseline/raw/seed-0"
    run.mkdir(parents=True)
    kept = {
        "config.yaml",
        "amago_config.gin",
        "checkpoint.pt",
        "train.csv",
        "metrics.json",
        "training_metrics.jsonl",
        "provenance.json",
    }
    for name in kept:
        (run / name).write_text(name, encoding="utf-8")
    for name in ("ckpts", "replay", "wandb_logs"):
        (run / name).mkdir()
        (run / name / "temporary").write_text("temporary", encoding="utf-8")
    compact_successful_run(run)
    assert {path.name for path in run.iterdir()} == kept | {"wandb_logs"}


def test_runs_record_reproducibility_provenance(tmp_path: Path) -> None:
    from reasoned_icrl.experiments.benchmarks import experiment_config
    from reasoned_icrl.experiments.config import dump_config
    from reasoned_icrl.runtime.devices import resolve_runtime
    from tests.experiments.fixtures import load_fixture_study

    study = load_fixture_study("stage1")
    config = experiment_config(
        study.contract("darkroom"),
        study,
        condition="raw",
        seed=0,
        repository=ROOT,
        device="cpu",
        output_root=tmp_path,
        smoke=True,
    )
    config.run_directory.mkdir(parents=True)
    selection = resolve_runtime(config)
    config = selection.config
    dump_config(config, config.run_directory / "config.yaml")
    path = _write_provenance(
        config, run=config.run_directory, actual_device="cpu", selection=selection
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "reasoned-icrl-provenance.v1"
    assert payload["protocol"] == config.environment.benchmark
    assert payload["actual_device"] == "cpu"
    assert payload["runtime_selection"] == selection.metadata()
    assert set(payload["source_sha256"]) and all(
        name.startswith("reasoned_icrl/") for name in payload["source_sha256"]
    )


def test_old_policy_checkpoints_and_runtime_states_are_rejected() -> None:
    with pytest.raises(ContractError, match="policy checkpoint contract"):
        read_policy_checkpoint(
            {"weight": torch.ones(1)},
            condition="raw",
            architecture_id="amago-history-v1",
        )
    adapter = AMAGORuntimeState(SimpleNamespace(env_mode="sync"), None)
    for schema in ("amago-runtime-state.v1", "amago-runtime-state.v3"):
        with pytest.raises(ContractError, match="Unsupported AMAGO runtime checkpoint"):
            adapter.validate_state_dict({"schema": schema})
    from reasoned_icrl.runtime.experiment import ReasonedExperiment

    experiment = object.__new__(ReasonedExperiment)
    with pytest.raises(ContractError, match="requires the runtime-state adapter"):
        experiment.load_checkpoint(0, resume_training_state=True)
