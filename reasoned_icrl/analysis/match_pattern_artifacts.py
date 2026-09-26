"""Read the diagnostic's own contract and saved evidence for figure notebooks.

No model is constructed, no evaluation is launched, and incomplete three-seed
panels remain incomplete. Notebook selectors operate on these computed tables.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from reasoned_icrl.analysis.match_pattern import learning_summary, stratified_interval
from reasoned_icrl.experiments.benchmarks import BenchmarkContract, Study
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import evaluation_directory
from reasoned_icrl.experiments.match_pattern import (
    MANIFEST_PATH,
    PROTOCOL,
    canonical_hash,
)
from reasoned_icrl.experiments.records import (
    BenchmarkEvent,
    cell_values,
    read_benchmark_results,
)
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.jobs import completed_training_budget

if TYPE_CHECKING:
    from reasoned_icrl.analysis.tier import TierReport

Row = dict[str, Any]
PALETTE = {
    "full_context": "#56B4E9",
    "full_dual_relational": "#000000",
    "full_dual_content": "#E69F00",
    "full_gru": "#009E73",
}


def load_match_pattern_artifacts(
    root: Path,
    repository: Path,
    *,
    source: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Load independent learning, endpoint, lifetime and cost evidence tables."""
    sources: dict[str, str] = {}

    def track(path: Path) -> None:
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        if source is not None:
            source(path)

    def read(path: Path) -> Any:
        track(path)
        return json.loads(path.read_text())

    study_path = repository / "configs/match_pattern_8m.yaml"
    track(study_path)
    for source_file in (
        "configs/environments/8m/match_pattern.yaml",
        "reasoned_icrl/analysis/match_pattern.py",
        "reasoned_icrl/analysis/match_pattern_artifacts.py",
    ):
        track(repository / source_file)
    study = load_summary_memory_study(study_path)
    contract = study.contract("match_pattern")
    corpus_hash = canonical_hash(read(repository / MANIFEST_PATH))
    tables: dict[str, list[Row]] = {
        key: []
        for key in (
            "inventory",
            "native_returns",
            "native_mean",
            "development_mean",
            "development",
            "learning_summary",
            "endpoints",
            "seed_scores",
            "selected_companions",
            "contrasts",
            "interventions",
            "costs",
            "mechanisms",
            "mechanism_effects",
            "qualification",
        )
    }
    panels: dict[tuple[str, str, str], list[BenchmarkEvent]] = {}
    hashes: dict[tuple[str, int], str] = {}
    gate_path = root / f"qualification-{PROTOCOL}-8m.json"
    gate = (
        read(gate_path)
        if gate_path.is_file()
        else {"status": "unqualified: gate missing"}
    )
    qualification_records = {}
    for path in sorted((root / "qualification").glob("*.json")):
        record = read(path)
        qualification_records[path.stem] = record
        tables["qualification"].append({"record": path.stem, "evidence": record})
    for condition in study.cells(contract):
        for seed in study.training_seeds:
            run = root / PROTOCOL / condition / f"seed-{seed}"
            base = {"condition": condition, "seed": seed}
            try:
                budget = completed_training_budget(run, contract)
                status = "endpoint reached"
            except ContractError:
                budget, status = None, "incomplete"
            tables["inventory"].append(
                base | {"status": status, "charged_calls": budget}
            )
            for name in (
                "config.yaml",
                "metrics.json",
                "systems.json",
                "provenance.json",
            ):
                if (run / name).is_file():
                    track(run / name)
            log = run / "training_metrics.jsonl"
            if log.is_file():
                track(log)
                # Exact resume truncates this log to its checkpoint boundary.
                # Repeated clocks from other resumes are explicit last records.
                rows: dict[tuple[str, int], Row] = {}
                for line in log.read_text().splitlines():
                    row = json.loads(line)
                    if row.get("panel") not in ("train-rollout", "val"):
                        continue
                    value = row.get("Average Total Return (Across All Env Names)")
                    if value is None or "charged_calls" not in row:
                        continue
                    calls = int(row["charged_calls"])
                    rows[(row["panel"], calls)] = base | {
                        "panel": row["panel"],
                        "charged_calls": calls,
                        "native_return": float(value),
                        "chance": 0.0,
                    }
                tables["native_returns"].extend(rows.values())
            development = run / "development.json"
            points = {}
            if development.is_file():
                for row in read(development)["series"]:
                    if row.get("charged_calls") is None:
                        continue
                    calls, accuracy = int(row["charged_calls"]), float(row["primary"])
                    points[calls] = accuracy
                    tables["development"].append(
                        base
                        | {
                            "charged_calls": calls,
                            "accuracy": accuracy,
                            "checkpoint": row["epoch"],
                            "checkpoint_sha256": row.get("checkpoint_sha256"),
                            "chance": 0.5,
                        }
                    )
            tables["learning_summary"].append(base | learning_summary(points))
            system_path = run / "systems.json"
            if system_path.is_file():
                system = read(system_path)
                provenance = (
                    read(run / "provenance.json")
                    if (run / "provenance.json").is_file()
                    else {}
                )
                tables["costs"].append(
                    base
                    | {
                        "status": status,
                        "parameters_by_module": system.get("parameters"),
                        "parameters_optimized": qualification_records.get(
                            "construction", {}
                        )
                        .get("conditions", {})
                        .get(condition, {})
                        .get("parameters_optimized"),
                        "persistent_state_bytes": system.get("persistent_state_bytes"),
                        "latency_seconds": system.get("decision_latency_seconds"),
                        "peak_gpu_bytes": system.get("peak_gpu_bytes"),
                        "training_seconds": system.get("runtime_seconds"),
                        "sessions": system.get("sessions"),
                        "hardware": provenance.get("gpu"),
                        "provenance": provenance,
                        "memory_writes": None,
                        "eviction": None,
                    }
                )
            for split in contract.evaluation.splits:
                for history in ("retained", "current-token"):
                    path = (
                        run
                        / "eval"
                        / evaluation_directory(split, history, "endpoint")
                        / "benchmark_results.json"
                    )
                    if not path.is_file():
                        continue
                    raw = read(path)
                    if "partial_task_cap" in raw:
                        continue
                    runs, loaded_events = read_benchmark_results(path, [contract])
                    if (
                        len(runs) != 1
                        or runs[0].checkpoint != "policy_epoch_999"
                        or budget != 8_000_000
                    ):
                        raise ContractError(
                            "Match-pattern figures require measured 8M endpoint panels."
                        )
                    record = runs[0]
                    if (
                        record.corpus_sha256 != corpus_hash
                        or not record.checkpoint_sha256
                    ):
                        raise ContractError(
                            "Evaluation is missing its frozen corpus/model identity."
                        )
                    key = (condition, seed)
                    if (
                        hashes.setdefault(key, record.checkpoint_sha256)
                        != record.checkpoint_sha256
                    ):
                        raise ContractError(
                            "IID/binding panels do not share the same frozen model."
                        )
                    panels.setdefault((split, condition, history), []).extend(
                        loaded_events
                    )
                    tables["seed_scores"].append(
                        base
                        | {
                            "split": split,
                            "history": history,
                            ("accuracy"): float(
                                np.mean([e.numerator for e in loaded_events])
                            ),
                            "checkpoint_sha256": record.checkpoint_sha256,
                        }
                    )
            # Development-selected companions remain separate from the 8M
            # endpoint tables, including when both happen to use label 999.
            if development.is_file():
                selection = read(development)
                epoch = selection["selected_epoch"]
                name = (
                    "initial_checkpoint.pt" if epoch == -1 else f"policy_epoch_{epoch}"
                )
                expected_hash = next(
                    (
                        r.get("checkpoint_sha256")
                        for r in selection["series"]
                        if r["epoch"] == epoch
                    ),
                    None,
                )
                for split in contract.evaluation.splits:
                    path = (
                        run
                        / "eval"
                        / evaluation_directory(split, "retained")
                        / "benchmark_results.json"
                    )
                    if not path.is_file() or "partial_task_cap" in read(path):
                        continue
                    runs, selected_events = read_benchmark_results(path, [contract])
                    if (
                        len(runs) != 1
                        or runs[0].checkpoint != name
                        or not expected_hash
                        or runs[0].checkpoint_sha256 != expected_hash
                        or runs[0].corpus_sha256 != corpus_hash
                    ):
                        raise ContractError("Selected companion identity mismatch.")
                    tables["selected_companions"].append(
                        base
                        | {
                            "split": split,
                            "checkpoint_rule": "selected",
                            "checkpoint": name,
                            "checkpoint_sha256": expected_hash,
                            "accuracy": float(
                                np.mean([e.numerator for e in selected_events])
                            ),
                            "role": "development-selected companion",
                        }
                    )
    for source_key, destination, metric in (
        ("native_returns", "native_mean", "native_return"),
        ("development", "development_mean", "accuracy"),
    ):
        groups = {
            (r["condition"], r.get("panel", "development"), r["charged_calls"])
            for r in tables[source_key]
        }
        for condition, panel, calls in sorted(groups):
            mean_rows = [
                r
                for r in tables[source_key]
                if (r["condition"], r.get("panel", "development"), r["charged_calls"])
                == (condition, panel, calls)
            ]
            if {r["seed"] for r in mean_rows} == set(study.training_seeds):
                tables[destination].append(
                    {
                        "condition": condition,
                        "panel": panel,
                        "charged_calls": calls,
                        metric: float(np.mean([r[metric] for r in mean_rows])),
                        "statistic": "all-three-seed mean",
                    }
                )
    for split in contract.evaluation.splits:
        for condition in study.cells(contract):
            events = panels.get((split, condition, "retained"), [])
            cells = cell_values(events, "exact_accuracy")
            complete = {s for s, _, _ in cells} == set(study.training_seeds)
            if not complete:
                tables["endpoints"].append(
                    {"condition": condition, "split": split, "status": "incomplete"}
                )
                continue
            estimate, lower, upper = stratified_interval(cells, events)
            _, crossed_lower, crossed_upper = stratified_interval(
                cells, events, crossed=True
            )
            tables["endpoints"].append(
                {
                    "condition": condition,
                    "split": split,
                    "status": "complete",
                    "accuracy": estimate,
                    "conditional_lower": lower,
                    "conditional_upper": upper,
                    "crossed_lower": crossed_lower,
                    "crossed_upper": crossed_upper,
                }
            )
            removed = cell_values(
                panels.get((split, condition, "current-token"), []), "exact_accuracy"
            )
            if set(removed) == set(cells):
                difference = {k: cells[k] - removed[k] for k in cells}
                mean, lo, hi = stratified_interval(difference, events)
                tables["interventions"].append(
                    {
                        "condition": condition,
                        "split": split,
                        "intervention": "query-only-current-token",
                        "effect": mean,
                        "lower": lo,
                        "upper": hi,
                    }
                )
        left_events = panels.get((split, "full_dual_relational", "retained"), [])
        left = cell_values(left_events, "exact_accuracy")
        for comparator in ("full_context", "full_dual_content", "full_gru"):
            right = cell_values(
                panels.get((split, comparator, "retained"), []), "exact_accuracy"
            )
            base_contrast = {
                "split": split,
                "left": "full_dual_relational",
                "right": comparator,
                "role": "companion" if comparator == "full_gru" else "primary",
                "practical_effect": 0.05,
            }
            if (
                not left
                or set(left) != set(right)
                or {s for s, _, _ in left} != set(study.training_seeds)
            ):
                tables["contrasts"].append(base_contrast | {"status": "incomplete"})
                continue
            difference = {k: left[k] - right[k] for k in left}
            mean, lo, hi = stratified_interval(difference, left_events)
            _, crossed_lo, crossed_hi = stratified_interval(
                difference, left_events, crossed=True
            )
            tables["contrasts"].append(
                base_contrast
                | {
                    "status": "complete",
                    "effect": mean,
                    "conditional_lower": lo,
                    "conditional_upper": hi,
                    "crossed_lower": crossed_lo,
                    "crossed_upper": crossed_hi,
                    "per_seed": {
                        s: float(
                            np.mean([v for k, v in difference.items() if k[0] == s])
                        )
                        for s in study.training_seeds
                    },
                }
            )
    for split in contract.evaluation.splits:
        retained_events = panels.get((split, "full_dual_relational", "retained"), [])
        retained_cells = cell_values(retained_events, "exact_accuracy")
        for mode in ("relation-zero", "channel-zero", "causal-permute", "content-norm"):
            changed_events: list[BenchmarkEvent] = []
            for seed in study.training_seeds:
                path = (
                    root
                    / "mechanisms"
                    / f"seed-{seed}"
                    / split
                    / mode
                    / "eval"
                    / evaluation_directory(split, "retained", "endpoint")
                    / "benchmark_results.json"
                )
                if path.is_file():
                    track(path)
                    runs, events_for_mode = read_benchmark_results(path, [contract])
                    if runs[0].checkpoint_sha256 != hashes.get(
                        ("full_dual_relational", seed)
                    ):
                        raise ContractError(
                            "Mechanism and retained panels use different frozen models."
                        )
                    changed_events.extend(events_for_mode)
            changed = cell_values(changed_events, "exact_accuracy")
            base_effect = {
                "condition": "full_dual_relational",
                "split": split,
                "intervention": mode,
            }
            if (
                not retained_cells
                or set(changed) != set(retained_cells)
                or {s for s, _, _ in changed} != set(study.training_seeds)
            ):
                tables["mechanism_effects"].append(
                    base_effect | {"status": "incomplete"}
                )
                continue
            mean, low, high = stratified_interval(
                {k: retained_cells[k] - changed[k] for k in changed}, retained_events
            )
            tables["mechanism_effects"].append(
                base_effect
                | {"status": "complete", "effect": mean, "lower": low, "upper": high}
            )
    for path in sorted((root / "mechanisms").glob("*.json")):
        tables["mechanisms"].append({"path": str(path), "record": read(path)})
    return {
        "tables": tables,
        "gate": gate,
        "sources": sources,
        "metadata": {
            "input_root": str(root),
            "protocol": PROTOCOL,
            "corpus_sha256": corpus_hash,
            "study": str(study_path),
            "seeds": list(study.training_seeds),
            "split_counts": {
                k: len(contract.roster(k)) for k in contract.evaluation.splits
            },
            "bootstrap_draws": 10000,
            "bootstrap_seed": 20260916,
            "accuracy_chance": 0.5,
            "return_chance": 0.0,
            "practical_effect": 0.05,
            "qualification": gate.get("status", "unqualified"),
            "limits": [
                (
                    "IID/binding examples are independent populations; training seeds "
                    "shared."
                ),
                "Scalar native logs have no task-bootstrap bands.",
                "Binding generalization is not unseen-rule inference.",
            ],
        },
    }


def write_match_pattern_report(
    study: Study,
    contract: BenchmarkContract,
    study_root: Path,
    *,
    split: str,
    output: str | Path | None = None,
) -> TierReport:
    """Task-specific tables under the shared tier-report layout, with no door curves."""
    from reasoned_icrl.analysis.plotting import _write_table
    from reasoned_icrl.analysis.tier import TierReport, completeness_rows
    from reasoned_icrl.utils import repository_root

    bundle = load_match_pattern_artifacts(study_root, repository_root())
    root = (
        Path(output)
        if output is not None
        else study_root / "reports" / PROTOCOL / split / "tier"
    )
    root.mkdir(parents=True, exist_ok=True)
    ledger = completeness_rows(study, contract, study_root, split=split)
    for row in ledger:
        path = (
            study_root
            / PROTOCOL
            / str(row["condition"])
            / f"seed-{row['seed']}"
            / "systems.json"
        )
        measured = (
            json.loads(path.read_text()).get("measured", {}) if path.is_file() else {}
        )
        row.update(
            {
                key: measured.get(key)
                for key in (
                    "completed_decisions",
                    "unique_completed_examples",
                    "forced_advances",
                )
            }
        )
    tables = {
        "completeness": _write_table(
            root, "completeness", ledger, title="Match-pattern run completeness"
        )
    }
    for name, records in bundle["tables"].items():
        rows = [r for r in records if "split" not in r or r["split"] == split]
        serial = [
            {
                k: json.dumps(v, sort_keys=True)
                if isinstance(v, (dict, list, tuple))
                else v
                for k, v in r.items()
            }
            for r in rows
        ]
        tables[name] = _write_table(
            root, name, serial, title=f"Match-pattern {split}: {name}"
        )
    complete = tuple(
        r["condition"]
        for r in bundle["tables"]["endpoints"]
        if r["split"] == split and r["status"] == "complete"
    )
    pending = tuple(c for c in study.cells(contract) if c not in complete)
    notes = (
        "Conditional stratified example intervals and crossed seed/example "
        "sensitivity are separate.",
        "IID and bindings are independent populations with shared fitted seeds.",
        "Memory writes, eviction, attempt curves and 99% crossings do not apply.",
    )
    (root / "notes.json").write_text(
        json.dumps(
            bundle["metadata"] | {"sources": bundle["sources"], "notes": notes},
            indent=2,
        )
        + "\n"
    )
    return TierReport(
        root, contract.protocol, split, tables, {}, notes, complete, pending
    )


def write_match_pattern_evidence(root: Path, repository: Path) -> Path:
    """Append a dated execution report, including failures and unallocated cells."""
    from datetime import UTC, datetime

    study = load_summary_memory_study(repository / "configs/match_pattern_8m.yaml")
    contract = study.contract("match_pattern")
    timestamp = datetime.now(UTC)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    report = repository / "docs/gates" / f"MATCH_PATTERN_{stamp}.md"
    lines = [
        f"# Match-pattern execution evidence — {timestamp.isoformat()}",
        "",
        "This dated record preserves the outcome of one authorized allocation. "
        "It does not promote engineering smokes into scientific fits.",
        "",
        f"Study root: `{root}`. Protocol: `{PROTOCOL}`.",
        "",
        "| Condition | Seed | Charged calls | Status | Job | Node | GPU | W&B |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    completed = 0
    for condition in study.cells(contract):
        for seed in study.training_seeds:
            run = root / PROTOCOL / condition / f"seed-{seed}"
            try:
                calls = completed_training_budget(run, contract)
                completed += 1
                status = "measured endpoint"
            except ContractError:
                calls = None
                status = "incomplete" if run.exists() else "unallocated"
            path = run / "provenance.json"
            provenance = json.loads(path.read_text()) if path.is_file() else {}
            wandb = provenance.get("wandb") or {}
            lines.append(
                f"| `{condition}` | {seed} | {calls if calls is not None else ''} | "
                f"{status} | {provenance.get('slurm_job_id', '')} | "
                f"{provenance.get('host', '')} | {provenance.get('gpu', '')} | "
                f"{wandb.get('url', '')} |"
            )
    lines += ["", f"Measured 8M endpoints: **{completed}/12**.", ""]
    for name in (
        "execution/stage.json",
        "execution/forecast.json",
        "qualification/engineering.json",
        "qualification/pilot-seed42.json",
        "qualification/ordinary-three-seed.json",
    ):
        path = root / name
        if path.is_file():
            # The record and its digest make the claim auditable without copying
            # complete engineering telemetry or all per-example observations.
            data = path.read_bytes()
            record = json.loads(data)
            compact = {k: v for k, v in record.items() if k != "single_systems"}
            lines += [
                f"Evidence `{name}`, SHA256 `{hashlib.sha256(data).hexdigest()}`:",
                "",
                "```json",
                json.dumps(compact, indent=2, sort_keys=True),
                "```",
                "",
            ]
    lines += [
        "Exact attempts, scheduler output, failures and resumes remain under "
        "`execution/`. Per-fit provenance and report `completeness.csv` retain "
        "hardware and tracking identities. No incomplete or failed fit is "
        "replaced by a favorable seed.",
        "",
        "Only complete frozen-panel records support endpoint estimates. Native "
        "collection/validation return has chance 0; accuracy has chance 0.5. "
        "IID and binding panels are distinct populations. Passing engineering "
        "checks is not evidence of learning or three-seed qualification.",
        "",
    ]
    with report.open("x") as stream:
        stream.write("\n".join(lines))
    return report
