"""The tier report of the 8M study from saved records (R4/R6).

One call writes, for one protocol and split, the completeness ledger of every
declared cell and seed, the cell estimates of every saved panel (the exact
endpoint primary and the development-selected companion, each history mode
apart), the tier's predeclared contrast family with seed differences and
dispositions, the history-dependence contrasts, the complete-record curves
(cumulative successes by charged call, success by attempt with risk sets,
successes per fixed call interval, first-success times with censoring), the
costs per GPU model, and a notes file naming the resampling seed and
construction. Nothing here trains, evaluates or reads a checkpoint; a missing
cell is a pending row and an incomplete run is marked, never scored.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from reasoned_icrl.analysis.plotting import _plot_series, _write_table, condition_style
from reasoned_icrl.analysis.runs import (
    RunRecords,
    collect,
    complete_events,
    cost_by_gpu,
)
from reasoned_icrl.analysis.statistics import (
    SIGN_FLIP_MINIMUM_P,
    attempt_curves,
    concentration_flip_curves,
    concentration_retrieval_rows,
    count_recall_query_curves,
    count_recall_segment_rows,
    cumulative_curves,
    estimates,
    first_event_times,
    history_contrasts,
    interval_successes,
    tier_contrasts,
)
from reasoned_icrl.experiments.artifacts import (
    checkpoint_labels,
    collected_at_label,
    endpoint_label,
)
from reasoned_icrl.experiments.benchmarks import BenchmarkContract, Study
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import (
    SUMMARY_CLEARED,
    history_modes,
)
from reasoned_icrl.experiments.records import BenchmarkEvent
from reasoned_icrl.experiments.resumes import gpu_chain, session_hours
from reasoned_icrl.experiments.summary_memory.jobs import fit_status
from reasoned_icrl.experiments.summary_memory.revised import (
    read_run_development,
)

TIER_TABLES = (
    "completeness",
    "cells",
    "contrasts",
    "interventions",
    "cumulative",
    "attempts",
    "intervals",
    "first_success",
    "costs",
)


@dataclass(frozen=True, slots=True)
class TierReport:
    """What one call wrote."""

    root: Path
    protocol: str
    split: str
    tables: Mapping[str, Path]
    figures: Mapping[str, Path]
    notes: tuple[str, ...]
    complete_primary_cells: tuple[str, ...]
    pending_primary_cells: tuple[str, ...]


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else None


def _seeds(values: Mapping[int, float], *, signed: bool = True) -> str:
    form = "{seed}: {value:+.3f}" if signed else "{seed}: {value:.3f}"
    return ", ".join(
        form.format(seed=seed, value=value) for seed, value in sorted(values.items())
    )


def _seed_intervals(lower: Mapping[int, float], upper: Mapping[int, float]) -> str:
    return "; ".join(
        f"{seed}: [{lower[seed]:+.3f}, {upper[seed]:+.3f}]" for seed in sorted(lower)
    )


def _sessions(systems: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = systems.get("sessions")
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return []
    return [cast(Mapping[str, object], s) for s in raw if isinstance(s, Mapping)]


def _session_text(session: Mapping[str, object]) -> str:
    labels = session.get("labels")
    span = (
        f"{labels[0]}-{labels[1]}"
        if isinstance(labels, Sequence) and len(labels) == 2
        else "?"
    )
    return f"{session.get('gpu') or 'unknown'} labels {span} {_hours_text(session)}"


def _hours_text(session: Mapping[str, object]) -> str:
    value = session.get("hours", session.get("hours_to_last_saved_label"))
    return "? h" if value is None else f"{float(value):.2f} h"  # type: ignore[arg-type]


def _resumes(directory: Path) -> str:
    """The run's resumes (R6), one clause per record, empty when uninterrupted.

    Each clause names the label resumed from and, for a lenient resume, how
    many of the checkpoint's replay files the FIFO had evicted and were
    dropped; an exact resume says so. Read from ``resumes.jsonl`` so a running
    resumed fit shows it before its ``systems.json`` exists.
    """
    path = directory / "resumes.jsonl"
    if not path.is_file():
        return ""
    clauses: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        deviation = entry.get("replay_deviation")
        clause = f"label {entry.get('resumed_label')}"
        if isinstance(deviation, Mapping):
            clause += (
                f": {deviation.get('missing_files')} of "
                f"{deviation.get('expected_files')} replay files evicted and dropped"
            )
        else:
            clause += ": exact"
        if entry.get("gpu"):
            clause += f" ({entry['gpu']}, job {entry.get('slurm_job_id')})"
        clauses.append(clause)
    return "; ".join(clauses)


def completeness_rows(
    study: Study, contract: BenchmarkContract, study_root: Path, *, split: str
) -> list[dict[str, object]]:
    """One row per declared cell and seed: training state, measured counters,
    development record, saved panels, resumes and provenance."""
    plan = study.tier(contract.name)
    labels = checkpoint_labels(
        contract.training.epochs,
        contract.training.checkpoint_interval,
        start_learning=contract.training.start_learning_epoch,
    )
    endpoint = endpoint_label(contract.training.epochs)
    rows: list[dict[str, object]] = []
    for condition in plan.compared_cells:
        # A frozen reference cell of a comparator tier is read from the other
        # study's root; its row says so in `group`.
        base = study.cell_root(contract, condition, study_root) / contract.protocol
        for seed in study.training_seeds:
            directory = base / condition / f"seed-{seed}"
            status = fit_status(directory)
            saved = sorted(
                int(p.stem.rsplit("_", 1)[1])
                for p in (directory / "ckpts" / "policy_weights").glob(
                    "policy_epoch_*.pt"
                )
            )
            metrics = _read(directory / "metrics.json") or {}
            systems = _read(directory / "systems.json") or {}
            provenance = _read(directory / "provenance.json") or {}
            measured = systems.get("measured") or {}
            sessions = _sessions(systems)
            development = read_run_development(directory)
            wandb = provenance.get("wandb") or {}
            panels = {
                rule: (
                    directory
                    / "eval"
                    / (
                        f"{split}-retained"
                        if rule == "selected"
                        else f"{split}-retained-{rule}"
                    )
                    / "benchmark_results.json"
                ).is_file()
                for rule in contract.evaluation.panel_rules
            }
            last = saved[-1] if saved else None
            rows.append(
                {
                    "protocol": contract.protocol,
                    "tier": plan.tier,
                    "condition": condition,
                    "group": plan.role_of(condition),
                    "seed": seed,
                    "status": status,
                    "last_saved_label": last,
                    "collected_at_last_label": None
                    if last is None
                    else collected_at_label(
                        last,
                        timesteps_per_epoch=contract.training.timesteps_per_epoch,
                        actors=contract.environment.parallel_envs,
                    ),
                    "endpoint_reached": status == "complete" and endpoint in saved,
                    "scheduled_labels_saved": f"{len(saved)}/{len(labels)}",
                    "measured_charged_calls": measured.get("charged_calls"),
                    "measured_physical_actions": measured.get("physical_actions"),
                    "measured_reset_only_steps": measured.get("reset_only_steps"),
                    "validation_charged_calls": (measured.get("validation") or {}).get(
                        "charged_calls"
                    ),
                    "gradient_steps": metrics.get("gradient_steps"),
                    # A resumed run's hours sum its sessions (systems.json
                    # `sessions`); an uninterrupted run's come from metrics.
                    "training_hours": (
                        session_hours(sessions)
                        if sessions
                        else None
                        if metrics.get("runtime_seconds") is None
                        else float(metrics["runtime_seconds"]) / 3600.0
                    ),
                    "sessions": (
                        "; ".join(_session_text(s) for s in sessions)
                        if sessions
                        else ""
                    ),
                    "development_checkpoints": None
                    if development is None
                    else len(development.series),
                    "development_selected_label": None
                    if development is None
                    else development.selected_epoch,
                    **{
                        f"panel_{rule}_retained": present
                        for rule, present in panels.items()
                    },
                    "gpu": gpu_chain(sessions) if sessions else provenance.get("gpu"),
                    "host": provenance.get("host"),
                    "slurm_job_id": provenance.get("slurm_job_id"),
                    "wandb_url": wandb.get("url"),
                    # R6: a died pack's fits resume from their latest training
                    # state; the ledger says from which label and whether the
                    # FIFO had evicted part of that state's replay buffer.
                    "resumes": _resumes(directory),
                }
            )
    return rows


def _panels(
    study: Study, contract: BenchmarkContract, study_root: Path, *, split: str
) -> dict[tuple[str, str], list[RunRecords]]:
    panels: dict[tuple[str, str], list[RunRecords]] = {}
    for rule in contract.evaluation.panel_rules:
        for history in history_modes(contract.name):
            panels[(history, rule)] = collect(
                study,
                contract,
                study_root,
                split=split,
                history=history,
                checkpoint_rule=rule,
            )
    return panels


def _figure_cumulative(
    root: Path, contract: BenchmarkContract, points: Sequence[Any]
) -> Path | None:
    if not points:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for condition in sorted({p.condition for p in points}):
        rows = [p for p in points if p.condition == condition]
        _plot_series(
            ax,
            condition,
            [p.step for p in rows],
            [p.estimate for p in rows],
            [p.lower for p in rows],
            [p.upper for p in rows],
            markers=False,
        )
    concentration = contract.name == "concentration"
    ax.set_xlabel(
        "cumulative flips"
        if concentration
        else "scored query index"
        if contract.name == "count_recall"
        else "charged evaluation calls (resets included)"
    )
    ax.set_ylabel(
        "matched-pair fraction"
        if concentration
        else "cumulative exact accuracy"
        if contract.name == "count_recall"
        else "cumulative completed doors per task"
    )
    ax.set_title(
        f"{contract.protocol}: adaptation (endpoint, retained)",
        fontsize=10,
    )
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = root / "figure_cumulative.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


SUPPORTED_RISK_FRACTION = 0.5
"""An attempt index is supported when at least this fraction of a condition's
seed x task cells finished it; the attempt figure's success panel stops there."""


def supported_attempt_cutoff(points: Sequence[Any]) -> int:
    """Largest attempt index that every condition supports (at least 1).

    Per condition the cutoff is the largest index whose risk set is at least
    SUPPORTED_RISK_FRACTION of that condition's largest risk set; the figure
    uses the smallest such cutoff so no condition is drawn past its support.
    """
    cutoffs: list[int] = []
    for condition in sorted({p.condition for p in points}):
        rows = [p for p in points if p.condition == condition]
        largest = max((p.risk_set or 0) for p in rows)
        supported = [
            int(p.event_index)
            for p in rows
            if (p.risk_set or 0) >= SUPPORTED_RISK_FRACTION * largest
        ]
        cutoffs.append(max(supported, default=1))
    return max(1, min(cutoffs, default=1))


def budget_attempt_cutoff(contract: BenchmarkContract) -> int:
    """Last attempt index at which a failure can still finish inside the budget.

    A failed attempt costs ``horizon`` physical steps plus one reset-only call,
    so a task that fails every attempt finishes attempt k at call
    k * (horizon + 1) - 1. Past the largest k with that inside ``outer_length``
    every failure is cut off by the outer budget and censored as partial, and
    the success rate among *finished* attempts is 1.0 by construction. Attempt
    tasks only; other event kinds have no attempt budget and return 1.
    """
    if contract.event_kind != "attempt":
        return 1
    horizon = int(contract.environment.horizon)
    outer = int(contract.environment.outer_length)
    return max(1, (outer + 1) // (horizon + 1))


def _figure_attempts(
    root: Path, contract: BenchmarkContract, points: Sequence[Any]
) -> Path | None:
    if not points:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The left panel shows only supported attempts: those reached by at least
    # SUPPORTED_RISK_FRACTION of each condition's seed x task cells. Later
    # attempts exist only for cells that already finished many short, successful
    # attempts, so their conditional success is close to one by construction;
    # and past the budget bound no failure can finish at all, so the rate among
    # finished attempts is exactly one. The panel stops at the earlier of the
    # two cuts; the right panel keeps every attempt with its risk set.
    cutoff = min(supported_attempt_cutoff(points), budget_attempt_cutoff(contract))
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for condition in sorted({p.condition for p in points}):
        rows = sorted(
            (p for p in points if p.condition == condition), key=lambda p: p.event_index
        )
        shown = [p for p in rows if p.event_index <= cutoff]
        _plot_series(
            left,
            condition,
            [p.event_index for p in shown],
            [p.estimate for p in shown],
            [p.lower for p in shown],
            [p.upper for p in shown],
        )
        style = condition_style(condition)
        right.plot(
            [p.event_index for p in rows],
            [p.risk_set or 0 for p in rows],
            color=style.color,
            linestyle=style.linestyle,
            linewidth=1.3,
            label=condition,
        )
    left.set_xlabel(
        f"attempt 1-{cutoff} (risk set >= {SUPPORTED_RISK_FRACTION:.0%} of cells; "
        "a failure can still finish in the budget)"
    )
    left.set_ylabel("door success (finished attempts)")
    left.set_ylim(-0.02, 1.02)
    left.set_xlim(0.5, cutoff + 0.5)
    right.axvline(cutoff + 0.5, color="0.4", linestyle=":", linewidth=1.0)
    right.set_xlabel("attempt (every finished attempt)")
    right.set_ylabel("risk set (seed x task cells reaching it)")
    for ax in (left, right):
        ax.grid(alpha=0.2, linewidth=0.6)
    left.legend(fontsize=7, frameon=False)
    fig.suptitle(
        f"{contract.protocol}: success by attempt and risk sets (endpoint, retained)",
        fontsize=10,
    )
    fig.tight_layout()
    path = root / "figure_attempts.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_tier_report(
    study: Study,
    contract: BenchmarkContract,
    study_root: str | Path,
    *,
    split: str = "development",
    samples: int = 2000,
    seed: int = 0,
    output: str | Path | None = None,
    grid_step: int = 25,
    interval: int = 50,
    qualified: bool | None = None,
) -> TierReport:
    """Write one protocol's tier report under
    ``<study_root>/reports/<protocol>/<split>/tier/`` (or ``output``).

    ``qualified`` is the tier gate's C2 verdict when known (it feeds every
    contrast's disposition); ``None`` leaves the reference unqualified only
    if the gate says so elsewhere. ``seed`` is the resampling seed of every
    interval, recorded in ``notes.json`` beside the construction.
    """
    if contract.name == "match_pattern":
        from reasoned_icrl.analysis.match_pattern_artifacts import (
            write_match_pattern_report,
        )

        return write_match_pattern_report(
            study, contract, Path(study_root), split=split, output=output
        )
    if not study.tiered:
        raise ContractError("The tier report reads a tiered study.")
    study_root = Path(study_root)
    plan = study.tier(contract.name)
    if contract.name in ("concentration", "count_recall") and study.tiered:
        from reasoned_icrl.experiments.summary_memory.jobs import (
            require_primary_training_complete,
        )

        require_primary_training_complete(study, contract, study_root)
        if qualified is None:
            # A comparator tier inherits its environment's qualification through
            # its frozen reference, so the gate is read from the reference root.
            gate_root = study.cell_root(
                contract, plan.qualification_reference, study_root
            )
            gate = _read(gate_root / f"qualification-{contract.protocol}-8m.json") or {}
            qualified = (gate.get("c2") or {}).get("passed")
    root = (
        Path(output)
        if output is not None
        else study_root / "reports" / contract.protocol / split / "tier"
    )
    root.mkdir(parents=True, exist_ok=True)
    metric = contract.evaluation.primary_metric
    delta = plan.practical_effect
    expected = len(study.training_seeds)
    primary_rule = contract.evaluation.panel_rules[0]
    intervention = history_modes(contract.name)[1]
    notes: list[str] = []
    tables: dict[str, Path] = {}

    ledger = completeness_rows(study, contract, study_root, split=split)
    tables["completeness"] = _write_table(
        root,
        "completeness",
        ledger,
        title=f"Completeness — {contract.protocol} ({split})",
        note="One row per declared cell and seed: what the run directory holds "
        "(complete / resumable / started / missing), the last saved label and "
        "its nominal collection, the measured collection and validation clocks, "
        "the development record and the saved panels. A cell is scored only from "
        "saved panels; an incomplete run is never substituted.",
    )
    incomplete = [
        f"{row['condition']} seed {row['seed']}: {row['status']}"
        for row in ledger
        if row["group"] == "primary" and row["status"] != "complete"
    ]
    if incomplete:
        notes.append("incomplete primary fits: " + "; ".join(incomplete))

    panels = _panels(study, contract, study_root, split=split)
    every: list[BenchmarkEvent] = []
    for records in panels.values():
        every.extend(complete_events(records))
    complete_cells = {
        (history, rule, record.condition, record.seed)
        for (history, rule), records in panels.items()
        for record in records
        if record.run is not None and not record.partial
    }
    primary_complete = tuple(
        c
        for c in plan.primary
        if all(
            ("retained", primary_rule, c, s) in complete_cells
            for s in study.training_seeds
        )
    )
    primary_pending = tuple(c for c in plan.primary if c not in primary_complete)
    if primary_pending:
        notes.append(
            f"{contract.protocol} {split}: pending {primary_rule} records for "
            + ", ".join(primary_pending)
        )

    cell_rows: list[dict[str, object]] = [
        {
            "protocol": row.protocol,
            "split": row.split,
            "history": row.history,
            "checkpoint_rule": row.checkpoint_rule,
            "condition": row.condition,
            "group": plan.role_of(row.condition),
            "metric": metric,
            "estimate": row.estimate,
            "lower": row.lower,
            "upper": row.upper,
            "per_seed": _seeds(row.per_seed, signed=False),
            "per_seed_task_bootstrap": _seed_intervals(
                row.per_seed_lower, row.per_seed_upper
            ),
            "seeds": row.seeds,
            "tasks": row.tasks,
            "status": "complete" if row.seeds == expected else "partial seeds",
        }
        for row in estimates(every, metric=metric, samples=samples, seed=seed)
    ]
    for condition in primary_pending:
        cell_rows.append(
            {
                "protocol": contract.protocol,
                "split": split,
                "history": "retained",
                "checkpoint_rule": primary_rule,
                "condition": condition,
                "group": "primary",
                "metric": metric,
                "status": "pending",
            }
        )
    tables["cells"] = _write_table(
        root,
        "cells",
        cell_rows,
        title=f"Cells — {contract.protocol}: `{metric}` per cell and panel ({split})",
        note=f"Panels: `{primary_rule}` is the primary checkpoint rule, "
        f"`{contract.evaluation.panel_rules[1]}` the companion; every history "
        "mode is its own panel. Joint seed/task bootstrap 95 % intervals and "
        "per-seed task-bootstrap intervals conditional on each trained seed.",
    )

    contrast_rows: list[dict[str, object]] = []
    for rule in contract.evaluation.panel_rules:
        retained = [
            e for e in every if e.history == "retained" and e.checkpoint_rule == rule
        ]
        for row in tier_contrasts(
            retained,
            plan.primary_contrasts,
            companions=plan.companion_contrasts,
            delta=delta,
            metric=metric,
            expected_seeds=expected,
            qualified=qualified,
            samples=samples,
            seed=seed,
        ):
            contrast_rows.append(
                {
                    "protocol": contract.protocol,
                    "split": split,
                    "checkpoint_rule": rule,
                    "panel_role": "primary" if rule == primary_rule else "companion",
                    "contrast": row.name,
                    "contrast_role": row.role,
                    "left": row.contrast.left,
                    "right": row.contrast.right,
                    "metric": metric,
                    "estimate": row.contrast.estimate,
                    "lower": row.contrast.lower,
                    "upper": row.contrast.upper,
                    "per_seed": _seeds(row.contrast.per_seed),
                    "per_seed_task_bootstrap": _seed_intervals(
                        row.contrast.per_seed_lower, row.contrast.per_seed_upper
                    ),
                    "positive_seeds": row.contrast.positive_seeds,
                    "seeds": row.contrast.seeds,
                    "tasks": row.contrast.tasks,
                    "delta": row.effect_of_interest,
                    "meets_delta": row.meets_effect,
                    "excludes_zero": row.contrast.excludes_zero,
                    "disposition": row.disposition,
                }
            )
    declared = {c.name for c in plan.contrasts}
    reported = {
        str(r["contrast"])
        for r in contrast_rows
        if r["checkpoint_rule"] == primary_rule
    }
    for missing in sorted(declared - reported):
        contrast_rows.append(
            {
                "protocol": contract.protocol,
                "split": split,
                "checkpoint_rule": primary_rule,
                "contrast": missing,
                "metric": metric,
                "disposition": "pending",
            }
        )
    tables["contrasts"] = _write_table(
        root,
        "contrasts",
        contrast_rows,
        title=f"Contrasts — {contract.protocol}: the tier's declared family "
        f"(delta = {delta:g} in `{metric}` units; {split})",
        note="Absolute differences on paired rosters; every seed difference is "
        "shown. `disposition` follows EXPERIMENTS section 7: a consistent "
        "practical gain needs every seed, the point estimate at delta, all seed "
        "differences positive and the joint interval above zero. With three "
        f"seeds a sign-flip test has minimum p = {SIGN_FLIP_MINIMUM_P}; no row "
        "supports a conventional significance claim. A pending row names a "
        "contrast whose cells lack complete records.",
    )

    dependence_rows: list[dict[str, object]] = []
    for rule in contract.evaluation.panel_rules:
        on_rule = [e for e in every if e.checkpoint_rule == rule]
        for right in (intervention, SUMMARY_CLEARED):
            for dependence in history_contrasts(
                on_rule, "retained", right, metric=metric, samples=samples, seed=seed
            ):
                dependence_rows.append(
                    {
                        "protocol": dependence.protocol,
                        "split": dependence.split,
                        "checkpoint_rule": dependence.checkpoint_rule,
                        "condition": dependence.condition,
                        "dependence": f"retained - {right}",
                        "metric": metric,
                        "estimate": dependence.estimate,
                        "lower": dependence.lower,
                        "upper": dependence.upper,
                        "per_seed": _seeds(dependence.per_seed),
                        "seeds": dependence.seeds,
                        "tasks": dependence.tasks,
                    }
                )
    tables["interventions"] = _write_table(
        root,
        "interventions",
        dependence_rows,
        title=f"History dependence — {contract.protocol}: retained minus "
        f"`{intervention}` and minus `{SUMMARY_CLEARED}` ({split})",
        note="Paired inside every seed/task cell; the intervention measures "
        "dependence on retained history, not a specific inference algorithm.",
    )

    endpoint_retained = [
        e
        for e in every
        if e.history == "retained" and e.checkpoint_rule == primary_rule
    ]
    outer = contract.environment.outer_length
    grid = tuple(range(grid_step, outer + 1, grid_step))
    if grid and grid[-1] != outer:
        grid = (*grid, outer)
    attempt_kind = contract.event_kind == "attempt"
    cumulative = (
        cumulative_curves(endpoint_retained, grid=grid, samples=samples, seed=seed)
        if attempt_kind and endpoint_retained
        else ()
    )
    if contract.name == "concentration":
        cumulative = (
            concentration_flip_curves(endpoint_retained, samples=samples, seed=seed)
            if endpoint_retained
            else ()
        )
        tables["retrieval"] = _write_table(
            root,
            "retrieval",
            concentration_retrieval_rows(every),
            title=f"Retrieval and matching efficiency — {contract.protocol}",
            note="Per-seed on-policy opportunities by binding age in flips. "
            "Matching efficiency is twice matched pairs divided by executed flips. "
            "Zero opportunities have no rate; opportunities differ between policies.",
        )
    if contract.protocol == "count-recall-medium":
        cumulative = count_recall_query_curves(
            endpoint_retained, samples=samples, seed=seed
        )
        for measure in ("query_accuracy", "absolute_error"):
            tables[measure] = _write_table(
                root,
                measure,
                [
                    asdict(p) | {"per_seed": _seeds(p.per_seed, signed=False)}
                    for p in count_recall_query_curves(
                        endpoint_retained, measure=measure, samples=samples, seed=seed
                    )
                ],
                title=f"{measure} by scored query — {contract.protocol}",
                note="Bootstrap whole streams with equal task/seed weights; "
                "103 queries are not independent tasks.",
            )
        tables["query_segments"] = _write_table(
            root,
            "query_segments",
            count_recall_segment_rows(every),
            title="C32 query blocks and pre-segment evidence",
            note="Queries 1-32 / 33-64 / 65-96 / 97-103; native return = "
            "2 * accuracy - 1. Opportunity counts accompany retention accuracy.",
        )
    curve_title = (
        "Matched-pair fraction by flip"
        if contract.name == "concentration"
        else "Cumulative exact accuracy by query"
        if contract.name == "count_recall"
        else "Cumulative successes by charged call"
    )
    tables["cumulative"] = _write_table(
        root,
        "cumulative",
        [
            asdict(p) | {"per_seed": _seeds(p.per_seed, signed=False)}
            for p in cumulative
        ],
        title=f"{curve_title} — {contract.protocol} "
        f"({primary_rule}, retained, {split})",
        note=(
            "Every board stays in the denominator through flip 104; natural completion "
            "carries its terminal fraction forward without calls or rewards."
            if contract.name == "concentration"
            else (
                "All 103 scored queries; reset query included, terminal observation "
                "excluded; whole-stream resampling."
            )
            if contract.name == "count_recall"
            else "Every attempt counted at its end; resets are inside the clock. "
            "Units are completed doors per task, not a rate."
        ),
    )
    attempts = (
        attempt_curves(endpoint_retained, samples=samples, seed=seed)
        if (attempt_kind and endpoint_retained)
        else ()
    )
    tables["attempts"] = _write_table(
        root,
        "attempts",
        [
            asdict(p)
            | {
                "risk_set_per_seed": _seeds(
                    {k: float(v) for k, v in p.risk_set_per_seed.items()}, signed=False
                )
            }
            for p in attempts
        ],
        title=f"Success by attempt with risk sets — {contract.protocol} "
        f"({primary_rule}, retained, {split})",
        note="Finished attempts only; the risk set is the number of seed x task "
        "cells that finished that attempt. Late attempts are conditional on the "
        "tasks that reached them.",
    )
    intervals = (
        interval_successes(
            endpoint_retained,
            interval=interval,
            outer_length=outer,
            samples=samples,
            seed=seed,
        )
        if attempt_kind and endpoint_retained
        else ()
    )
    tables["intervals"] = _write_table(
        root,
        "intervals",
        [asdict(p) | {"per_seed": _seeds(p.per_seed, signed=False)} for p in intervals],
        title=f"Successes per {interval}-call interval — {contract.protocol} "
        f"({primary_rule}, retained, {split})",
        note="Fixed denominators: successes per task in each interval of charged "
        f"calls (doors per 100 calls = 2 x the {interval}-call value).",
    )
    first = (
        first_event_times(
            endpoint_retained,
            grid=grid,
            outer_length=outer,
            samples=samples,
            seed=seed,
        )
        if attempt_kind and endpoint_retained
        else ()
    )
    tables["first_success"] = _write_table(
        root,
        "first_success",
        [asdict(p) | {"per_seed": _seeds(p.per_seed, signed=False)} for p in first],
        title=f"First success by call, right-censored — {contract.protocol} "
        f"({primary_rule}, retained, {split})",
        note="The fraction of tasks whose first success ended by each call; "
        "tasks without a success are censored at the budget, never counted as "
        "late successes. The median is the first grid call at which half the "
        "tasks succeeded, blank when censored.",
    )
    tables["costs"] = _write_table(
        root,
        "costs",
        [
            asdict(row)
            | {
                "seeds": ", ".join(str(s) for s in row.seeds),
                "per_seed_training_hours": _seeds(
                    row.per_seed_training_hours, signed=False
                ),
                "slurm_jobs": ", ".join(row.slurm_jobs),
            }
            for row in cost_by_gpu(study, contract, study_root)
        ],
        title=f"Costs per GPU model — {contract.protocol}",
        note="Hours and latencies are reported per GPU model and never pooled "
        "across hardware; all cells run FP32.",
    )

    if contract.name == "concentration":
        scripted_rows: list[dict[str, object]] = []
        for condition in plan.primary:
            for training_seed in study.training_seeds:
                for rule in contract.evaluation.panel_rules:
                    suffix = "" if rule == "selected" else f"-{rule}"
                    path = (
                        study_root
                        / contract.protocol
                        / condition
                        / f"seed-{training_seed}"
                        / "eval"
                        / f"{split}-retained{suffix}"
                        / "scripted_retrieval.json"
                    )
                    probe = _read(path)
                    if probe is None:
                        notes.append(
                            f"missing scripted probe: {condition} "
                            f"seed {training_seed} {rule}"
                        )
                        continue
                    count = probe["opportunity_count"]
                    scripted_rows.append(
                        {
                            "condition": condition,
                            "seed": training_seed,
                            "checkpoint_rule": rule,
                            "opportunities": count,
                            "hits": probe["hits"],
                            "retrieval_rate": probe["hits"] / count if count else None,
                            "charged_calls": probe["charged_calls"],
                            "tasks": len(probe["task_ids"]),
                        }
                    )
        tables["scripted_retrieval"] = _write_table(
            root,
            "scripted_retrieval",
            scripted_rows,
            title="Partner predictions on identical scripted public histories",
            note="Every cell receives the same 104-flip script and board roster. "
            "Predictions do not affect the script; this is a mechanism diagnostic, "
            "separate from on-policy returns.",
        )

    if contract.protocol == "count-recall-medium":
        count_scripted_rows: list[dict[str, object]] = []
        input_rosters: dict[str, dict[tuple[int, int], str]] = {}
        for condition in plan.primary:
            for training_seed in study.training_seeds:
                histories: tuple[str, ...] = ("retained", "current-token")
                if condition == "fixed_summary":
                    histories += ("summary-cleared",)
                for history in histories:
                    path = (
                        study_root
                        / contract.protocol
                        / condition
                        / f"seed-{training_seed}"
                        / "eval"
                        / f"{split}-{history}-endpoint"
                        / "scripted_counts.json"
                    )
                    probe = _read(path)
                    if probe is None or probe.get("partial_task_cap") is not None:
                        notes.append(
                            f"missing shared-stream probe: {condition} "
                            f"seed {training_seed} {history}"
                        )
                        continue
                    queries = probe["queries"]
                    inputs = {
                        (q["task_id"], q["query_index"]): q["input_sha256"]
                        for q in queries
                    }
                    if set(inputs) != {
                        (task, index)
                        for task in contract.roster(split)
                        for index in range(1, 104)
                    }:
                        raise ContractError(
                            "Shared CountRecall probe lacks the full query roster."
                        )
                    if input_rosters and inputs != next(iter(input_rosters.values())):
                        raise ContractError("CountRecall public probe inputs differ.")
                    input_rosters[f"{condition}-{training_seed}-{history}"] = inputs
                    for low, high in ((1, 32), (33, 64), (65, 96), (97, 103)):
                        rows = [q for q in queries if low <= q["query_index"] <= high]
                        older = [q for q in rows if q["count_before_segment"] > 0]
                        count_scripted_rows.append(
                            {
                                "condition": condition,
                                "seed": training_seed,
                                "history": history,
                                "first_query": low,
                                "last_query": high,
                                "queries": len(rows),
                                "accuracy": sum(q["correct"] for q in rows) / len(rows),
                                "absolute_error": sum(q["absolute_error"] for q in rows)
                                / len(rows),
                                "pre_segment_opportunities": len(older),
                                "pre_segment_accuracy": sum(q["correct"] for q in older)
                                / len(older)
                                if older
                                else None,
                            }
                        )
        tables["scripted_counts"] = _write_table(
            root,
            "scripted_counts",
            count_scripted_rows,
            title="Count predictions on identical public answer-zero histories",
            note="Input hashes are checked across all cells, seeds and interventions. "
            "Predictions do not change native RL2 or timer. This is a diagnostic, "
            "separate from policy return, with retention opportunity counts.",
        )

    figures: dict[str, Path] = {}
    figure = _figure_cumulative(root, contract, cumulative)
    if figure is not None:
        figures["cumulative"] = figure
    figure = _figure_attempts(root, contract, attempts)
    if figure is not None:
        figures["attempts"] = figure

    (root / "notes.json").write_text(
        json.dumps(
            {
                "protocol": contract.protocol,
                "split": split,
                "tier": plan.tier,
                "metric": metric,
                "delta": delta,
                "primary_rule": primary_rule,
                "resampling_seed": seed,
                "samples": samples,
                "construction": (
                    "joint seed/task bootstrap (seeds and (task, rollout) units "
                    "drawn with replacement, paired differences formed per cell "
                    "before resampling) plus per-seed task bootstraps with the "
                    "seed fixed; ragged seed/unit bootstrap for attempt curves"
                ),
                "checkpoint_labels": (
                    "AMAGO labels epochs from zero: label N holds N + 1 epochs; "
                    "the endpoint is label epochs - 1"
                ),
                "complete_primary_cells": list(primary_complete),
                "pending_primary_cells": list(primary_pending),
                "notes": notes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TierReport(
        root,
        contract.protocol,
        split,
        tables,
        figures,
        tuple(notes),
        primary_complete,
        primary_pending,
    )


__all__ = ["TIER_TABLES", "TierReport", "completeness_rows", "write_tier_report"]
