"""Fit planning for the summary-memory study (R4/R6).

A tier's fits are its declared group's cells at the study's training seeds,
ordered condition-major so that a condition's three seeds sit together in a
job file and a pack starts them concurrently. Every planned fit reports what
its run directory already holds, so a job file never doubles a completed fit
and a died pack's run is resumed rather than restarted. Nothing here submits,
trains or evaluates; the Slurm launchers consume the written job file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from reasoned_icrl.experiments.artifacts import CHECKPOINT_FILE, endpoint_training_epoch
from reasoned_icrl.experiments.benchmarks import (
    EVENT_KINDS,
    BenchmarkContract,
    Study,
    experiment_config,
)
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.resumes import reconcile_systems_file

FitStatus = Literal["complete", "resumable", "started", "missing"]
"""``complete``: the portable checkpoint exists (the queue skips it);
``resumable``: a resolved config and an AMAGO training state exist (the queue
resumes it); ``started``: a resolved config but no training state yet (the
queue restarts it with ``--overwrite``); ``missing``: nothing on disk."""

FitGroup = Literal["primary", "supplementary", "all"]

DEPENDENCIES: tuple[str, ...] = (
    "train: scripts/train.py summary_memory --benchmark <b> --condition <c> "
    "--seed <s> --device cuda --wandb (checkpoints every 50 epochs)",
    "development series: every scheduled policy_epoch_N on the 64 development "
    "tasks, retained (and the declared interventions at the selected epoch)",
    "endpoint panel: policy_epoch_<epochs> only when the run reached the "
    "full budget; an early stop is incomplete and never substituted",
    "final panel: the 256 final-revised tasks after roster, metric, horizon and "
    "checkpoint rules are recorded; never before every primary fit completes",
)
"""The explicit train -> evaluate dependencies every planned fit carries."""


def evaluation_dependencies(contract: BenchmarkContract) -> tuple[str, ...]:
    """Resolve panel sizes and sources from this study's own frozen contract."""
    panels = ", ".join(
        f"{name}: {len(contract.roster(name))} examples ({split.source})"
        for name, split in contract.evaluation.splits.items()
    )
    return (
        DEPENDENCIES[0],
        f"development: every scheduled checkpoint; {panels}",
        DEPENDENCIES[2],
        "final populations: only after every required primary endpoint; "
        "one shared frozen model per seed across evaluation groups",
    )


@dataclass(frozen=True, slots=True)
class FitPlan:
    """One planned (environment, condition, seed) fit and its disk state."""

    benchmark: str
    protocol: str
    condition: str
    seed: int
    group: str
    run_directory: Path
    status: FitStatus

    @property
    def line(self) -> str:
        """The pool launcher's job line: ``benchmark condition seed``."""
        return f"{self.benchmark} {self.condition} {self.seed}"

    @property
    def pending(self) -> bool:
        return self.status != "complete"


def fit_status(run_directory: Path) -> FitStatus:
    """What one run directory holds, in the pool launcher's terms."""
    if (run_directory / CHECKPOINT_FILE).is_file():
        return "complete"
    if (run_directory / "config.yaml").is_file():
        states = run_directory / "ckpts" / "training_states"
        if any(states.glob("*_epoch_*")):
            return "resumable"
        return "started"
    return "missing"


def plan_fits(
    study: Study,
    contract: BenchmarkContract,
    *,
    repository: str | Path,
    group: FitGroup = "primary",
    seeds: Sequence[int] | None = None,
    conditions: Sequence[str] | None = None,
    output_root: str | Path | None = None,
) -> tuple[FitPlan, ...]:
    """The fits one tier group needs on ``contract``, condition-major.

    ``seeds`` and ``conditions`` restrict the plan to declared training seeds
    and to cells of the requested group; anything else is refused rather than
    silently added. A tier whose contract is still pending (CountRecallMedium
    before R8) cannot be planned.
    """
    tier = study.tier(contract.name) if study.tiered else None
    if tier is not None and tier.contract_pending:
        raise ContractError(
            f"The {contract.name} tier's contract is pending; nothing can be planned."
        )
    if tier is None:
        cells: tuple[str, ...] = (
            study.all_conditions if group != "supplementary" else ()
        )
    else:
        cells = {
            "primary": tier.primary,
            "supplementary": tier.supplementary,
            "all": tier.cells,
        }[group]
    if conditions is not None:
        unknown = [name for name in conditions if name not in cells]
        if unknown:
            raise ContractError(
                f"{unknown} are not {group} cells of {contract.name} "
                f"({', '.join(cells) or 'none'})."
            )
        cells = tuple(name for name in cells if name in conditions)
    chosen = study.training_seeds if seeds is None else tuple(int(s) for s in seeds)
    outside = [seed for seed in chosen if seed not in study.training_seeds]
    if outside:
        raise ContractError(
            f"Seeds {outside} are not training seeds of {study.name} "
            f"({list(study.training_seeds)})."
        )
    plans: list[FitPlan] = []
    for condition in cells:
        for seed in chosen:
            config = experiment_config(
                contract,
                study,
                condition=condition,
                seed=seed,
                repository=repository,
                output_root=output_root,
            )
            plans.append(
                FitPlan(
                    benchmark=contract.name,
                    protocol=contract.protocol,
                    condition=condition,
                    seed=seed,
                    group=tier.group_of(condition) if tier is not None else "primary",
                    run_directory=config.run_directory,
                    status=fit_status(config.run_directory),
                )
            )
    return tuple(plans)


def write_jobs_file(
    path: str | Path,
    plans: Sequence[FitPlan],
    *,
    header: str = "",
    include_complete: bool = False,
) -> Path:
    """Write the pool launcher's job file: one ``benchmark condition seed`` line
    per pending fit (completed fits are listed as comments unless included)."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if header:
        lines += [f"# {row}" for row in header.splitlines()]
    for plan in plans:
        if plan.pending or include_complete:
            lines.append(f"{plan.line}  # {plan.group}, {plan.status}")
        else:
            lines.append(f"# {plan.line}  # {plan.group}, complete: skipped")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


__all__ = [
    "DEPENDENCIES",
    "FitGroup",
    "FitPlan",
    "FitStatus",
    "fit_status",
    "plan_fits",
    "reconcile_resumed_runs",
    "write_jobs_file",
]


def reconcile_resumed_runs(
    study: Study, contract: BenchmarkContract, root: Path
) -> list[Path]:
    """Reconcile the counters and sessions of every resumed run of the tier
    (primary and supplementary cells) whose ``systems.json`` still reports a
    single session; returns the run directories that changed (R6)."""
    changed: list[Path] = []
    for condition in study.cells(contract):
        for seed in study.training_seeds:
            directory = root / contract.protocol / condition / f"seed-{seed}"
            if reconcile_systems_file(
                directory,
                epochs=contract.training.epochs,
                timesteps_per_epoch=contract.training.timesteps_per_epoch,
                actors=contract.environment.parallel_envs,
                reset_capable=EVENT_KINDS.get(contract.name) == "attempt",
            ):
                changed.append(directory)
    return changed


def completed_training_budget(run_directory: Path, contract: BenchmarkContract) -> int:
    """Require endpoint artifacts and measured training calls."""
    endpoint_training_epoch(run_directory, epochs=contract.training.epochs)
    try:
        metrics = json.loads((run_directory / "metrics.json").read_text())
        measured = json.loads((run_directory / "systems.json").read_text())["measured"]
        expected = (
            contract.training.epochs
            * contract.training.timesteps_per_epoch
            * contract.environment.parallel_envs
        )
        charged = measured["charged_calls"]
        physical = measured["physical_actions"]
        reset = measured["reset_only_steps"]
        if (
            metrics["status"] != "trained"
            or type(charged) is not int
            or charged != expected
            or charged != physical + reset
            or metrics["scalar_training_transitions"] != expected
        ):
            raise ValueError("counter mismatch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ContractError(
            f"{run_directory}: incomplete or inconsistent measured training budget"
        ) from error
    return charged


def require_primary_training_complete(
    study: Study, contract: BenchmarkContract, root: Path
) -> None:
    """Guard held-out evaluation against missing, early and miscounted primary fits."""
    pending: list[str] = []
    for condition in study.primary_cells(contract):
        for seed in study.training_seeds:
            directory = root / contract.protocol / condition / f"seed-{seed}"
            try:
                completed_training_budget(directory, contract)
            except ContractError as error:
                pending.append(str(error))
    if pending:
        raise ContractError(
            "Every primary fit must complete its measured training budget "
            "before final evaluation/report: " + "; ".join(pending)
        )
