"""The 8M study's references, development series, C2/C3 and panels (R4/R5).

Everything here reads completed or in-progress runs under the active study
root and writes records beside them; nothing trains. The legacy 4M gate stays
in :mod:`qualification` with its own thresholds, and none of it is reused to
pass or fail the revised protocol.

* :func:`measure_references` records the evaluation-only references of one
  environment on one split before any revised outcome is read: on Key-to-Door
  the random policy at three declared action seeds and the declared
  public-position sweep; elsewhere the contract's random floor. The larger
  mean is the C2 comparison level (EXPERIMENTS section 8).
* :func:`develop_run` scores every scheduled checkpoint of one run on the
  development split with the contract's primary metric, extending an existing
  record as new checkpoints appear while training continues, and writes the
  selected checkpoint's retained and intervention records.
* :func:`decide_revised_c2` and :func:`decide_revised_c3` apply the 0.4M-grid
  learning rule and the history-necessity rule with the tier's delta.
* :func:`finalize_revised_cell` writes the endpoint and selected panels on a
  declared split; the caller decides when the final split may be read.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from reasoned_icrl.analysis.statistics import PlateauVerdict, plateau_screen
from reasoned_icrl.environments.concentration import ConcentrationEnv
from reasoned_icrl.experiments.artifacts import (
    CHECKPOINT_FILE,
    endpoint_label,
    endpoint_training_epoch,
)
from reasoned_icrl.experiments.benchmarks import (
    BenchmarkContract,
    Study,
    saved_config,
)
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import ContractError, condition_spec
from reasoned_icrl.experiments.evaluation import (
    SUMMARY_CLEARED,
    ReferenceRollout,
    concentration_references,
    evaluation_directory,
    evaluation_environment,
    evaluation_replicates,
    random_policy_door_counts,
    sweep_policy_door_counts,
    task_intervention,
    write_evaluation,
    xland_one_rule_references,
)
from reasoned_icrl.experiments.qualification import (
    CheckpointScore,
    TaskDiagnostic,
    measure_reference,
    scheduled_epochs,
)
from reasoned_icrl.experiments.records import (
    BenchmarkEvent,
    Cell,
    CheckpointRule,
    cell_values,
    primary_value,
    read_benchmark_results,
)
from reasoned_icrl.runtime.diagnostics import task_diagnostic
from reasoned_icrl.runtime.rollout import (
    checkpoint_series,
    evaluate,
    scripted_concentration_retrieval,
    scripted_count_recall,
)
from reasoned_icrl.runtime.training import (
    close_experiment,
    load_experiment,
    load_initial_checkpoint,
)

REFERENCES_SCHEMA = "summary-memory-references.v1"
DEVELOPMENT_SCHEMA_V2 = "summary-memory-development.v2"
RUN_DEVELOPMENT_FILE = "development.json"
GATE_SCHEMA_V2 = "summary-memory-tier-gate.v1"
MINIMUM_VERIFIED_PAIRS = 32
MINIMUM_PAIRED_TASKS = 16
C2_CHECKPOINTS: tuple[int, ...] = (900, 950, 999)
"""The saved labels nearest the nominal 7.2/7.6/8M development checkpoints
of a 1,000-epoch run (a label ``N`` holds ``N + 1`` epochs; 999 is the 8M
endpoint). EXPERIMENTS section 3's comparison windows map likewise."""
LATE_WINDOW: tuple[int, ...] = (850, 900, 950, 999)
EARLY_WINDOW: tuple[int, ...] = (650, 700, 750, 800)
"""The plateau screen's windows in saved labels: the nominal 6.8/7.2/7.6/8M
and 5.2/5.6/6/6.4M points, each within one epoch (8,000 decisions) of the
saved label except the endpoint, which is exact."""
"""C3 instrument coverage (EXPERIMENTS section 8): verified pairs of public
histories with identical complete inputs and incompatible correct actions."""


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------------
# References
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReferencePanel:
    """The declared evaluation-only references of one environment and split."""

    protocol: str
    split: str
    metric: str
    task_ids: tuple[int, ...]
    random: ReferenceRollout
    reactive: ReferenceRollout | None
    measured_at: str
    additional: tuple[ReferenceRollout, ...] = ()

    @property
    def random_level(self) -> float:
        return self._mean(self.random)

    @property
    def reactive_level(self) -> float | None:
        return None if self.reactive is None else self._mean(self.reactive)

    @property
    def level(self) -> float:
        """The comparison level: the larger reference mean."""
        levels = [self.random_level]
        if self.reactive is not None:
            levels.append(self._mean(self.reactive))
        levels.extend(self._mean(reference) for reference in self.additional)
        return max(levels)

    def _mean(self, rollout: ReferenceRollout) -> float:
        if self.metric == "exact_accuracy" and rollout.exact_accuracies is not None:
            return float(np.mean(list(rollout.exact_accuracies.values())))
        if self.metric == "pair_fraction" and rollout.pair_fractions is not None:
            return float(np.mean(list(rollout.pair_fractions.values())))
        if self.metric == "doors_completed":
            return rollout.mean_doors_completed
        if self.metric == "success_last2":
            return rollout.mean_attempt_success
        return rollout.mean_door_success_first8

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": REFERENCES_SCHEMA,
            "protocol": self.protocol,
            "split": self.split,
            "metric": self.metric,
            "task_ids": list(self.task_ids),
            "level": self.level,
            "random": _rollout_dict(self.random),
            "reactive": None if self.reactive is None else _rollout_dict(self.reactive),
            "measured_at": self.measured_at,
            "additional": [_rollout_dict(r) for r in self.additional],
        }


def _rollout_dict(rollout: ReferenceRollout) -> dict[str, Any]:
    return {
        "name": rollout.name,
        "generator_seeds": list(rollout.generator_seeds),
        "doors_completed": {str(k): v for k, v in rollout.doors_completed.items()},
        "door_success_first8": {
            str(k): v for k, v in rollout.door_success_first8.items()
        },
        "pair_fractions": rollout.pair_fractions,
        "exact_accuracies": rollout.exact_accuracies,
        "attempt_success": None
        if rollout.attempt_success is None
        else {str(k): v for k, v in rollout.attempt_success.items()},
        "mean_attempt_success": None
        if rollout.attempt_success is None
        else rollout.mean_attempt_success,
        "mean_doors_completed": rollout.mean_doors_completed
        if rollout.doors_completed
        else None,
        "mean_door_success_first8": rollout.mean_door_success_first8
        if rollout.door_success_first8
        else None,
        "charged_calls": rollout.charged_calls,
        "physical_actions": rollout.physical_actions,
    }


def _rollout_from_dict(raw: Mapping[str, Any]) -> ReferenceRollout:
    return ReferenceRollout(
        name=str(raw["name"]),
        generator_seeds=tuple(int(s) for s in raw["generator_seeds"]),
        doors_completed={int(k): float(v) for k, v in raw["doors_completed"].items()},
        door_success_first8={
            int(k): float(v) for k, v in raw["door_success_first8"].items()
        },
        charged_calls=int(raw["charged_calls"]),
        physical_actions=int(raw["physical_actions"]),
        exact_accuracies=None
        if raw.get("exact_accuracies") is None
        else {int(k): float(v) for k, v in raw["exact_accuracies"].items()},
        pair_fractions=None
        if raw.get("pair_fractions") is None
        else {int(k): float(v) for k, v in raw["pair_fractions"].items()},
        attempt_success=None
        if raw.get("attempt_success") is None
        else {int(k): float(v) for k, v in raw["attempt_success"].items()},
    )


def references_file(root: str | Path, protocol: str, split: str) -> Path:
    return Path(root) / "references" / f"{protocol}-{split}.json"


def measure_references(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    split: str = "development",
    task_cap: int | None = None,
    generator_seeds: Sequence[int] = (0, 1, 2),
) -> ReferencePanel:
    """Measure the environment's declared references on one split.

    Key-to-Door: the uniform random policy at the declared action seeds and
    the public-position sweep, both scored in completed doors per 500-call
    task (and, for the historical panel, first-eight success). Other
    environments: the contract's random floor through the legacy measurer,
    with no reactive reference declared yet (Concentration's visible-board
    heuristic is R7 work).
    """
    roster = contract.roster(split)
    if task_cap is not None:
        roster = roster[:task_cap]
    metric = contract.evaluation.primary_metric
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if contract.name == "match_pattern":
        from reasoned_icrl.experiments.match_pattern import load_corpus, pattern_indices

        corpus = load_corpus()
        source = contract.evaluation.splits[split].source
        scores = {
            task: float(len(set(pattern_indices(corpus.example(source, task)))) == 2)
            for task in roster
        }
        random = ReferenceRollout(
            "uniform-binary-exact-expectation",
            (),
            {},
            {},
            0,
            0,
            exact_accuracies={task: 0.5 for task in roster},
        )
        current = ReferenceRollout(
            "public-current-always-nonmatch", (), {}, {}, 0, 0, exact_accuracies=scores
        )
        return ReferencePanel(
            contract.protocol, split, metric, tuple(roster), random, current, stamp
        )
    if contract.name == "dark_key_to_door":
        environment = evaluation_environment(contract, config, split=split, seed=0)
        try:
            random = random_policy_door_counts(
                environment, task_ids=roster, generator_seeds=generator_seeds
            )
            sweep = sweep_policy_door_counts(environment, task_ids=roster)
        finally:
            environment.close()
        return ReferencePanel(
            contract.protocol, split, metric, tuple(roster), random, sweep, stamp
        )
    if contract.name == "concentration":
        environment = evaluation_environment(contract, config, split=split, seed=0)
        try:
            assert isinstance(environment, ConcentrationEnv)
            random, reactive = concentration_references(
                environment, task_ids=roster, generator_seeds=generator_seeds
            )
        finally:
            environment.close()
        return ReferencePanel(
            contract.protocol, split, metric, tuple(roster), random, reactive, stamp
        )
    if contract.name == "xland_one_rule":
        # One environment per declared layout root of the split: the
        # development panel scores two replicates, the final panel three.
        roots = evaluation_replicates(contract, split)
        environments = [
            evaluation_environment(contract, config, split=split, seed=root)
            for root in roots
        ]
        try:
            random, reactive = xland_one_rule_references(
                environments, task_ids=roster, generator_seeds=generator_seeds
            )
        finally:
            for environment in environments:
                environment.close()
        return ReferencePanel(
            contract.protocol, split, metric, tuple(roster), random, reactive, stamp
        )
    if contract.protocol == "count-recall-medium":
        from reasoned_icrl.experiments.count_recall_controls import (
            frozen_count_time_prior,
            score_count_time_prior,
        )

        prior = frozen_count_time_prior(config.run_directory.parents[2])
        source = contract.evaluation.splits[split].source
        time_scores, equality_scores = score_count_time_prior(
            prior, tuple(roster), split=source
        )
        random = ReferenceRollout(
            "uniform-answer-exact-1/27",
            (),
            {},
            {},
            0,
            0,
            exact_accuracies={task: 1 / 27 for task in roster},
        )
        timed = ReferenceRollout(
            "training-only-time-prior.v1",
            (),
            {},
            {},
            len(roster) * 103,
            len(roster) * 103,
            exact_accuracies=time_scores,
        )
        equality = ReferenceRollout(
            "training-only-time-equality-prior.v1",
            (),
            {},
            {},
            0,
            0,
            exact_accuracies=equality_scores,
        )
        return ReferencePanel(
            contract.protocol,
            split,
            metric,
            tuple(roster),
            random,
            timed,
            stamp,
            (equality,),
        )
    floor = measure_reference(contract, config, task_cap=task_cap)
    random = ReferenceRollout(
        name=floor.name,
        generator_seeds=(0,),
        doors_completed={int(t): floor.primary for t in roster},
        door_success_first8={int(t): floor.primary for t in roster},
        charged_calls=0,
        physical_actions=0,
    )
    return ReferencePanel(
        contract.protocol, split, metric, tuple(roster), random, None, stamp
    )


def write_references(root: str | Path, panel: ReferencePanel) -> Path:
    return _write_json(
        references_file(root, panel.protocol, panel.split), panel.as_dict()
    )


def read_references(
    root: str | Path, protocol: str, split: str
) -> ReferencePanel | None:
    path = references_file(root, protocol, split)
    if not path.is_file():
        return None
    raw = _read_json(path)
    if raw.get("schema") != REFERENCES_SCHEMA:
        raise ContractError(f"Unsupported references schema at {path}.")
    reactive = raw.get("reactive")
    return ReferencePanel(
        protocol=str(raw["protocol"]),
        split=str(raw["split"]),
        metric=str(raw["metric"]),
        task_ids=tuple(int(t) for t in raw["task_ids"]),
        random=_rollout_from_dict(raw["random"]),
        reactive=None if reactive is None else _rollout_from_dict(reactive),
        measured_at=str(raw["measured_at"]),
        additional=tuple(_rollout_from_dict(row) for row in raw.get("additional", [])),
    )


# ----------------------------------------------------------------------
# Development series, extended while training continues
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunDevelopment:
    """One run's development series under the contract's metric (R4).

    ``series`` holds every scheduled checkpoint scored so far; the record is
    extended as new checkpoints appear while the fit trains. ``selected`` is
    the best development checkpoint (ties earliest) among those scored, and
    ``modes`` the metric at that checkpoint under every written history mode.
    ``endpoint_reached`` says whether the declared final epoch is among them.
    """

    protocol: str
    condition: str
    seed: int
    metric: str
    run_directory: str
    series: tuple[CheckpointScore, ...]
    selected_epoch: int
    selected_primary: float
    modes: Mapping[str, float]
    endpoint_epoch: int
    endpoint_reached: bool
    development_evaluation_seconds: float
    task_cap: int | None = None

    @property
    def checkpoint(self) -> str:
        return (
            "initial_checkpoint.pt"
            if self.selected_epoch == -1
            else f"policy_epoch_{self.selected_epoch}"
        )

    @property
    def by_epoch(self) -> dict[int, float]:
        return {score.epoch: score.primary for score in self.series}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": DEVELOPMENT_SCHEMA_V2,
            "protocol": self.protocol,
            "condition": self.condition,
            "seed": self.seed,
            "metric": self.metric,
            "run_directory": self.run_directory,
            "series": [asdict(score) for score in self.series],
            "selected_epoch": self.selected_epoch,
            "selected_primary": self.selected_primary,
            "modes": dict(self.modes),
            "endpoint_epoch": self.endpoint_epoch,
            "endpoint_reached": self.endpoint_reached,
            "development_evaluation_seconds": self.development_evaluation_seconds,
            "task_cap": self.task_cap,
        }


def read_run_development(run_directory: str | Path) -> RunDevelopment | None:
    path = Path(run_directory) / RUN_DEVELOPMENT_FILE
    if not path.is_file():
        return None
    raw = _read_json(path)
    if raw.get("schema") != DEVELOPMENT_SCHEMA_V2:
        return None  # a legacy (v1) record belongs to the retired gate
    cap = raw.get("task_cap")
    return RunDevelopment(
        protocol=str(raw["protocol"]),
        condition=str(raw["condition"]),
        seed=int(raw["seed"]),
        metric=str(raw["metric"]),
        run_directory=str(raw["run_directory"]),
        series=tuple(
            CheckpointScore(
                int(row["epoch"]),
                float(row["primary"]),
                float(row["evaluation_seconds"]),
                row.get("charged_calls"),
                row.get("checkpoint_sha256"),
            )
            for row in raw["series"]
        ),
        selected_epoch=int(raw["selected_epoch"]),
        selected_primary=float(raw["selected_primary"]),
        modes={str(k): float(v) for k, v in dict(raw["modes"]).items()},
        endpoint_epoch=int(raw["endpoint_epoch"]),
        endpoint_reached=bool(raw["endpoint_reached"]),
        development_evaluation_seconds=float(raw["development_evaluation_seconds"]),
        task_cap=None if cap is None else int(cap),
    )


def reference_configs(
    study: Study, contract: BenchmarkContract, *, repository: str | Path, device: str
) -> list[ExperimentConfig]:
    """The saved recipes of a comparator tier's frozen reference cells.

    Each reference run's own ``config.yaml`` under the tier's ``reference_root``
    is reloaded (never re-resolved from a roster), so its panels are scored
    on exactly the saved recipe and written beside its existing panels in
    that root; the other study's final reports are untouched. Every declared
    reference cell must exist at every training seed.
    """
    from reasoned_icrl.experiments.config import load_resolved_config

    if not study.tiered:
        raise ContractError("Frozen reference cells belong to a tiered study.")
    plan = study.tier(contract.name)
    configs: list[ExperimentConfig] = []
    for condition in plan.reference_cells:
        base = study.cell_root(contract, condition, repository) / contract.protocol
        for seed in study.training_seeds:
            path = base / condition / f"seed-{seed}" / "config.yaml"
            if not path.is_file():
                raise ContractError(
                    f"Frozen reference {condition} seed {seed} has no saved run at "
                    f"{path.parent}."
                )
            saved = load_resolved_config(path, repository=repository)
            if saved.condition != condition or saved.seed != seed:
                raise ContractError(
                    f"{path} does not describe {condition} seed {seed}."
                )
            configs.append(replace(saved, device=cast(Any, device)))
    return configs


def development_modes(contract: BenchmarkContract, condition: str) -> tuple[str, ...]:
    """The history modes written at a run's selected development checkpoint:
    retained, the task's own intervention (C3 on every cell) and, on a summary
    cell, the summary intervention."""
    modes = ["retained", task_intervention(contract.environment.name)]
    if condition_spec(condition).memory in ("summary", "accumulated"):
        modes.append(SUMMARY_CLEARED)
    return tuple(modes)


def develop_run(
    study: Study,
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    task_cap: int | None = None,
    refresh: bool = False,
) -> RunDevelopment:
    """Score the scheduled checkpoints a run has saved so far and select.

    Checkpoints already in the saved record are not re-evaluated unless
    ``refresh``; new ones are appended, then the selection is redone over the
    whole series and the selected checkpoint's history modes are (re)written
    only when the selection moved. A run still training is developed
    provisionally; ``endpoint_reached`` says whether the final epoch is in.
    """
    if config.condition not in study.cells(contract):
        raise ContractError(
            f"{config.condition!r} is not a cell of {study.name} on {contract.name}."
        )
    if config.seed not in study.training_seeds:
        raise ContractError("Development reads the study's training seeds.")
    run_directory = config.run_directory
    if not (run_directory / "config.yaml").is_file():
        raise ContractError(f"No run at {run_directory}; train it first.")
    config = saved_config(config)
    metric = contract.evaluation.primary_metric
    epochs = scheduled_epochs(run_directory)
    if (
        contract.name == "match_pattern"
        and (run_directory / "initial_checkpoint.pt").is_file()
    ):
        epochs = [-1, *epochs]
    if not epochs:
        raise ContractError("The run has saved no scheduled policy checkpoint yet.")
    existing = None if refresh else read_run_development(run_directory)
    if existing is not None and (
        existing.task_cap != task_cap or existing.metric != metric
    ):
        existing = None
    scored: dict[int, CheckpointScore] = (
        {} if existing is None else {s.epoch: s for s in existing.series}
    )
    pending = [epoch for epoch in epochs if epoch not in scored]
    started = time.perf_counter()
    previous_selected = None if existing is None else existing.selected_epoch
    modes: dict[str, float] = {} if existing is None else dict(existing.modes)
    if pending or existing is None:
        # A run still training has no portable checkpoint.pt yet: seed the
        # evaluator from the earliest saved label (every scheduled label is
        # then loaded by number); a completed run's checkpoint.pt is identical
        # to its last label. Loading touches no replay file of the live run.
        seed_checkpoint = (
            None
            if (run_directory / CHECKPOINT_FILE).is_file()
            else run_directory
            / "ckpts"
            / "policy_weights"
            / f"policy_epoch_{epochs[0]}.pt"
        )
        if epochs[0] == -1 and seed_checkpoint is not None:
            seed_checkpoint = run_directory / "initial_checkpoint.pt"
        experiment = load_experiment(
            config, seed_checkpoint, work_directory=run_directory
        )
        try:
            for score in (
                checkpoint_series(
                    contract, config, experiment, epochs=pending, task_cap=task_cap
                )
                if pending
                else ()
            ):
                scored[score.epoch] = score
            series = tuple(scored[e] for e in sorted(scored))
            selected = max(series, key=lambda s: (s.primary, -s.epoch))
            if selected.epoch != previous_selected or not modes:
                modes = {}
                if selected.epoch == -1:
                    load_initial_checkpoint(experiment, config)
                else:
                    experiment.load_checkpoint(
                        int(selected.epoch), resume_training_state=False
                    )
                for history in development_modes(contract, config.condition):
                    run, events, secondary = evaluate(
                        contract,
                        config,
                        experiment,
                        checkpoint="initial_checkpoint.pt"
                        if selected.epoch == -1
                        else f"policy_epoch_{selected.epoch}",
                        split="development",
                        history=history,
                        task_cap=task_cap,
                    )
                    write_evaluation(
                        run_directory,
                        contract,
                        run,
                        events,
                        secondary,
                        split="development",
                        history=history,
                        task_cap=task_cap,
                    )
                    if contract.name == "concentration" and history == "retained":
                        _write_json(
                            run_directory
                            / "eval"
                            / "development-retained"
                            / "scripted_retrieval.json",
                            scripted_concentration_retrieval(
                                experiment,
                                contract,
                                config,
                                split="development",
                                checkpoint="initial_checkpoint.pt"
                                if selected.epoch == -1
                                else f"policy_epoch_{selected.epoch}",
                                task_cap=task_cap,
                            ),
                        )
                    if contract.protocol == "count-recall-medium":
                        _write_json(
                            run_directory
                            / "eval"
                            / evaluation_directory("development", history, "selected")
                            / "scripted_counts.json",
                            scripted_count_recall(
                                experiment,
                                contract,
                                config,
                                split="development",
                                checkpoint="initial_checkpoint.pt"
                                if selected.epoch == -1
                                else f"policy_epoch_{selected.epoch}",
                                history=history,
                                task_cap=task_cap,
                            ),
                        )
                    modes[history] = primary_value(events, metric)
        finally:
            close_experiment(experiment)
    else:
        series = tuple(scored[e] for e in sorted(scored))
        selected = max(series, key=lambda s: (s.primary, -s.epoch))
    record = RunDevelopment(
        protocol=contract.protocol,
        condition=config.condition,
        seed=config.seed,
        metric=metric,
        run_directory=str(run_directory),
        series=series,
        selected_epoch=int(selected.epoch),
        selected_primary=float(selected.primary),
        modes=modes,
        endpoint_epoch=endpoint_label(config.training.epochs),
        endpoint_reached=endpoint_label(config.training.epochs) in scored
        and (run_directory / CHECKPOINT_FILE).is_file(),
        development_evaluation_seconds=round(
            (0.0 if existing is None else existing.development_evaluation_seconds)
            + time.perf_counter()
            - started,
            2,
        ),
        task_cap=task_cap,
    )
    _write_json(run_directory / RUN_DEVELOPMENT_FILE, record.as_dict())
    return record


def collect_developments(
    study: Study, contract: BenchmarkContract, root: str | Path
) -> tuple[RunDevelopment, ...]:
    """Every v2 development record under ``<root>/<protocol>/<cell>/seed-N``."""
    base = Path(root) / contract.protocol
    found: list[RunDevelopment] = []
    for condition in study.cells(contract):
        for directory in sorted(base.glob(f"{condition}/seed-*")):
            record = read_run_development(directory)
            if record is not None:
                found.append(record)
    return tuple(found)


# ----------------------------------------------------------------------
# C2 and C3 under the revised rules
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RevisedC2:
    """EXPERIMENTS section 8: at each of the last three development
    checkpoints the across-seed reference mean exceeds the comparison level
    by at least delta, with at least two of three seeds clearing it."""

    protocol: str
    condition: str
    metric: str
    delta: float
    reference_level: float
    reference_names: tuple[str, ...]
    checkpoints: tuple[int, ...]
    per_seed: Mapping[int, Mapping[int, float]]
    across_seed: Mapping[int, float]
    seeds_clearing: Mapping[int, int]
    expected_seeds: int
    complete: bool
    provisional: bool
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            **{
                k: v
                for k, v in asdict(self).items()
                if k not in ("per_seed", "across_seed", "seeds_clearing")
            },
            "per_seed": {
                str(s): {str(e): v for e, v in row.items()}
                for s, row in self.per_seed.items()
            },
            "across_seed": {str(e): v for e, v in self.across_seed.items()},
            "seeds_clearing": {str(e): v for e, v in self.seeds_clearing.items()},
        }


def decide_revised_c2(
    contract: BenchmarkContract,
    condition: str,
    series: Mapping[int, Mapping[int, float]],
    *,
    references: ReferencePanel,
    delta: float,
    checkpoints: Sequence[int] = C2_CHECKPOINTS,
    expected_seeds: int = 3,
) -> RevisedC2:
    """Decide C2 for the designated reference cell from per-seed series.

    ``checkpoints`` are saved labels: the last three development checkpoints
    of a 1,000-epoch run are labels 900, 950 and 999 (7.208M, 7.608M and 8M
    collected), the nearest saved weights to the nominal 7.2/7.6/8M grid.
    """
    level = references.level
    if contract.name == "match_pattern":
        delta = 0.40  # Admission is independent of the .05 comparison margin.
    required = (
        max(0.90, level + delta) if contract.name == "match_pattern" else level + delta
    )
    per_seed = {s: dict(points) for s, points in series.items()}
    complete_seeds = [
        s for s, points in per_seed.items() if all(e in points for e in checkpoints)
    ]
    reasons: list[str] = []
    across: dict[int, float] = {}
    clearing: dict[int, int] = {}
    for epoch in checkpoints:
        values = [per_seed[s][epoch] for s in complete_seeds]
        if values:
            across[epoch] = float(np.mean(values))
            clearing[epoch] = sum(v >= required for v in values)
    complete = len(complete_seeds) >= expected_seeds
    if not complete:
        reasons.append(
            f"only {len(complete_seeds)} of {expected_seeds} seeds have scored "
            f"every checkpoint in {list(checkpoints)}"
        )
    for epoch in checkpoints:
        if epoch not in across:
            continue
        if across[epoch] < required:
            reasons.append(
                f"epoch {epoch}: across-seed mean {across[epoch]:.3f} below the "
                f"required {required:.3f} (level {level:.3f} + delta {delta:.3f})"
            )
        if clearing[epoch] < min(2, expected_seeds):
            reasons.append(
                f"epoch {epoch}: only {clearing[epoch]} seed(s) clear {required:.3f}"
            )
    names = [references.random.name]
    if references.reactive is not None:
        names.append(references.reactive.name)
    names.extend(reference.name for reference in references.additional)
    return RevisedC2(
        protocol=contract.protocol,
        condition=condition,
        metric=references.metric,
        delta=float(delta),
        reference_level=level,
        reference_names=tuple(names),
        checkpoints=tuple(int(e) for e in checkpoints),
        per_seed=per_seed,
        across_seed=across,
        seeds_clearing=clearing,
        expected_seeds=expected_seeds,
        complete=complete,
        provisional=not complete and bool(complete_seeds),
        passed=bool(complete and not reasons),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class RevisedC3:
    """EXPERIMENTS section 8: instrument coverage plus the reference cell's
    loss under the environment's history reset."""

    protocol: str
    condition: str
    metric: str
    delta: float
    intervention: str
    verified_pairs: int | None
    paired_tasks: int | None
    coverage_passed: bool
    per_seed_dependence: Mapping[int, float]
    dependence: float
    conditional_lower: float
    conditional_upper: float
    positive_seeds: int
    expected_seeds: int
    complete: bool
    passed: bool
    reasons: tuple[str, ...]
    diagnostic: TaskDiagnostic

    def as_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in asdict(self).items() if k != "diagnostic"}
        payload["per_seed_dependence"] = {
            str(s): v for s, v in self.per_seed_dependence.items()
        }
        payload["required_dependence"] = (
            0.10 if self.protocol == "match-pattern-symbolic-v1" else self.delta / 2
        )
        payload["diagnostic"] = self.diagnostic.as_dict()
        return payload


def _conditional_interval(
    cells: Mapping[Cell, float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    """A paired task bootstrap conditional on the trained agents: the seeds
    are fixed and the shared task-index vector is resampled per replicate."""
    seeds = sorted({s for s, _, _ in cells})
    units = sorted({(t, r) for _, t, r in cells})
    matrix = np.asarray(
        [[cells[(s, *u)] for u in units] for s in seeds], dtype=np.float64
    )
    estimate = float(matrix.mean())
    if samples == 0 or matrix.shape[1] == 0:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, matrix.shape[1], size=(samples, matrix.shape[1]))
    draws = matrix[:, picks].mean(axis=(0, 2))
    tail = (1.0 - confidence) / 2.0
    return (
        estimate,
        float(np.quantile(draws, tail)),
        float(np.quantile(draws, 1.0 - tail)),
    )


def decide_revised_c3(
    contract: BenchmarkContract,
    condition: str,
    *,
    retained: Sequence[BenchmarkEvent],
    intervened: Sequence[BenchmarkEvent],
    diagnostic: TaskDiagnostic,
    delta: float,
    expected_seeds: int = 3,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RevisedC3:
    """Decide C3 from saved retained/intervened records and the diagnostic."""
    metric = contract.evaluation.primary_metric
    intervention = task_intervention(contract.environment.name)
    measured = diagnostic.measurements
    pairs = measured.get("verified_pairs")
    tasks = measured.get("tasks_in_verified_pairs")
    coverage = bool(
        pairs is not None
        and tasks is not None
        and pairs >= MINIMUM_VERIFIED_PAIRS
        and tasks >= MINIMUM_PAIRED_TASKS
    )
    reasons: list[str] = []
    if pairs is None or tasks is None:
        reasons.append("the task diagnostic reports no verified-pair coverage")
    elif not coverage:
        reasons.append(
            f"coverage {int(pairs)} verified pairs over {int(tasks)} tasks is below "
            f"the required {MINIMUM_VERIFIED_PAIRS} pairs over {MINIMUM_PAIRED_TASKS} "
            "tasks"
        )
    left = cell_values(retained, metric)
    right = cell_values(intervened, metric)
    if not left or set(left) != set(right):
        raise ContractError(
            "C3 needs the retained and intervened records on one seed/task roster."
        )
    cells = {key: left[key] - right[key] for key in left}
    per_seed: dict[int, float] = {}
    for s in sorted({key[0] for key in cells}):
        per_seed[s] = float(np.mean([v for k, v in cells.items() if k[0] == s]))
    if contract.name == "match_pattern":
        from reasoned_icrl.analysis.match_pattern import stratified_interval

        estimate, lower, upper = stratified_interval(cells, retained)
    else:
        estimate, lower, upper = _conditional_interval(
            cells, samples=samples, confidence=confidence, seed=seed
        )
    positives = sum(v > 0.0 for v in per_seed.values())
    complete = len(per_seed) >= expected_seeds
    if not complete:
        reasons.append(f"only {len(per_seed)} of {expected_seeds} seeds are evaluated")
    required = 0.10 if contract.name == "match_pattern" else delta / 2
    if estimate < required:
        threshold_name = "required" if contract.name == "match_pattern" else "delta/2 ="
        reasons.append(
            f"the {intervention} loss {estimate:+.3f} is below "
            f"{threshold_name} {required:.3f}"
        )
    if positives < min(2, expected_seeds):
        reasons.append(f"only {positives} seed(s) lose under {intervention}")
    if lower <= 0.0:
        reasons.append(
            f"the paired task-bootstrap lower bound {lower:+.3f} does not exceed zero"
        )
    return RevisedC3(
        protocol=contract.protocol,
        condition=condition,
        metric=metric,
        delta=float(delta),
        intervention=intervention,
        verified_pairs=None if pairs is None else int(pairs),
        paired_tasks=None if tasks is None else int(tasks),
        coverage_passed=coverage,
        per_seed_dependence=per_seed,
        dependence=estimate,
        conditional_lower=lower,
        conditional_upper=upper,
        positive_seeds=positives,
        expected_seeds=expected_seeds,
        complete=complete,
        passed=bool(complete and coverage and not reasons),
        reasons=tuple(reasons),
        diagnostic=diagnostic,
    )


# ----------------------------------------------------------------------
# Panels on a declared split
# ----------------------------------------------------------------------


def finalize_revised_cell(
    study: Study,
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    split: str,
    task_cap: int | None = None,
    refresh: bool = False,
    rules: Sequence[CheckpointRule] | None = None,
) -> dict[str, Path]:
    """Write one run's panels on ``split`` under the contract's checkpoint rules.

    The endpoint panel (every history mode of :func:`development_modes`) needs
    the run to have reached the declared final epoch; the selected panel
    (retained) needs a development record. Records that exist are kept unless
    ``refresh``. The caller alone decides when ``split="final"`` may be read.
    """
    if (
        contract.name in ("concentration", "count_recall", "match_pattern")
        and split in ("final", "final-bindings")
        and study.tiered
    ):
        from reasoned_icrl.experiments.summary_memory.jobs import (
            require_primary_training_complete,
        )

        require_primary_training_complete(
            study, contract, config.run_directory.parents[2]
        )
    if config.condition not in study.compared_cells(contract):
        raise ContractError(
            f"{config.condition!r} is not a cell of {study.name} on {contract.name}."
        )
    config = saved_config(config)
    run_directory = config.run_directory
    chosen = (
        tuple(rules)
        if rules is not None
        else tuple(
            cast(CheckpointRule, rule) for rule in contract.evaluation.panel_rules
        )
    )
    plan: list[tuple[str, str, CheckpointRule]] = []
    for rule in chosen:
        if rule == "endpoint":
            epoch = endpoint_training_epoch(
                run_directory, epochs=config.training.epochs
            )
            for history in development_modes(contract, config.condition):
                plan.append((history, f"policy_epoch_{epoch}", "endpoint"))
        elif rule == "selected":
            development = read_run_development(run_directory)
            if development is None:
                raise ContractError(
                    f"No development selection at {run_directory}; develop it first."
                )
            plan.append(("retained", development.checkpoint, "selected"))
        else:
            raise ContractError(f"The revised panels do not use the {rule!r} rule.")
    written: dict[str, Path] = {}
    pending: list[tuple[str, str, str, CheckpointRule]] = []
    for history, name, rule in plan:
        key = f"{history}-{rule}"
        destination = (
            run_directory / "eval" / evaluation_directory(split, history, rule)
        )
        probe_ready = (
            contract.name != "concentration"
            or history != "retained"
            or (destination / "scripted_retrieval.json").is_file()
        )
        if contract.protocol == "count-recall-medium":
            probe_ready = (destination / "scripted_counts.json").is_file()
        if (
            not refresh
            and probe_ready
            and (destination / "benchmark_results.json").is_file()
        ):
            written[key] = destination
            continue
        pending.append((key, history, name, rule))
    if not pending:
        return written
    for _, _, name, rule in pending:
        # A panel that exists is kept as it is; only a panel still to be
        # written needs its checkpoint on disk (intermediate policies are
        # pruned once a report is saved).
        weights = run_directory / "ckpts" / "policy_weights" / f"{name}.pt"
        if not weights.is_file():
            raise ContractError(
                f"{run_directory}: the {rule} panel needs {weights.name}, which is "
                "no longer saved; restrict the pass to the panels whose weights "
                "exist (--panel-rules endpoint)."
            )
    experiment = load_experiment(config, work_directory=run_directory)
    try:
        loaded: str | None = None
        for key, history, name, rule in pending:
            if loaded != name:
                if name == "initial_checkpoint.pt":
                    load_initial_checkpoint(experiment, config)
                else:
                    experiment.load_checkpoint(
                        int(name.rsplit("_", 1)[1]), resume_training_state=False
                    )
                loaded = name
            run, events, secondary = evaluate(
                contract,
                config,
                experiment,
                checkpoint=name,
                split=split,
                history=history,
                task_cap=task_cap,
                checkpoint_rule=rule,
            )
            written[key] = write_evaluation(
                run_directory,
                contract,
                run,
                events,
                secondary,
                split=split,
                history=history,
                task_cap=task_cap,
                checkpoint_rule=rule,
            )
            if contract.protocol == "count-recall-medium":
                _write_json(
                    written[key] / "scripted_counts.json",
                    scripted_count_recall(
                        experiment,
                        contract,
                        config,
                        split=split,
                        checkpoint=name,
                        history=history,
                        task_cap=task_cap,
                    ),
                )
            if contract.name == "concentration" and history == "retained":
                _write_json(
                    written[key] / "scripted_retrieval.json",
                    scripted_concentration_retrieval(
                        experiment,
                        contract,
                        config,
                        split=split,
                        checkpoint=name,
                        task_cap=task_cap,
                    ),
                )
    finally:
        close_experiment(experiment)
    return written


def panel_events(
    run_directory: str | Path,
    contract: BenchmarkContract,
    *,
    split: str,
    history: str,
    rule: CheckpointRule,
) -> tuple[BenchmarkEvent, ...]:
    """The validated events of one saved panel, or none when it is absent."""
    path = (
        Path(run_directory)
        / "eval"
        / evaluation_directory(split, history, rule)
        / "benchmark_results.json"
    )
    if not path.is_file():
        return ()
    if "partial_task_cap" in json.loads(path.read_text(encoding="utf-8")):
        return ()  # a capped roster never qualifies; leave the panel pending
    _, events = read_benchmark_results(path, [contract])
    return events


# ----------------------------------------------------------------------
# The tier gate
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TierGate:
    """One environment's revised qualification on the development split."""

    protocol: str
    reference_condition: str
    metric: str
    delta: float
    references: ReferencePanel
    c2: RevisedC2
    c3: RevisedC3 | None
    plateau: Mapping[str, PlateauVerdict]
    developed: tuple[RunDevelopment, ...]

    @property
    def status(self) -> str:
        if not self.c2.complete:
            return "C2 provisional" if self.c2.provisional else "C2 pending"
        if not self.c2.passed:
            return "C2 fail"
        if self.c3 is None:
            return "C2 pass, C3 pending"
        return "C2 pass, C3 pass" if self.c3.passed else "C2 pass, C3 unqualified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": GATE_SCHEMA_V2,
            "protocol": self.protocol,
            "reference_condition": self.reference_condition,
            "metric": self.metric,
            "delta": self.delta,
            "status": self.status,
            "references": self.references.as_dict(),
            "c2": self.c2.as_dict(),
            "c3": None if self.c3 is None else self.c3.as_dict(),
            "plateau": {k: asdict(v) for k, v in self.plateau.items()},
            "developed": [d.as_dict() for d in self.developed],
        }


def gate_tier(
    study: Study,
    contract: BenchmarkContract,
    root: str | Path,
    *,
    repository: str | Path,
    device: str = "auto",
    task_cap: int | None = None,
    checkpoints: Sequence[int] = C2_CHECKPOINTS,
    late: Sequence[int] = LATE_WINDOW,
    early: Sequence[int] = EARLY_WINDOW,
    samples: int = 2000,
    seed: int = 0,
) -> TierGate:
    """Decide the tier's C2 and C3 from saved development records and the
    declared references, and write the dashboard under ``root``.

    References are measured once per split and reused; development records
    are read, never created here (call :func:`develop_run` as checkpoints
    appear). C3 needs the reference cell's retained and intervention records
    at its selected checkpoints and the diagnostic coverage.
    """
    from reasoned_icrl.experiments.benchmarks import experiment_config

    plan = study.tier(contract.name)
    reference = plan.qualification_reference
    metric = contract.evaluation.primary_metric
    any_config = experiment_config(
        contract,
        study,
        condition=reference,
        seed=study.training_seeds[0],
        repository=repository,
        device=device,
        output_root=root,
    )
    roster = contract.roster("development")
    expected_tasks = tuple(roster[:task_cap] if task_cap else roster)
    references = read_references(root, contract.protocol, "development")
    if task_cap is not None:
        # A capped pass measures in memory only; the recorded file holds the
        # declared full-roster references measured before any fit was read.
        if references is None or references.task_ids != expected_tasks:
            references = measure_references(contract, any_config, task_cap=task_cap)
    elif references is None or references.task_ids != expected_tasks:
        references = measure_references(contract, any_config, task_cap=None)
        write_references(root, references)
    developed = collect_developments(study, contract, root)
    reference_runs = {d.seed: d for d in developed if d.condition == reference}
    c2 = decide_revised_c2(
        contract,
        reference,
        {s: d.by_epoch for s, d in reference_runs.items()},
        references=references,
        delta=plan.practical_effect,
        checkpoints=checkpoints,
        expected_seeds=len(study.training_seeds),
    )
    plateau = {
        condition: plateau_screen(
            condition,
            {d.seed: d.by_epoch for d in developed if d.condition == condition},
            late=late,
            early=early,
            delta=plan.practical_effect,
            expected_seeds=len(study.training_seeds),
        )
        for condition in sorted({d.condition for d in developed})
    }
    c3: RevisedC3 | None = None
    retained: list[BenchmarkEvent] = []
    intervened: list[BenchmarkEvent] = []
    intervention = task_intervention(contract.environment.name)
    # A capped roster writes partial records and can never qualify; C3 reads
    # the full-roster panels only.
    for development in reference_runs.values() if task_cap is None else ():
        run_directory = Path(development.run_directory)
        retained.extend(
            panel_events(
                run_directory,
                contract,
                split="development",
                history="retained",
                rule="endpoint"
                if contract.protocol
                in ("count-recall-medium", "match-pattern-symbolic-v1")
                else "selected",
            )
        )
        intervened.extend(
            panel_events(
                run_directory,
                contract,
                split="development",
                history=intervention,
                rule="endpoint"
                if contract.protocol
                in ("count-recall-medium", "match-pattern-symbolic-v1")
                else "selected",
            )
        )
    if retained and intervened and task_cap is None:
        diagnostic = task_diagnostic(contract, saved_config(any_config), task_cap=None)
        c3 = decide_revised_c3(
            contract,
            reference,
            retained=retained,
            intervened=intervened,
            diagnostic=diagnostic,
            delta=plan.practical_effect,
            expected_seeds=len(study.training_seeds),
            samples=samples,
            seed=seed,
        )
    gate = TierGate(
        contract.protocol,
        reference,
        metric,
        plan.practical_effect,
        references,
        c2,
        c3,
        plateau,
        developed,
    )
    write_tier_gate(Path(root) / f"qualification-{contract.protocol}-8m.md", gate)
    return gate


def _plateau_label(verdict: PlateauVerdict) -> str:
    if not verdict.complete:
        return "incomplete"
    return "yes" if verdict.still_improving else "no"


def write_tier_gate(path: str | Path, gate: TierGate) -> Path:
    """The revised dashboard: references, C2 per checkpoint and seed, C3,
    the plateau screen and every development series."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    refs = gate.references
    lines = [
        f"# Tier qualification — `{gate.protocol}` (8M revision)",
        "",
        f"Status: **{gate.status}**. Reference cell `{gate.reference_condition}`, "
        f"metric `{gate.metric}`, delta {gate.delta:g}. Every number is a saved "
        "development record; the final roster is never read here.",
        "",
        "## References (development split, evaluation-only)",
        "",
        f"- {refs.random.name}: {refs.random_level:.3f} (action seeds "
        f"{list(refs.random.generator_seeds)}; "
        f"charged calls {refs.random.charged_calls})",
    ]
    if refs.reactive is not None:
        lines.append(
            f"- {refs.reactive.name}: {refs.reactive_level:.3f} "
            f"(charged calls {refs.reactive.charged_calls})"
        )
    for extra in refs.additional:
        lines.append(
            f"- {extra.name}: {refs._mean(extra):.3f} "
            "(same scored public streams; no additional calls)"
        )
    lines += [
        f"- comparison level (larger mean): {refs.level:.3f}; required "
        f"{refs.level + gate.c2.delta:.3f}; measured {refs.measured_at} on "
        f"{len(refs.task_ids)} tasks",
        "",
        "## C2 — sustained learning of the reference cell",
        "",
        "| Checkpoint | Across-seed mean | Seeds clearing | "
        + " | ".join(f"seed {s}" for s in sorted(gate.c2.per_seed))
        + " |",
        "|---|---:|---:|" + "---:|" * len(gate.c2.per_seed),
    ]
    for epoch in gate.c2.checkpoints:
        cells = [
            f"{gate.c2.per_seed[s].get(epoch, float('nan')):.3f}"
            for s in sorted(gate.c2.per_seed)
        ]
        lines.append(
            f"| {epoch} | {gate.c2.across_seed.get(epoch, float('nan')):.3f} | "
            f"{gate.c2.seeds_clearing.get(epoch, 0)}/{gate.c2.expected_seeds} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    lines.append(
        f"- verdict: **{'pass' if gate.c2.passed else 'not passed'}**"
        + (" (provisional: not every seed scored)" if gate.c2.provisional else "")
    )
    lines += [f"  - {reason}" for reason in gate.c2.reasons]
    lines += ["", "## C3 — history necessity", ""]
    if gate.c3 is None:
        lines.append(
            "- pending: the reference cell's retained and intervention records at "
            "its selected checkpoints are not all saved yet"
        )
    else:
        c3 = gate.c3
        lines += [
            f"- coverage: {c3.verified_pairs} verified pairs over {c3.paired_tasks} "
            f"tasks ({'pass' if c3.coverage_passed else 'insufficient'}; "
            f"required {MINIMUM_VERIFIED_PAIRS} over {MINIMUM_PAIRED_TASKS})",
            f"- {c3.intervention} loss: {c3.dependence:+.3f} "
            f"[{c3.conditional_lower:+.3f}, {c3.conditional_upper:+.3f}] "
            f"(paired task bootstrap conditional on the trained seeds); per seed "
            + ", ".join(
                f"{s}: {v:+.3f}" for s, v in sorted(c3.per_seed_dependence.items())
            )
            + f"; {c3.positive_seeds}/{c3.expected_seeds} seeds positive; "
            "required loss = " + str(c3.as_dict()["required_dependence"]),
            f"- verdict: **{'pass' if c3.passed else 'unqualified'}**",
        ]
        lines += [f"  - {reason}" for reason in c3.reasons]
        lines.append(f"- task diagnostic: {c3.diagnostic.statement}")
    lines += ["", "## Development plateau screen", ""]
    lines += [
        "| Condition | Seeds | Late - early (across seeds) | Per seed | "
        "Late range | Still improving | Variable |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for condition, verdict in gate.plateau.items():
        across = verdict.across_seed_difference
        late_range = verdict.late_range
        lines.append(
            f"| `{condition}` | {len(verdict.per_seed_late)} | "
            f"{'-' if across is None else f'{across:+.3f}'} | "
            + ", ".join(
                f"{s}: {v:+.3f}" for s, v in sorted(verdict.per_seed_difference.items())
            )
            + f" | {'-' if late_range is None else f'{late_range:.3f}'} | "
            f"{_plateau_label(verdict)} | {'yes' if verdict.variable else 'no'} |"
        )
    lines += ["", "## Development series", ""]
    lines += [
        "| Condition | Seed | Checkpoints | Selected | Selected value | Modes | "
        "Endpoint reached |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for record in sorted(gate.developed, key=lambda r: (r.condition, r.seed)):
        lines.append(
            f"| `{record.condition}` | {record.seed} | {len(record.series)} | "
            f"e{record.selected_epoch} | {record.selected_primary:.3f} | "
            + ", ".join(f"{k}: {v:.3f}" for k, v in sorted(record.modes.items()))
            + f" | {'yes' if record.endpoint_reached else 'no'} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output.with_suffix(".json"), gate.as_dict())
    return output


__all__ = [
    "C2_CHECKPOINTS",
    "DEVELOPMENT_SCHEMA_V2",
    "EARLY_WINDOW",
    "GATE_SCHEMA_V2",
    "LATE_WINDOW",
    "MINIMUM_PAIRED_TASKS",
    "MINIMUM_VERIFIED_PAIRS",
    "REFERENCES_SCHEMA",
    "RUN_DEVELOPMENT_FILE",
    "ReferencePanel",
    "RevisedC2",
    "RevisedC3",
    "RunDevelopment",
    "TierGate",
    "collect_developments",
    "decide_revised_c2",
    "decide_revised_c3",
    "develop_run",
    "development_modes",
    "finalize_revised_cell",
    "gate_tier",
    "measure_references",
    "panel_events",
    "read_references",
    "read_run_development",
    "reference_configs",
    "references_file",
    "write_references",
    "write_tier_gate",
]
