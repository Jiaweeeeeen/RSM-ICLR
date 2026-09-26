"""Train one resolved configuration with AMAGO, and reload its policy.

:func:`train_experiment` writes a self-describing run directory: the resolved
``config.yaml``, AMAGO's operative gin, provenance, the initialization audit,
the portable ``checkpoint.pt`` and the training summary. :func:`load_experiment`
rebuilds the same experiment weights-only for evaluation.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import shutil
import socket
import sys
import time
import warnings
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r".*Missing FlashAttention.*",
    module=r"amago\.utils",
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from reasoned_icrl.experiments.artifacts import (  # noqa: E402
    AMAGO_CONFIG_FILE,
    AMAGO_WORK_DIRECTORIES,
    CHECKPOINT_FILE,
    METRICS_FILE,
    PROVENANCE_FILE,
    TRAIN_FILE,
    latest_training_epoch,
)
from reasoned_icrl.experiments.benchmarks import EVENT_KINDS  # noqa: E402
from reasoned_icrl.experiments.config import (  # noqa: E402
    ExperimentConfig,
    dump_config,
    load_resolved_config,
)
from reasoned_icrl.experiments.contracts import (  # noqa: E402
    ContractError,
    architecture_label,
    architecture_uses_history_packet,
    condition_label,
)
from reasoned_icrl.experiments.resumes import (  # noqa: E402
    append_resume,
    read_resumes,
    reconcile_measured,
    resumes_for_systems,
    sessions_from_disk,
)
from reasoned_icrl.runtime.checkpointing import (  # noqa: E402
    policy_checkpoint,
    read_policy_checkpoint,
    validate_checkpoint_architecture,
)
from reasoned_icrl.runtime.devices import (  # noqa: E402
    RuntimeSelection,
    resolve_runtime,
    validate_requested_device,
)
from reasoned_icrl.runtime.environments import environment_builders  # noqa: E402
from reasoned_icrl.runtime.experiment import (  # noqa: E402
    build_experiment,
    cache_measurements,
    carrier_cost,
    dat_audit,
    initialization_audit,
    parameter_count,
    persist_operative_gin,
    start_experiment,
)
from reasoned_icrl.utils import (  # noqa: E402
    file_sha256,
    package_versions,
    source_identity,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_run_directory(
    config: ExperimentConfig, *, overwrite: bool, resume: bool
) -> Path:
    """Protect readable run directories from accidental replacement."""
    run = config.run_directory.resolve()
    output = config.output_root.resolve()
    if not run.is_relative_to(output):
        raise ContractError("Run directory escaped the configured output root.")
    exists = run.is_dir() and any(run.iterdir())
    if resume:
        if not exists:
            raise ContractError("Cannot resume a missing or empty run.")
        return run
    if exists and not overwrite:
        raise ContractError(f"Run already exists: {run}. Use --overwrite explicitly.")
    if exists:
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=True)
    return run


def _write_training_row(path: Path, row: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def _write_provenance(
    config: ExperimentConfig,
    *,
    run: Path,
    actual_device: str,
    selection: RuntimeSelection,
    wandb: Mapping[str, object] | None = None,
) -> Path:
    """Record enough immutable context to audit and reproduce a future run.

    ``host``, ``slurm_job_id``, ``gpu`` and ``wandb`` were added for the
    cluster runs; files written before them load unchanged.
    """
    output = run / PROVENANCE_FILE
    if output.is_file():
        return output
    payload = {
        "schema": "reasoned-icrl-provenance.v1",
        "experiment": config.experiment,
        **source_identity(config.repository),
        "resolved_config_sha256": file_sha256(run / "config.yaml"),
        "protocol": config.environment.benchmark,
        "architecture_id": config.model.architecture_id,
        "architecture_label": architecture_label(config.model.architecture_id),
        "condition": config.condition,
        "condition_label": condition_label(config.condition),
        "training_seed": config.seed,
        "requested_device": selection.requested_device,
        "runtime_selection": selection.metadata(),
        "actual_device": actual_device,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": (
            torch.cuda.get_device_name(0)
            if actual_device == "cuda" and torch.cuda.is_available()
            else None
        ),
        "wandb": None if wandb is None else dict(wandb),
        "packages": package_versions(),
        "command": sys.argv,
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return output


def compact_successful_run(run: Path) -> None:
    """Remove AMAGO internals after a portable checkpoint has been written."""
    for name in AMAGO_WORK_DIRECTORIES:
        directory = run / name
        if directory.is_dir():
            shutil.rmtree(directory)


def _validate_resume_config(config: ExperimentConfig, path: Path) -> None:
    if not path.is_file():
        raise ContractError("Cannot resume a run without its resolved config.yaml.")
    restored = load_resolved_config(path, repository=config.repository)
    # A study root reached through a symbolic link (the projects pool) spells the same
    # run directory two ways: the saved
    # config keeps the repository-relative path, the fresh resolution the
    # physical one. The location must be the same directory; the recipe must
    # be identical once the location is normalised.
    same_location = restored.output_root.resolve() == config.output_root.resolve()
    if not same_location or replace(restored, output_root=config.output_root) != config:
        raise ContractError(
            "Resume configuration differs from the resolved configuration in the run."
        )


def close_experiment(experiment: object) -> None:
    """Close both environments and flush owned trackers before interpreter exit."""
    with ExitStack() as cleanup:
        accelerator = getattr(experiment, "accelerator", None)
        trackers = getattr(accelerator, "trackers", None)
        if accelerator is not None and trackers:
            cleanup.callback(accelerator.end_training)
            if sys.exception() is not None:
                for tracker in trackers:
                    if getattr(tracker, "name", None) == "wandb":
                        # Accelerate's no-argument finish defaults to success.
                        # W&B finish is idempotent; retain failure before it runs.
                        cleanup.callback(tracker.run.finish, exit_code=1)
        # ExitStack still flushes trackers if an environment close raises.
        for name in ("val_envs", "train_envs"):
            environment = getattr(experiment, name, None)
            close = getattr(environment, "close", None)
            if callable(close):
                cleanup.callback(close)


def wandb_run_info(experiment: object) -> dict[str, object] | None:
    """The live W&B run's identity, or None when no tracker was started."""
    accelerator = getattr(experiment, "accelerator", None)
    for tracker in getattr(accelerator, "trackers", ()):
        if getattr(tracker, "name", "") != "wandb":
            continue
        run = getattr(tracker, "run", None)
        if run is None:
            return None
        return {
            key: (None if value is None else str(value))
            for key, value in (
                ("id", getattr(run, "id", None)),
                ("name", getattr(run, "name", None)),
                ("project", getattr(run, "project", None)),
                ("url", getattr(run, "url", None)),
            )
        }
    return None


def _record_resume(
    run: Path,
    config: ExperimentConfig,
    experiment: Any,
    *,
    resumed_label: int,
    allow_missing_replay: bool,
    actual_device: str,
) -> dict[str, object]:
    """Append this resume to ``resumes.jsonl`` and return the entry (R6)."""
    dataset = getattr(experiment, "reasoned_dataset", None)
    deviation = getattr(dataset, "resume_deviation", None)
    entry: dict[str, object] = {
        "schema": "reasoned-icrl-resume.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "resumed_label": int(resumed_label),
        "next_epoch": int(resumed_label) + 1,
        "allow_missing_replay": bool(allow_missing_replay),
        # None for an exact resume; otherwise the evicted files the FIFO had
        # dropped after the checkpoint, listed by name.
        "replay_deviation": None if deviation is None else dict(deviation),
        **source_identity(config.repository),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "actual_device": actual_device,
        "gpu": (
            torch.cuda.get_device_name(0)
            if actual_device == "cuda" and torch.cuda.is_available()
            else None
        ),
        "wandb": wandb_run_info(experiment) if config.tracking.wandb else None,
        "command": sys.argv,
    }
    append_resume(run, entry)
    return entry


def train_experiment(
    config: ExperimentConfig,
    *,
    overwrite: bool = False,
    resume: bool = False,
    allow_missing_replay: bool = False,
) -> dict[str, object]:
    """Train one supported condition and save one evaluation checkpoint.

    ``resume`` continues from the latest AMAGO training state exactly;
    ``allow_missing_replay`` additionally lets a resume drop replay files the
    FIFO evicted after that state was saved (a died pack), recording the
    deviation in ``resumes.jsonl`` and ``systems.json``.
    """
    if allow_missing_replay and not resume:
        raise ContractError("allow_missing_replay applies only to a resume.")
    environment = config.environment
    if environment.name == "mazerunner" and (
        environment.meta_horizon is not None or environment.protocol_size is not None
    ):
        # The larger-maze and repeated-laps axes are evaluation-only contracts
        # derived by the horizon adapter; a fit never runs one.
        raise ContractError(
            "MazeRunner trains one native episode per task on the trained maze; the "
            "larger-maze and repeated-laps contracts are evaluation-only."
        )
    selection = resolve_runtime(config)
    config = selection.config
    print(
        f"runtime: {selection.reason}; "
        f"{config.device}/{config.model.attention_backend}/{config.training.mixed_precision}"
    )
    mapping = config.as_runtime_mapping()
    validate_requested_device(mapping)
    run = prepare_run_directory(config, overwrite=overwrite, resume=resume)
    config_path = run / "config.yaml"
    if resume:
        _validate_resume_config(config, config_path)
    else:
        dump_config(config, config_path)
    seed_everything(config.seed)
    train_env, validation_env = environment_builders(mapping)
    experiment = build_experiment(
        config=mapping,
        run_directory=run,
        make_train_env=train_env,
        make_validation_env=validation_env,
    )
    try:
        actual = start_experiment(experiment, mapping, checkpoint_runtime=True)
        uses_history = architecture_uses_history_packet(config.model.architecture_id)
        persist_operative_gin(experiment, run)
        _write_provenance(
            config,
            run=run,
            actual_device=actual,
            selection=selection,
            wandb=wandb_run_info(experiment) if config.tracking.wandb else None,
        )
        if not resume:
            (run / "initialization.json").write_text(
                json.dumps(
                    {
                        "schema": "darkroom-init.v1",
                        "training_seed": config.seed,
                        **initialization_audit(experiment.policy),
                    },
                    indent=2,
                )
                + "\n"
            )
            torch.save(
                policy_checkpoint(
                    experiment.policy.state_dict(),
                    condition=config.condition,
                    architecture_id=config.model.architecture_id,
                ),
                run / "initial_checkpoint.pt",
            )
        resumed_epoch: int | None = None
        if resume:
            resumed_epoch = latest_training_epoch(
                run, retained=config.training.retain_training_state
            )
            adapter = getattr(experiment, "reasoned_runtime_state", None)
            if adapter is not None:
                adapter.allow_missing_replay = allow_missing_replay
            try:
                experiment.load_checkpoint(resumed_epoch, resume_training_state=True)
            except (RuntimeError, ValueError) as error:
                raise ContractError(
                    "The AMAGO training checkpoint does not match this resolved "
                    f"config: {error}"
                ) from error
            experiment.epoch = resumed_epoch + 1
            _record_resume(
                run,
                config,
                experiment,
                resumed_label=resumed_epoch,
                allow_missing_replay=allow_missing_replay,
                actual_device=actual,
            )
            if experiment.epoch >= experiment.epochs and not (
                uses_history or config.training.retain_training_state
            ):
                raise ContractError(
                    "The latest checkpoint already reaches the configured final epoch."
                )
        if uses_history:
            encoder = experiment.policy.tstep_encoder
            (run / "packet_spec.json").write_text(
                json.dumps(
                    {
                        **asdict(encoder.spec),
                        "sha256": encoder.spec.sha256,
                    },
                    indent=2,
                )
                + "\n"
            )
            if actual == "cuda":
                torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        completed = False
        try:
            if experiment.epoch < experiment.epochs:
                experiment.learn()
                if uses_history or config.training.retain_training_state:
                    experiment.save_checkpoint()
            if config.environment.name == "match_pattern":
                # The final native validation return is a separate evaluation
                # clock, not an extra collection epoch or checkpoint selection.
                experiment.evaluate_val()
            completed = True
        finally:
            if config.training.retain_training_state:
                with (run / "training_sessions.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "runtime_seconds": time.perf_counter() - started,
                                "completed": completed,
                                "resumed_epoch": resumed_epoch,
                                "gradient_steps": experiment.grad_update_counter,
                            }
                        )
                        + "\n"
                    )
        elapsed = time.perf_counter() - started
        checkpoint = run / CHECKPOINT_FILE
        portable_started = time.perf_counter()
        torch.save(
            policy_checkpoint(
                experiment.policy.state_dict(),
                condition=config.condition,
                architecture_id=config.model.architecture_id,
            ),
            checkpoint,
        )
        portable_seconds = time.perf_counter() - portable_started
        row: dict[str, object] = {
            "experiment": config.experiment,
            "condition": config.condition,
            "condition_label": condition_label(config.condition),
            "architecture_id": config.model.architecture_id,
            "architecture_label": architecture_label(config.model.architecture_id),
            "seed": config.seed,
            "device": actual,
            "epochs": config.training.epochs,
            "resumed_epoch": "" if resumed_epoch is None else resumed_epoch,
            "parameters": parameter_count(experiment.policy),
            "runtime_seconds": elapsed,
            "checkpoint": CHECKPOINT_FILE,
            "amago_config": AMAGO_CONFIG_FILE,
            "provenance": PROVENANCE_FILE,
            "gradient_steps": experiment.grad_update_counter,
            "scalar_training_transitions": int(
                experiment.x_axis_metrics()["total_frames"]
            ),
        }
        _write_training_row(run / TRAIN_FILE, row)
        metrics = {
            "experiment": config.experiment,
            "condition": config.condition,
            "condition_label": condition_label(config.condition),
            "architecture_id": config.model.architecture_id,
            "architecture_label": architecture_label(config.model.architecture_id),
            "seed": config.seed,
            "status": "trained",
            "runtime_seconds": elapsed,
            "parameters": row["parameters"],
            "gradient_steps": row["gradient_steps"],
            "scalar_training_transitions": row["scalar_training_transitions"],
            "checkpoint_selection": "final",
            "training_seed_count": 1,
            "checkpoint": CHECKPOINT_FILE,
            "amago_config": AMAGO_CONFIG_FILE,
            "provenance": PROVENANCE_FILE,
        }
        (run / METRICS_FILE).write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if uses_history or config.training.retain_training_state:
            carrier = experiment.policy.traj_encoder
            hidden = experiment.hidden_state
            groups = {
                "tokenizer": parameter_count(experiment.policy.tstep_encoder),
                "fusion": parameter_count(carrier.fusion)
                if uses_history and hasattr(carrier, "fusion")
                else 0,
                "backbone": parameter_count(
                    carrier.backbone if uses_history else carrier
                ),
            }
            groups["heads_and_other"] = parameter_count(experiment.policy) - sum(
                groups.values()
            )
            resumes = read_resumes(run)
            first_provenance: dict[str, object] = (
                json.loads((run / PROVENANCE_FILE).read_text(encoding="utf-8"))
                if (run / PROVENANCE_FILE).is_file()
                else {}
            )
            systems = {
                "parameters": groups,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated()
                if actual == "cuda"
                else 0,
                **cache_measurements(hidden),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()
                if actual == "cuda"
                else 0,
                "checkpoint_seconds": getattr(experiment, "checkpoint_seconds", 0.0)
                + portable_seconds,
                "validation_seconds": getattr(experiment, "validation_seconds", 0.0),
                "attention": dat_audit(experiment.policy, config)
                if config.model.dat is not None
                else {"backend_map": {}}
                if config.condition_spec.trajectory_encoder == "gru"
                else {
                    "backend_map": {
                        str(index): {"content": config.model.attention_backend}
                        for index in range(config.model.layers)
                    },
                },
                "runtime_seconds": elapsed,
                "gradient_steps": experiment.grad_update_counter,
                "context_actions": config.training.max_sequence_length,
                # Renamed from `physical_steps` in R1: this was always the
                # recipe's nominal product and never a measured count. Records
                # written before R1 keep the old key and the old meaning.
                "nominal_collection_product": config.training.epochs
                * config.training.timesteps_per_epoch
                * config.environment.parallel_envs,
                # R6: a resumed run's live counters may cover only the last
                # session (snapshots before R6 carried none); the totals are
                # reconciled from the label arithmetic, and every session's
                # hardware and hours are listed under `sessions`.
                "measured": reconcile_measured(
                    {
                        "scalar_training_transitions": row[
                            "scalar_training_transitions"
                        ],
                        **getattr(experiment, "collection_counters", {}),
                        # R4: AMAGO's in-training validation rollouts, counted
                        # apart from the collection clock.
                        "validation": dict(
                            getattr(experiment, "validation_counters", {})
                        ),
                    },
                    resumes=resumes,
                    epochs=config.training.epochs,
                    timesteps_per_epoch=config.training.timesteps_per_epoch,
                    actors=config.environment.parallel_envs,
                    reset_capable=EVENT_KINDS.get(config.environment.name) == "attempt",
                ),
                "resumes": resumes_for_systems(resumes),
                "sessions": sessions_from_disk(
                    run,
                    resumes,
                    epochs=config.training.epochs,
                    first_gpu=first_provenance.get("gpu"),
                    first_job=first_provenance.get("slurm_job_id"),
                    runtime_seconds=elapsed,
                ),
                # R3: the dual-attention window records the summary reference
                # its W was matched against and the residual bytes; every
                # other cell records None.
                "state_match": (
                    None
                    if config.model.window_match is None
                    else config.model.window_match.to_dict()
                ),
                **carrier_cost(
                    experiment.policy,
                    rows=config.environment.parallel_envs,
                    device=experiment.DEVICE,
                ),
            }
            (run / "epoch_timings.json").write_text(
                json.dumps(getattr(experiment, "history_epoch_timings", []), indent=2)
                + "\n"
            )
            (run / "systems.json").write_text(json.dumps(systems, indent=2) + "\n")
        elif not config.training.retain_training_state:
            compact_successful_run(run)
        return metrics
    finally:
        close_experiment(experiment)


def load_experiment(
    config: ExperimentConfig,
    checkpoint: str | Path | None = None,
    *,
    work_directory: str | Path | None = None,
    persist_configuration: bool = True,
) -> Any:
    """Load a weights-only policy without starting a training/W&B tracker."""
    selection = resolve_runtime(config)
    config = replace(
        selection.config, tracking=replace(selection.config.tracking, wandb=False)
    )
    mapping = config.as_runtime_mapping()
    validate_requested_device(mapping)
    train_env, validation_env = environment_builders(mapping)
    runtime_directory = (
        config.run_directory if work_directory is None else Path(work_directory)
    )
    experiment = build_experiment(
        config=mapping,
        run_directory=runtime_directory,
        make_train_env=train_env,
        make_validation_env=validation_env,
    )
    try:
        start_experiment(experiment, mapping, checkpoint_runtime=False)
        experiment.runtime_selection = selection.metadata()
        if persist_configuration:
            evaluation_runtime = runtime_directory / "eval" / "runtime"
            evaluation_runtime.mkdir(parents=True, exist_ok=True)
            persist_operative_gin(experiment, evaluation_runtime)
            (evaluation_runtime / "selection.json").write_text(
                json.dumps(selection.metadata(), indent=2) + "\n", encoding="utf-8"
            )
        selected = (
            config.run_directory / CHECKPOINT_FILE
            if checkpoint is None
            else Path(checkpoint)
        )
        if not selected.is_file():
            raise ContractError(f"Checkpoint is missing: {selected}.")
        payload = torch.load(
            selected, map_location=experiment.DEVICE, weights_only=True
        )
        state = read_policy_checkpoint(
            payload,
            condition=config.condition,
            architecture_id=config.model.architecture_id,
        )
        validate_checkpoint_architecture(
            state,
            config.model.architecture_id,
            expected_state=experiment.policy.state_dict(),
        )
        try:
            experiment.policy.load_state_dict(state)
        except RuntimeError as error:
            raise ContractError(
                "Checkpoint tensor shapes do not match the resolved model "
                "configuration."
            ) from error
    except Exception:
        close_experiment(experiment)
        raise
    return experiment


def load_initial_checkpoint(experiment: Any, config: ExperimentConfig) -> None:
    """Restore the separately retained, unfitted policy with identity validation."""
    payload = torch.load(
        config.run_directory / "initial_checkpoint.pt",
        map_location=experiment.DEVICE,
        weights_only=True,
    )
    state = read_policy_checkpoint(
        payload,
        condition=config.condition,
        architecture_id=config.model.architecture_id,
    )
    validate_checkpoint_architecture(
        state,
        config.model.architecture_id,
        expected_state=experiment.policy.state_dict(),
    )
    experiment.policy.load_state_dict(state)


def load_selected_checkpoint(experiment: Any, checkpoint: str) -> None:
    """Swap in one scheduled ``policy_epoch_N`` checkpoint, if requested."""
    if checkpoint == CHECKPOINT_FILE:
        return
    if not checkpoint.startswith("policy_epoch_"):
        raise ContractError("Use checkpoint.pt or policy_epoch_N.")
    experiment.load_checkpoint(
        int(checkpoint.removeprefix("policy_epoch_")), resume_training_state=False
    )


__all__ = [
    "close_experiment",
    "compact_successful_run",
    "load_experiment",
    "load_selected_checkpoint",
    "prepare_run_directory",
    "seed_everything",
    "train_experiment",
]
