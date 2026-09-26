"""Run-directory layout and checkpoint-label arithmetic.

The file names a run writes, the measured interaction counters every actor
reports, and the arithmetic that maps AMAGO's zero-based ``policy_epoch_N``
labels to collected decisions and to the collection endpoint. Everything here
is path and integer work over a saved run; nothing imports the learner, so the
study drivers, the job planner and the analysis can read a run's layout
without loading AMAGO or torch.
"""

from __future__ import annotations

from pathlib import Path

from reasoned_icrl.experiments.contracts import ContractError

CHECKPOINT_FILE = "checkpoint.pt"
TRAIN_FILE = "train.csv"
METRICS_FILE = "metrics.json"
PROVENANCE_FILE = "provenance.json"
AMAGO_CONFIG_FILE = "amago_config.gin"
TRAINING_METRICS_FILE = "training_metrics.jsonl"
AMAGO_WORK_DIRECTORIES = ("ckpts", "replay")

COLLECTION_COUNTERS = (
    "charged_calls",
    "physical_actions",
    "reset_only_steps",
    "tasks_started",
    "tasks_completed",
)
"""The measured interaction counters every actor reports."""


def latest_training_epoch(run_directory: str | Path, *, retained: bool = False) -> int:
    """Resolve the greatest readable ordinal AMAGO training checkpoint."""
    root = Path(run_directory)
    weights = root / "ckpts" / "policy_weights"
    training_states = root / "ckpts" / "training_states"
    epochs: list[int] = []
    for path in weights.glob("policy_epoch_*.pt"):
        suffix = path.stem.removeprefix("policy_epoch_")
        if not suffix.isdecimal():
            continue
        epoch = int(suffix)
        full = training_states / f"{root.name}_epoch_{epoch}"
        if (
            full.is_dir()
            and any(full.iterdir())
            and (not retained or (full / "reproduction-complete.json").is_file())
        ):
            epochs.append(epoch)
    if not epochs:
        raise ContractError("No complete AMAGO training checkpoint is available.")
    return max(epochs)


def checkpoint_labels(
    epochs: int, interval: int, *, start_learning: int = 0
) -> tuple[int, ...]:
    """The ``policy_epoch_N`` labels a completed run saves.

    AMAGO numbers epochs from zero, skips the save (with the update) for every
    epoch before ``start_learning``, saves after every later epoch whose label
    is a multiple of the interval, and the trainer saves once more after the
    last epoch: a 1,000-epoch run at interval 50 that starts learning at epoch
    1 holds labels 50, 100, ..., 950 and 999. Label ``N`` has collected
    ``N + 1`` epochs (:func:`collected_at_label`).
    """
    if epochs < 1 or interval < 1 or start_learning < 0:
        raise ContractError("Checkpoint labels need positive epochs and interval.")
    labels = [label for label in range(0, epochs, interval) if label >= start_learning]
    if epochs - 1 not in labels:
        labels.append(epochs - 1)
    return tuple(labels)


def collected_at_label(label: int, *, timesteps_per_epoch: int, actors: int) -> int:
    """Nominal scalar decisions collected when ``policy_epoch_<label>`` was
    saved: ``(label + 1)`` epochs of ``timesteps_per_epoch x actors``. The
    measured count is in the run's telemetry; this is the label's meaning."""
    return (int(label) + 1) * int(timesteps_per_epoch) * int(actors)


def endpoint_label(epochs: int) -> int:
    """The label of the collection endpoint's weights: ``epochs - 1``."""
    if epochs < 1:
        raise ContractError("A run needs at least one epoch.")
    return int(epochs) - 1


def endpoint_training_epoch(run_directory: str | Path, *, epochs: int) -> int:
    """The endpoint label, only when the run actually reached it (R4).

    The endpoint panel reads the weights saved at the collection endpoint,
    never a stopped run's last checkpoint: the run must hold the portable
    ``checkpoint.pt`` the trainer writes on completion, its ``metrics.json``,
    and ``policy_epoch_<epochs - 1>.pt`` (AMAGO's zero-based final epoch).
    Anything less is incomplete.
    """
    root = Path(run_directory)
    label = endpoint_label(epochs)
    weights = root / "ckpts" / "policy_weights" / f"policy_epoch_{label}.pt"
    missing = [
        name
        for name, path in (
            ("checkpoint.pt", root / "checkpoint.pt"),
            ("metrics.json", root / "metrics.json"),
            (weights.name, weights),
        )
        if not path.is_file()
    ]
    if missing:
        raise ContractError(
            f"{root} has not reached the collection endpoint (label {label}, "
            f"{epochs} epochs): missing {', '.join(missing)}; the run is "
            "incomplete and its last checkpoint is never substituted into an "
            "endpoint panel."
        )
    return label


__all__ = [
    "AMAGO_CONFIG_FILE",
    "AMAGO_WORK_DIRECTORIES",
    "CHECKPOINT_FILE",
    "COLLECTION_COUNTERS",
    "METRICS_FILE",
    "PROVENANCE_FILE",
    "TRAINING_METRICS_FILE",
    "TRAIN_FILE",
    "checkpoint_labels",
    "collected_at_label",
    "endpoint_label",
    "endpoint_training_epoch",
    "latest_training_epoch",
]
