"""Train and evaluate the summary-memory study.

Every run is one (environment contract, condition, seed) of the study roster
through the shared trainer and evaluator. There is no pilot-seed branch and no
cost ledger: the C2/C3 gate reads the ``raw`` reference's own development
series, and the gate, references and reports are library functions in the
study's ``qualification`` module and :mod:`reasoned_icrl.analysis` that the
per-environment notebooks call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reasoned_icrl.experiments.artifacts import (
    endpoint_training_epoch,
    latest_training_epoch,
)
from reasoned_icrl.experiments.benchmarks import (
    BenchmarkContract,
    Study,
    experiment_config,
    saved_config,
)
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import (
    ALL_HISTORY_MODES,
    write_evaluation,
)
from reasoned_icrl.experiments.horizon import (
    continued_laps,
    continued_streams,
    extended_horizon,
)
from reasoned_icrl.experiments.records import CheckpointRule
from reasoned_icrl.experiments.summary_memory.configs import (
    load_summary_memory_study,
)
from reasoned_icrl.runtime.rollout import evaluate as evaluate_checkpoint
from reasoned_icrl.runtime.training import (
    close_experiment,
    load_experiment,
    load_selected_checkpoint,
    train_experiment,
)
from reasoned_icrl.utils import repository_root


def resolve(
    study: Study,
    *,
    benchmark: str,
    condition: str,
    seed: int,
    device: str = "auto",
    output_root: str | Path | None = None,
    smoke: bool = False,
    wandb: bool = False,
) -> tuple[BenchmarkContract, ExperimentConfig]:
    contract = study.contract(benchmark)
    config = experiment_config(
        contract,
        study,
        condition=condition,
        seed=seed,
        repository=repository_root(),
        device=device,
        output_root=output_root,
        smoke=smoke,
        wandb=wandb,
    )
    return contract, config


def train(
    study: Study,
    *,
    benchmark: str,
    condition: str,
    seed: int,
    device: str = "auto",
    output_root: str | Path | None = None,
    smoke: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    resume_allow_evicted_replay: bool = False,
    preflight: bool = False,
    wandb: bool = False,
) -> Path | dict[str, object]:
    """Resolve a contract into an experiment and run its declared recipe.

    ``resume_allow_evicted_replay`` lets a resume drop the replay files the
    FIFO evicted after the latest training state was saved (a died pack); the
    deviation is recorded in the run (R6). It requires ``resume``.
    """
    if resume_allow_evicted_replay and not resume:
        raise ContractError("--resume-allow-evicted-replay requires --resume.")
    _, config = resolve(
        study,
        benchmark=benchmark,
        condition=condition,
        seed=seed,
        device=device,
        output_root=output_root,
        smoke=smoke,
        wandb=wandb,
    )
    if preflight:
        from reasoned_icrl.runtime.devices import resolve_runtime
        from reasoned_icrl.runtime.experiment import preflight_config

        return preflight_config(resolve_runtime(config).config)
    if config.environment.name == "match_pattern" and not smoke:
        from reasoned_icrl.experiments.summary_memory.match_pattern import (
            require_fit_admission,
        )

        require_fit_admission(config.run_directory.parents[2], condition, seed)
    train_experiment(
        config,
        overwrite=overwrite,
        resume=resume,
        allow_missing_replay=resume_allow_evicted_replay,
    )
    return config.run_directory


def evaluate(
    study: Study,
    *,
    benchmark: str,
    condition: str,
    seed: int,
    device: str = "auto",
    output_root: str | Path | None = None,
    split: str = "development",
    history: str = "retained",
    checkpoint: str = "checkpoint.pt",
    task_cap: int | None = None,
    checkpoint_rule: CheckpointRule = "selected",
    horizon: int | None = None,
    layout_period: int | None = None,
) -> Path:
    """Evaluate one saved checkpoint and write validated benchmark records.

    ``checkpoint_rule="final-epoch"`` is the legacy supplement: it resolves
    the run's last scheduled ``policy_epoch_N`` itself and writes beside the
    selected-checkpoint panel, never over it. ``"endpoint"`` (R4) is the 8M
    study's primary panel: the weights at the declared final epoch, refused
    for a run that stopped early.
    """
    contract, config = resolve(
        study,
        benchmark=benchmark,
        condition=condition,
        seed=seed,
        device=device,
        output_root=output_root,
    )
    if (
        contract.name in ("concentration", "count_recall", "match_pattern", "tmaze")
        and split in ("final", "final-bindings")
        and study.tiered
    ):
        from reasoned_icrl.experiments.summary_memory.jobs import (
            require_primary_training_complete,
        )

        require_primary_training_complete(
            study, contract, config.run_directory.parents[2]
        )
    config = saved_config(config)
    return evaluate_saved(
        contract,
        config,
        split=split,
        history=history,
        checkpoint=checkpoint,
        task_cap=task_cap,
        checkpoint_rule=checkpoint_rule,
        horizon=horizon,
        layout_period=layout_period,
    )


def evaluate_saved(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    split: str,
    history: str = "retained",
    checkpoint: str = "checkpoint.pt",
    task_cap: int | None = None,
    checkpoint_rule: CheckpointRule = "selected",
    horizon: int | None = None,
    layout_period: int | None = None,
    streams: int | None = None,
    laps: int | None = None,
) -> Path:
    """Evaluate one run from its saved recipe (``saved_config`` or a frozen
    reference's own ``config.yaml``) and write its panel. ``laps`` (the MazeRunner
    repeated-laps axis) evaluates the frozen
    endpoint weights on every roster map replayed in laps until that many
    charged calls through
    :func:`~reasoned_icrl.experiments.horizon.continued_laps`; the panel lands
    under the ``-calls<B>`` suffix. ``streams`` (the continued-stream axis) evaluates
    the frozen endpoint
    weights over that many consecutive CountRecall deck pairs through
    :func:`~reasoned_icrl.experiments.horizon.continued_streams`; the panel
    lands under the ``-s<N>`` suffix.

    ``horizon`` (EXPERIMENTS section 5) evaluates the
    endpoint weights over a longer outer task through
    :func:`~reasoned_icrl.experiments.horizon.extended_horizon`; the panel lands
    under the ``-h<H>`` suffix beside the trained-horizon panels and carries
    ``outer_length`` on its run and events.
    """
    native_budget = contract.environment.outer_length
    if horizon is not None:
        if checkpoint_rule not in ("endpoint", "selected"):
            raise ContractError(
                "Extended-horizon panels evaluate the frozen endpoint weights or "
                "the development-selected checkpoint (the secondary rule) only."
            )
        if checkpoint_rule == "selected" and not checkpoint.startswith("policy_epoch_"):
            raise ContractError(
                "An extended-horizon panel under the selected rule names the "
                "development-selected policy_epoch_N checkpoint explicitly."
            )
        contract, config = extended_horizon(contract, config, horizon)
    if streams is not None:
        if checkpoint_rule != "endpoint" or horizon is not None:
            raise ContractError(
                "Continued-stream panels evaluate the frozen endpoint weights over "
                "consecutive deck pairs; they take no horizon."
            )
        contract, config = continued_streams(contract, config, streams)
    if laps is not None:
        if checkpoint_rule != "endpoint" or horizon is not None or streams is not None:
            raise ContractError(
                "Repeated-laps panels evaluate the frozen endpoint weights over a "
                "budget of calls; they take no horizon or streams."
            )
        contract, config = continued_laps(contract, config, laps)
    if layout_period is not None:
        if checkpoint_rule != "endpoint" or (horizon is None and laps is None):
            raise ContractError(
                "Layout-change panels evaluate the frozen endpoint weights over "
                "an extended horizon or a repeated-laps budget only."
            )
        if int(layout_period) < native_budget:
            raise ContractError(
                "The layout period must not fall inside the trained budget: the "
                "first change keeps the 500-call panel as the acceptance base."
            )
    run_directory = config.run_directory
    if checkpoint_rule in ("final-epoch", "endpoint"):
        if checkpoint != "checkpoint.pt":
            raise ContractError(
                f"The {checkpoint_rule} rule resolves the checkpoint itself; do "
                "not pass one."
            )
        epoch = (
            latest_training_epoch(run_directory)
            if checkpoint_rule == "final-epoch"
            else endpoint_training_epoch(run_directory, epochs=config.training.epochs)
        )
        checkpoint = f"policy_epoch_{epoch}"
    elif checkpoint_rule != "selected":
        raise ContractError(f"Unknown checkpoint rule: {checkpoint_rule!r}.")
    experiment = load_experiment(
        config,
        work_directory=run_directory,
        persist_configuration=horizon is None and streams is None and laps is None,
    )
    try:
        load_selected_checkpoint(experiment, checkpoint)
        run, events, secondary = evaluate_checkpoint(
            contract,
            config,
            experiment,
            checkpoint=checkpoint,
            split=split,
            history=history,
            task_cap=task_cap,
            checkpoint_rule=checkpoint_rule,
            layout_period=layout_period,
        )
    finally:
        close_experiment(experiment)
    return write_evaluation(
        run_directory,
        contract,
        run,
        events,
        secondary,
        split=split,
        history=history,
        task_cap=task_cap,
        checkpoint_rule=checkpoint_rule,
        horizon=horizon,
        layout_period=layout_period,
        streams=streams,
        laps=laps,
    )


# ----------------------------------------------------------------------
# Script registration
# ----------------------------------------------------------------------


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study", type=Path, default=None)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--output-root", type=Path, default=None)


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    _common_arguments(parser)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-allow-evicted-replay",
        action="store_true",
        help=(
            "With --resume: drop the replay files the FIFO evicted after the "
            "latest training state (a died pack) and record the deviation."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Read-only resolved config, model and budget audit; no rollout.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Mirror telemetry to Weights & Biases (needs `wandb login`).",
    )


def train_main(args: argparse.Namespace) -> Path | dict[str, object]:
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    return train(
        load_summary_memory_study(args.study),
        benchmark=args.benchmark,
        condition=args.condition,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
        smoke=args.smoke,
        overwrite=args.overwrite,
        resume=args.resume,
        resume_allow_evicted_replay=args.resume_allow_evicted_replay,
        preflight=args.preflight,
        wandb=args.wandb,
    )


def add_evaluate_arguments(parser: argparse.ArgumentParser) -> None:
    _common_arguments(parser)
    parser.add_argument("--split", default="development")
    parser.add_argument("--history", default="retained", choices=ALL_HISTORY_MODES)
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--task-cap", type=int, default=None)
    parser.add_argument(
        "--checkpoint-rule",
        default="selected",
        choices=("selected", "final-epoch", "endpoint"),
        help="endpoint evaluates the weights at the declared final epoch (the 8M "
        "study's primary panel; refused for an incomplete run); final-epoch "
        "evaluates the last scheduled checkpoint as the legacy supplement. Both "
        "are written beside the selected-checkpoint panel.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="evaluate the endpoint weights over this many charged calls in the "
        "same task (Key-to-Door length extrapolation, EXPERIMENTS section 5); "
        "the panel is written under the -h<H> suffix",
    )
    parser.add_argument(
        "--layout-period",
        type=int,
        default=None,
        metavar="P",
        help="Key-to-Door layout-change continuation: a new hidden start, key and "
        "door at the first attempt boundary after every P charged calls, with "
        "--horizon (evaluation only; the panel is written under -relayout<P>)",
    )


def evaluate_main(args: argparse.Namespace) -> Path:
    return evaluate(
        load_summary_memory_study(args.study),
        benchmark=args.benchmark,
        condition=args.condition,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
        split=args.split,
        history=args.history,
        checkpoint=args.checkpoint,
        task_cap=args.task_cap,
        checkpoint_rule=args.checkpoint_rule,
        horizon=args.horizon,
        layout_period=args.layout_period,
    )


__all__ = [
    "add_evaluate_arguments",
    "add_train_arguments",
    "evaluate",
    "evaluate_main",
    "evaluate_saved",
    "resolve",
    "train",
    "train_main",
]
