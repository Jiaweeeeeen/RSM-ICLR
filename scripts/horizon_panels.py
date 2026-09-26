"""Extended-horizon panels for the bounded-summary paper (EXPERIMENTS section 5).

    python scripts/horizon_panels.py --study configs/memo_key_to_door_8m.yaml \
        --benchmark dark_key_to_door --split confirmation \
        --horizons 500 1000 2000 --device cuda \
        [--cells memo memo_fixed raw_summary raw_segment full_context full_gru] \
        [--seeds 42 100 2026] [--task-cap N] [--check-only]

Evaluates the frozen endpoint weights of the study's primary cells and of the
declared frozen reference cells over ``--horizons`` charged calls in the same
Key-to-Door task, writing ``<run>/eval/<split>-retained-endpoint-h<H>/`` beside
the trained-horizon panels (an existing complete panel is never recomputed).
After every panel the adapter's acceptance check runs: the panel's first 500
calls must reproduce the 500-call panel (the ``-h500`` panel of the same run,
or the trained-horizon panel when that exists); the result is written to
``prefix_check.json`` inside the panel.
On CountRecall (``--benchmark count_recall``) ``--horizons`` are stream
lengths in records (default 104 208 416 832): the frozen weights play the trained
104-record stream and then a
query-only tail (a blank value slot, a query per record drawn by the task
identity, counts fixed), the panel lands under ``-h<H>``, its first 103
decisions must reproduce the plain panel query for query (the acceptance
check) and every panel records its exact accuracy per 104-record window.

On MazeRunner (``--study configs/mazerunner_8m.yaml --benchmark mazerunner``)
``--horizons`` are odd maze sizes (default 15 17 19 21 25, declared before any
MazeRunner endpoint existed): the frozen weights
trained on 15x15 mazes play every roster map drawn at the larger size under the
area-scaled step budget (500 steps at 15; 642, 802, 980 and 1,389 at 17, 19,
21 and 25), the panel lands under ``-h<S>`` with that budget as the run's
``outer_length``, the trained-size panel must reproduce the plain panel
episode for episode (the acceptance check) and every panel records its goal
fraction and goals per 500 steps.

``--laps B ...`` (MazeRunner only) evaluates the
frozen endpoint weights on every roster map replayed in laps until ``B``
charged calls (default 1000 2000 4000): a lap is one native episode of the
trained task on the map's own seed, a reset-only call separates laps and the
carried memory continues; the panel lands under ``-calls<B>``, its first lap
must reproduce the plain panel map for map (the acceptance check) and every
panel records goals reached per 500-call window. ``--history attempt-cleared``
is the lap-cleared companion (the memory wiped at every lap boundary).

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from reasoned_icrl.experiments.benchmarks import BenchmarkContract, saved_config
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import (
    ALL_HISTORY_MODES,
    RESULTS_FILE,
    evaluation_directory,
)
from reasoned_icrl.experiments.horizon import (
    DECLARED_HORIZONS,
    LAP_BUDGETS,
    MAZE_SIZES,
    STREAM_HORIZONS,
    check_episode_panel,
    check_horizon_prefix,
    check_laps_prefix,
    check_stream_prefix,
    continued_laps,
    continued_streams,
    extended_horizon,
    horizon_kind,
    laps_per_unit,
    maze_goal_fraction,
    maze_size_ladder,
    native_horizon,
    stream_window_accuracy,
    streams_accuracy,
    window_door_counts,
    window_goal_counts,
)
from reasoned_icrl.experiments.records import (
    compact_benchmark_results,
    read_benchmark_results,
    read_results_text,
    results_file,
)
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.experiments import (
    evaluate_saved,
    resolve,
)
from reasoned_icrl.experiments.summary_memory.revised import (
    read_run_development,
    reference_configs,
)
from reasoned_icrl.utils import repository_root

PAPER_CELLS = (
    "raw_summary_residual",
    "full_context",
    "full_gru",
    "raw_summary",
    "raw_segment",
    "memo",
    "memo_fixed",
)
"""The seven cells of the bounded-summary paper (the method first), in the paper's
order."""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--study", type=Path, required=True)
    result.add_argument("--benchmark", default="dark_key_to_door")
    result.add_argument("--split", default="confirmation")
    result.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Key-to-Door: outer budgets in charged calls (default "
        f"{list(DECLARED_HORIZONS)}); CountRecall: stream lengths in "
        f"records (default {list(STREAM_HORIZONS)}, a query-only tail after "
        "the native deck); MazeRunner: odd maze sizes (default the trained size "
        f"and +2/+4/+6/+10, {list(MAZE_SIZES)} for the 15x15 protocol; the frozen "
        "weights play a larger maze under the area-scaled step budget, the panel "
        "suffix is -h<S>)",
    )
    result.add_argument(
        "--streams",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="CountRecall only: write the continued-stream panels (-s<N>) for these "
        "numbers of consecutive deck pairs instead of horizon panels (endpoint rule)",
    )
    result.add_argument(
        "--compact",
        action="store_true",
        help="gzip each panel's per-record file in place once its check has been "
        "written (the readers accept the compacted twin; a 32-pair CountRecall "
        "panel is 1.1 GB per seed plain)",
    )
    result.add_argument(
        "--laps",
        type=int,
        nargs="+",
        default=None,
        help="MazeRunner: replay every roster map in laps until these budgets of "
        f"charged calls (the repeated-laps axis; declared {list(LAP_BUDGETS)}); "
        "panels under the -calls<B> suffix, endpoint rule, histories retained, "
        "attempt-cleared (lap-cleared) or summary-cleared",
    )
    result.add_argument("--cells", nargs="+", default=list(PAPER_CELLS))
    result.add_argument("--seeds", type=int, nargs="*", default=None)
    result.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    result.add_argument("--task-cap", type=int, default=None)
    result.add_argument(
        "--check-only",
        action="store_true",
        help="run the acceptance checks on the panels that exist; evaluate nothing",
    )
    result.add_argument(
        "--checkpoint-rules",
        nargs="+",
        default=["endpoint"],
        choices=("endpoint", "selected"),
        help="which checkpoint each panel evaluates: the frozen endpoint (the "
        "primary rule) and/or the development-selected checkpoint recorded in "
        "the run's development.json (the declared secondary rule; the panel "
        "then lands without a rule suffix, beside the selected trained-length "
        "panel it is checked against)",
    )
    result.add_argument(
        "--layout-period",
        type=int,
        default=None,
        metavar="P",
        help="Key-to-Door layout-change continuation: a new hidden start, key "
        "and door at the first attempt boundary after every P charged calls "
        "(panels under the -relayout<P> suffix; endpoint rule only)",
    )
    result.add_argument(
        "--history",
        default="retained",
        choices=ALL_HISTORY_MODES,
        help="the history intervention of the panels (default retained); e.g. "
        "summary-cleared on the summary carrier at 2,000 calls, the split "
        "record's companion reading; the acceptance check then compares with "
        "the 500-call panel of the same intervention",
    )
    return result


def _configs(args: argparse.Namespace) -> list[tuple[str, ExperimentConfig]]:
    study = load_summary_memory_study(args.study)
    contract = study.contract(args.benchmark)
    plan = study.tier(contract.name)
    seeds = tuple(study.training_seeds if not args.seeds else args.seeds)
    fitted = tuple(plan.primary) + tuple(plan.supplementary)
    references = reference_configs(
        study, contract, repository=repository_root(), device=args.device
    )
    chosen: list[tuple[str, ExperimentConfig]] = []
    for condition in args.cells:
        if condition in fitted:
            for seed in seeds:
                _, config = resolve(
                    study,
                    benchmark=args.benchmark,
                    condition=condition,
                    seed=seed,
                    device=args.device,
                )
                chosen.append((condition, saved_config(config)))
        elif condition in plan.reference_cells:
            for config in references:
                if config.condition == condition and config.seed in seeds:
                    chosen.append((condition, config))
        else:
            raise ContractError(
                f"{condition!r} is neither fitted under {study.name} nor one of its "
                f"frozen reference cells on {contract.name}."
            )
    return chosen


def _panel(
    config: ExperimentConfig,
    split: str,
    horizon: int | None,
    history: str = "retained",
    rule: str = "endpoint",
    layout_period: int | None = None,
    streams: int | None = None,
    laps: int | None = None,
) -> Path:
    return (
        config.run_directory
        / "eval"
        / evaluation_directory(
            split,
            history,
            rule,
            horizon=horizon,
            layout_period=layout_period,
            streams=streams,
            laps=laps,
        )
    )


def _complete(panel: Path) -> bool:
    path = panel / RESULTS_FILE
    if results_file(path) is None:
        return False
    raw = json.loads(read_results_text(path))
    return isinstance(raw, dict) and "partial_task_cap" not in raw


def _compact(panel: Path, label: str) -> None:
    """Gzip the panel's per-record file in place after its check is written."""
    plain = panel / RESULTS_FILE
    if plain.is_file():
        before = plain.stat().st_size
        packed = compact_benchmark_results(plain)
        print(
            f"{time.strftime('%T')} {label}: compacted {before / 1e6:.0f} MB -> "
            f"{packed.stat().st_size / 1e6:.0f} MB",
            flush=True,
        )


def _check(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    split: str,
    horizon: int,
    history: str = "retained",
    rule: str = "endpoint",
    layout_period: int | None = None,
) -> dict[str, Any]:
    """Compare a horizon panel's prefix with the 500-call panel of the same run,
    history intervention and checkpoint rule.

    A layout-change panel keeps the native task for its first period, so its
    prefix is checked against the plain 500-call panel like any other.
    """
    native = native_horizon(contract)
    kind = horizon_kind(contract)
    extended = _panel(config, split, horizon, history, rule, layout_period)
    base = _panel(config, split, native, history=history, rule=rule)
    if kind != "calls" or not _complete(base):
        base = _panel(config, split, None, history=history, rule=rule)
    if not _complete(base) or not _complete(extended):
        return {
            "status": "skipped",
            "reason": "a complete base or extended panel is missing",
        }
    _, base_events = read_benchmark_results(base / RESULTS_FILE, [contract])
    # The extended panel was written under the adapter's evaluation-only
    # contract (a larger maze, or a `horizon`-call budget), so it is read back
    # under the same contract: the native one rejects any episode longer than
    # the trained budget as exceeding its horizon.
    extended_contract, _ = extended_horizon(contract, config, horizon)
    _, events = read_benchmark_results(extended / RESULTS_FILE, [extended_contract])
    applicable = True
    units = len(contract.roster(split)) * len(contract.evaluation.rollout_seeds)
    counts: dict[str, Any]
    estimand: dict[str, Any]
    if kind == "maze":
        # MazeRunner: the adapter reproduces the trained-size panel exactly; a
        # larger maze changes every episode by design, so only the panel's
        # estimand (the goal fraction by size) is recorded.
        applicable = applicable and horizon == native
        counts = (
            check_episode_panel(base_events, events)
            if applicable
            else {
                "reason": "a larger maze changes every episode by design; "
                "the trained-size panel is the acceptance check"
            }
        )
        estimand = {"goal_fraction": maze_goal_fraction(events)}
    elif kind == "stream":
        # CountRecall: the first 103 decisions reproduce the plain panel query
        # for query; the tail is read per 104-record window.
        counts = check_stream_prefix(base_events, events, native=native)
        estimand = {
            "exact_accuracy_per_window": stream_window_accuracy(
                events, native=native, horizon=horizon
            ),
            "window_records": native,
        }
    else:
        counts = (
            check_horizon_prefix(base_events, events, native=native)
            if applicable
            else {
                "reason": "the cap is exceeded inside the trained horizon, so the "
                "prefix differs from the plain panel by design"
            }
        )
        windows = window_door_counts(events, outer_length=horizon)
        # Mean over every (task, rollout seed) unit of the roster, so units
        # that completed no door count as zero and panels are comparable.
        estimand = {
            "mean_doors_per_500_call_window": [
                sum(values[index] for values in windows.values()) / max(units, 1)
                for index in range(horizon // 500)
            ]
        }
    record = {
        "status": "passed" if applicable else "not-applicable",
        "base_panel": base.name,
        "native": native,
        "horizon": horizon,
        "history": history,
        "checkpoint_rule": rule,
        "layout_period": layout_period,
        **counts,
        "roster_units": units,
        **estimand,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (extended / "prefix_check.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _check_streams(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    split: str,
    streams: int,
    rule: str = "endpoint",
    history: str = "retained",
) -> dict[str, Any]:
    """The continued-stream panel's acceptance check: its first stream reproduces
    the plain panel query for query (the stream prefix check, flips bounded);
    the estimand is exact accuracy per deck pair."""
    native = native_horizon(contract)
    decisions = native - 1
    # A stream-cleared panel has no boundary before its first pair ends, so
    # its first pair is checked against the plain retained panel as well.
    extended = _panel(config, split, None, history=history, rule=rule, streams=streams)
    base = _panel(config, split, None, rule=rule)
    if not _complete(base) or not _complete(extended):
        return {
            "status": "skipped",
            "reason": "a complete base or extended panel is missing",
        }
    _, base_events = read_benchmark_results(base / RESULTS_FILE, [contract])
    longer, _ = continued_streams(contract, config, streams)
    _, events = read_benchmark_results(extended / RESULTS_FILE, [longer])
    counts = check_stream_prefix(base_events, events, native=native)
    record = {
        "status": "passed",
        "base_panel": base.name,
        "native": native,
        "streams": streams,
        "decisions_per_stream": decisions,
        "history": history,
        "checkpoint_rule": rule,
        **counts,
        "roster_units": len(contract.roster(split))
        * len(contract.evaluation.rollout_seeds),
        "exact_accuracy_per_stream": streams_accuracy(
            events, decisions=decisions, streams=streams
        ),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (extended / "prefix_check.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _run_streams(args: argparse.Namespace) -> int:
    """Write and check the continued-stream panels (CountRecall, endpoint rule)."""
    study = load_summary_memory_study(args.study)
    contract = study.contract(args.benchmark)
    if horizon_kind(contract) != "stream":
        raise ContractError("Continued-stream panels exist on CountRecall only.")
    if args.layout_period is not None:
        raise ContractError("Continued-stream panels take no layout period.")
    history = args.history
    if history not in ("retained", "attempt-cleared") or list(
        dict.fromkeys(args.checkpoint_rules)
    ) != ["endpoint"]:
        raise ContractError(
            "Continued-stream panels evaluate the endpoint weights on the retained "
            "history or under the stream-cleared (attempt-cleared) intervention."
        )
    failures = 0
    for streams in sorted(int(n) for n in args.streams):
        for condition, config in _configs(args):
            label = f"{condition} seed {config.seed} s{streams} endpoint"
            if history != "retained":
                label = f"{label} {history}"
            panel = _panel(config, args.split, None, history=history, streams=streams)
            if not args.check_only:
                if _complete(panel):
                    print(f"{time.strftime('%T')} {label}: exists, skipped", flush=True)
                else:
                    started = time.perf_counter()
                    try:
                        evaluate_saved(
                            contract,
                            config,
                            split=args.split,
                            history=history,
                            checkpoint="checkpoint.pt",
                            task_cap=args.task_cap,
                            checkpoint_rule="endpoint",
                            streams=streams,
                        )
                    except ContractError as error:
                        failures += 1
                        print(
                            f"{time.strftime('%T')} {label}: FAILED {error}", flush=True
                        )
                        continue
                    print(
                        f"{time.strftime('%T')} {label}: written in "
                        f"{time.perf_counter() - started:.0f} s",
                        flush=True,
                    )
            try:
                record = _check_streams(
                    contract, config, args.split, streams, history=history
                )
            except ContractError as error:
                failures += 1
                print(
                    f"{time.strftime('%T')} {label}: CHECK FAILED {error}", flush=True
                )
                continue
            print(f"{time.strftime('%T')} {label}: check {record}", flush=True)
            # Compact whatever the check said. A skipped check
            # (the plain panel not yet written, e.g. before a study's finalize)
            # is re-run by --check-only later and reads the compacted twin;
            # leaving the plain file until then cost the pool gigabytes.
            if args.compact:
                _compact(panel, label)
    return 1 if failures else 0


def _check_laps(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    split: str,
    budget: int,
    history: str = "retained",
    layout_period: int | None = None,
) -> dict[str, Any]:
    """The repeated-laps panel's acceptance check: its first lap reproduces
    the plain one-episode panel map for map; the estimand is goals reached
    per 500-call window over every roster unit.

    A lap-cleared panel has no boundary before its first lap ends, so its
    first lap is checked against the plain retained panel; a summary-cleared
    laps panel is checked against the plain summary-cleared panel."""
    base_history = "retained" if history == "attempt-cleared" else history
    extended = _panel(
        config, split, None, history=history, layout_period=layout_period, laps=budget
    )
    base = _panel(config, split, None, history=base_history, rule="endpoint")
    if not _complete(base) or not _complete(extended):
        return {
            "status": "skipped",
            "reason": "a complete base or laps panel is missing",
        }
    _, base_events = read_benchmark_results(base / RESULTS_FILE, [contract])
    longer, _ = continued_laps(contract, config, budget)
    _, events = read_benchmark_results(extended / RESULTS_FILE, [longer])
    counts = check_laps_prefix(base_events, events)
    units = len(contract.roster(split)) * len(contract.evaluation.rollout_seeds)
    windows = window_goal_counts(events, outer_length=budget)
    finished = laps_per_unit(events)
    record = {
        "status": "passed",
        "base_panel": base.name,
        "native": contract.environment.horizon,
        "budget": budget,
        "history": history,
        "layout_period": layout_period,
        "checkpoint_rule": "endpoint",
        **counts,
        "roster_units": units,
        "mean_goals_per_500_call_window": [
            sum(values[index] for values in windows.values()) / max(units, 1)
            for index in range(budget // 500)
        ],
        "mean_finished_laps": sum(finished.values()) / max(units, 1),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (extended / "prefix_check.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _run_laps(args: argparse.Namespace) -> int:
    """Write and check the repeated-laps panels (MazeRunner, endpoint rule)."""
    study = load_summary_memory_study(args.study)
    contract = study.contract(args.benchmark)
    if contract.environment.name != "mazerunner":
        raise ContractError("Repeated-laps panels exist on MazeRunner only.")
    relayout = args.layout_period
    history = args.history
    if history not in ("retained", "attempt-cleared", "summary-cleared") or list(
        dict.fromkeys(args.checkpoint_rules)
    ) != ["endpoint"]:
        raise ContractError(
            "Repeated-laps panels evaluate the endpoint weights on the retained "
            "history, under the lap-cleared (attempt-cleared) intervention or "
            "summary-cleared."
        )
    failures = 0
    for budget in sorted(int(n) for n in args.laps):
        if budget % 500:
            raise ContractError("A laps budget is a whole number of 500-call windows.")
        for condition, config in _configs(args):
            label = f"{condition} seed {config.seed} calls{budget} endpoint"
            if history != "retained":
                label = f"{label} {history}"
            if relayout is not None:
                label = f"{label} relayout{relayout}"
            panel = _panel(
                config,
                args.split,
                None,
                history=history,
                layout_period=relayout,
                laps=budget,
            )
            if not args.check_only:
                if _complete(panel):
                    print(f"{time.strftime('%T')} {label}: exists, skipped", flush=True)
                else:
                    started = time.perf_counter()
                    try:
                        evaluate_saved(
                            contract,
                            config,
                            split=args.split,
                            history=history,
                            checkpoint="checkpoint.pt",
                            task_cap=args.task_cap,
                            checkpoint_rule="endpoint",
                            laps=budget,
                            layout_period=relayout,
                        )
                    except ContractError as error:
                        failures += 1
                        print(
                            f"{time.strftime('%T')} {label}: FAILED {error}", flush=True
                        )
                        continue
                    print(
                        f"{time.strftime('%T')} {label}: written in "
                        f"{time.perf_counter() - started:.0f} s",
                        flush=True,
                    )
            try:
                record = _check_laps(
                    contract,
                    config,
                    args.split,
                    budget,
                    history=history,
                    layout_period=relayout,
                )
            except ContractError as error:
                failures += 1
                print(
                    f"{time.strftime('%T')} {label}: CHECK FAILED {error}", flush=True
                )
                continue
            print(f"{time.strftime('%T')} {label}: check {record}", flush=True)
            if args.compact:
                _compact(panel, label)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.streams is not None:
        return _run_streams(args)
    if args.laps is not None:
        return _run_laps(args)
    study = load_summary_memory_study(args.study)
    contract = study.contract(args.benchmark)
    chosen = _configs(args)
    failures = 0
    history = args.history
    horizons = args.horizons
    if horizons is None:
        kind = horizon_kind(contract)
        horizons = list(
            STREAM_HORIZONS
            if kind == "stream"
            else maze_size_ladder(native_horizon(contract))
            if kind == "maze"
            else DECLARED_HORIZONS
        )
    rules = list(dict.fromkeys(args.checkpoint_rules))
    relayout = args.layout_period
    if relayout is not None and rules != ["endpoint"]:
        raise ContractError("Layout-change panels evaluate the endpoint weights only.")
    for horizon in sorted(horizons):
        for rule in rules:
            for condition, config in chosen:
                label = f"{condition} seed {config.seed} h{horizon} {rule}"
                if history != "retained":
                    label = f"{label} {history}"
                if relayout is not None:
                    label = f"{label} relayout{relayout}"
                checkpoint = "checkpoint.pt"
                if rule == "selected":
                    development = read_run_development(config.run_directory)
                    if development is None or development.selected_epoch == -1:
                        print(
                            f"{time.strftime('%T')} {label}: no development selection, "
                            "skipped",
                            flush=True,
                        )
                        continue
                    checkpoint = development.checkpoint
                    label = f"{label} ({checkpoint})"
                panel = _panel(config, args.split, horizon, history, rule, relayout)
                if not args.check_only:
                    if _complete(panel):
                        print(
                            f"{time.strftime('%T')} {label}: exists, skipped",
                            flush=True,
                        )
                    else:
                        started = time.perf_counter()
                        try:
                            evaluate_saved(
                                contract,
                                config,
                                split=args.split,
                                history=history,
                                checkpoint=checkpoint,
                                task_cap=args.task_cap,
                                checkpoint_rule=rule,
                                horizon=horizon,
                                layout_period=relayout,
                            )
                        except ContractError as error:
                            failures += 1
                            print(
                                f"{time.strftime('%T')} {label}: FAILED {error}",
                                flush=True,
                            )
                            continue
                        print(
                            f"{time.strftime('%T')} {label}: written in "
                            f"{time.perf_counter() - started:.0f} s",
                            flush=True,
                        )
                try:
                    record = _check(
                        contract,
                        config,
                        args.split,
                        horizon,
                        history,
                        rule,
                        relayout,
                    )
                except ContractError as error:
                    failures += 1
                    print(
                        f"{time.strftime('%T')} {label}: CHECK FAILED {error}",
                        flush=True,
                    )
                    continue
                print(f"{time.strftime('%T')} {label}: check {record}", flush=True)
                if args.compact:
                    _compact(panel, label)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"horizon_panels: {error}", file=sys.stderr)
        raise SystemExit(2) from error
