"""Retired study rosters kept as test fixtures, and the DAT-era result fixture.

``stage1.yaml`` and ``dat_benchmarks.yaml`` are the rosters of the two retired
studies. They stay here so the frozen Stage-1 history conditions and the
transition-DAT arms of :mod:`reasoned_icrl.experiments.contracts` keep a study
to resolve against in tests; neither is launchable.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from reasoned_icrl.experiments.benchmarks import EvaluationSplit, Study, load_study
from reasoned_icrl.experiments.records import BenchmarkEvent, BenchmarkRun

FIXTURES = Path(__file__).resolve().parent


def fixture_study_path(name: str) -> Path:
    return FIXTURES / f"{name}.yaml"


def load_fixture_study(name: str) -> Study:
    """Load a retired roster kept as a fixture: ``stage1`` or ``dat_benchmarks``."""
    return load_study(fixture_study_path(name))


def fixture_results(
    study: Study | None = None,
) -> tuple[Study, list[BenchmarkRun], list[BenchmarkEvent]]:
    """Two final tasks per contract; DAT arms score one, ordinary arms zero."""
    study = load_fixture_study("dat_benchmarks") if study is None else study
    contracts = tuple(
        replace(
            c,
            evaluation=replace(
                c.evaluation,
                splits={
                    name: EvaluationSplit(split.source, 2, split.offset)
                    for name, split in c.evaluation.splits.items()
                },
            ),
        )
        for c in study.contracts
    )
    study = replace(study, contracts=contracts)
    winners = {"transition_dat", "transition_bypass"}
    runs, events = [], []
    for contract in contracts:
        env = contract.environment
        kind = contract.event_kind
        count = {"attempt": env.attempts, "query": env.horizon, "episode": 1}[kind]
        if env.name == "mazerunner":
            denominator = env.goals
        elif env.name == "concentration":
            denominator = env.size // 2
        else:
            denominator = 1
        full = 2 if env.name == "dark_key_to_door" else 1
        for seed in study.training_seeds:
            for condition in study.conditions:
                run = BenchmarkRun(
                    contract.protocol,
                    env.name,
                    condition,
                    seed,
                    f"{condition}-fixture-{seed}",
                    "final",
                    "retained",
                    "completed",
                    100,
                    10,
                    1000,
                    2000,
                    100,
                    "fixture CPU",
                    "synthetic",
                )
                runs.append(run)
                for task in contract.roster("final"):
                    for i in range(1, count + 1):
                        numerator = denominator if condition in winners else 0
                        if kind == "query":
                            reward = (2 * numerator - 1) / env.horizon
                        elif kind == "attempt":
                            reward = full * numerator
                        elif env.name == "concentration":
                            reward = numerator / denominator
                        else:
                            reward = numerator
                        events.append(
                            BenchmarkEvent(
                                contract.protocol,
                                env.name,
                                condition,
                                seed,
                                run.checkpoint,
                                "final",
                                "retained",
                                task,
                                task,
                                0,
                                kind,
                                i,
                                i if kind == "query" else 10,
                                numerator,
                                denominator,
                                reward,
                                1 if kind == "query" else None,
                            )
                        )
    return study, runs, events
