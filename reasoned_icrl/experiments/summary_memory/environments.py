"""The environments of the summary-memory study and how they are evaluated.

The Gymnasium-to-AMAGO adaptation is shared by every study and lives in
:mod:`reasoned_icrl.experiments.environments`; this module names the roster
the study runs and builds evaluation environments for its notebooks.
"""

from __future__ import annotations

from dataclasses import asdict

from reasoned_icrl.environments.base import BaseEnv
from reasoned_icrl.experiments.benchmarks import Study
from reasoned_icrl.experiments.environments import build_environment
from reasoned_icrl.experiments.evaluation import HISTORY_MODES

ENVIRONMENTS = (
    "match_pattern",
    "dark_key_to_door",
    "concentration",
    "xland_minigrid",
    "xland_one_rule",
    "count_recall",
    "mazerunner",
    "tmaze",
)
"""Registered study environments, including historical protocols.

The active 8M YAML selects Key-to-Door (tier 0), ConcentrationEasy (tier 1),
official CountRecallMedium (tier 2) and XLand (tier 3, migration deferred R9).
The explicit retired YAML retains CountRecallHard and MazeRunner-15 and their
original identities. Each study still selects one contract per environment.
"""


def environment(study: Study, name: str, *, split: str, seed: int = 0) -> BaseEnv:
    """Build one study environment over the full roster of ``split``."""
    contract = study.contract(name)
    source = contract.evaluation.splits.get(split)
    return build_environment(
        {"environment": asdict(contract.environment)},
        split=split if source is None else source.source,
        seed=seed,
    )


def interventions(name: str) -> tuple[str, ...]:
    """The cache interventions declared for ``name``."""
    return HISTORY_MODES[name]


__all__ = ["ENVIRONMENTS", "environment", "interventions"]
