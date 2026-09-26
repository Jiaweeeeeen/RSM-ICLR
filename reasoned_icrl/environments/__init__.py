"""Gymnasium environments that publish the shared five-key packet.

Every environment extends :class:`~reasoned_icrl.environments.base.BaseEnv`,
which owns the task roster, the packet contract and the restorable state.
"""

from reasoned_icrl.environments.base import (
    Attempt,
    BaseEnv,
    PublicDecision,
    PublicField,
    benchmark_task_sources,
)
from reasoned_icrl.environments.concentration import ConcentrationEnv
from reasoned_icrl.environments.count_recall import CountRecallEnv
from reasoned_icrl.environments.dark_key_to_door import DarkKeyToDoorEnv
from reasoned_icrl.environments.darkroom import DarkRoomEnv
from reasoned_icrl.environments.match_pattern import MatchPatternEnv
from reasoned_icrl.environments.mazerunner import MazeRunnerEnv
from reasoned_icrl.environments.tmaze import TMazeEnv
from reasoned_icrl.environments.xland_minigrid import XLandMiniGridEnv
from reasoned_icrl.environments.xland_one_rule import XLandOneRuleEnv

ENVIRONMENT_NAMES = (
    "darkroom",
    "dark_key_to_door",
    "count_recall",
    "mazerunner",
    "concentration",
    "xland_minigrid",
    "xland_one_rule",
    "match_pattern",
    "tmaze",
)

__all__ = [
    "ENVIRONMENT_NAMES",
    "Attempt",
    "BaseEnv",
    "ConcentrationEnv",
    "CountRecallEnv",
    "DarkKeyToDoorEnv",
    "DarkRoomEnv",
    "MatchPatternEnv",
    "MazeRunnerEnv",
    "PublicDecision",
    "PublicField",
    "TMazeEnv",
    "XLandMiniGridEnv",
    "XLandOneRuleEnv",
    "benchmark_task_sources",
]
