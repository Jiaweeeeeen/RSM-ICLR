"""The interactive player: the evaluator's rollout, one decision at a time.

The parity test trains the CPU smoke recipe and checks that a task driven
through :class:`PolicyPlayer` with every proposal accepted ends in exactly the
episode record the evaluator's :func:`rollout` produces, for a recurrent
carrier and for the summary carrier with its cached hidden state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from reasoned_icrl.environments.base import DEVELOPMENT_TASKS
from reasoned_icrl.environments.tmaze import DOWN, FORWARD, UP, TMazeEnv
from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import evaluation_environment
from reasoned_icrl.experiments.summary_memory.configs import load_tmaze_v3_study
from reasoned_icrl.runtime.environments import amago_environment
from reasoned_icrl.runtime.play import (
    PolicyPlayer,
    close_player_experiment,
    load_player,
    resolve_checkpoint,
)
from reasoned_icrl.runtime.rollout import rollout
from reasoned_icrl.runtime.training import train_experiment

ROOT = Path(__file__).resolve().parents[2]


def contract() -> Any:
    return load_tmaze_v3_study().contract("tmaze")


def resolved(condition: str, **overrides: Any) -> Any:
    settings: dict[str, Any] = {"seed": 42, "repository": ROOT, "device": "cpu"}
    settings.update(overrides)
    return experiment_config(
        contract(), load_tmaze_v3_study(), condition=condition, **settings
    )


def test_a_policy_free_player_steps_one_task_by_hand() -> None:
    env = TMazeEnv(corridor_length=6, split="development", initial_seed=0)
    player = PolicyPlayer(None, env, rollout_seed=7)
    assert not player.has_policy and player.done
    with pytest.raises(ContractError, match="reset the player"):
        player.propose()
    player.reset(DEVELOPMENT_TASKS[3])
    assert player.task == DEVELOPMENT_TASKS[3] and player.steps == 0
    assert set(player.observation()) == {
        "current",
        "previous",
        "outcome",
        "event",
        "valid",
    }
    proposal = player.propose()
    assert 0 <= proposal < 4 and player.propose() == proposal  # cached per step
    for _ in range(6):
        reward, done, info = player.step(FORWARD)
        assert reward == 0.0 and not done and info["step_in_task"] == player.steps
    reward, done, _ = player.step(UP if env.cue == 1 else DOWN)
    assert done and reward == 1.0 and player.done
    assert player.episode_return == 1.0 and player.steps == 7
    assert env.episode is not None and env.episode.success
    assert "success" in player.render("ansi")
    with pytest.raises(ContractError, match="reset the player"):
        player.step(FORWARD)
    with pytest.raises(ContractError, match="outside"):
        player.reset(DEVELOPMENT_TASKS[0])
        player.step(9)
    player.close()


def test_random_proposals_are_reproducible_per_rollout_seed() -> None:
    draws = []
    for _ in range(2):
        env = TMazeEnv(corridor_length=4, split="development", initial_seed=0)
        player = PolicyPlayer(None, env, rollout_seed=3)
        player.reset(DEVELOPMENT_TASKS[0])
        proposals = []
        while not player.done:
            proposals.append(player.propose())
            player.step()
        draws.append(proposals)
        player.close()
    assert draws[0] == draws[1]


def test_checkpoint_rules_resolve_as_the_evaluator_does(tmp_path: Path) -> None:
    config = resolved("full_gru", output_root=tmp_path, smoke=True)
    assert resolve_checkpoint(
        config, checkpoint="policy_epoch_3", checkpoint_rule="selected"
    )
    with pytest.raises(ContractError, match="resolves the checkpoint itself"):
        resolve_checkpoint(
            config, checkpoint="policy_epoch_3", checkpoint_rule="endpoint"
        )
    with pytest.raises(ContractError, match="Unknown checkpoint rule"):
        resolve_checkpoint(config, checkpoint="checkpoint.pt", checkpoint_rule="best")  # type: ignore[arg-type]


def test_a_policy_free_player_loads_from_the_contract_alone() -> None:
    player, roster, checkpoint = load_player(contract(), None, split="development")
    try:
        assert checkpoint is None and not player.has_policy
        assert roster == contract().roster("development")
        assert isinstance(player.environment, TMazeEnv)
        assert player.environment.render_mode == "ansi"
        assert player.environment.corridor_length == 128
        player.reset(roster[0])
        assert player.render("rgb_array").shape[0] == 21
    finally:
        player.close()
    with pytest.raises(ContractError, match="declares no"):
        load_player(contract(), None, split="nowhere")


@pytest.mark.parametrize("condition", ["full_gru", "raw_summary"])
def test_the_player_reproduces_the_evaluator_rollout(
    condition: str, tmp_path: Path
) -> None:
    active = contract()
    config = resolved(condition, output_root=tmp_path, smoke=True)
    train_experiment(config)
    player, roster, checkpoint = load_player(
        active, config, split="development", checkpoint_rule="final-epoch"
    )
    assert checkpoint is not None and checkpoint.startswith("policy_epoch_")
    try:
        assert player.has_policy
        records = []
        for task in roster[:2]:
            player.reset(task)
            while not player.done:
                player.step()
            records.append(player.environment.episode)
        env = evaluation_environment(
            active, config, split="development", seed=player.rollout_seed
        )
        wrapped = amago_environment(env, name="check", seed=player.rollout_seed)
        try:
            results, _ = rollout(
                player.experiment,
                wrapped,
                task_ids=list(roster[:2]),
                rollout_seed=player.rollout_seed,
            )
        finally:
            wrapped.close()
        assert [result.episode for result in results] == records
    finally:
        player.close()
        close_player_experiment(player.experiment)
