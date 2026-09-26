"""Drive one task, one decision at a time, under a frozen policy or by hand.

:class:`PolicyPlayer` is the interactive counterpart of
:func:`reasoned_icrl.runtime.rollout.rollout`: the same AMAGO boundary
(:class:`SequenceWrapper` over the project adapter), the same inference call
and hidden-state carry, but one environment, one task and one step per call,
so the frames can be watched, the policy's choice can be read before it is
executed, and any decision can be overridden. No record is written; the
evaluator's rollout stays the only source of benchmark results.

:func:`load_player` loads a saved run the way the evaluator does (the run's
own saved recipe, the endpoint or a named checkpoint, weights only) and
builds the player around one rendering environment of the requested split.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from amago.envs.amago_env import SequenceWrapper
from numpy.typing import NDArray

from reasoned_icrl.environments.base import BaseEnv
from reasoned_icrl.experiments.artifacts import (
    endpoint_training_epoch,
    latest_training_epoch,
)
from reasoned_icrl.experiments.benchmarks import BenchmarkContract
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import (
    ContractError,
    architecture_uses_history_packet,
)
from reasoned_icrl.experiments.environments import build_environment
from reasoned_icrl.experiments.records import CHECKPOINT_RULES, CheckpointRule
from reasoned_icrl.runtime.environments import amago_environment


class PolicyPlayer:
    """One task under a frozen policy, stepped from outside.

    ``experiment`` is a loaded AMAGO experiment (:func:`load_experiment`) or
    ``None`` for a policy-free player whose proposals are uniform random
    draws. Call :meth:`reset` with a task identity, then alternate
    :meth:`propose` (the policy's action for the current timestep, computed
    once per timestep and cached) and :meth:`step` (execute any action). The
    policy reads the executed action through RL2 at the next timestep, so an
    override is consistent with what the rollout would have fed it.
    """

    def __init__(
        self,
        experiment: Any | None,
        environment: BaseEnv,
        *,
        rollout_seed: int,
        sample_actions: bool = False,
    ) -> None:
        self.environment = environment
        self.rollout_seed = int(rollout_seed)
        self.sample_actions = sample_actions
        self._wrapped = amago_environment(
            environment,
            name=f"{type(environment).__name__}-play",
            seed=self.rollout_seed,
        )
        self._sequence = SequenceWrapper(
            self._wrapped, save_trajs_to=None, save_every=None
        )
        self._experiment = experiment
        self._random = np.random.default_rng(self.rollout_seed)
        if experiment is not None:
            self._policy = experiment.policy
            self._policy.eval()
            self._encoder = self._policy.traj_encoder
            self._device = experiment.DEVICE
            self._precision: Callable[[], Any] = (
                experiment.caster
                if architecture_uses_history_packet(
                    str(experiment.encoder_architecture_id)
                )
                else nullcontext
            )
        self._hidden: Any = None
        self._proposal: int | None = None
        self._steps = 0
        self._return = 0.0
        self._done = True
        self._task: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def has_policy(self) -> bool:
        return self._experiment is not None

    @property
    def experiment(self) -> Any | None:
        """The loaded experiment, for :func:`close_player_experiment`."""
        return self._experiment

    @property
    def task(self) -> int | None:
        return self._task

    @property
    def steps(self) -> int:
        """Decisions executed in the live task."""
        return self._steps

    @property
    def episode_return(self) -> float:
        """Native reward accumulated over the live task."""
        return self._return

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, task_id: int) -> None:
        """Start ``task_id`` with a fresh trajectory cache and hidden state."""
        self.environment.set_task(int(task_id))
        self._sequence.reset(seed=self.rollout_seed)
        if self._experiment is not None:
            self._hidden = self._encoder.init_hidden_state(1, self._device)
        if self.sample_actions:
            torch.manual_seed(self.rollout_seed)
        self._proposal = None
        self._steps = 0
        self._return = 0.0
        self._done = False
        self._task = int(task_id)

    def observation(self) -> dict[str, NDArray[np.float32]]:
        """The packet the next decision is chosen from, one row."""
        packet, _, _ = self._sequence.current_timestep
        return {key: np.asarray(value)[0] for key, value in packet.items()}

    def propose(self) -> int:
        """The action the policy would take now; random without a policy.

        Computed once per timestep: the inference call also advances the
        carried hidden state, exactly as one rollout step does, so a second
        call in the same timestep returns the cached proposal.
        """
        if self._done:
            raise ContractError("The task is over; reset the player.")
        if self._proposal is not None:
            return self._proposal
        if self._experiment is None:
            self._proposal = int(self._random.integers(self.environment.action_count))
            return self._proposal
        packet, rl2, time_index = self._sequence.current_timestep
        tensors = {
            key: torch.from_numpy(np.array(value, copy=True))
            .to(self._device)
            .unsqueeze(1)
            for key, value in packet.items()
        }
        with torch.inference_mode(), self._precision():
            actions, self._hidden = self._policy.get_actions(
                obs=tensors,
                rl2s=torch.from_numpy(np.array(rl2, copy=True))
                .to(self._device)
                .unsqueeze(1),
                time_idxs=torch.from_numpy(np.array(time_index, copy=True))
                .to(self._device)
                .unsqueeze(1),
                sample=self.sample_actions,
                hidden_state=self._hidden,
            )
        self._proposal = int(
            np.asarray(actions.squeeze(1).cpu().numpy()).reshape(-1)[0]
        )
        return self._proposal

    def step(self, action: int | None = None) -> tuple[float, bool, dict[str, Any]]:
        """Execute ``action`` (the proposal when ``None``) and return
        ``(reward, done, info)`` from the environment."""
        if self._done:
            raise ContractError("The task is over; reset the player.")
        chosen = self.propose() if action is None else int(action)
        if not 0 <= chosen < self.environment.action_count:
            raise ContractError("The action lies outside the environment's space.")
        if self._proposal is None:
            # Keep the policy's hidden state in step with the timestep even
            # when the human chose without asking it.
            self.propose()
        _, reward, terminated, truncated, info = self._sequence.step(
            np.array([chosen], dtype=np.int64)
        )
        value = float(np.asarray(reward).reshape(-1)[0])
        self._return += value
        self._steps += 1
        self._done = bool(np.logical_or(terminated, truncated).reshape(-1)[0])
        self._proposal = None
        return value, self._done, dict(info)

    def render(self, mode: str | None = None) -> Any:
        return self.environment.render(mode)

    def close(self) -> None:
        self._wrapped.close()


def resolve_checkpoint(
    config: ExperimentConfig, *, checkpoint: str, checkpoint_rule: CheckpointRule
) -> str:
    """The checkpoint name a rule selects, as the evaluator resolves it."""
    if checkpoint_rule not in CHECKPOINT_RULES:
        raise ContractError(f"Unknown checkpoint rule: {checkpoint_rule!r}.")
    if checkpoint_rule == "selected":
        return checkpoint
    if checkpoint != "checkpoint.pt":
        raise ContractError(
            f"The {checkpoint_rule} rule resolves the checkpoint itself; "
            "do not pass one."
        )
    epoch = (
        latest_training_epoch(config.run_directory)
        if checkpoint_rule == "final-epoch"
        else endpoint_training_epoch(
            config.run_directory, epochs=config.training.epochs
        )
    )
    return f"policy_epoch_{epoch}"


def load_player(
    contract: BenchmarkContract,
    config: ExperimentConfig | None,
    *,
    split: str,
    checkpoint: str = "checkpoint.pt",
    checkpoint_rule: CheckpointRule = "endpoint",
    sample_actions: bool = False,
    layout_period: int | None = None,
) -> tuple[PolicyPlayer, Sequence[int], str | None]:
    """Build a player over one rendering environment of ``split``.

    With ``config`` the run's weights are loaded (weights only, no tracker,
    nothing written under the run); without it the player is policy-free.
    Returns the player, the split's ordered task roster and the checkpoint
    name that was loaded.
    """
    plan = contract.evaluation
    if len(plan.rollout_seeds) != 1:
        raise ContractError("Every protocol declares exactly one rollout seed.")
    rollout_seed = int(plan.rollout_seeds[0])
    roster = contract.roster(split)
    experiment = None
    selected: str | None = None
    if config is not None:
        from reasoned_icrl.runtime.training import (
            load_experiment,
            load_selected_checkpoint,
        )

        selected = resolve_checkpoint(
            config, checkpoint=checkpoint, checkpoint_rule=checkpoint_rule
        )
        experiment = load_experiment(
            config, work_directory=config.run_directory, persist_configuration=False
        )
        try:
            load_selected_checkpoint(experiment, selected)
        except Exception:
            close_player_experiment(experiment)
            raise
    # The evaluator builds from the resolved config's runtime mapping; the
    # policy-free player has no run, so the contract's own environment section
    # (the same section the roster is sliced from) stands in.
    mapping = (
        config.as_runtime_mapping()
        if config is not None
        else {"environment": asdict(contract.environment)}
    )
    environment = build_environment(
        mapping, split=plan.splits[split].source, seed=rollout_seed
    )
    if layout_period is not None:
        setter = getattr(environment, "set_layout_period", None)
        if setter is None:
            raise ContractError(f"{contract.name!r} has no layout-change continuation.")
        setter(int(layout_period))
    environment.render_mode = "ansi"
    player = PolicyPlayer(
        experiment,
        environment,
        rollout_seed=rollout_seed,
        sample_actions=sample_actions,
    )
    return player, roster, selected


def close_player_experiment(experiment: Any) -> None:
    """Close the actors a loaded experiment opened."""
    from reasoned_icrl.runtime.training import close_experiment

    close_experiment(experiment)


__all__ = [
    "PolicyPlayer",
    "close_player_experiment",
    "load_player",
    "resolve_checkpoint",
]
