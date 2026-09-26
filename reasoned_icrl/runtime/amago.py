"""Configure pinned AMAGO components and restorable seeded exploration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata
from typing import Any, cast

import gin
import numpy as np
import torch
from amago import cli_utils
from amago.agent import Agent
from amago.envs.exploration import (
    BilevelEpsilonGreedy,
    EpsilonGreedy,
    ExplorationWrapper,
)
from amago.nets.traj_encoders import TrajEncoder
from amago.nets.transformer import FlashAttention, VanillaAttention
from amago.nets.tstep_encoders import FFTstepEncoder, TstepEncoder
from torch import nn

from reasoned_icrl.experiments.contracts import (
    DAT_ARCHITECTURE_ID,
    DAT_SUMMARY_ARCHITECTURE_ID,
    DAT_SUMMARY_V2_ARCHITECTURE_ID,
    DAT_WINDOW_ARCHITECTURE_ID,
    DUAL_CONTENT_ARCHITECTURE_ID,
    DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID,
    FEEDFORWARD_ARCHITECTURE_ID,
    GRU_HISTORY_ARCHITECTURE_ID,
    HISTORY_ARCHITECTURE_ID,
    MEMO_ARCHITECTURE_ID,
    SUMMARY_ARCHITECTURE_ID,
    WINDOW_ARCHITECTURE_ID,
    AttentionVariant,
    ContractError,
    DATSpec,
    MemoSpec,
    SummarySpec,
    WindowSpec,
    architecture_uses_dat,
    architecture_uses_history_packet,
    architecture_uses_memo,
    architecture_uses_summary,
    dat_architecture_id,
    summary_architecture_id,
)
from reasoned_icrl.model.step_encoder import (
    TransitionTstepEncoder,
)
from reasoned_icrl.model.trajectory_encoder import (
    GRUHistoryTrajEncoder,
    HistoryTrajEncoder,
)

BOUND_CARRIERS = frozenset(
    {
        FEEDFORWARD_ARCHITECTURE_ID,
        HISTORY_ARCHITECTURE_ID,
        DAT_ARCHITECTURE_ID,
        DUAL_CONTENT_ARCHITECTURE_ID,
        GRU_HISTORY_ARCHITECTURE_ID,
        SUMMARY_ARCHITECTURE_ID,
        DAT_SUMMARY_ARCHITECTURE_ID,
        DUAL_CONTENT_SUMMARY_ARCHITECTURE_ID,
        DAT_SUMMARY_V2_ARCHITECTURE_ID,
        WINDOW_ARCHITECTURE_ID,
        DAT_WINDOW_ARCHITECTURE_ID,
        MEMO_ARCHITECTURE_ID,
    }
)
"""Architecture identities with a trajectory carrier bound in this module.

A condition may be registered in the contract tables before its carrier lands;
:func:`configure_amago` refuses such an identity rather than binding a carrier
that would run different semantics under its name. ``amago-dat-summary-v2``
(R2) is the summary carrier under the timestep-record relational route;
``amago-dat-window-v1`` (R3) is the window carrier with dual attention over
its per-layer band.
"""


def _dat_spec(settings: Mapping[str, Any], *, width: int, heads: int) -> DATSpec:
    """Rebuild the attention identity from a resolved runtime mapping."""
    values = {
        key: value
        for key, value in settings.items()
        if key not in ("sha256", "layer_indices")
    }
    spec = DATSpec(layer_indices=tuple(settings["layer_indices"]), **values)
    if spec.d_model != width or spec.total_heads != heads:
        raise ContractError("DAT attention geometry disagrees with the model.")
    recorded = settings.get("sha256")
    if recorded is not None and recorded != spec.sha256:
        raise ContractError("Resolved DAT attention identity does not match.")
    return spec


def _summary_spec(settings: Mapping[str, Any], *, width: int) -> SummarySpec:
    """Rebuild the summary identity from a resolved runtime mapping."""
    values = {key: value for key, value in settings.items() if key != "sha256"}
    spec = SummarySpec(**values)
    if spec.d_model != width:
        raise ContractError("Summary geometry disagrees with the model.")
    recorded = settings.get("sha256")
    if recorded is not None and recorded != spec.sha256:
        raise ContractError("Resolved summary identity does not match.")
    return spec


def _memo_spec(settings: Mapping[str, Any], *, width: int) -> MemoSpec:
    """Rebuild the Memo identity from a resolved runtime mapping."""
    values = {key: value for key, value in settings.items() if key != "sha256"}
    spec = MemoSpec(**values)
    if spec.d_model != width:
        raise ContractError("Memo geometry disagrees with the model.")
    recorded = settings.get("sha256")
    if recorded is not None and recorded != spec.sha256:
        raise ContractError("Resolved Memo identity does not match.")
    return spec


def _window_spec(settings: Mapping[str, Any]) -> WindowSpec:
    """Rebuild the window identity from a resolved runtime mapping."""
    values = {key: value for key, value in settings.items() if key != "sha256"}
    spec = WindowSpec(**values)
    recorded = settings.get("sha256")
    if recorded is not None and recorded != spec.sha256:
        raise ContractError("Resolved window identity does not match.")
    return spec


def _wrapped_state(wrapper: ExplorationWrapper) -> object:
    reader = getattr(wrapper.env, "state_dict", None)
    if not callable(reader):
        raise ContractError("The exploration environment is not restorable.")
    return deepcopy(reader())


def _restore_wrapped_state(wrapper: ExplorationWrapper, state: object) -> None:
    loader = getattr(wrapper.env, "load_state_dict", None)
    if not callable(loader) or not isinstance(state, Mapping):
        raise ContractError("The exploration environment is not restorable.")
    loader(deepcopy(state))


class _SeededExploration:
    """Explicit per-environment RNG plus restorable state for AMAGO explorers.

    AMAGO's explorers seed themselves from global NumPy state, which makes a
    resumed run diverge from an uninterrupted one. This re-seeds from the
    environment's own research seed and serializes the generator.
    """

    # Provided by the AMAGO explorer this is mixed into.
    schema: str
    env: Any
    rng: np.random.Generator
    randomize_eps: bool
    batched_envs: int
    global_step: int
    global_multiplier: Any

    @property
    def env_name(self) -> str:
        return cast(str, self.env.get_wrapper_attr("env_name"))

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        output = super().reset(*args, **kwargs)
        seed = int(getattr(self.env, "research_seed", 0))
        self.rng = np.random.default_rng(seed)
        if self.randomize_eps:
            self.global_multiplier = cast(Any, self.rng.random(self.batched_envs))
        return output

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rng": deepcopy(self.rng.bit_generator.state),
            "global_step": self.global_step,
            "global_multiplier": np.asarray(self.global_multiplier).copy(),
            "environment_state": _wrapped_state(cast(ExplorationWrapper, self)),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema") != self.schema:
            raise ContractError("Unsupported exploration state.")
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = cast(dict[str, Any], state["rng"])
        self.global_step = int(cast(int, state["global_step"]))
        self.global_multiplier = cast(
            Any, np.asarray(state["global_multiplier"]).copy()
        )
        _restore_wrapped_state(
            cast(ExplorationWrapper, self), state.get("environment_state")
        )


@gin.configurable
class SeededEpsilonGreedy(_SeededExploration, EpsilonGreedy):
    """AMAGO epsilon-greedy with an explicit per-environment generator."""

    schema = "seeded-epsilon-greedy.v0.3"


@gin.configurable
class SeededBilevelEpsilonGreedy(_SeededExploration, BilevelEpsilonGreedy):
    """AMAGO bilevel exploration with explicit, restorable NumPy state."""

    schema = "seeded-bilevel-epsilon-greedy.v0.3"


@gin.configurable
class SeededTMazeEpsilonGreedy(_SeededExploration, EpsilonGreedy):
    """AMAGO's T-Maze exploration schedule (``examples/04_tmaze.py``), seeded.

    The passive T-Maze tests recall over a corridor of deterministic forward
    moves; plain epsilon-greedy on that corridor is the worst case for the
    task (one random action fails the episode), so AMAGO holds the noise at
    ``0.5 / corridor_length`` inside the corridor and keeps the ordinary
    annealed schedule for the first ``start_window`` and last ``end_window``
    decisions, where the cue is shown and the turn is taken. This is the
    published recipe Memo's fork inherits; the paper's v2 contract binds it.
    """

    schema = "seeded-tmaze-epsilon-greedy.v0.3"

    def __init__(
        self,
        env: Any,
        *,
        corridor_length: int = gin.REQUIRED,
        start_window: int = 0,
        end_window: int = 3,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        steps_anneal: int = 1_000_000,
    ) -> None:
        if int(corridor_length) < 1:
            raise ContractError("The T-Maze exploration schedule needs a corridor.")
        self.corridor_length = int(corridor_length)
        self.start_window = int(start_window)
        self.end_window = int(end_window)
        super().__init__(
            env, eps_start=eps_start, eps_end=eps_end, steps_anneal=steps_anneal
        )

    def live_corridor(self) -> int:
        """The corridor of the wrapped environment's live task when it exposes
        one (the v3 protocol draws it per task), else the bound length."""
        try:
            return int(self.env.get_wrapper_attr("corridor_length"))
        except AttributeError:
            return self.corridor_length

    def current_eps(self, local_step: Any) -> Any:
        current = np.asarray(super().current_eps(local_step), dtype=np.float64)
        steps = np.asarray(local_step).reshape(-1)
        corridor_length = self.live_corridor()
        corridor = (steps > self.start_window) & (
            steps < corridor_length - self.end_window
        )
        return np.where(corridor, 0.5 / corridor_length, current)


def exploration_wrapper_type(
    training: Mapping[str, Any],
) -> type[ExplorationWrapper]:
    """Resolve the two explicit exploration schedules used by presets."""
    name = str(training.get("exploration", "epsilon_greedy"))
    if name == "epsilon_greedy":
        return SeededEpsilonGreedy
    if name == "bilevel_epsilon_greedy":
        return SeededBilevelEpsilonGreedy
    if name == "tmaze_epsilon_greedy":
        return SeededTMazeEpsilonGreedy
    raise ContractError(f"Unknown exploration schedule: {name}.")


@dataclass(frozen=True, slots=True)
class AMAGOComponents:
    """Component classes passed to :class:`amago.Experiment`."""

    timestep_encoder: type[TstepEncoder]
    trajectory_encoder: type[TrajEncoder]
    agent: type[Agent]
    exploration: type[ExplorationWrapper]
    architecture_id: str


def assert_amago_version() -> None:
    installed = metadata.version("amago")
    if installed != "3.4.0":
        raise ContractError(f"AMAGO 3.4.0 is required, found {installed}.")
    if metadata.version("gymnasium") != "0.29.1":
        raise ContractError("Gymnasium 0.29.1 is required by AMAGO 3.4.0.")


def configure_amago(
    model: Mapping[str, Any],
    training: Mapping[str, Any],
) -> AMAGOComponents:
    """Translate the resolved YAML mapping to one finalized Gin configuration."""
    assert_amago_version()
    compile_value = training.get("torch_compile")
    if not isinstance(compile_value, bool):
        raise ContractError("Training presets must explicitly select torch_compile.")
    torch.compiler.set_stance("default" if compile_value else "force_eager")

    width = int(model["width"])
    layers = int(model["layers"])
    heads = int(model["heads"])
    multiplier = int(model["feedforward_multiplier"])
    if min(width, layers, heads, multiplier) <= 0 or width % heads:
        raise ContractError("Model width, layers, and heads are invalid.")

    gin.clear_config()
    architecture_id = str(model["architecture_id"])
    if architecture_id not in BOUND_CARRIERS:
        raise ContractError(f"No trajectory carrier is bound for {architecture_id!r}.")
    params: dict[str, object] = {}
    timestep_type: type[TstepEncoder]
    if architecture_uses_history_packet(architecture_id):
        timestep_type = TransitionTstepEncoder
        target = "reasoned_icrl.model.step_encoder.TransitionTstepEncoder"
        params.update(
            {
                f"{target}.evidence": model["evidence"],
                f"{target}.bypass": model["bypass"],
                f"{target}.trunk": model.get("trunk", "ff"),
                f"{target}.public_contract": model["public_contract"],
                f"{target}.initialization_seed": int(
                    model.get("initialization_seed", 0)
                ),
            }
        )
    elif architecture_id == FEEDFORWARD_ARCHITECTURE_ID:
        timestep_type = FFTstepEncoder
        target = "amago.nets.tstep_encoders.FFTstepEncoder"
        params.update(
            {
                f"{target}.n_layers": 2,
                f"{target}.d_hidden": 128,
                f"{target}.d_output": 64,
                f"{target}.normalize_inputs": True,
                # The memoryless reference sees the same normalized current
                # state as every history condition, and nothing else.
                f"{target}.specify_obs_keys": ["current"],
            }
        )
    else:
        raise ContractError(f"Unknown encoder architecture: {architecture_id!r}.")
    trajectory_name = str(model["trajectory_encoder"])
    attention_backend = model["attention_backend"]
    attention_type: type[nn.Module]
    if attention_backend == "vanilla":
        attention_type = VanillaAttention
    elif attention_backend == "flash":
        attention_type = FlashAttention
    else:
        raise ContractError("Unknown attention backend in resolved model settings.")
    if trajectory_name == "transformer":
        transformer_kwargs: dict[str, object] = {
            "d_ff": width * multiplier,
            "n_heads": heads,
        }
        transformer_kwargs["attention_type"] = attention_type
        trajectory_type = cli_utils.switch_traj_encoder(
            params,
            "transformer",
            memory_size=width,
            layers=layers,
            **transformer_kwargs,
        )
    elif trajectory_name == "feedforward":
        trajectory_type = cli_utils.switch_traj_encoder(
            params,
            "ff",
            memory_size=width,
            layers=layers,
            d_ff=width * multiplier,
        )
    elif trajectory_name == "gru":
        # Bound explicitly, as the DAT carrier is: switch_traj_encoder("rnn")
        # would configure AMAGO's bare GRUTrajEncoder, which reads no packet.
        if architecture_id != GRU_HISTORY_ARCHITECTURE_ID or model["bypass"]:
            raise ContractError("The GRU carrier runs the raw packet without bypass.")
        trajectory_type = GRUHistoryTrajEncoder
        target = "reasoned_icrl.model.trajectory_encoder.GRUHistoryTrajEncoder"
        params.update({f"{target}.d_model": width, f"{target}.n_layers": layers})
    else:
        raise ContractError(f"Unknown trajectory encoder: {trajectory_name!r}.")
    if architecture_id == GRU_HISTORY_ARCHITECTURE_ID and trajectory_name != "gru":
        raise ContractError("The GRU architecture needs the gru trajectory encoder.")

    if architecture_id == HISTORY_ARCHITECTURE_ID:
        trajectory_type = HistoryTrajEncoder
        target = "reasoned_icrl.model.trajectory_encoder.HistoryTrajEncoder"
        params.update({f"{target}.bypass": model["bypass"], f"{target}.d_model": width})

    if architecture_id in (WINDOW_ARCHITECTURE_ID, DAT_WINDOW_ARCHITECTURE_ID):
        # The sliding-window cells: the backbone under a per-layer band, bound
        # explicitly with the window identity and, for the revised band, the
        # attention identity whose clipping distance is the window length.
        from reasoned_icrl.model.trajectory_encoder import WindowTrajEncoder

        window_settings = model.get("window")
        if not isinstance(window_settings, Mapping):
            raise ContractError("The window architecture needs model.window.")
        window = _window_spec(window_settings)
        if model["bypass"] or trajectory_name != "transformer":
            raise ContractError("The window cell runs the Transformer without bypass.")
        window_dat_settings = model.get("dat")
        window_dat: DATSpec | None = None
        if architecture_id == DAT_WINDOW_ARCHITECTURE_ID:
            if not isinstance(window_dat_settings, Mapping):
                raise ContractError("The dual-attention window needs model.dat.")
            window_dat = _dat_spec(window_dat_settings, width=width, heads=heads)
            if window_dat.mode != "dat":
                raise ContractError(
                    "The revised band runs dual relational attention; the "
                    "dual-content control is not a window cell."
                )
            if window_dat.max_relative_distance != window.capacity:
                raise ContractError(
                    "A window cell's DAT clipping distance must be its window length."
                )
        elif window_dat_settings is not None:
            raise ContractError("The ordinary window carries no dual-attention block.")
        trajectory_type = WindowTrajEncoder
        target = "reasoned_icrl.model.trajectory_encoder.WindowTrajEncoder"
        params.update(
            {
                f"{target}.spec": window,
                f"{target}.dat": window_dat,
                f"{target}.d_model": width,
                f"{target}.initialization_seed": int(
                    model.get("initialization_seed", 0)
                ),
            }
        )

    if architecture_uses_memo(architecture_id):
        # The Memo comparator: the ordinary backbone under accumulated
        # summaries, bound explicitly with its own identity (ME1).
        from reasoned_icrl.model.trajectory_encoder import MemoTrajEncoder

        memo_settings = model.get("memo")
        if not isinstance(memo_settings, Mapping):
            raise ContractError("The Memo architecture needs model.memo.")
        if model.get("dat") is not None:
            raise ContractError("The Memo comparator carries no dual-attention block.")
        if model["bypass"] or trajectory_name != "transformer":
            raise ContractError("The Memo cell runs the Transformer without bypass.")
        trajectory_type = MemoTrajEncoder
        target = "reasoned_icrl.model.trajectory_encoder.MemoTrajEncoder"
        params.update(
            {
                f"{target}.spec": _memo_spec(memo_settings, width=width),
                f"{target}.d_model": width,
                f"{target}.initialization_seed": int(
                    model.get("initialization_seed", 0)
                ),
            }
        )

    if architecture_uses_summary(architecture_id):
        # The segment/summary carrier: bound explicitly like the DAT carrier,
        # with its summary identity and, for the dual-attention cells, the
        # attention identity whose clipping distance is the segment capacity.
        from reasoned_icrl.model.trajectory_encoder import SummaryTrajEncoder

        summary_settings = model.get("summary")
        if not isinstance(summary_settings, Mapping):
            raise ContractError("A summary architecture needs model.summary.")
        summary = _summary_spec(summary_settings, width=width)
        dat_settings = model.get("dat")
        dat_spec: DATSpec | None = None
        if dat_settings is not None:
            if not isinstance(dat_settings, Mapping):
                raise ContractError("model.dat must be a mapping when configured.")
            dat_spec = _dat_spec(dat_settings, width=width, heads=heads)
            if dat_spec.max_relative_distance != summary.capacity:
                raise ContractError(
                    "A summary cell's DAT clipping distance must be its capacity."
                )
        attention: AttentionVariant = "ordinary"
        if dat_spec is not None:
            attention = "dual_content" if dat_spec.mode == "dual_content" else "dat"
        if summary.relational_sources == "timestep_records" and dat_spec is None:
            raise ContractError(
                "The timestep-record relational route needs a dual-attention block."
            )
        if architecture_id != summary_architecture_id(
            attention, summary.regime, summary.relational_sources
        ):
            raise ContractError("Resolved summary regime disagrees with the identity.")
        if model["bypass"] or trajectory_name != "transformer":
            raise ContractError("Summary cells run the Transformer without bypass.")
        trajectory_type = SummaryTrajEncoder
        target = "reasoned_icrl.model.trajectory_encoder.SummaryTrajEncoder"
        params.update(
            {
                f"{target}.spec": summary,
                f"{target}.dat": dat_spec,
                f"{target}.d_model": width,
                f"{target}.initialization_seed": int(
                    model.get("initialization_seed", 0)
                ),
            }
        )
    elif (
        architecture_uses_dat(architecture_id)
        and architecture_id != DAT_WINDOW_ARCHITECTURE_ID
    ):
        # switch_traj_encoder would bind `memory_size` for an unknown registered
        # architecture; this carrier takes `d_model`, so bind it explicitly.
        from reasoned_icrl.model.trajectory_encoder import DATTrajEncoder

        settings = model.get("dat")
        if not isinstance(settings, Mapping):
            raise ContractError("A dual-attention architecture needs model.dat.")
        spec = _dat_spec(settings, width=width, heads=heads)
        if architecture_id != dat_architecture_id(spec.mode):
            raise ContractError("Resolved DAT mode disagrees with the architecture.")
        if model["bypass"]:
            raise ContractError("Dual-attention conditions run without the bypass.")
        trajectory_type = DATTrajEncoder
        target = "reasoned_icrl.model.trajectory_encoder.DATTrajEncoder"
        params.update(
            {
                f"{target}.spec": spec,
                f"{target}.d_model": width,
                f"{target}.initialization_seed": int(
                    model.get("initialization_seed", 0)
                ),
            }
        )

    # switch_agent registers AMAGO's native agent gin parameters; the
    # concrete class is then MatchedBaselineAgent, whose component-isolated
    # initialization keeps the shared parts of every condition identical at
    # step zero. That is what makes the 2x2 matched, not merely same-seeded.
    from reasoned_icrl.model.agent import MatchedBaselineAgent

    cli_utils.switch_agent(
        params,
        "agent",
        reward_multiplier=float(training["reward_multiplier"]),
    )
    agent_type = MatchedBaselineAgent
    public_contract = model.get("public_contract", {})
    if (
        isinstance(public_contract, Mapping)
        and public_contract.get("environment_protocol") == "match-pattern-symbolic-v1"
    ):
        from reasoned_icrl.experiments.match_pattern import QUERY_FIELD

        params["reasoned_icrl.model.agent.MatchedBaselineAgent.query_loss_field"] = (
            QUERY_FIELD
        )
    params["reasoned_icrl.model.agent.MatchedBaselineAgent.initialization_seed"] = int(
        model["initialization_seed"]
    )
    exploration_type = exploration_wrapper_type(training)
    exploration_path = f"{exploration_type.__module__}.{exploration_type.__name__}"
    if str(training["exploration"]) == "epsilon_greedy":
        params.update(
            {
                f"{exploration_path}.eps_start": 1.0,
                f"{exploration_path}.eps_end": 0.05,
                f"{exploration_path}.steps_anneal": int(
                    training["epsilon_anneal_steps"]
                ),
            }
        )
    elif str(training["exploration"]) == "tmaze_epsilon_greedy":
        # The rollout horizon is the corridor plus the turn; the corridor is
        # the window AMAGO holds at low noise.
        params.update(
            {
                f"{exploration_path}.corridor_length": int(
                    training["exploration_rollout_horizon"]
                )
                - 1,
                f"{exploration_path}.eps_start": 1.0,
                f"{exploration_path}.eps_end": 0.05,
                f"{exploration_path}.steps_anneal": int(
                    training["epsilon_anneal_steps"]
                ),
            }
        )
    else:
        params.update(
            {
                f"{exploration_path}.rollout_horizon": int(
                    training["exploration_rollout_horizon"]
                ),
                f"{exploration_path}.steps_anneal": int(
                    training["epsilon_anneal_steps"]
                ),
            }
        )

    critic = model.get("critic")
    if critic is not None:
        if not isinstance(critic, Mapping):
            raise ContractError("model.critic must be a mapping when configured.")
        target = "amago.nets.actor_critic.NCriticsTwoHot"
        params.update(
            {
                f"{target}.min_return": float(critic["min_return"]),
                f"{target}.max_return": float(critic["max_return"]),
                f"{target}.output_bins": int(critic["output_bins"]),
            }
        )

    native_agent = training.get("native_agent")
    if native_agent is not None:
        # Already validated by NativeAgentConfig; just bind the coefficients.
        if not isinstance(native_agent, Mapping):
            raise ContractError("training.native_agent must be a mapping.")
        for name, value in native_agent.items():
            params[f"amago.agent.Agent.{name}"] = value
    if "popart_initial_second_moment" in training:
        moment = float(training["popart_initial_second_moment"])
        if not 0 < moment < float("inf"):
            raise ContractError("PopArt initial second moment must be positive/finite.")
        params["amago.nets.actor_critic.PopArtLayer.init_nu"] = moment
    cli_utils.use_config(params, finalize=True)
    return AMAGOComponents(
        timestep_encoder=cast(type[TstepEncoder], timestep_type),
        trajectory_encoder=trajectory_type,
        agent=agent_type,
        exploration=exploration_type,
        architecture_id=architecture_id,
    )
