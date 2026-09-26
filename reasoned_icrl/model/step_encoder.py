"""Project-owned timestep encoders.

One encoder implements every history condition. It is an ordinary AMAGO
``TstepEncoder`` built out of AMAGO's own registered trunks -- ``FFTstepEncoder``
today, ``CNNTstepEncoder`` when observations become pixels -- and adds exactly
two things on top of that contract:

1. **Transition evidence.** The token may be built from the causal transition
   ``(current, previous, outcome, event)`` rather than the current timestep
   alone, so one token carries what the action actually did.
2. **State bypass.** The current observation may additionally be encoded by a
   second trunk whose output travels *around* the trajectory encoder, straight
   to the actor and critic, so control does not have to route through memory.

Timestep encoders stay stateless. All recurrence lives in the trajectory encoder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import gin
import gymnasium as gym
import numpy as np
import torch
from amago.loading import MAGIC_PAD_VAL
from amago.nets.cnn import GridworldCNN
from amago.nets.ff import Normalization
from amago.nets.tstep_encoders import (
    CNNTstepEncoder,
    FFTstepEncoder,
    TstepEncoder,
    register_tstep_encoder,
)
from amago.nets.utils import add_activation_log, symlog
from torch import nn
from torch.nn import functional as F

from reasoned_icrl.experiments.contracts import ContractError, Evidence
from reasoned_icrl.model.utils import reset_named_linears

PACKET_KEYS = ("current", "previous", "outcome", "event", "valid")

TOKEN_KEYS: dict[Evidence, tuple[str, ...]] = {
    "raw": ("current",),
    "transition": ("current", "previous", "outcome", "event"),
}
"""Which observation fields reach the token trunk, per evidence setting."""

BYPASS_KEYS = ("current",)
"""What the state-bypass trunk sees: the current observation, nothing else."""

Trunk = Literal["ff", "cnn", "xland"]
_EMPTY_RL2 = gym.spaces.Box(-1.0, 1.0, (0,), dtype=np.float32)

XLAND_TRUNK_FIELDS = (
    "grid_tile",
    "grid_color",
    "direction",
    "attempt_done",
    "goal",
    "attempt_time",
)
"""The packet layout the XLand trunk reads, in order
(``environments/xland_minigrid.py``)."""


class XLandGridTrunk(TstepEncoder):
    """AMAGO's ``XLandMGTstepEncoder`` (examples/11_xland_minigrid.py, v3.4.0)
    over the packet's ``current`` vector.

    The packet stores the native ids rescaled to ``[-1, 1]``; the trunk decodes
    them back with the bounds the public contract declares, embeds the two view
    layers (``Embedding(15, 8)`` each, the example's ``grid_id_dim``), runs the
    example's ``GridworldCNN(32, 48, 64)`` and projects to ``grid_emb_dim``;
    embeds the goal ids and projects them to ``goal_emb_dim``; then merges grid,
    goal, the direction/done flags and ``symlog`` of the RL2 features through
    ``ff_dim`` into ``d_output``, layer-normalised. The example computes the
    grid embedding and then overwrites it with the CNN of the raw ids; this
    trunk keeps the embedding path the example declares.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.Space[Any],
        *,
        public_contract: dict[str, Any],
        d_output: int,
        grid_id_dim: int = 8,
        grid_emb_dim: int = 128,
        goal_id_dim: int = 8,
        goal_emb_dim: int = 32,
        ff_dim: int = 256,
    ) -> None:
        super().__init__(obs_space, rl2_space)
        if tuple(obs_space.spaces) != ("current",):
            raise ContractError("The XLand trunk reads the current token only.")
        layout: dict[str, tuple[int, int, float, float]] = {}
        offset = 0
        for field in public_contract["fields"]:
            width = len(field["low"])
            low, high = set(field["low"]), set(field["high"])
            if len(low) != 1 or len(high) != 1:
                raise ContractError("XLand packet fields share one bound per field.")
            layout[str(field["name"])] = (offset, width, low.pop(), high.pop())
            offset += width
        if tuple(layout) != XLAND_TRUNK_FIELDS:
            raise ContractError("The XLand trunk needs the XLand packet layout.")
        assert obs_space["current"].shape is not None
        if offset != int(obs_space["current"].shape[0]):
            raise ContractError("XLand packet width disagrees with its contract.")
        self.layout = layout
        cells = layout["grid_tile"][1]
        side = round(cells**0.5)
        if side * side != cells or layout["grid_color"][1] != cells:
            raise ContractError("The XLand view is square with two layers.")
        self.side = side
        tile_tokens = int(layout["grid_tile"][3] - layout["grid_tile"][2]) + 1
        color_tokens = int(layout["grid_color"][3] - layout["grid_color"][2]) + 1
        goal_tokens = int(layout["goal"][3] - layout["goal"][2]) + 1
        # The example sizes one vocabulary from the wrapper's Box(0, 14).
        grid_tokens = max(tile_tokens, color_tokens, goal_tokens)
        self.tile_embedding = nn.Embedding(grid_tokens, grid_id_dim)
        self.color_embedding = nn.Embedding(grid_tokens, grid_id_dim)
        self.grid_processor = GridworldCNN(
            img_shape=(side, side, 2 * grid_id_dim),
            channels_first=False,
            activation="leaky_relu",
            channels=[32, 48, 64],
        )
        with torch.no_grad():
            blank = torch.zeros((1, 1, side, side, 2 * grid_id_dim))
            grid_out_dim = self.grid_processor(blank, from_float=True).shape[-1]
        self.grid_rep_ff = nn.Linear(grid_out_dim, grid_emb_dim)
        self.goal_embedding = nn.Embedding(grid_tokens, goal_id_dim)
        goal_inp_dim = goal_id_dim * layout["goal"][1]
        self.goal_rep_ff = nn.Sequential(
            nn.Linear(goal_inp_dim, goal_inp_dim),
            nn.LeakyReLU(),
            nn.Linear(goal_inp_dim, goal_emb_dim),
        )
        extras = layout["direction"][1] + layout["attempt_done"][1]
        extras += layout["attempt_time"][1]
        assert rl2_space.shape is not None
        self.merge = nn.Sequential(
            nn.Linear(
                grid_emb_dim + goal_emb_dim + extras + rl2_space.shape[-1], ff_dim
            ),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, d_output),
        )
        self.out_norm = Normalization("layer", d_output)
        self.out_dim = d_output

    @property
    def emb_dim(self) -> int:
        return self.out_dim

    def _slice(self, current: torch.Tensor, name: str) -> torch.Tensor:
        offset, width, _, _ = self.layout[name]
        return current[..., offset : offset + width]

    def _ids(self, current: torch.Tensor, name: str) -> torch.Tensor:
        """Invert ``normalize_fields``: ``[-1, 1]`` back to the declared ids."""
        _, _, low, high = self.layout[name]
        values = (self._slice(current, name) + 1.0) / 2.0 * max(high - low, 1.0) + low
        return torch.round(values).long().clamp_(int(low), int(high))

    def inner_forward(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        log_dict: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        current = obs["current"].float()
        # AMAGO pads invalid steps with MAGIC_PAD_VAL; the ids are clamped and
        # every output row is masked again by the caller.
        current = torch.where(
            current == MAGIC_PAD_VAL, torch.zeros_like(current), current
        )
        side = self.side
        tiles = self._ids(current, "grid_tile").reshape(*current.shape[:-1], side, side)
        colors = self._ids(current, "grid_color").reshape(
            *current.shape[:-1], side, side
        )
        grid = torch.cat((self.tile_embedding(tiles), self.color_embedding(colors)), -1)
        grid_rep = self.grid_processor(grid, from_float=True)
        add_activation_log("encoder-grid-rep", grid_rep, log_dict)
        grid_rep = F.leaky_relu(self.grid_rep_ff(grid_rep))
        add_activation_log("encoder-grid-rep-ff", grid_rep, log_dict)
        goal = self.goal_embedding(self._ids(current, "goal"))
        goal = goal.reshape(*goal.shape[:-2], -1)
        goal_rep = F.leaky_relu(self.goal_rep_ff(goal))
        add_activation_log("encoder-goal-rep-ff", goal_rep, log_dict)
        extras = torch.cat(
            (
                self._slice(current, "direction"),
                self._slice(current, "attempt_done"),
                self._slice(current, "attempt_time"),
                symlog(rl2s),
            ),
            -1,
        )
        merged = self.merge(torch.cat((grid_rep, goal_rep, extras), -1))
        add_activation_log("encoder-merged-rep", merged, log_dict)
        return cast(torch.Tensor, self.out_norm(merged))


@dataclass(frozen=True, slots=True)
class PacketSpec:
    """The tensor contract between the timestep and trajectory encoders.

    A packet is ``[token | state | valid]``: ``token_dim`` columns of history
    evidence, ``token_dim`` columns of bypassed current state when ``bypass`` is
    on, and one validity column. Its ``sha256`` is the encoder identity written
    into checkpoints.
    """

    observation_dim: int
    action_dim: int
    public_contract: dict[str, Any]
    evidence: Evidence
    bypass: bool
    trunk: Trunk = "ff"
    token_dim: int = 64
    schema: str = "history-packet.v2"

    @property
    def state_dim(self) -> int:
        return self.token_dim if self.bypass else 0

    @property
    def flat_dim(self) -> int:
        return self.token_dim + self.state_dim + 1

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


def _subspace(obs_space: gym.spaces.Dict, keys: tuple[str, ...]) -> gym.spaces.Dict:
    return gym.spaces.Dict({key: obs_space[key] for key in keys})


def _build_trunk(
    trunk: Trunk,
    obs_space: gym.spaces.Dict,
    rl2_space: gym.Space[Any],
    *,
    d_output: int,
    d_hidden: int,
    n_layers: int,
    public_contract: dict[str, Any] | None = None,
) -> TstepEncoder:
    """Instantiate one AMAGO trunk over a restricted view of the observation."""
    if trunk == "ff":
        return FFTstepEncoder(
            obs_space,
            rl2_space,
            n_layers=n_layers,
            d_hidden=d_hidden,
            d_output=d_output,
        )
    if trunk == "cnn":
        return CNNTstepEncoder(obs_space, rl2_space, d_output=d_output)
    if trunk == "xland":
        if public_contract is None:
            raise ContractError("The XLand trunk needs the public contract.")
        return XLandGridTrunk(
            obs_space, rl2_space, public_contract=public_contract, d_output=d_output
        )
    raise ContractError(f"Unknown timestep trunk: {trunk!r}.")


@gin.configurable
@register_tstep_encoder("transition")
class TransitionTstepEncoder(TstepEncoder):
    """Timestep encoder with selectable transition evidence and state bypass.

    Args:
        obs_space: must be the public outcome adapter's five-key space.
        rl2_space: AMAGO's previous action and reward features.
        evidence: ``"raw"`` for a timestep token, ``"transition"`` for a
            transition token that also sees the previous and outcome states.
        bypass: whether to add the current-state branch around memory.
        trunk: which AMAGO encoder builds each branch.
        public_contract: the environment's declared public field contract,
            recorded in the packet identity.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.Space[Any],
        evidence: Evidence = "transition",
        bypass: bool = True,
        trunk: Trunk = "ff",
        token_dim: int = 64,
        d_hidden: int = 128,
        n_layers: int = 2,
        public_contract: dict[str, Any] | None = None,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__(obs_space, rl2_space)
        if set(obs_space.spaces) != set(PACKET_KEYS):
            raise ContractError(
                "History conditions require the public outcome adapter's fields."
            )
        if not isinstance(public_contract, dict):
            raise ContractError("History observation space lacks its public contract.")
        if evidence not in TOKEN_KEYS:
            raise ContractError(f"Unknown history evidence: {evidence!r}.")
        if trunk == "xland" and (evidence != "raw" or bypass):
            raise ContractError("The XLand trunk reads the raw current token only.")
        assert obs_space["current"].shape is not None
        assert rl2_space.shape is not None
        self.spec = PacketSpec(
            observation_dim=int(obs_space["current"].shape[0]),
            action_dim=int(rl2_space.shape[-1]) - 1,
            public_contract=public_contract,
            evidence=evidence,
            bypass=bypass,
            trunk=trunk,
            token_dim=token_dim,
        )
        self.register_buffer(
            "protocol_identity",
            torch.tensor(list(bytes.fromhex(self.spec.sha256)), dtype=torch.uint8),
        )
        self.token_keys = TOKEN_KEYS[evidence]
        self.token_encoder = _build_trunk(
            trunk,
            _subspace(obs_space, self.token_keys),
            rl2_space,
            d_output=token_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            public_contract=public_contract,
        )
        reset_named_linears(self.token_encoder, initialization_seed, "timestep")
        self.state_encoder: TstepEncoder | None = None
        if bypass:
            self.state_encoder = _build_trunk(
                trunk,
                _subspace(obs_space, BYPASS_KEYS),
                _EMPTY_RL2,
                d_output=token_dim,
                d_hidden=d_hidden,
                n_layers=n_layers,
            )
            reset_named_linears(self.state_encoder, initialization_seed, "current")

    @property
    def emb_dim(self) -> int:
        return self.spec.flat_dim

    def inner_forward(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        log_dict: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        valid = obs["valid"] == 1
        if not valid.any():
            # An entirely padded batch must not move AMAGO's input statistics.
            return rl2s.new_zeros((*rl2s.shape[:2], self.emb_dim))
        token_inputs = {
            key: torch.where(valid, obs[key].float(), MAGIC_PAD_VAL)
            for key in self.token_keys
        }
        feedback = rl2s
        if self.spec.evidence == "transition":
            # Previous, outcome and the causal feedback only exist once the
            # environment has actually produced a physical transition tuple.
            available = valid & (obs["event"][..., :1] == 1)
            for key in ("previous", "outcome"):
                token_inputs[key] = torch.where(
                    valid, torch.where(available, obs[key].float(), 0.0), MAGIC_PAD_VAL
                )
            feedback = torch.where(available, feedback, 0.0)
        feedback = torch.where(valid, feedback, MAGIC_PAD_VAL)
        token = self.token_encoder(token_inputs, feedback, log_dict=log_dict)
        parts = [token]
        if self.state_encoder is not None:
            state = {
                key: torch.where(valid, obs[key].float(), MAGIC_PAD_VAL)
                for key in BYPASS_KEYS
            }
            parts.append(self.state_encoder(state, rl2s[..., :0], log_dict=log_dict))
        parts.append(valid.to(token.dtype))
        return torch.where(valid, torch.cat(parts, -1), 0.0)


__all__ = [
    "BYPASS_KEYS",
    "PACKET_KEYS",
    "TOKEN_KEYS",
    "XLAND_TRUNK_FIELDS",
    "PacketSpec",
    "TransitionTstepEncoder",
    "Trunk",
    "XLandGridTrunk",
]
