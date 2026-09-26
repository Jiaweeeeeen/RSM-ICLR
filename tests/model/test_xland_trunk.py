"""The XLand grid trunk: AMAGO's example encoder inside the packet contract.

The trunk reads the ``current`` token only, decodes the declared ids exactly,
reproduces the example's layer structure, keeps the encoder stateless per
timestep, masks padded rows through the shared ``TransitionTstepEncoder`` and
changes the packet identity exactly when it is selected.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from amago.loading import MAGIC_PAD_VAL
from amago.nets.cnn import GridworldCNN

from reasoned_icrl.environments.base import OUTCOME_PROTOCOL
from reasoned_icrl.environments.utils import normalize_fields
from reasoned_icrl.environments.xland_minigrid import (
    XLAND_ACTIONS,
    XLAND_GOAL_LENGTH,
    XLAND_PROTOCOL,
    public_fields,
)
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.model.step_encoder import (
    XLAND_TRUNK_FIELDS,
    TransitionTstepEncoder,
    XLandGridTrunk,
)

FIELDS = public_fields(XLAND_GOAL_LENGTH)
WIDTH = sum(len(field.low) for field in FIELDS)
LOW = np.concatenate([np.asarray(f.low) for f in FIELDS])
HIGH = np.concatenate([np.asarray(f.high) for f in FIELDS])
CONTRACT = {
    "schema": OUTCOME_PROTOCOL,
    "fields": [
        {
            "name": f.name,
            "shape": f.shape,
            "dtype": f.dtype,
            "low": f.low,
            "high": f.high,
        }
        for f in FIELDS
    ],
    "environment_protocol": XLAND_PROTOCOL,
    "goal_schedule_sha256": None,
}
RL2 = gym.spaces.Box(-np.inf, np.inf, (XLAND_ACTIONS + 1,), np.float32)


def packet_space() -> gym.spaces.Dict:
    box = gym.spaces.Box(-1, 1, (WIDTH,), np.float32)
    return gym.spaces.Dict(
        {
            "current": box,
            "previous": box,
            "outcome": box,
            "event": gym.spaces.Box(0, 1, (3,), np.float32),
            "valid": gym.spaces.Box(0, 1, (1,), np.float32),
        }
    )


def native_values(rng: np.random.Generator) -> np.ndarray:
    """One random native observation: ids within bounds, unit fields in [0, 1]."""
    values = np.zeros(WIDTH)
    offset = 0
    for field in FIELDS:
        width = len(field.low)
        high = np.asarray(field.high)
        if field.name == "direction":
            values[offset + int(rng.integers(width))] = 1.0
        elif high.max() > 1.0:
            values[offset : offset + width] = rng.integers(0, high + 1)
        else:
            values[offset : offset + width] = rng.random(width)
        offset += width
    return values


def batch(rows: int, seed: int = 0) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rng = np.random.default_rng(seed)
    current = np.stack(
        [normalize_fields(native_values(rng), LOW, HIGH) for _ in range(rows)]
    )
    obs = {
        "current": torch.tensor(current).unsqueeze(0),
        "previous": torch.zeros(1, rows, WIDTH),
        "outcome": torch.zeros(1, rows, WIDTH),
        "event": torch.zeros(1, rows, 3),
        "valid": torch.ones(1, rows, 1),
    }
    rl2 = torch.zeros(1, rows, XLAND_ACTIONS + 1)
    rl2[:, 1:, 2] = 1
    rl2[:, :, 0] = torch.linspace(0, 1, rows)
    return obs, rl2


def trunk(d_output: int = 64) -> XLandGridTrunk:
    torch.manual_seed(0)
    space = gym.spaces.Dict({"current": packet_space()["current"]})
    return XLandGridTrunk(space, RL2, public_contract=CONTRACT, d_output=d_output)


def test_the_trunk_reproduces_the_example_layer_structure() -> None:
    encoder = trunk()
    assert tuple(encoder.layout) == XLAND_TRUNK_FIELDS
    assert encoder.side == 5
    # One vocabulary of 15 ids (the wrapper's Box(0, 14)), 8-wide embeddings.
    assert encoder.tile_embedding.weight.shape == (15, 8)
    assert encoder.color_embedding.weight.shape == (15, 8)
    assert encoder.goal_embedding.weight.shape == (15, 8)
    assert isinstance(encoder.grid_processor, GridworldCNN)
    assert [
        c.out_channels
        for c in (
            encoder.grid_processor.conv1,
            encoder.grid_processor.conv2,
            encoder.grid_processor.conv3,
        )
    ] == [32, 48, 64]
    assert encoder.grid_processor.conv1.in_channels == 16
    assert encoder.grid_rep_ff.out_features == 128
    assert encoder.goal_rep_ff[-1].out_features == 32
    assert encoder.merge[0].in_features == 128 + 32 + 4 + 1 + 1 + XLAND_ACTIONS + 1
    assert encoder.merge[-1].out_features == 64 and encoder.emb_dim == 64


def test_ids_decode_exactly_from_the_normalised_packet() -> None:
    encoder = trunk()
    rng = np.random.default_rng(3)
    values = native_values(rng)
    current = torch.tensor(normalize_fields(values, LOW, HIGH)).reshape(1, 1, -1)
    offset = 0
    for field in FIELDS:
        width = len(field.low)
        if np.asarray(field.high).max() > 1.0:
            decoded = encoder._ids(current, field.name).reshape(-1).numpy()
            np.testing.assert_array_equal(decoded, values[offset : offset + width])
        offset += width


def test_the_trunk_is_stateless_per_timestep_and_masks_padding() -> None:
    obs, rl2 = batch(6)
    step = TransitionTstepEncoder(
        packet_space(),
        RL2,
        evidence="raw",
        bypass=False,
        trunk="xland",
        public_contract=CONTRACT,
    ).eval()
    assert step.spec.trunk == "xland"
    whole = step(obs, rl2)
    assert whole.shape == (1, 6, step.emb_dim)
    incremental = torch.cat(
        [
            step({k: v[:, i : i + 1] for k, v in obs.items()}, rl2[:, i : i + 1])
            for i in range(6)
        ],
        1,
    )
    torch.testing.assert_close(whole, incremental, rtol=1e-5, atol=1e-6)
    # Endpoint and event payloads never reach a raw token.
    altered = {k: v.clone() for k, v in obs.items()}
    altered["previous"] += 17
    altered["outcome"][:, 0] = torch.inf
    altered["event"] = 1 - altered["event"]
    torch.testing.assert_close(step(altered, rl2), whole, rtol=0, atol=0)
    # A padded row is zero, whatever AMAGO wrote into it.
    padded = {k: v.clone() for k, v in obs.items()}
    padded["valid"][:, -1] = 0
    padded["current"][:, -1] = MAGIC_PAD_VAL
    out = step(padded, rl2)
    assert not out[:, -1].any()
    torch.testing.assert_close(out[:, :-1], whole[:, :-1])
    # An entirely padded batch is zero without touching the input statistics.
    empty = {k: v.clone() for k, v in obs.items()}
    empty["valid"][:] = 0
    assert not step(empty, rl2).any()


def test_the_trunk_changes_the_packet_identity_only_when_selected() -> None:
    kwargs = dict(evidence="raw", bypass=False, public_contract=CONTRACT)
    shared = TransitionTstepEncoder(packet_space(), RL2, trunk="ff", **kwargs)
    grid = TransitionTstepEncoder(packet_space(), RL2, trunk="xland", **kwargs)
    assert shared.spec.sha256 != grid.spec.sha256
    assert shared.spec.token_dim == grid.spec.token_dim
    assert shared.emb_dim == grid.emb_dim


def test_the_trunk_refuses_other_evidence_and_layouts() -> None:
    with pytest.raises(ContractError, match="raw current token only"):
        TransitionTstepEncoder(
            packet_space(),
            RL2,
            evidence="transition",
            bypass=False,
            trunk="xland",
            public_contract=CONTRACT,
        )
    with pytest.raises(ContractError, match="raw current token only"):
        TransitionTstepEncoder(
            packet_space(),
            RL2,
            evidence="raw",
            bypass=True,
            trunk="xland",
            public_contract=CONTRACT,
        )
    other = {**CONTRACT, "fields": CONTRACT["fields"][:-1]}
    space = gym.spaces.Dict({"current": packet_space()["current"]})
    with pytest.raises(ContractError, match="XLand packet layout"):
        XLandGridTrunk(space, RL2, public_contract=other, d_output=8)
    with pytest.raises(ContractError, match="current token only"):
        XLandGridTrunk(packet_space(), RL2, public_contract=CONTRACT, d_output=8)
