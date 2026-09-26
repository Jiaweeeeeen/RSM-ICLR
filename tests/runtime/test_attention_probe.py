"""The summary carrier's attention probe.

CPU, FP32 and deterministic: an unbiased probe changes no value; the recorded
masses of every decision sum to one; blocking a key group removes exactly its
mass; a finite bias on the first block follows the closed form of a biased
softmax; every decision is placed at its segment and position; the probe is
refused where it cannot see every block and detaches after use.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.model.summary_transformer import MaskedOrdinaryBlock
from reasoned_icrl.model.trajectory_encoder import SummaryTrajEncoder
from reasoned_icrl.runtime.attention import ReadBias, SummaryAttentionProbe
from tests.model.test_summary_transformer import LAYERS, TOKEN, C, M, _carrier, _packet

HEADS = 2
STEPS = 3 * C + 2  # three boundaries crossed, a partial fourth segment


def _probe(bias: ReadBias | None = None) -> SummaryAttentionProbe:
    return SummaryAttentionProbe(
        memory_tokens=M,
        segment_length=C,
        layers=LAYERS,
        heads=HEADS,
        bias=ReadBias() if bias is None else bias,
    )


def _probed_rollout(
    carrier: SummaryTrajEncoder,
    probe: SummaryAttentionProbe | None,
    *,
    batch: int = 2,
) -> tuple[torch.Tensor, Any]:
    """Cached decisions on every row, the probe fed as the rollout feeds it."""
    seq, times = _packet(batch, STEPS)
    hidden = carrier.init_hidden_state(batch, torch.device("cpu"))
    outputs = []
    if probe is not None:
        probe.begin_chunk(list(range(100, 100 + batch)))
    context = carrier.backbone.probed(probe) if probe is not None else None
    with torch.no_grad():
        if context is not None:
            context.__enter__()
        try:
            for step in range(STEPS):
                record = seq[:, step : step + 1].clone()
                record[..., -1] = 1.0
                out, hidden = carrier(record, times[:, step : step + 1], hidden)
                outputs.append(out)
                if probe is not None:
                    probe.after_policy(
                        list(range(batch)), np.full(batch, step, dtype=np.int64), hidden
                    )
        finally:
            if context is not None:
                context.__exit__(None, None, None)
    return torch.cat(outputs, 1), hidden


def test_an_unbiased_probe_changes_no_value() -> None:
    carrier = _carrier()
    plain, plain_hidden = _probed_rollout(carrier, None)
    probed, probed_hidden = _probed_rollout(carrier, _probe())
    torch.testing.assert_close(probed, plain, rtol=0, atol=0)
    torch.testing.assert_close(
        probed_hidden.memory, plain_hidden.memory, rtol=0, atol=0
    )
    assert all(
        block.probe is None
        for block in carrier.backbone.layers
        if isinstance(block, MaskedOrdinaryBlock)
    )


def test_every_decision_is_recorded_with_masses_that_sum_to_one() -> None:
    probe = _probe()
    _probed_rollout(_carrier(), probe)
    got = probe.decisions()
    assert got["summary"].shape == (2 * STEPS, LAYERS, HEADS)
    total = got["summary"] + got["buffer"] + got["own"]
    np.testing.assert_allclose(total, 1.0, atol=1e-6)
    steps = got["step"].reshape(STEPS, 2)[:, 0]
    assert steps.tolist() == list(range(STEPS))
    positions = got["position"].reshape(STEPS, 2)[:, 0]
    segments = got["segment"].reshape(STEPS, 2)[:, 0]
    assert positions.tolist() == [step % C + 1 for step in range(STEPS)]
    assert segments.tolist() == [step // C for step in range(STEPS)]
    # The first record of a segment has no earlier record to read.
    first = got["position"] == 1
    assert np.all(got["buffer"][first] == 0.0)
    assert sorted(set(got["task_id"].tolist())) == [100, 101]


@pytest.mark.parametrize("target", ["summary", "buffer"])
def test_blocking_a_key_group_removes_exactly_its_mass(target: str) -> None:
    probe = _probe(ReadBias(target, math.inf))  # type: ignore[arg-type]
    outputs, _ = _probed_rollout(_carrier(), probe)
    assert torch.isfinite(outputs).all()
    got = probe.decisions()
    assert np.all(got[target] == 0.0)
    other = "buffer" if target == "summary" else "summary"
    assert float(got[other].sum()) > 0.0


def test_a_finite_bias_on_the_first_block_follows_the_biased_softmax() -> None:
    """In the first segment both runs read the initial memory and layer 0's
    keys are embeddings, so layer 0 sees the same scores under any bias and its
    biased summary mass is exactly m e^-b / (m e^-b + 1 - m). Later segments
    read a summary written from the biased records, so they differ."""
    beta = 2.0
    plain = _probe()
    biased = _probe(ReadBias("summary", beta))
    carrier = _carrier()
    _probed_rollout(carrier, plain)
    _probed_rollout(carrier, biased)
    first = plain.decisions()["segment"] == 0
    assert np.array_equal(first, biased.decisions()["segment"] == 0)
    mass = plain.decisions()["summary"][first, 0, :].astype(np.float64)
    expected = mass * math.exp(-beta) / (mass * math.exp(-beta) + 1.0 - mass)
    np.testing.assert_allclose(
        biased.decisions()["summary"][first, 0, :], expected, rtol=1e-5, atol=1e-6
    )


def test_the_bias_reaches_decisions_but_the_writer_still_reads_the_summary() -> None:
    carrier = _carrier()
    plain, _ = _probed_rollout(carrier, _probe())
    blocked, hidden = _probed_rollout(carrier, _probe(ReadBias("summary", math.inf)))
    assert not torch.allclose(blocked, plain)
    # The carried memory is still written from the previous summary, not reset.
    assert not torch.equal(
        hidden.memory[0], hidden.initial_memory.to(hidden.memory.dtype)
    )


def test_the_probe_is_refused_where_it_cannot_see_every_block() -> None:
    dat = _carrier(dat=True).backbone
    with pytest.raises(ContractError, match="ordinary blocks"), dat.probed(_probe()):
        pass
    carrier = _carrier()
    with carrier.backbone.probed(_probe()):
        with (
            pytest.raises(ContractError, match="already attached"),
            carrier.backbone.probed(_probe()),
        ):
            pass
        tokens = torch.randn(1, C, TOKEN)
        valid = torch.ones(1, C, dtype=torch.bool)
        with pytest.raises(ContractError, match="cached path"), torch.no_grad():
            carrier.backbone.training_forward(tokens, valid)
    assert all(
        block.probe is None
        for block in carrier.backbone.layers
        if isinstance(block, MaskedOrdinaryBlock)
    )


@pytest.mark.parametrize(
    "target,beta",
    [
        ("none", 1.0),
        ("summary", 0.0),
        ("buffer", -1.0),
        ("summary", math.nan),
        ("x", 1.0),
    ],
)
def test_the_read_bias_refuses_ill_formed_settings(target: str, beta: float) -> None:
    with pytest.raises(ContractError):
        ReadBias(target, beta)  # type: ignore[arg-type]


def test_read_bias_labels() -> None:
    assert ReadBias().label == "retained"
    assert ReadBias("summary", 2.0).label == "summary-read-bias-2"
    assert ReadBias("buffer", math.inf).label == "buffer-read-bias-inf"
