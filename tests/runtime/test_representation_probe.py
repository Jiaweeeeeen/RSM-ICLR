"""The summary capture and the summary transplant.

CPU, FP32 and deterministic: a memory transform that changes nothing changes
no value; the capture keeps, once per task and segment, the summary that
segment reads; a transplant replaces exactly the summary one segment reads
(everything before it is unchanged, the donor's summary is read there), and
*cleared once* writes the initial memory there; a partial crossing, a carrier
that writes no summary and a second transform are refused; the donor order
shifts the roster; the capture keeps each decision's inputs and action.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.model.summary_transformer import SummaryHiddenState
from reasoned_icrl.model.trajectory_encoder import SummaryTrajEncoder
from reasoned_icrl.runtime.representation import (
    ProbeGroup,
    SummaryCapture,
    SummaryTransplant,
    donor_order,
)
from tests.model.test_summary_transformer import C, M, _carrier, _packet

STEPS = 3 * C + 2  # three boundaries crossed, a partial fourth segment
TASKS = [100, 101]


class _Idle:
    def after_write(self, hidden: SummaryHiddenState) -> None:
        del hidden


def _rollout(
    carrier: SummaryTrajEncoder,
    capture: SummaryCapture,
    *,
    transform: Any = None,
) -> torch.Tensor:
    """Cached decisions on every row, the probes fed as the rollout feeds them."""
    batch = len(TASKS)
    seq, times = _packet(batch, STEPS)
    hidden = carrier.init_hidden_state(batch, torch.device("cpu"))
    probes: list[Any] = [capture]
    if isinstance(transform, SummaryTransplant):
        probes.append(transform)
    group = ProbeGroup(probes)
    group.begin_chunk(TASKS)
    outputs = []
    context = carrier.backbone.transformed(transform) if transform else None
    with torch.no_grad():
        if context is not None:
            context.__enter__()
        try:
            for step in range(STEPS):
                record = seq[:, step : step + 1].clone()
                record[..., -1] = 1.0
                out, hidden = carrier(record, times[:, step : step + 1], hidden)
                outputs.append(out)
                steps = np.full(batch, step, dtype=np.int64)
                group.after_policy(list(range(batch)), steps, hidden)
                group.after_actions(
                    list(range(batch)),
                    steps,
                    {"current": record[:, 0, :4].numpy()},
                    np.full((batch, 1), step % 5),
                )
        finally:
            if context is not None:
                context.__exit__(None, None, None)
    return torch.cat(outputs, 1)


def _by_segment(capture: SummaryCapture) -> dict[tuple[int, int], np.ndarray]:
    got = capture.summaries()
    return {
        (int(task), int(segment)): memory
        for task, segment, memory in zip(
            got["task_id"], got["segment"], got["memory"], strict=True
        )
    }


def test_an_idle_transform_changes_no_value() -> None:
    carrier = _carrier()
    plain = _rollout(carrier, SummaryCapture())
    idle = _rollout(carrier, SummaryCapture(), transform=_Idle())
    torch.testing.assert_close(idle, plain, rtol=0, atol=0)
    assert carrier.backbone.memory_transform is None


def test_the_capture_keeps_the_summary_each_segment_reads_once() -> None:
    carrier = _carrier()
    capture = SummaryCapture()
    _rollout(carrier, capture)
    got = capture.summaries()
    assert got["memory"].shape == (2 * 4, M, carrier.backbone.memory_init.shape[1])
    assert got["memory"].dtype == np.float32
    assert sorted(
        zip(got["task_id"].tolist(), got["segment"].tolist(), strict=True)
    ) == [(task, segment) for task in TASKS for segment in range(4)]
    assert sorted(set(got["step"].tolist())) == [0, C, 2 * C, 3 * C]
    initial = carrier.backbone.memory_init.detach().numpy()
    summaries = _by_segment(capture)
    for task in TASKS:
        np.testing.assert_array_equal(summaries[(task, 0)], initial)
        assert not np.allclose(summaries[(task, 1)], initial)


def test_a_transplant_replaces_exactly_the_summary_one_segment_reads() -> None:
    carrier = _carrier()
    plain_capture = SummaryCapture()
    plain = _rollout(carrier, plain_capture)
    plain_summaries = _by_segment(plain_capture)
    donors = {
        TASKS[0]: plain_summaries[(TASKS[1], 2)],
        TASKS[1]: plain_summaries[(TASKS[0], 2)],
    }
    transplant = SummaryTransplant(segment=2, donors=donors)
    capture = SummaryCapture()
    moved = _rollout(carrier, capture, transform=transplant)
    torch.testing.assert_close(moved[:, : 2 * C], plain[:, : 2 * C], rtol=0, atol=0)
    assert not torch.allclose(moved[:, 2 * C :], plain[:, 2 * C :])
    summaries = _by_segment(capture)
    for task in TASKS:
        for segment in (0, 1):
            np.testing.assert_array_equal(
                summaries[(task, segment)], plain_summaries[(task, segment)]
            )
        np.testing.assert_array_equal(summaries[(task, 2)], donors[task])
    assert sorted(transplant.applied) == TASKS
    assert transplant.label == "transplant-b2"


def test_cleared_once_writes_the_initial_memory_at_its_boundary_only() -> None:
    carrier = _carrier()
    plain_capture = SummaryCapture()
    _rollout(carrier, plain_capture)
    cleared = SummaryTransplant(segment=2)
    capture = SummaryCapture()
    _rollout(carrier, capture, transform=cleared)
    initial = carrier.backbone.memory_init.detach().numpy()
    summaries = _by_segment(capture)
    plain_summaries = _by_segment(plain_capture)
    for task in TASKS:
        np.testing.assert_array_equal(summaries[(task, 2)], initial)
        np.testing.assert_array_equal(summaries[(task, 1)], plain_summaries[(task, 1)])
        assert not np.allclose(summaries[(task, 3)], initial)
    assert cleared.label == "cleared-once-b2"


def test_a_transplant_refuses_a_partial_crossing_and_a_missing_donor() -> None:
    carrier = _carrier()
    hidden = carrier.init_hidden_state(2, torch.device("cpu"))
    hidden.segment[:] = torch.tensor([2, 1])
    transplant = SummaryTransplant(segment=2)
    transplant.begin_chunk(TASKS)
    with pytest.raises(ContractError, match="together"):
        transplant.after_write(hidden)
    with pytest.raises(ContractError, match="No donor"):
        SummaryTransplant(
            segment=2, donors={100: np.zeros((M, 1), dtype=np.float32)}
        ).begin_chunk(TASKS)
    with pytest.raises(ContractError, match="after the first"):
        SummaryTransplant(segment=0)


def test_transforms_need_a_written_summary_and_attach_once() -> None:
    segment = _carrier("segment").backbone
    with (
        pytest.raises(ContractError, match="writes a summary"),
        segment.transformed(_Idle()),
    ):
        pass
    backbone = _carrier().backbone
    with (
        backbone.transformed(_Idle()),
        pytest.raises(ContractError, match="already attached"),
        backbone.transformed(_Idle()),
    ):
        pass
    assert backbone.memory_transform is None


def test_the_donor_order_shifts_the_roster() -> None:
    assert donor_order([5, 6, 7, 8], 2) == {5: 7, 6: 8, 7: 5, 8: 6}
    for offset in (0, 4):
        with pytest.raises(ContractError, match="offset"):
            donor_order([5, 6, 7, 8], offset)
    with pytest.raises(ContractError, match="distinct"):
        donor_order([5, 5, 7], 1)


def test_the_capture_keeps_each_decisions_inputs_and_action() -> None:
    capture = SummaryCapture()
    capture.begin_chunk([1, 2, 3])
    current = np.arange(12, dtype=np.float32).reshape(3, 4)
    capture.after_actions(
        [0, 2], np.array([7, 7, 7]), {"current": current}, np.array([[4], [0], [2]])
    )
    got = capture.inputs()
    assert got["task_id"].tolist() == [1, 3]
    assert got["step"].tolist() == [7, 7]
    assert got["action"].tolist() == [4, 2]
    np.testing.assert_array_equal(got["current"], current[[0, 2]])
