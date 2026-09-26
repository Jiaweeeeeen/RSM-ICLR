"""What the summary carries, and a summary transplant (evaluation only).

The representation declaration asks four displays
and one intervention of a frozen summary-carrier policy: the carried summary
projected and read out against the hidden task variable, its similarity across
rewrites, where in the room and when in a long task the decisions read it, and
a transplant of one task's summary into another at one boundary.

:class:`SummaryCapture` (a rollout
:class:`~reasoned_icrl.runtime.rollout.DecisionProbe`) keeps, per task, the
summary each segment reads (the carrier's ``memory`` at the segment's first
decision) and every decision's public input packet and chosen action.
:class:`SummaryTransplant` (a carrier
:class:`~reasoned_icrl.model.summary_transformer.MemoryTransform`) replaces, at
the boundary that opens one segment, every row's written summary with its
donor's or, with no donors, with the initial memory (*cleared once*); the rows
of a retained rollout cross their boundaries together, which it checks.
:func:`representation_evaluation` runs one checkpoint through the shared
evaluator with the unbiased attention probe and the capture attached, and the
transplant when one is given. :func:`keydoor_layouts` reads the evaluator-side
hidden cells of Key-to-Door tasks after the rollout's own reset; nothing here
reaches the policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from typing import Any

import numpy as np
import torch
from amago.envs.amago_env import SequenceWrapper
from numpy.typing import NDArray

from reasoned_icrl.experiments.benchmarks import BenchmarkContract
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import evaluation_environment
from reasoned_icrl.experiments.records import BenchmarkEvent, CheckpointRule
from reasoned_icrl.model.summary_transformer import SummaryHiddenState
from reasoned_icrl.runtime.attention import (
    ReadBias,
    SummaryAttentionProbe,
    summary_backbone,
)
from reasoned_icrl.runtime.environments import amago_environment
from reasoned_icrl.runtime.rollout import DecisionProbe, evaluate

Columns = dict[str, NDArray[Any]]


class ProbeGroup:
    """Several decision probes on one rollout, called in the given order."""

    def __init__(self, probes: Sequence[DecisionProbe]) -> None:
        if not probes:
            raise ContractError("A probe group needs at least one probe.")
        self.probes = tuple(probes)

    def begin_chunk(self, task_ids: Sequence[int]) -> None:
        for probe in self.probes:
            probe.begin_chunk(task_ids)

    def after_policy(
        self, live_rows: Sequence[int], steps: NDArray[np.int64], hidden: Any
    ) -> None:
        for probe in self.probes:
            probe.after_policy(live_rows, steps, hidden)

    def after_actions(
        self,
        live_rows: Sequence[int],
        steps: NDArray[np.int64],
        observation: Mapping[str, NDArray[Any]],
        actions: NDArray[Any],
    ) -> None:
        for probe in self.probes:
            probe.after_actions(live_rows, steps, observation, actions)


class SummaryCapture:
    """The summary each segment reads, and every decision's inputs and action.

    A summary is taken once per task and segment, at the segment's first
    decision (the carrier's ``memory`` is the summary the whole segment reads),
    as float32 ``[M, d]``. Decisions keep the ``current`` public packet and the
    chosen action; the other packet keys (previous action and reward) follow
    from these.
    """

    def __init__(self) -> None:
        self._task_ids: list[int] = []
        self._last = np.zeros(0, dtype=np.int64)
        self._summaries: dict[str, list[NDArray[Any]]] = {
            name: [] for name in ("task_id", "segment", "step", "memory")
        }
        self._inputs: dict[str, list[NDArray[Any]]] = {
            name: [] for name in ("task_id", "step", "current", "action")
        }

    def begin_chunk(self, task_ids: Sequence[int]) -> None:
        self._task_ids = [int(task_id) for task_id in task_ids]
        self._last = np.full(len(self._task_ids), -1, dtype=np.int64)

    def after_policy(
        self, live_rows: Sequence[int], steps: NDArray[np.int64], hidden: Any
    ) -> None:
        segments = hidden.segment.detach().cpu().numpy().astype(np.int64)
        rows = np.asarray(
            [row for row in live_rows if segments[row] != self._last[row]],
            dtype=np.int64,
        )
        if rows.size == 0:
            return
        index = torch.as_tensor(rows, device=hidden.memory.device)
        memory = hidden.memory.index_select(0, index).detach().float().cpu().numpy()
        tasks = np.asarray(self._task_ids, dtype=np.int64)
        self._summaries["task_id"].append(tasks[rows])
        self._summaries["segment"].append(segments[rows])
        self._summaries["step"].append(np.asarray(steps, dtype=np.int64)[rows])
        self._summaries["memory"].append(memory.astype(np.float32))
        self._last[rows] = segments[rows]

    def after_actions(
        self,
        live_rows: Sequence[int],
        steps: NDArray[np.int64],
        observation: Mapping[str, NDArray[Any]],
        actions: NDArray[Any],
    ) -> None:
        rows = np.asarray(list(live_rows), dtype=np.int64)
        width = len(self._task_ids)
        current = np.asarray(observation["current"], dtype=np.float32).reshape(
            width, -1
        )
        chosen = np.asarray(actions).reshape(width, -1)[:, 0].astype(np.int64)
        tasks = np.asarray(self._task_ids, dtype=np.int64)
        self._inputs["task_id"].append(tasks[rows])
        self._inputs["step"].append(np.asarray(steps, dtype=np.int64)[rows])
        self._inputs["current"].append(current[rows])
        self._inputs["action"].append(chosen[rows])

    @staticmethod
    def _stack(parts: dict[str, list[NDArray[Any]]]) -> Columns:
        return {
            name: np.concatenate(values, axis=0) if values else np.zeros(0)
            for name, values in parts.items()
        }

    def summaries(self) -> Columns:
        """``task_id``, ``segment``, ``step`` (the segment's first decision)
        and ``memory`` ``[N, M, d]``, one row per task and segment."""
        return self._stack(self._summaries)

    def inputs(self) -> Columns:
        """``task_id``, ``step``, ``current`` ``[N, width]`` and ``action``."""
        return self._stack(self._inputs)


class SummaryTransplant:
    """Replace, at the boundary opening ``segment``, each row's written summary.

    ``donors`` maps a recipient task to the ``[M, d]`` summary it receives;
    ``None`` writes the carrier's initial memory instead (*cleared once*).
    One-shot: the segment counter reaches ``segment`` once per task. The
    rollout's rows must cross that boundary together, as every row of a
    retained rollout does; a partial crossing is refused.
    """

    def __init__(
        self,
        *,
        segment: int,
        donors: Mapping[int, NDArray[np.float32]] | None = None,
    ) -> None:
        if isinstance(segment, bool) or int(segment) < 1:
            raise ContractError("A transplant opens a segment after the first.")
        self.segment = int(segment)
        self.donors = None if donors is None else dict(donors)
        self._task_ids: list[int] = []
        self.applied: list[int] = []

    @property
    def label(self) -> str:
        kind = "cleared-once" if self.donors is None else "transplant"
        return f"{kind}-b{self.segment}"

    def begin_chunk(self, task_ids: Sequence[int]) -> None:
        self._task_ids = [int(task_id) for task_id in task_ids]
        if self.donors is not None:
            missing = [task for task in self._task_ids if task not in self.donors]
            if missing:
                raise ContractError(f"No donor summary for tasks {missing[:5]}.")

    def after_policy(
        self, live_rows: Sequence[int], steps: NDArray[np.int64], hidden: Any
    ) -> None:
        del live_rows, steps, hidden

    def after_actions(
        self,
        live_rows: Sequence[int],
        steps: NDArray[np.int64],
        observation: Mapping[str, NDArray[Any]],
        actions: NDArray[Any],
    ) -> None:
        del live_rows, steps, observation, actions

    def after_write(self, hidden: SummaryHiddenState) -> None:
        crossing = hidden.segment == self.segment
        if not bool(crossing.any()):
            return
        if hidden.batch_size != len(self._task_ids) or not bool(crossing.all()):
            raise ContractError(
                "A transplant needs every row of a chunk to cross its boundary "
                "together."
            )
        if self.donors is None:
            replacement = hidden.initial_memory.unsqueeze(0).expand(
                hidden.batch_size, -1, -1
            )
        else:
            replacement = torch.as_tensor(
                np.stack([self.donors[task] for task in self._task_ids]),
                device=hidden.memory.device,
            )
        if replacement.shape != hidden.memory.shape:
            raise ContractError("A donor summary disagrees with the carrier's memory.")
        hidden.memory.copy_(replacement.to(hidden.memory.dtype))
        self.applied.extend(self._task_ids)


def donor_order(task_ids: Sequence[int], offset: int) -> dict[int, int]:
    """Recipient task -> donor task: the task ``offset`` roster places later."""
    tasks = [int(task) for task in task_ids]
    if len(set(tasks)) != len(tasks):
        raise ContractError("Roster task identities must be distinct.")
    if not 0 < int(offset) < len(tasks):
        raise ContractError("The donor offset must lie strictly inside the roster.")
    return {
        task: tasks[(index + int(offset)) % len(tasks)]
        for index, task in enumerate(tasks)
    }


def representation_evaluation(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    experiment: Any,
    *,
    checkpoint: str,
    split: str,
    transplant: SummaryTransplant | None = None,
    checkpoint_rule: CheckpointRule = "endpoint",
    task_cap: int | None = None,
    batch_size: int | None = None,
) -> tuple[tuple[BenchmarkEvent, ...], Columns, Columns, Columns]:
    """One checkpoint over a roster with the capture attached.

    Returns the evaluator's events (history ``retained``; a transplant is
    carried by the caller's labels, never by these records), the attention
    probe's decisions (unbiased), the summaries and the decisions' inputs.
    Without a transplant the events equal an ordinary retained evaluation.
    """
    backbone = summary_backbone(experiment)
    spec = backbone.spec
    attention = SummaryAttentionProbe(
        memory_tokens=spec.memory_tokens,
        segment_length=spec.segment_length,
        layers=backbone.n_layers,
        heads=backbone.backbone_heads,
        bias=ReadBias(),
    )
    capture = SummaryCapture()
    probes: list[DecisionProbe] = [attention, capture]
    if transplant is not None:
        probes.append(transplant)
    with ExitStack() as stack:
        stack.enter_context(backbone.probed(attention))
        if transplant is not None:
            stack.enter_context(backbone.transformed(transplant))
        _, events, _ = evaluate(
            contract,
            config,
            experiment,
            checkpoint=checkpoint,
            split=split,
            history="retained",
            task_cap=task_cap,
            checkpoint_rule=checkpoint_rule,
            batch_size=batch_size,
            decision_probe=ProbeGroup(probes),
        )
    return events, attention.decisions(), capture.summaries(), capture.inputs()


def keydoor_layouts(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    split: str,
    task_ids: Sequence[int],
) -> dict[int, dict[str, tuple[int, int]]]:
    """Each Key-to-Door task's hidden ``start``, ``key`` and ``door`` cells as
    ``(row, column)``, read from the environment's ``state_dict`` after the
    rollout's own reset (the evaluator's labels; never an input)."""
    if contract.environment.name != "dark_key_to_door":
        raise ContractError("Layouts are read on Key-to-Door only.")
    rollout_seed = contract.evaluation.rollout_seeds[0]
    environment = evaluation_environment(
        contract, config, split=split, seed=rollout_seed
    )
    wrapped = amago_environment(
        environment, name=f"{type(environment).__name__}-labels", seed=rollout_seed
    )
    base = wrapped.unwrapped
    sequence = SequenceWrapper(wrapped, save_trajs_to=None, save_every=None)
    layouts: dict[int, dict[str, tuple[int, int]]] = {}
    try:
        for task in task_ids:
            base.set_task(int(task))
            sequence.reset(seed=rollout_seed)
            native = base.state_dict()["native"]
            layouts[int(task)] = {
                name: (int(native[key][0]), int(native[key][1]))
                for name, key in (("start", "start"), ("key", "key"), ("door", "goal"))
            }
    finally:
        wrapped.close()
    return layouts


__all__ = [
    "ProbeGroup",
    "SummaryCapture",
    "SummaryTransplant",
    "donor_order",
    "keydoor_layouts",
    "representation_evaluation",
]
