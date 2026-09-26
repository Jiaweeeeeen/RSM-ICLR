"""Attention diagnostics of the summary carrier (evaluation only).

The study protocol's attention declaration asks two things of
a frozen summary-carrier policy. Where do its decisions read from: how much of
each decision row's attention falls on the carried summary (READ slots), on the
earlier records of the working buffer and on the decision's own record? And
what is reading the summary worth: how does the policy perform when an additive
bias ``-beta`` lowers the decision rows' scores on the summary keys (or, as the
companion, on the earlier buffer keys)? ``beta = inf`` blocks those keys.

Only decision (RECORD) rows are biased. The WRITE rows still read the previous
summary directly, so information can still be carried from one segment to the
next; what the bias removes is the decisions' access to it. Record states at
deeper layers change with the bias, and the writer reads those, so the carried
memory is not held fixed; the intervention is on reading, not on carrying.

:class:`SummaryAttentionProbe` implements both the carrier's
:class:`~reasoned_icrl.model.summary_transformer.AttentionProbe` and the
rollout's :class:`~reasoned_icrl.runtime.rollout.DecisionProbe`, so the masses
of every decision are stored beside its task, decision index, segment and
position in the segment; :func:`attention_evaluation` runs one checkpoint over
a roster through the shared evaluator and returns its events with them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from reasoned_icrl.experiments.benchmarks import BenchmarkContract
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.records import BenchmarkEvent, CheckpointRule
from reasoned_icrl.model.summary_transformer import SummaryTransformer
from reasoned_icrl.runtime.rollout import evaluate

ReadTarget = Literal["none", "summary", "buffer"]
READ_TARGETS: tuple[ReadTarget, ...] = ("none", "summary", "buffer")


@dataclass(frozen=True, slots=True)
class ReadBias:
    """An additive ``-beta`` on the decision rows' scores for one key group.

    ``target = "summary"`` biases the READ keys, ``"buffer"`` the earlier
    RECORD keys of the current segment (never the decision's own record);
    ``"none"`` with ``beta = 0`` is the unbiased capture. ``beta = inf``
    blocks the keys.
    """

    target: ReadTarget = "none"
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.target not in READ_TARGETS:
            raise ContractError(f"Unknown read-bias target: {self.target!r}.")
        if math.isnan(self.beta) or self.beta < 0.0:
            raise ContractError("A read bias is a non-negative beta (inf blocks).")
        if (self.target == "none") != (self.beta == 0.0):
            raise ContractError(
                "The unbiased capture is target 'none' with beta 0; every other "
                "target needs a positive beta."
            )

    @property
    def label(self) -> str:
        """``retained`` or ``<target>-read-bias-<beta>`` (``inf`` blocks)."""
        if self.target == "none":
            return "retained"
        beta = "inf" if math.isinf(self.beta) else f"{self.beta:g}"
        return f"{self.target}-read-bias-{beta}"


class SummaryAttentionProbe:
    """Record each decision's attention masses and apply one :class:`ReadBias`.

    For every decision row and block the probe keeps, per head, the attention
    mass on the READ slots (the carried summary), on the earlier RECORD slots of
    the segment (the working buffer) and on the decision's own record; the
    three sum to one. READ and WRITE queries (the read pass and the boundary)
    are neither biased nor recorded.
    """

    def __init__(
        self,
        *,
        memory_tokens: int,
        segment_length: int,
        layers: int,
        heads: int,
        bias: ReadBias,
    ) -> None:
        if min(memory_tokens, segment_length, layers, heads) < 1:
            raise ContractError("The probe needs a positive carrier geometry.")
        self.memory_tokens = memory_tokens
        self.record_start = memory_tokens
        self.write_start = memory_tokens + segment_length
        self.layers = layers
        self.heads = heads
        self.bias = bias
        self._task_ids: list[int] = []
        self._current: list[tuple[NDArray[np.float32], ...] | None] = [None] * layers
        self._columns: dict[str, list[NDArray[Any]]] = {
            name: []
            for name in (
                "task_id",
                "step",
                "segment",
                "position",
                "summary",
                "buffer",
                "own",
            )
        }

    # -- the carrier's AttentionProbe ------------------------------------

    def _record_rows(self, query_slots: torch.Tensor) -> torch.Tensor:
        slots = query_slots.long()
        return (slots >= self.record_start) & (slots < self.write_start)

    def logit_bias(
        self, layer: int, query_slots: torch.Tensor, window: int
    ) -> torch.Tensor | None:
        del layer
        if self.bias.target == "none":
            return None
        slots = query_slots.long()
        keys = torch.arange(window, device=slots.device)
        if self.bias.target == "summary":
            chosen = (keys < self.memory_tokens).unsqueeze(0).expand(len(slots), -1)
        else:
            chosen = (keys >= self.record_start).unsqueeze(0) & (
                keys.unsqueeze(0) < slots.unsqueeze(1)
            )
        chosen = chosen & self._record_rows(slots).unsqueeze(1)
        value = -math.inf if math.isinf(self.bias.beta) else -float(self.bias.beta)
        bias = torch.zeros((len(slots), window), device=slots.device)
        bias = bias.masked_fill(chosen, value)
        return bias.view(len(slots), 1, 1, window)

    def observe(
        self, layer: int, query_slots: torch.Tensor, weights: torch.Tensor
    ) -> None:
        records = self._record_rows(query_slots)
        if not bool(records.any()):
            return
        if not bool(records.all()):
            raise ContractError("A decision pass mixes decision and memory rows.")
        if self._current[layer] is not None:
            raise ContractError("Two decision passes before the probe was read.")
        rows = weights[:, :, 0, :].float()
        slots = query_slots.long()
        keys = torch.arange(rows.shape[-1], device=rows.device)
        earlier = (keys >= self.record_start).unsqueeze(0) & (
            keys.unsqueeze(0) < slots.unsqueeze(1)
        )
        summary = rows[..., : self.memory_tokens].sum(-1)
        buffer = (rows * earlier.unsqueeze(1)).sum(-1)
        own = rows.gather(
            -1, slots.view(-1, 1, 1).expand(-1, rows.shape[1], 1)
        ).squeeze(-1)
        self._current[layer] = tuple(
            tensor.detach().cpu().numpy().astype(np.float32)
            for tensor in (summary, buffer, own)
        )

    # -- the rollout's DecisionProbe --------------------------------------

    def begin_chunk(self, task_ids: Sequence[int]) -> None:
        self._task_ids = [int(task_id) for task_id in task_ids]
        self._current = [None] * self.layers

    def after_policy(
        self, live_rows: Sequence[int], steps: NDArray[np.int64], hidden: Any
    ) -> None:
        if any(current is None for current in self._current):
            raise ContractError("A policy call left no decision pass to record.")
        rows = np.asarray(list(live_rows), dtype=np.int64)
        lengths = hidden.lengths.detach().cpu().numpy().astype(np.int64)
        segments = hidden.segment.detach().cpu().numpy().astype(np.int64)
        stacked = [
            np.stack([layer[kind] for layer in self._current if layer is not None], 1)
            for kind in range(3)
        ]
        columns = self._columns
        columns["task_id"].append(np.asarray(self._task_ids, dtype=np.int64)[rows])
        columns["step"].append(np.asarray(steps, dtype=np.int64)[rows])
        columns["segment"].append(segments[rows])
        # The decision's record sat at slot lengths - 1 before the counter moved.
        columns["position"].append(lengths[rows] - self.memory_tokens)
        for name, values in zip(("summary", "buffer", "own"), stacked, strict=True):
            columns[name].append(values[rows])
        self._current = [None] * self.layers

    def after_actions(
        self,
        live_rows: Sequence[int],
        steps: NDArray[np.int64],
        observation: Mapping[str, NDArray[Any]],
        actions: NDArray[Any],
    ) -> None:
        """The masses are complete after the policy call; nothing to add."""
        del live_rows, steps, observation, actions

    # -- results ------------------------------------------------------------

    def decisions(self) -> dict[str, NDArray[Any]]:
        """Every recorded decision: ids, counters and ``[N, layers, heads]``
        masses on the summary, the earlier buffer records and the own record."""
        out: dict[str, NDArray[Any]] = {}
        for name, parts in self._columns.items():
            if parts:
                out[name] = np.concatenate(parts, axis=0)
            elif name in ("summary", "buffer", "own"):
                out[name] = np.zeros((0, self.layers, self.heads), dtype=np.float32)
            else:
                out[name] = np.zeros(0, dtype=np.int64)
        return out


def summary_backbone(experiment: Any) -> SummaryTransformer:
    """The summary carrier of a loaded experiment, or a refusal."""
    encoder = experiment.policy.traj_encoder
    backbone = getattr(encoder, "backbone", None)
    if not isinstance(backbone, SummaryTransformer):
        raise ContractError(
            "Attention diagnostics apply to the summary carrier's cells only."
        )
    return backbone


def attention_evaluation(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    experiment: Any,
    *,
    checkpoint: str,
    split: str,
    bias: ReadBias,
    checkpoint_rule: CheckpointRule = "endpoint",
    task_cap: int | None = None,
    batch_size: int | None = None,
) -> tuple[tuple[BenchmarkEvent, ...], dict[str, NDArray[Any]]]:
    """One checkpoint over a roster under ``bias``, through the shared evaluator.

    Returns the evaluator's events (history ``retained``; the bias is carried
    by the caller's labels, never by these records) and the probe's decisions.
    With ``ReadBias()`` the events equal an ordinary retained evaluation.
    """
    backbone = summary_backbone(experiment)
    spec = backbone.spec
    probe = SummaryAttentionProbe(
        memory_tokens=spec.memory_tokens,
        segment_length=spec.segment_length,
        layers=backbone.n_layers,
        heads=backbone.backbone_heads,
        bias=bias,
    )
    with backbone.probed(probe):
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
            decision_probe=probe,
        )
    return events, probe.decisions()
