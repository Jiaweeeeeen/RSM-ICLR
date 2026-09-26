"""Evaluate one checkpoint over a declared task roster, environment by environment.

The greedy rollout that drives every environment lives in
:mod:`reasoned_icrl.runtime.rollout`; this module owns what differs per
environment: which record each task yields, how those records become
:class:`BenchmarkEvent` rows, the evaluation-only references, and which cache
*interventions* the task declares. An intervention clears the trajectory cache
at a task boundary while leaving the environment, rewards, budget and the
current transition packet untouched; it measures dependence on retained
history, not a specific inference algorithm.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from reasoned_icrl.environments.base import Attempt, BaseEnv
from reasoned_icrl.environments.concentration import (
    ConcentrationEnv,
    ConcentrationFlip,
)
from reasoned_icrl.environments.count_recall import (
    COUNT_RECALL_ACTIONS,
    CountRecallQuery,
    count_recall_variant,
)
from reasoned_icrl.environments.match_pattern import MatchPatternDecision
from reasoned_icrl.environments.mazerunner import (
    MAZERUNNER_ACTIONS,
    GoalCompletion,
    MazeRunnerEnv,
)
from reasoned_icrl.environments.tmaze import FORWARD, UP, TMazeEpisode
from reasoned_icrl.experiments.benchmarks import BenchmarkContract
from reasoned_icrl.experiments.config import EnvironmentConfig, ExperimentConfig
from reasoned_icrl.experiments.contracts import (
    ContractError,
    ResultValidationError,
)
from reasoned_icrl.experiments.environments import build_environment
from reasoned_icrl.experiments.records import (
    CHECKPOINT_RULES,
    BenchmarkEvent,
    BenchmarkRun,
    CheckpointRule,
    write_benchmark_results,
)

SUMMARY_CLEARED = "summary-cleared"
"""The summary carrier's intervention: the carried summary is replaced by the
task-independent initial memory at every boundary while the current segment is
kept. Available on every environment; refused unless the carrier's regime is
``summary`` (on a ``segment`` cell it would be a no-op with a misleading record)."""

HISTORY_MODES = {
    "darkroom": ("retained", "attempt-cleared", SUMMARY_CLEARED),
    "dark_key_to_door": ("retained", "attempt-cleared", SUMMARY_CLEARED),
    "count_recall": ("retained", "current-token", SUMMARY_CLEARED),
    "mazerunner": ("retained", "goal-cleared", SUMMARY_CLEARED),
    "concentration": ("retained", "current-token", SUMMARY_CLEARED),
    "xland_minigrid": ("retained", "attempt-cleared", SUMMARY_CLEARED),
    "xland_one_rule": ("retained", "attempt-cleared", SUMMARY_CLEARED),
    "match_pattern": ("retained", "current-token"),
    "tmaze": ("retained", "current-token", SUMMARY_CLEARED),
}
"""History interventions per environment; ``retained`` is the ordinary rollout,
the second entry the task's declared cache boundary, the third the summary
carrier's own intervention."""

ALL_HISTORY_MODES = (
    "retained",
    "attempt-cleared",
    "current-token",
    "goal-cleared",
    SUMMARY_CLEARED,
)
HistoryMode = Literal[
    "retained", "attempt-cleared", "current-token", "goal-cleared", "summary-cleared"
]


def history_modes(environment: str) -> tuple[str, ...]:
    """Return the history interventions declared for one environment."""
    try:
        return HISTORY_MODES[environment]
    except KeyError as error:
        raise ContractError(
            f"No evaluator is implemented for {environment!r}."
        ) from error


def continued_history_modes(environment: Any) -> tuple[str, ...]:
    """The history interventions of one resolved environment: the declared
    modes and, on a CountRecall task continued over several deck pairs (the
    continued-stream axis), ``attempt-cleared``: the carried
    memory reset at every pair boundary (the stream-cleared companion). A
    single stream has no boundary, so the mode is refused there."""
    modes = history_modes(str(environment.name))
    if environment.name == "count_recall" and int(environment.attempts) > 1:
        modes = (*modes, "attempt-cleared")
    if environment.name == "mazerunner" and environment.meta_horizon is not None:
        # The repeated-laps axis: the carried memory reset
        # at every lap boundary (the lap-cleared companion).
        modes = (*modes, "attempt-cleared")
    return modes


def task_intervention(environment: str) -> str:
    """The task's declared cache-boundary intervention, used by the C3 comparison."""
    return history_modes(environment)[1]


# ----------------------------------------------------------------------
# Per-task results
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttemptTaskResult:
    """One completed K-attempt task over a single hidden layout."""

    task_id: int
    rollout_seed: int
    attempts: tuple[Attempt, ...]
    partial: Attempt | None
    task_return: float
    decisions: int
    writes: tuple[int, ...] | None = None
    """Summary boundaries crossed before each decision (1-based), or None."""

    def first_event_step(self, name: str) -> int | None:
        """Global step at which a named one-based attempt event first happened."""
        for attempt in (
            *self.attempts,
            *(() if self.partial is None else (self.partial,)),
        ):
            local = getattr(attempt, name, None)
            if local is not None:
                return int(attempt.first_step + local - 1)
        return None


@dataclass(frozen=True, slots=True)
class MatchPatternResult:
    task_id: int
    rollout_seed: int
    decision: MatchPatternDecision
    decisions: int


@dataclass(frozen=True, slots=True)
class CountRecallStreamResult:
    """One completed stream over a single pair of decks."""

    task_id: int
    rollout_seed: int
    queries: tuple[CountRecallQuery, ...]
    stream_return: float
    decisions: int
    writes: tuple[int, ...] | None = None
    """Summary boundaries crossed before each decision (1-based), or None."""
    values: tuple[int, ...] = ()
    """The dealt value of every observation token, in order, from public packets."""
    streams: int = 1
    """Deck pairs in the outer task (the continued-stream axis); one in training."""

    @property
    def accuracy(self) -> float:
        if not self.queries:
            raise ResultValidationError("A scored stream needs at least one query.")
        return sum(query.correct for query in self.queries) / len(self.queries)

    @property
    def decisions_per_stream(self) -> int:
        if self.streams < 1 or len(self.queries) % self.streams:
            raise ResultValidationError("The streams do not divide the scored queries.")
        return len(self.queries) // self.streams


@dataclass(frozen=True, slots=True)
class TMazeEpisodeResult:
    """One completed passive T-Maze episode over a single cue."""

    task_id: int
    rollout_seed: int
    episode: TMazeEpisode
    decisions: int
    writes: tuple[int, ...] | None = None
    """Summary boundaries crossed before each decision (1-based), or None."""

    def __post_init__(self) -> None:
        if self.episode.steps != self.decisions:
            raise ResultValidationError("T-Maze episode steps disagree with decisions.")


@dataclass(frozen=True, slots=True)
class ConcentrationLedger:
    """The retrieval ledger totals of one board (spec §7.2, decision 14).

    ``evicted_*`` count the opportunities whose partners' most recent reveal
    was consumed under an earlier summary segment than the flip; they are None
    off the summary carriers, where no boundary exists.
    """

    opportunities: int
    successes: int
    evicted_opportunities: int | None
    evicted_successes: int | None
    invalid: int
    redundant: int


@dataclass(frozen=True, slots=True)
class ConcentrationResult:
    """One completed board over a single shuffled deck."""

    task_id: int
    rollout_seed: int
    flips: tuple[ConcentrationFlip, ...]
    matched_pairs: int
    pairs: int
    episode_return: float
    decisions: int
    writes: tuple[int, ...] | None = None
    """Summary boundaries crossed before each decision (1-based), or None."""

    def __post_init__(self) -> None:
        if not 0 <= self.matched_pairs <= self.pairs or self.pairs < 1:
            raise ResultValidationError("Concentration pair counters are invalid.")
        if len(self.flips) != self.decisions:
            raise ResultValidationError("Concentration records one flip per decision.")
        if sum(flip.matched for flip in self.flips) != self.matched_pairs:
            raise ResultValidationError("Concentration matches disagree with flips.")

    @property
    def pair_fraction(self) -> float:
        return self.matched_pairs / self.pairs

    @property
    def board_complete(self) -> bool:
        return self.matched_pairs == self.pairs

    def ledger(self) -> ConcentrationLedger:
        """Totals of the retrieval ledger; eviction needs the write counters."""
        opportunities = successes = evicted = evicted_hits = 0
        for flip in self.flips:
            if not flip.opportunity:
                continue
            opportunities += 1
            successes += int(flip.hit)
            if self.writes is not None:
                assert flip.partner_last_reveal is not None
                # Observation k feeds decision k + 1, whose write counter is
                # writes[k]; the flip's own counter is writes[index - 1].
                seen = _writes_at(self.writes, flip.partner_last_reveal + 1)
                now = _writes_at(self.writes, flip.index)
                assert seen is not None and now is not None
                if seen < now:
                    evicted += 1
                    evicted_hits += int(flip.hit)
        tracked = self.writes is not None
        return ConcentrationLedger(
            opportunities=opportunities,
            successes=successes,
            evicted_opportunities=evicted if tracked else None,
            evicted_successes=evicted_hits if tracked else None,
            invalid=sum(flip.invalid for flip in self.flips),
            redundant=sum(flip.redundant for flip in self.flips),
        )


@dataclass(frozen=True, slots=True)
class MazeRunnerEpisodeResult:
    """One completed episode over a single maze and goal sequence."""

    task_id: int
    rollout_seed: int
    goals_total: int
    completions: tuple[GoalCompletion, ...]
    episode_return: float
    decisions: int
    writes: tuple[int, ...] | None = None
    """Summary boundaries crossed before each decision (1-based), or None."""

    def __post_init__(self) -> None:
        if self.goals_total < 1 or len(self.completions) > self.goals_total:
            raise ResultValidationError("MazeRunner goal counters are invalid.")
        if self.episode_return != float(len(self.completions)):
            raise ResultValidationError(
                "MazeRunner return must equal the goals completed."
            )

    @property
    def goal_fraction(self) -> float:
        return len(self.completions) / self.goals_total

    @property
    def full_sequence(self) -> bool:
        return len(self.completions) == self.goals_total


# ----------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------


def _event(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    checkpoint: str,
    split: str,
    history: str,
    result: Any,
    kind: Literal["attempt", "query", "episode", "decision"],
    event_index: int,
    step: int,
    numerator: int,
    denominator: int,
    native_return: float,
    true_count: int | None = None,
    answer: int | None = None,
    left_pattern: int | None = None,
    right_pattern: int | None = None,
    writes_before_decision: int | None = None,
    evidence_age_writes: int | None = None,
    evidence_age_writes_recent: int | None = None,
    evidence_in_current_segment: bool | None = None,
    count_before_current_segment: int | None = None,
    count_in_current_segment: int | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    checkpoint_rule: CheckpointRule = "selected",
    retrieval_opportunities: int | None = None,
    retrieval_successes: int | None = None,
    evicted_opportunities: int | None = None,
    evicted_successes: int | None = None,
    invalid_flips: int | None = None,
    redundant_flips: int | None = None,
    board_complete: bool | None = None,
    complete: bool | None = None,
    key_step: int | None = None,
    flips: tuple[ConcentrationFlip, ...] | None = None,
    layout_index: int | None = None,
    goal_steps: tuple[int, ...] | None = None,
) -> BenchmarkEvent:
    return BenchmarkEvent(
        protocol=contract.protocol,
        benchmark=contract.name,
        condition=config.condition,
        training_seed=config.seed,
        checkpoint=checkpoint,
        split=split,
        history=history,
        task_id=result.task_id,
        cluster_id=result.task_id,
        rollout_seed=result.rollout_seed,
        kind=kind,
        event_index=event_index,
        step=step,
        numerator=numerator,
        denominator=denominator,
        native_return=native_return,
        true_count=true_count,
        answer=answer,
        left_pattern=left_pattern,
        right_pattern=right_pattern,
        writes_before_decision=writes_before_decision,
        evidence_age_writes=evidence_age_writes,
        evidence_age_writes_recent=evidence_age_writes_recent,
        evidence_in_current_segment=evidence_in_current_segment,
        count_before_current_segment=count_before_current_segment,
        count_in_current_segment=count_in_current_segment,
        start_step=start_step,
        end_step=end_step,
        checkpoint_rule=checkpoint_rule,
        retrieval_opportunities=retrieval_opportunities,
        retrieval_successes=retrieval_successes,
        evicted_opportunities=evicted_opportunities,
        evicted_successes=evicted_successes,
        invalid_flips=invalid_flips,
        redundant_flips=redundant_flips,
        board_complete=board_complete,
        complete=complete,
        key_step=key_step,
        flips=flips,
        outer_length=contract.environment.outer_length,
        layout_index=layout_index,
        goal_steps=goal_steps,
    )


def _attempt_at(result: AttemptTaskResult, ordinal: int) -> Attempt:
    """The 1-based attempt of a task: a completed one, or the partial attempt
    the outer budget cut short as the ordinal after the last completed one."""
    if 1 <= ordinal <= len(result.attempts):
        return result.attempts[ordinal - 1]
    if ordinal == len(result.attempts) + 1 and result.partial is not None:
        return result.partial
    raise ResultValidationError(f"Task {result.task_id} has no attempt {ordinal}.")


def retained_attempts(result: AttemptTaskResult) -> tuple[Attempt, ...]:
    """Every attempt of the task, the partial one last (R4 complete retention)."""
    return (
        *result.attempts,
        *(() if result.partial is None else (result.partial,)),
    )


class AttemptWriteFields(NamedTuple):
    """Summary-carrier bookkeeping of one Key-to-Door attempt (spec §9)."""

    writes: int | None
    age: int | None
    age_recent: int | None
    in_current_segment: bool | None


class QueryWriteFields(NamedTuple):
    """Summary-carrier bookkeeping of one CountRecall query (spec §9)."""

    writes: int | None
    age: int | None
    count_before_segment: int | None
    count_in_segment: int | None


def _writes_at(writes: tuple[int, ...] | None, decision: int) -> int | None:
    """Summary boundaries crossed before the 1-based ``decision``, if tracked."""
    if writes is None:
        return None
    if not 1 <= decision <= len(writes):
        raise ResultValidationError("Decision index outside the tracked rollout.")
    return writes[decision - 1]


def attempt_write_fields(result: AttemptTaskResult, ordinal: int) -> AttemptWriteFields:
    """Writes before an attempt's first decision, and since its evidence.

    In Dark Key-to-Door the key is public only when it is picked up and the
    door only when it is opened. For the attempt with 1-based ``ordinal``, the
    *first* evidence is the earliest key pickup of any earlier attempt and the
    *most recent* evidence is the latest key pickup or door opening of any
    earlier attempt; the packet that shows an event is the input of the next
    decision, so an age counts boundaries between that decision and this
    attempt's first decision. "Since the first observation" overstates the age
    of the necessary evidence when the door was met again recently, so both
    ages are kept, and ``in_current_segment`` says whether the most recent
    evidence still lies in the open segment. Every age is ``None`` when
    nothing earlier found the key.
    """
    attempt = _attempt_at(result, ordinal)
    writes = _writes_at(result.writes, attempt.first_step)
    if writes is None:
        return AttemptWriteFields(None, None, None, None)
    first: int | None = None
    recent: int | None = None
    for earlier in result.attempts[: ordinal - 1]:
        for name in ("key_step", "door_step"):
            local = getattr(earlier, name, None)
            if local is None:
                continue
            step = earlier.first_step + int(local) - 1
            if name == "key_step" and first is None:
                first = step
            recent = step if recent is None else max(recent, step)
    if first is None:
        return AttemptWriteFields(writes, None, None, None)
    assert recent is not None
    seen_first = _writes_at(result.writes, first + 1)
    seen_recent = _writes_at(result.writes, recent + 1)
    assert seen_first is not None and seen_recent is not None
    age_recent = writes - seen_recent
    return AttemptWriteFields(writes, writes - seen_first, age_recent, age_recent == 0)


def query_write_fields(
    result: CountRecallStreamResult, query: CountRecallQuery
) -> QueryWriteFields:
    """Writes before a query's decision, the age of the value's first appearance,
    and the queried count split across the open segment's boundary.

    Observation ``k`` (zero-based, the reset observation first) is the input of
    decision ``k + 1``; decision ``i`` answers the query visible in observation
    ``i - 1``, whose true count is the number of observations ``k < i`` that
    dealt the queried category. The first-appearance age is ``None`` when the
    value was never dealt. The count is split by each observation's segment:
    the part dealt before the segment open at the decision and the part inside
    it, so accuracy can be read against the count mass the model can no longer
    attend to directly.
    """
    # Under the continued-stream axis decision ``i`` of the outer task sits in
    # stream ``s`` (zero-based) after ``s`` reset-only calls, so its charged
    # call is ``i + s`` and its evidence is the current pair's observations
    # only: the counts restart with every pair.
    stream = (
        (query.index - 1) // result.decisions_per_stream if result.streams > 1 else 0
    )
    call = query.index + stream
    writes = _writes_at(result.writes, call)
    if writes is None:
        return QueryWriteFields(None, None, None, None)
    if len(result.values) < call:
        raise ResultValidationError("CountRecall stream lost its observation values.")
    assert result.writes is not None
    age: int | None = None
    total = inside = 0
    first = stream * (result.decisions_per_stream + 1) if result.streams > 1 else 0
    for observation in range(first, call):
        value = result.values[observation]
        if value != query.query:
            continue
        seen = result.writes[observation]  # the decision this observation feeds
        if age is None:
            age = writes - seen
        total += 1
        inside += int(seen == writes)
    if total != query.true_count:
        raise ResultValidationError(
            "CountRecall observations disagree with the query's true count."
        )
    return QueryWriteFields(writes, age, total - inside, inside)


def events(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    results: Sequence[Any],
    *,
    checkpoint: str,
    split: str,
    history: str,
    checkpoint_rule: CheckpointRule = "selected",
    layout_period: int | None = None,
) -> tuple[BenchmarkEvent, ...]:
    """Turn per-task results into scored evaluation events."""
    env = contract.environment
    rows: list[BenchmarkEvent] = []
    common: dict[str, Any] = {
        "checkpoint": checkpoint,
        "split": split,
        "history": history,
        "checkpoint_rule": checkpoint_rule,
    }
    complete_retention = contract.evaluation.retention == "complete"
    for result in results:
        if isinstance(result, AttemptTaskResult):
            if len(result.attempts) < env.attempts:
                raise ResultValidationError(
                    f"Task {result.task_id} completed {len(result.attempts)} "
                    f"attempts; the declared budget guarantees {env.attempts}."
                )
            # R4: a complete record keeps every attempt, the partial one last;
            # the legacy export keeps the declared scored band only.
            retained = (
                retained_attempts(result)
                if complete_retention
                else result.attempts[: env.attempts]
            )
            for ordinal, attempt in enumerate(retained, 1):
                if not complete_retention and ordinal < env.scored_from:
                    continue  # the endpoint scores a declared band of attempts
                fields = attempt_write_fields(result, ordinal)
                key_local = getattr(attempt, "key_step", None)
                lap = env.name == "mazerunner"
                rows.append(
                    _event(
                        contract,
                        config,
                        result=result,
                        kind="attempt",
                        event_index=ordinal,
                        step=attempt.steps,
                        numerator=int(getattr(attempt, "goals", 0))
                        if lap
                        else int(attempt.success),
                        denominator=int(env.goals or 0) if lap else 1,
                        native_return=attempt.native_return,
                        goal_steps=tuple(getattr(attempt, "goal_steps", ()))
                        if lap
                        else None,
                        writes_before_decision=fields.writes,
                        evidence_age_writes=fields.age,
                        evidence_age_writes_recent=fields.age_recent,
                        evidence_in_current_segment=fields.in_current_segment,
                        start_step=attempt.first_step,
                        layout_index=(
                            int(getattr(attempt, "layout", 0))
                            if layout_period is not None
                            else None
                        ),
                        end_step=attempt.last_step,
                        complete=attempt.complete if complete_retention else None,
                        key_step=(
                            None
                            if key_local is None or not complete_retention
                            else attempt.first_step + int(key_local) - 1
                        ),
                        **common,
                    )
                )
        elif isinstance(result, MatchPatternResult):
            if result.decisions != 7:
                raise ResultValidationError(
                    "Match-pattern evaluation must finish the seventh call."
                )
            decision = result.decision
            rows.append(
                _event(
                    contract,
                    config,
                    result=result,
                    kind="decision",
                    event_index=1,
                    step=7,
                    numerator=int(decision.correct),
                    denominator=1,
                    native_return=decision.native_reward,
                    answer=decision.answer,
                    left_pattern=decision.left_pattern,
                    right_pattern=decision.right_pattern,
                    start_step=7,
                    end_step=7,
                    complete=True,
                    **common,
                )
            )
        elif isinstance(result, CountRecallStreamResult):
            if len(result.queries) != env.horizon * env.attempts:
                raise ResultValidationError(
                    f"Stream {result.task_id} scored {len(result.queries)} queries; "
                    f"the native deck presents {env.horizon * env.attempts}."
                )
            for query in result.queries:
                counts = query_write_fields(result, query)
                rows.append(
                    _event(
                        contract,
                        config,
                        result=result,
                        kind="query",
                        event_index=query.index,
                        step=query.index,
                        numerator=int(query.correct),
                        denominator=1,
                        native_return=query.native_reward,
                        true_count=query.true_count,
                        answer=query.answer,
                        writes_before_decision=counts.writes,
                        evidence_age_writes=counts.age,
                        count_before_current_segment=counts.count_before_segment,
                        count_in_current_segment=counts.count_in_segment,
                        start_step=query.index,
                        end_step=query.index,
                        **common,
                    )
                )
        elif isinstance(result, TMazeEpisodeResult):
            if result.decisions != env.horizon:
                raise ResultValidationError(
                    f"T-Maze task {result.task_id} ran {result.decisions} decisions; "
                    f"the budget is exactly {env.horizon}."
                )
            rows.append(
                _event(
                    contract,
                    config,
                    result=result,
                    kind="episode",
                    event_index=1,
                    step=result.decisions,
                    numerator=int(result.episode.success),
                    denominator=1,
                    native_return=result.episode.native_return,
                    writes_before_decision=_writes_at(result.writes, result.decisions),
                    start_step=1,
                    end_step=result.decisions,
                    **common,
                )
            )
        elif isinstance(result, ConcentrationResult):
            if result.pairs != env.size // 2:
                raise ResultValidationError(
                    f"Board {result.task_id} holds {result.pairs} pairs; the "
                    f"contract declares {env.size // 2}."
                )
            ledger = result.ledger()
            rows.append(
                _event(
                    contract,
                    config,
                    result=result,
                    kind="episode",
                    event_index=1,
                    step=result.decisions,
                    numerator=result.matched_pairs,
                    denominator=result.pairs,
                    native_return=result.episode_return,
                    writes_before_decision=_writes_at(result.writes, result.decisions),
                    start_step=1,
                    end_step=result.decisions,
                    retrieval_opportunities=ledger.opportunities,
                    retrieval_successes=ledger.successes,
                    evicted_opportunities=ledger.evicted_opportunities,
                    evicted_successes=ledger.evicted_successes,
                    invalid_flips=ledger.invalid,
                    redundant_flips=ledger.redundant,
                    board_complete=result.board_complete,
                    flips=result.flips
                    if contract.evaluation.retention == "complete"
                    else None,
                    **common,
                )
            )
        else:
            if result.goals_total != env.goals:
                raise ResultValidationError(
                    f"Map {result.task_id} ran {result.goals_total} goals; the "
                    f"contract declares {env.goals}."
                )
            rows.append(
                _event(
                    contract,
                    config,
                    result=result,
                    kind="episode",
                    event_index=1,
                    step=result.decisions,
                    numerator=len(result.completions),
                    denominator=result.goals_total,
                    native_return=result.episode_return,
                    writes_before_decision=_writes_at(result.writes, result.decisions),
                    start_step=1,
                    end_step=result.decisions,
                    **common,
                )
            )
    return tuple(rows)


# ----------------------------------------------------------------------
# Secondary summaries and evaluation-only references
# ----------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def attempt_secondary(
    environment: EnvironmentConfig, results: Sequence[AttemptTaskResult]
) -> dict[str, float | None]:
    """Whole-budget records that the primary scored-attempt endpoint omits.

    Failures enter every fixed-budget timing summary at the full physical
    limit. Success-conditioned means are ``None``, not zero, when nothing
    succeeded. Key/door timings are reported when the attempts carry them.
    """
    if not results:
        raise ResultValidationError("Secondary summary needs at least one task.")
    limit = float(environment.horizon)
    budget = float(environment.outer_length)
    scored = environment.attempts
    all_attempts = [a for r in results for a in r.attempts]
    later = [a for r in results for a in r.attempts[scored:]]
    successes = [a for a in all_attempts if a.success]
    summary: dict[str, float | None] = {
        "tasks": float(len(results)),
        "full_budget_return_mean": float(np.mean([r.task_return for r in results])),
        "completed_attempts_mean": float(np.mean([len(r.attempts) for r in results])),
        "partial_attempt_fraction": float(
            np.mean([r.partial is not None for r in results])
        ),
        "later_attempt_count": float(len(later)),
        "later_attempt_success": _mean([float(a.success) for a in later]),
        "completion_steps_fixed_budget_mean": float(
            np.mean([float(a.steps) if a.success else limit for a in all_attempts])
        ),
        "completion_steps_success_only_mean": _mean(
            [float(a.steps) for a in successes]
        ),
    }
    if all_attempts and hasattr(all_attempts[0], "key_step"):
        key_steps = [r.first_event_step("key_step") for r in results]
        door_steps = [r.first_event_step("door_step") for r in results]
        summary.update(
            {
                "key_acquisition_rate": float(
                    np.mean([cast(Any, a).key_step is not None for a in all_attempts])
                ),
                "first_key_step_fixed_budget_mean": float(
                    np.mean([budget if s is None else float(s) for s in key_steps])
                ),
                "first_door_step_fixed_budget_mean": float(
                    np.mean([budget if s is None else float(s) for s in door_steps])
                ),
                "tasks_without_key": float(sum(s is None for s in key_steps)),
                "tasks_without_door": float(sum(s is None for s in door_steps)),
            }
        )
    else:
        first_success = [
            next(
                (float(a.first_step + a.steps - 1) for a in r.attempts if a.success),
                budget,
            )
            for r in results
        ]
        summary["first_success_step_fixed_budget_mean"] = float(np.mean(first_success))
    # Success by attempt over the whole declared budget, whatever band the
    # primary endpoint scores, and the in-context adaptation it implies.
    by_attempt = [
        [float(r.attempts[k].success) for r in results if len(r.attempts) > k]
        for k in range(scored)
    ]
    for k, column in enumerate(by_attempt, 1):
        summary[f"attempt_{k}_success"] = _mean(column)
    summary["success_first"] = _mean(by_attempt[0]) if by_attempt else None
    last_two = [v for column in by_attempt[-2:] for v in column]
    summary["success_last2"] = _mean(last_two)
    first_mean, last_mean = summary["success_first"], summary["success_last2"]
    summary["delta_adapt"] = (
        None if first_mean is None or last_mean is None else last_mean - first_mean
    )
    return summary


def tmaze_secondary(
    environment: EnvironmentConfig, results: Sequence[TMazeEpisodeResult]
) -> dict[str, float | None]:
    """Success by cue side, junction arrival and the forward-move budget.

    The corridor and the turn use the whole budget, so ``forward_moves``
    equals the corridor length on every success; a smaller mean says how
    much of the budget was wasted on lateral or backward moves.
    """
    if not results:
        raise ResultValidationError("A secondary summary needs at least one task.")
    episodes = [r.episode for r in results]
    up = [float(e.success) for e in episodes if e.cue == 1]
    down = [float(e.success) for e in episodes if e.cue == -1]
    reached = [e for e in episodes if e.junction_step is not None]
    return {
        "tasks": float(len(results)),
        "goal_success": float(np.mean([e.success for e in episodes])),
        "success_cue_up": _mean(up),
        "success_cue_down": _mean(down),
        "tasks_cue_up": float(len(up)),
        "tasks_cue_down": float(len(down)),
        "junction_reached_rate": float(len(reached) / len(episodes)),
        "junction_step_mean": _mean([float(e.junction_step or 0) for e in reached]),
        "forward_moves_mean": float(np.mean([e.forward_moves for e in episodes])),
        "corridor_length": float(environment.size),
        "native_return_mean": float(np.mean([e.native_return for e in episodes])),
        "decisions_mean": float(np.mean([r.decisions for r in results])),
    }


def cue_blind_tmaze_success(
    environment: Any, *, task_ids: Sequence[int], generator_seed: int = 0
) -> dict[str, float]:
    """The declared evaluation-only references of the passive T-Maze.

    ``random``: uniform actions, which almost never reach the junction in
    time. ``cue_blind``: the strongest memoryless policy, forward along the
    corridor and a fixed turn (up) at the junction, whose success is the
    fraction of tasks whose cue is up; the C2 level is the larger of the two.
    """
    if not task_ids:
        raise ResultValidationError("The T-Maze references need tasks to run.")
    generator = np.random.default_rng(generator_seed)
    random_success: list[float] = []
    blind_success: list[float] = []
    for task_id in task_ids:
        for policy in ("random", "cue_blind"):
            environment.set_task(int(task_id))
            packet, _ = environment.reset()
            done = False
            while not done:
                if policy == "random":
                    action = int(generator.integers(environment.action_count))
                else:
                    at_junction = environment.public_fields(packet)["at_junction"]
                    action = UP if at_junction else FORWARD
                packet, _, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
            episode = environment.episode
            if episode is None:
                raise ResultValidationError("The reference roster lost an episode.")
            (random_success if policy == "random" else blind_success).append(
                float(episode.success)
            )
    return {
        "random_reference_goal_success": float(np.mean(random_success)),
        "cue_blind_reference_goal_success": float(np.mean(blind_success)),
    }


def count_prior(
    results: Sequence[CountRecallStreamResult], *, horizon: int
) -> tuple[int, ...]:
    """Fit the modal true count at each episode position.

    This is the declared count/time-prior reference: it sees the position of a
    decision and nothing else, so it cannot separate two queries that arrive at
    the same position after different valid histories.
    """
    if not results:
        raise ResultValidationError("A count prior needs at least one stream.")
    modal: list[int] = []
    for position in range(horizon):
        counts: dict[int, int] = {}
        for result in results:
            value = result.queries[position].true_count
            counts[value] = counts.get(value, 0) + 1
        modal.append(max(sorted(counts), key=lambda key: counts[key]))
    return tuple(modal)


def prior_accuracy(
    prior: Sequence[int], results: Sequence[CountRecallStreamResult]
) -> float:
    """Exact accuracy of a fitted count/time prior on the given streams."""
    if not results:
        raise ResultValidationError("A prior needs streams to score.")
    correct = sum(
        query.true_count == prior[query.index - 1]
        for result in results
        for query in result.queries
    )
    return correct / sum(len(result.queries) for result in results)


def random_reference_accuracy(actions: int) -> float:
    """Exact accuracy of a uniform answer policy over ``actions`` answers.

    The answer space is the protocol's (27 for Easy and Medium, 17 for Hard),
    so callers name it; there is no default.
    """
    if type(actions) is not int or actions < 1:
        raise ResultValidationError("The action space must be positive.")
    return 1.0 / actions


def count_recall_actions(protocol: str) -> int:
    """The answer space of one CountRecall protocol identity."""
    return COUNT_RECALL_ACTIONS[count_recall_variant(protocol)]


def count_recall_secondary(
    environment: EnvironmentConfig,
    results: Sequence[CountRecallStreamResult],
    *,
    prior: Sequence[int] | None = None,
    position_bins: int = 4,
) -> dict[str, float | None]:
    """Position and count strata, plus the evaluation-only reference controls.

    The references are not information-free. Both see the current query token,
    and AMAGO's RL2 feedback additionally reveals whether the previous answer
    was exactly right.
    """
    if not results:
        raise ResultValidationError("A secondary summary needs at least one stream.")
    horizon = environment.horizon
    actions = count_recall_actions(environment.benchmark)
    queries = [query for result in results for query in result.queries]
    summary: dict[str, float | None] = {
        "streams": float(len(results)),
        "queries": float(len(queries)),
        "exact_accuracy": float(np.mean([query.correct for query in queries])),
        "native_return_mean": float(np.mean([r.stream_return for r in results])),
        "return_identity_gap": float(
            max(abs(r.stream_return - (2 * r.accuracy - 1)) for r in results)
        ),
        "answer_error_mean": float(
            np.mean([abs(query.answer - query.true_count) for query in queries])
        ),
        "random_reference_accuracy": random_reference_accuracy(actions),
    }
    edges = (
        np.asarray([0, 32, 64, 96, 103])
        if environment.benchmark == "count-recall-medium"
        else np.linspace(0, horizon, position_bins + 1)
    )
    for index in range(position_bins):
        low, high = int(edges[index]) + 1, int(edges[index + 1])
        selected = [q for q in queries if low <= q.index <= high]
        summary[f"accuracy_positions_{low}_{high}"] = _mean(
            [float(q.correct) for q in selected]
        )
        summary[f"absolute_error_positions_{low}_{high}"] = _mean(
            [float(abs(q.answer - q.true_count)) for q in selected]
        )
        summary[f"queries_positions_{low}_{high}"] = float(len(selected))
    # The count strata end at the protocol's largest possible count.
    for label, low, high in (("low", 0, 4), ("mid", 5, 12), ("high", 13, actions - 1)):
        selected = [q for q in queries if low <= q.true_count <= high]
        summary[f"accuracy_counts_{label}"] = _mean(
            [float(q.correct) for q in selected]
        )
        summary[f"queries_counts_{label}"] = float(len(selected))
    summary["prior_reference_accuracy"] = (
        prior_accuracy(prior, results) if prior is not None else None
    )
    return summary


def mazerunner_laps_secondary(
    environment: EnvironmentConfig, results: Sequence[AttemptTaskResult]
) -> dict[str, float | None]:
    """Laps finished, goals reached and calls per lap over the outer budget.

    ``goals_per_500_calls`` pools every goal of every unit over every charged
    call of the budget, so a unit that stalled counts in full; the lap
    fractions read finished laps only and the partial lap is counted apart.
    """
    if not results:
        raise ResultValidationError("A laps summary needs at least one map.")
    goals_total = environment.goals
    if goals_total is None:
        raise ResultValidationError("MazeRunner requires its declared goal count.")
    finished = [lap for result in results for lap in result.attempts]
    partial = [result.partial for result in results if result.partial is not None]
    goals = [float(result.task_return) for result in results]
    calls = [float(result.decisions) for result in results]
    first = [result.attempts[0] for result in results if result.attempts]
    return {
        "maps": float(len(results)),
        "laps_finished_mean": float(np.mean([len(r.attempts) for r in results])),
        "partial_laps": float(len(partial)),
        "goals_mean": float(np.mean(goals)),
        "goals_per_500_calls": 500.0 * float(np.sum(goals)) / float(np.sum(calls)),
        "decisions_mean": float(np.mean(calls)),
        "lap_goal_fraction_mean": _mean(
            [float(lap.native_return) / goals_total for lap in finished]
        ),
        "lap_full_sequence_rate": _mean([float(lap.success) for lap in finished]),
        "calls_per_finished_lap_mean": _mean([float(lap.steps) for lap in finished]),
        "first_lap_goal_fraction": _mean(
            [float(lap.native_return) / goals_total for lap in first]
        ),
        "first_lap_calls_mean": _mean([float(lap.steps) for lap in first]),
        "partial_lap_goal_fraction": _mean(
            [float(lap.native_return) / goals_total for lap in partial]
        ),
    }


def random_policy_goal_fraction(
    environment: MazeRunnerEnv, *, task_ids: Sequence[int], generator_seed: int = 0
) -> dict[str, float]:
    """The declared random-policy reference, measured on the same maps.

    Evaluation-only: no policy, no weights, no fit. It sees the same public
    packets a learner would and answers with a uniform action, so it is a
    behavioral floor rather than an information-free bound.
    """
    if not task_ids:
        raise ResultValidationError("The random reference needs maps to run.")
    generator = np.random.default_rng(generator_seed)
    fractions: list[float] = []
    completions: list[bool] = []
    for map_id in task_ids:
        environment.set_task(int(map_id))
        environment.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = environment.step(
                int(generator.integers(MAZERUNNER_ACTIONS))
            )
            done = terminated or truncated
        reached = len(environment.completed_goals)
        fractions.append(reached / environment.goals)
        completions.append(reached == environment.goals)
    return {
        "random_reference_goal_fraction": float(np.mean(fractions)),
        "random_reference_full_sequence": float(np.mean(completions)),
    }


WRITES_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3plus", 3, None),
)
"""Writes-before-the-flip bands of the retrieval rate (plan §5.3)."""


def concentration_secondary(
    environment: EnvironmentConfig, results: Sequence[ConcentrationResult]
) -> dict[str, float | None]:
    """Pairs, completion, native return, wasted flips and the retrieval ledger.

    The retrieval rate is the ratio of summed hits to summed opportunities over
    the roster (a per-task rate would be undefined on boards without an
    opportunity); by writes band it needs the summary carrier's counters and is
    None otherwise.
    """
    if not results:
        raise ResultValidationError("A secondary summary needs at least one board.")
    pairs = environment.size // 2
    flips = [flip for result in results for flip in result.flips]
    ledgers = [result.ledger() for result in results]
    opportunities = sum(ledger.opportunities for ledger in ledgers)
    successes = sum(ledger.successes for ledger in ledgers)
    tracked = all(result.writes is not None for result in results)
    summary: dict[str, float | None] = {
        "boards": float(len(results)),
        "pair_fraction": float(np.mean([r.pair_fraction for r in results])),
        "board_complete_rate": float(np.mean([r.board_complete for r in results])),
        "native_return_mean": float(np.mean([r.episode_return for r in results])),
        "decisions_mean": float(np.mean([r.decisions for r in results])),
        "flips": float(len(flips)),
        "invalid_flip_fraction": float(np.mean([f.invalid for f in flips])),
        "redundant_flip_fraction": float(np.mean([f.redundant for f in flips])),
        "retrieval_opportunities": float(opportunities),
        "retrieval_opportunities_per_board": opportunities / len(results),
        "retrieval_rate": successes / opportunities if opportunities else None,
        "random_reference_pair_fraction": None,
    }
    if tracked:
        evicted = sum(cast(int, ledger.evicted_opportunities) for ledger in ledgers)
        evicted_hits = sum(cast(int, ledger.evicted_successes) for ledger in ledgers)
        summary["evicted_retrieval_opportunities"] = float(evicted)
        summary["evicted_retrieval_rate"] = evicted_hits / evicted if evicted else None
        retained = opportunities - evicted
        summary["retained_retrieval_rate"] = (
            (successes - evicted_hits) / retained if retained else None
        )
        for label, low, high in WRITES_BANDS:
            band_hits = band_total = 0
            for result in results:
                for flip in result.flips:
                    if not flip.opportunity:
                        continue
                    writes = _writes_at(result.writes, flip.index)
                    assert writes is not None
                    if writes < low or (high is not None and writes > high):
                        continue
                    band_total += 1
                    band_hits += int(flip.hit)
            summary[f"retrieval_rate_writes_{label}"] = (
                band_hits / band_total if band_total else None
            )
            summary[f"retrieval_opportunities_writes_{label}"] = float(band_total)
    else:
        summary["evicted_retrieval_opportunities"] = None
        summary["evicted_retrieval_rate"] = None
        summary["retained_retrieval_rate"] = None
    for result in results:
        if result.pairs != pairs:
            raise ResultValidationError("Concentration board pairs disagree.")
    return summary


def random_policy_pair_fraction(
    environment: ConcentrationEnv, *, task_ids: Sequence[int], generator_seed: int = 0
) -> dict[str, float]:
    """The declared random-policy reference, measured on the same boards.

    Evaluation-only: uniform flips on the same public boards a learner would
    see; a behavioral floor rather than an information-free bound.
    """
    if not task_ids:
        raise ResultValidationError("The random reference needs boards to run.")
    generator = np.random.default_rng(generator_seed)
    fractions: list[float] = []
    completions: list[bool] = []
    for task_id in task_ids:
        environment.set_task(int(task_id))
        environment.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = environment.step(
                int(generator.integers(environment.cards))
            )
            done = terminated or truncated
        fractions.append(environment.matched_pairs / environment.pairs)
        completions.append(environment.board_complete)
    return {
        "random_reference_pair_fraction": float(np.mean(fractions)),
        "random_reference_board_complete": float(np.mean(completions)),
    }


KEY_TO_DOOR_MOVES: tuple[tuple[int, int], ...] = ((0, -1), (-1, 0), (0, 1), (1, 0))
"""The native left/up/right/down displacements in the public (x, y) grid, in
native action order (index 4 is stay), as the C3 diagnostic uses them."""


def sweep_policy_action(position: tuple[int, int], size: int) -> int:
    """The declared public-current-observation reactive reference (R5).

    A fixed Hamiltonian cycle over the ``size`` x ``size`` room, followed from
    whatever public position the agent reports: column 0 is walked upward, the
    other columns are swept row by row in a boustrophedon, so every cell is
    visited once per ``size**2`` physical steps whatever the start. The action
    is a function of the current public position alone; possession, time and
    history play no part. It is a memoryless floor with systematic coverage,
    declared before any revised fit is read; it is not a tuned competitor.
    """
    if size < 2 or size % 2:
        # The cycle closes only on an even side: an odd corner would step out
        # of the room and the clipped agent would sit still forever.
        raise ContractError("The sweep needs an even room side of at least 2.")
    a, b = position
    last = size - 1
    if a == 0:
        move = (0, -1) if b > 0 else (1, 0)
    elif b % 2 == 0:
        move = (1, 0) if a < last else (0, 1)
    elif a > 1:
        move = (-1, 0)
    else:
        move = (0, 1) if b < last else (-1, 0)
    return KEY_TO_DOOR_MOVES.index(move)


@dataclass(frozen=True, slots=True)
class ReferenceRollout:
    """One evaluation-only policy's outcomes on one task roster."""

    name: str
    generator_seeds: tuple[int, ...]
    doors_completed: Mapping[int, float]
    door_success_first8: Mapping[int, float]
    charged_calls: int
    physical_actions: int
    pair_fractions: Mapping[int, float] | None = None
    exact_accuracies: Mapping[int, float] | None = None
    attempt_success: Mapping[int, float] | None = None
    """Per-task success over the scored attempts (XLand's attempts 4-5),
    averaged over the layout roots and action seeds the reference ran."""

    @property
    def mean_doors_completed(self) -> float:
        return float(np.mean(list(self.doors_completed.values())))

    @property
    def mean_door_success_first8(self) -> float:
        return float(np.mean(list(self.door_success_first8.values())))

    @property
    def mean_attempt_success(self) -> float:
        if self.attempt_success is None:
            raise ResultValidationError("This reference scored no attempts.")
        return float(np.mean(list(self.attempt_success.values())))


def visible_board_action(
    board: NDArray[np.int64], *, sentinel: int, generator: np.random.Generator
) -> int:
    """Uniformly choose a currently face-down position, or any position if none.

    This reactive reference reads only the current visible board. It avoids
    visible cards, including matched cards and the pending first flip, without
    retaining revealed positions or consulting evaluator state. Mismatched
    cards stay visible for one observation, so they too are skipped then.
    """
    hidden = np.flatnonzero(board == sentinel)
    return int(generator.choice(hidden if len(hidden) else np.arange(len(board))))


def concentration_references(
    environment: ConcentrationEnv,
    *,
    task_ids: Sequence[int],
    generator_seeds: Sequence[int] = (0, 1, 2),
) -> tuple[ReferenceRollout, ReferenceRollout]:
    """Random and visible-board references, with actual per-board scores/clocks."""
    if not task_ids or not generator_seeds:
        raise ResultValidationError("Concentration references need tasks and seeds.")
    output: list[ReferenceRollout] = []
    for reactive in (False, True):
        fractions: defaultdict[int, list[float]] = defaultdict(list)
        charged = physical = 0
        for seed in generator_seeds:
            generator = np.random.default_rng(int(seed))
            for task in task_ids:
                environment.set_task(int(task))
                before = environment.collection_counters()
                packet, _ = environment.reset()
                done = False
                while not done:
                    action = (
                        visible_board_action(
                            environment.decode(packet)[0],
                            sentinel=environment.ranks,
                            generator=generator,
                        )
                        if reactive
                        else int(generator.integers(environment.cards))
                    )
                    packet, _, terminated, truncated, _ = environment.step(action)
                    done = terminated or truncated
                fractions[int(task)].append(
                    environment.matched_pairs / environment.pairs
                )
                after = environment.collection_counters()
                charged += after["charged_calls"] - before["charged_calls"]
                physical += after["physical_actions"] - before["physical_actions"]
        output.append(
            ReferenceRollout(
                name="visible-board uniform hidden flips"
                if reactive
                else "uniform random flips",
                generator_seeds=tuple(int(s) for s in generator_seeds),
                doors_completed={},
                door_success_first8={},
                charged_calls=charged,
                physical_actions=physical,
                pair_fractions={t: float(np.mean(v)) for t, v in fractions.items()},
            )
        )
    return output[0], output[1]


def _key_to_door_rollout(
    environment: Any, *, task_ids: Sequence[int], chooser: Any
) -> tuple[dict[int, float], dict[int, float], int, int]:
    doors: dict[int, float] = {}
    first8: dict[int, float] = {}
    charged = physical = 0
    for task_id in task_ids:
        environment.set_task(int(task_id))
        before = environment.collection_counters()
        packet, _ = environment.reset()
        done = False
        while not done:
            packet, _, terminated, truncated, _ = environment.step(chooser(packet))
            done = terminated or truncated
        attempts = environment.completed_attempts
        if len(attempts) < environment.attempts:
            raise ResultValidationError("The reference roster lost a scored attempt.")
        doors[int(task_id)] = float(sum(a.success for a in attempts))
        first8[int(task_id)] = float(
            np.mean([a.success for a in attempts[: environment.attempts]])
        )
        after = environment.collection_counters()
        charged += after["charged_calls"] - before["charged_calls"]
        physical += after["physical_actions"] - before["physical_actions"]
    return doors, first8, charged, physical


def random_policy_door_counts(
    environment: Any,
    *,
    task_ids: Sequence[int],
    generator_seeds: Sequence[int] = (0, 1, 2),
) -> ReferenceRollout:
    """The random-policy reference on Key-to-Door: uniform actions, three
    declared action seeds averaged within each task (R5)."""
    if not task_ids or not generator_seeds:
        raise ResultValidationError("The random reference needs tasks and seeds.")
    doors: defaultdict[int, list[float]] = defaultdict(list)
    first8: defaultdict[int, list[float]] = defaultdict(list)
    charged = physical = 0
    count = int(environment.action_count)
    for seed in generator_seeds:
        generator = np.random.default_rng(int(seed))

        def choose(
            packet: Mapping[str, Any], rng: np.random.Generator = generator
        ) -> int:
            del packet  # uniform actions read nothing
            return int(rng.integers(count))

        d, f, c, p = _key_to_door_rollout(
            environment, task_ids=task_ids, chooser=choose
        )
        for task, value in d.items():
            doors[task].append(value)
        for task, value in f.items():
            first8[task].append(value)
        charged += c
        physical += p
    return ReferenceRollout(
        name="random policy (uniform actions)",
        generator_seeds=tuple(int(s) for s in generator_seeds),
        doors_completed={t: float(np.mean(v)) for t, v in doors.items()},
        door_success_first8={t: float(np.mean(v)) for t, v in first8.items()},
        charged_calls=charged,
        physical_actions=physical,
    )


def sweep_policy_door_counts(
    environment: Any, *, task_ids: Sequence[int]
) -> ReferenceRollout:
    """The declared reactive reference on Key-to-Door: the deterministic sweep
    of :func:`sweep_policy_action` from the public position (R5)."""
    if not task_ids:
        raise ResultValidationError("The sweep reference needs tasks to run.")
    size = int(environment.size)

    def chooser(packet: Mapping[str, Any]) -> int:
        fields = environment.public_fields(packet)
        position = (
            round(fields["position_x"] * size),
            round(fields["position_y"] * size),
        )
        return sweep_policy_action(position, size)

    doors, first8, charged, physical = _key_to_door_rollout(
        environment, task_ids=task_ids, chooser=chooser
    )
    return ReferenceRollout(
        name="sweep policy (public position only)",
        generator_seeds=(),
        doors_completed=doors,
        door_success_first8=first8,
        charged_calls=charged,
        physical_actions=physical,
    )


XLAND_VIEW_AGENT = (4, 2)
"""The agent's cell in the ``5 x 5`` view: bottom row, centre column, facing up."""
XLAND_VIEW_WALKABLE = (1, 6, 10)
XLAND_VIEW_PICKABLE = (3, 4, 5, 7, 11, 12)


def xland_reactive_action(fields: Mapping[str, NDArray[np.float64]]) -> int:
    """The declared public-current reactive reference on the one-rule task.

    Reads only the decoded current packet: the ``5 x 5`` tile layer of the
    view, aligned so the agent sits at the bottom centre facing up. It walks
    towards the nearest visible pickable object (Manhattan distance in view
    cells, ties by row then column) and picks up whatever is directly ahead;
    with nothing visible it moves forward when the cell ahead is walkable and
    turns right otherwise. It keeps no memory, so it cannot tell a distractor
    from the precursor, cannot know that its pocket is full, and behaves the
    same on every attempt of a lifetime.
    """
    tiles = np.rint(fields["grid_tile"]).astype(int).reshape(5, 5)
    row, column = XLAND_VIEW_AGENT
    ahead = tiles[row - 1, column]
    ahead_walkable = int(ahead) in XLAND_VIEW_WALKABLE
    objects = [
        (r, c)
        for r in range(5)
        for c in range(5)
        if int(tiles[r, c]) in XLAND_VIEW_PICKABLE
    ]
    if not objects:
        return 0 if ahead_walkable else 1  # forward, else turn right
    target = min(
        objects, key=lambda cell: (abs(cell[0] - row) + abs(cell[1] - column), cell)
    )
    d_row, d_column = target[0] - row, target[1] - column
    if (d_row, d_column) == (-1, 0):
        return 3  # pick up
    if d_column == 0 or (d_row < 0 and ahead_walkable):
        return 0 if ahead_walkable else 1
    return 2 if d_column < 0 else 1  # turn towards the target's side


def _attempt_rollout(
    environment: Any,
    *,
    task_ids: Sequence[int],
    chooser: Callable[[Mapping[str, Any]], int],
) -> tuple[dict[int, float], dict[int, float], int, int]:
    """Drive one attempt task per identity; per-task scored and first success."""
    scored: dict[int, float] = {}
    first: dict[int, float] = {}
    band = int(getattr(environment, "scored_from", 1)) - 1
    before = environment.collection_counters()
    for task_id in task_ids:
        environment.set_task(int(task_id))
        packet, _ = environment.reset()
        done = False
        while not done:
            packet, _, terminated, truncated, _ = environment.step(chooser(packet))
            done = terminated or truncated
        attempts = environment.completed_attempts[: environment.attempts]
        if len(attempts) < environment.attempts:
            raise ResultValidationError("The reference roster lost an attempt.")
        scored[int(task_id)] = float(np.mean([a.success for a in attempts[band:]]))
        first[int(task_id)] = float(attempts[0].success)
    after = environment.collection_counters()
    return (
        scored,
        first,
        after["charged_calls"] - before["charged_calls"],
        after["physical_actions"] - before["physical_actions"],
    )


def xland_one_rule_references(
    environments: Sequence[Any],
    *,
    task_ids: Sequence[int],
    generator_seeds: Sequence[int] = (0, 1, 2),
) -> tuple[ReferenceRollout, ReferenceRollout]:
    """Random and public-current reactive references on the one-rule task.

    ``environments`` are the split's evaluation environments, one per
    declared layout root; the random policy also runs each declared action
    seed. Per-task scores are averaged over the roots (and seeds) so the
    panel pairs with the policies' replicate-averaged scores.
    """
    if not environments or not task_ids or not generator_seeds:
        raise ResultValidationError("The one-rule references need roots and tasks.")
    output: list[ReferenceRollout] = []
    for reactive in (False, True):
        scored: defaultdict[int, list[float]] = defaultdict(list)
        charged = physical = 0
        seeds = (0,) if reactive else tuple(int(s) for s in generator_seeds)
        for environment in environments:
            count = int(environment.action_count)
            for seed in seeds:
                generator = np.random.default_rng(int(seed))

                def choose(
                    packet: Mapping[str, Any],
                    rng: np.random.Generator = generator,
                    env: Any = environment,
                    scripted: bool = reactive,
                    actions: int = count,
                ) -> int:
                    if scripted:
                        return xland_reactive_action(env.public_fields(packet))
                    return int(rng.integers(actions))

                s, _, c, p = _attempt_rollout(
                    environment, task_ids=task_ids, chooser=choose
                )
                for task, value in s.items():
                    scored[task].append(value)
                charged += c
                physical += p
        output.append(
            ReferenceRollout(
                name="reactive policy (public current view only)"
                if reactive
                else "random policy (uniform actions)",
                generator_seeds=() if reactive else seeds,
                doors_completed={},
                door_success_first8={},
                charged_calls=charged,
                physical_actions=physical,
                attempt_success={t: float(np.mean(v)) for t, v in scored.items()},
            )
        )
    return output[0], output[1]


def random_policy_attempt_success(
    environment: Any, *, task_ids: Sequence[int], generator_seed: int = 0
) -> dict[str, float]:
    """The declared random-policy reference on a scored-band attempt task.

    Evaluation-only: uniform actions on the same rulesets and layout schedule a
    learner would meet; success is averaged over the attempts the primary
    endpoint scores (``scored_from .. attempts``) and, for context, over the
    first attempt.
    """
    if not task_ids:
        raise ResultValidationError("The random reference needs tasks to run.")
    generator = np.random.default_rng(generator_seed)
    scored: list[float] = []
    first: list[float] = []
    band = int(getattr(environment, "scored_from", 1)) - 1
    for task_id in task_ids:
        environment.set_task(int(task_id))
        environment.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = environment.step(
                int(generator.integers(environment.action_count))
            )
            done = terminated or truncated
        attempts = environment.completed_attempts[: environment.attempts]
        if len(attempts) < environment.attempts:
            raise ResultValidationError("The reference roster lost an attempt.")
        scored.append(float(np.mean([a.success for a in attempts[band:]])))
        first.append(float(attempts[0].success))
    return {
        "random_reference_scored_attempt_success": float(np.mean(scored)),
        "random_reference_first_attempt_success": float(np.mean(first)),
    }


def mazerunner_secondary(
    environment: EnvironmentConfig, results: Sequence[MazeRunnerEpisodeResult]
) -> dict[str, float | None]:
    """Goal completion, full-sequence success and time to each goal.

    Fixed-budget timings charge a goal that was never reached the whole native
    horizon, so failures are inside the summary rather than dropped.
    """
    if not results:
        raise ResultValidationError("A secondary summary needs at least one map.")
    if environment.goals is None:
        raise ResultValidationError("MazeRunner requires its declared goal count.")
    budget = float(environment.horizon)
    goals = environment.goals
    summary: dict[str, float | None] = {
        "maps": float(len(results)),
        "goal_fraction": float(np.mean([r.goal_fraction for r in results])),
        "full_sequence_rate": float(np.mean([r.full_sequence for r in results])),
        "native_return_mean": float(np.mean([r.episode_return for r in results])),
        "decisions_mean": float(np.mean([r.decisions for r in results])),
    }
    for index in range(1, goals):
        eligible = [r for r in results if len(r.completions) >= index]
        summary[f"goal_{index + 1}_success_given_goal_{index}"] = _mean(
            [float(len(r.completions) > index) for r in eligible]
        )
        summary[f"goal_{index + 1}_conditional_denominator"] = float(len(eligible))
    for index in range(goals):
        reached = [
            float(r.completions[index].step)
            for r in results
            if index < len(r.completions)
        ]
        steps = [
            float(r.completions[index].step) if index < len(r.completions) else budget
            for r in results
        ]
        summary[f"goal_{index}_reached_rate"] = float(len(reached) / len(results))
        summary[f"goal_{index}_steps_fixed_budget_mean"] = float(np.mean(steps))
        summary[f"goal_{index}_steps_success_only_mean"] = _mean(reached)
    return summary


def secondary(
    environment: EnvironmentConfig,
    results: Sequence[Any],
    *,
    prior: Sequence[int] | None = None,
) -> dict[str, float | None]:
    """Dispatch the environment's secondary summary."""
    if environment.name == "match_pattern":
        accuracy = float(np.mean([r.decision.correct for r in results]))
        return {
            "exact_accuracy": accuracy,
            "native_return": 2 * accuracy - 1,
            "completed_decisions": float(len(results)),
            "chance_accuracy": 0.5,
        }
    if environment.name == "count_recall":
        return count_recall_secondary(environment, results, prior=prior)
    if environment.name == "mazerunner":
        if environment.meta_horizon is not None:
            return mazerunner_laps_secondary(environment, results)
        return mazerunner_secondary(environment, results)
    if environment.name == "concentration":
        return concentration_secondary(environment, results)
    if environment.name == "tmaze":
        return tmaze_secondary(environment, results)
    return attempt_secondary(environment, results)


# ----------------------------------------------------------------------
# One evaluation
# ----------------------------------------------------------------------


def run_record(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    checkpoint: str,
    split: str,
    history: str,
    status: Literal["completed", "failed", "missing"] = "completed",
    metrics: Mapping[str, float] | None = None,
    parameter_count: int | None = None,
    hardware: str = "unmeasured",
    note: str = "",
    checkpoint_rule: CheckpointRule = "selected",
    layout_period: int | None = None,
) -> BenchmarkRun:
    """Build the run row that owns a set of evaluation events."""

    def clock(name: str) -> int | None:
        if metrics is None or name not in metrics:
            return None
        return int(metrics[name])

    return BenchmarkRun(
        protocol=contract.protocol,
        benchmark=contract.name,
        condition=config.condition,
        training_seed=config.seed,
        checkpoint=checkpoint,
        split=split,
        history=history,
        status=status,
        evaluation_seconds=None
        if metrics is None
        else float(metrics["runtime_seconds"]),
        parameter_count=parameter_count,
        hardware=hardware,
        note=note,
        checkpoint_rule=checkpoint_rule,
        retention=contract.evaluation.retention,
        metric=contract.evaluation.primary_metric,
        charged_calls=clock("charged_calls"),
        physical_actions=clock("physical_actions"),
        reset_only_steps=clock("reset_only_steps"),
        outer_length=contract.environment.outer_length,
        layout_period=layout_period,
    )


def evaluation_environment(
    contract: BenchmarkContract, config: ExperimentConfig, *, split: str, seed: int
) -> BaseEnv:
    """Build the scalar evaluation environment for one declared split."""
    source = contract.evaluation.splits[split].source
    return build_environment(config.as_runtime_mapping(), split=source, seed=seed)


DEVELOPMENT_REPLICATES = 2
"""The one-rule XLand protocol scores two layout roots on the development
split and every declared root on the final split."""


def evaluation_replicates(contract: BenchmarkContract, split: str) -> tuple[int, ...]:
    """The rollout seeds (layout roots) one split's panel replicates over."""
    seeds = tuple(contract.evaluation.rollout_seeds)
    if len(seeds) == 1:
        return seeds
    if contract.name != "xland_one_rule":
        raise ContractError("Only the one-rule XLand protocol declares several roots.")
    return seeds[:DEVELOPMENT_REPLICATES] if split == "development" else seeds


RESULTS_FILE = "benchmark_results.json"
SECONDARY_FILE = "benchmark_secondary.json"


def evaluation_directory(
    split: str,
    history: str,
    checkpoint_rule: str = "selected",
    horizon: int | None = None,
    layout_period: int | None = None,
    streams: int | None = None,
    laps: int | None = None,
) -> str:
    """``<split>-<history>``, suffixed by the rule when it is not ``selected``,
    by ``-h<H>`` for an extended-horizon panel (EXPERIMENTS section 5), by
    ``-relayout<P>`` for a layout-change continuation (a new hidden
    layout after every ``P`` calls), by ``-s<N>`` for a CountRecall task
    continued over ``N`` deck pairs (the continued-stream axis) and by
    ``-calls<B>`` for a MazeRunner map replayed in laps until ``B`` charged
    calls (the repeated-laps axis)."""
    if checkpoint_rule not in CHECKPOINT_RULES:
        raise ContractError(f"Unknown checkpoint rule: {checkpoint_rule!r}.")
    name = f"{split}-{history}"
    if checkpoint_rule != "selected":
        name = f"{name}-{checkpoint_rule}"
    if horizon is not None:
        if isinstance(horizon, bool) or int(horizon) < 1:
            raise ContractError("The panel horizon must be a positive integer.")
        name = f"{name}-h{int(horizon)}"
    if layout_period is not None:
        if isinstance(layout_period, bool) or int(layout_period) < 1:
            raise ContractError("The layout period is a positive number of calls.")
        name = f"{name}-relayout{int(layout_period)}"
    if streams is not None:
        if isinstance(streams, bool) or int(streams) < 2:
            raise ContractError("A continued-stream panel needs at least two streams.")
        if horizon is not None:
            raise ContractError(
                "A continued-stream panel names its streams, not a horizon."
            )
        name = f"{name}-s{int(streams)}"
    if laps is not None:
        if isinstance(laps, bool) or int(laps) < 1:
            raise ContractError("A repeated-laps panel names its budget in calls.")
        if horizon is not None or streams is not None:
            raise ContractError(
                "A repeated-laps panel names its budget, not a horizon or streams."
            )
        name = f"{name}-calls{int(laps)}"
    return name


def write_evaluation(
    run_directory: Path,
    contract: BenchmarkContract,
    run: BenchmarkRun,
    rows: Sequence[BenchmarkEvent],
    summary: Mapping[str, float | None],
    *,
    split: str,
    history: str,
    task_cap: int | None = None,
    checkpoint_rule: CheckpointRule = "selected",
    horizon: int | None = None,
    layout_period: int | None = None,
    streams: int | None = None,
    laps: int | None = None,
) -> Path:
    """Write one evaluation under
    ``<run>/eval/<split>-<history>[-<rule>][-h<H>][-relayout<P>][-s<N>][-calls<B>]/``.

    The development-selected panel keeps its directory; the fixed-final
    supplement is written beside it under the ``-final-epoch`` suffix, so
    neither overwrites the other. A capped roster cannot satisfy the
    completed-run validator, so it is written as an explicitly partial
    execution record instead.
    """
    import json
    from dataclasses import asdict

    if run.checkpoint_rule != checkpoint_rule:
        raise ContractError("The run's checkpoint rule disagrees with the destination.")
    if run.layout_period != layout_period:
        raise ContractError("The run's layout period disagrees with the destination.")
    destination = (
        run_directory
        / "eval"
        / evaluation_directory(
            split,
            history,
            checkpoint_rule,
            horizon=horizon,
            layout_period=layout_period,
            streams=streams,
            laps=laps,
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    if task_cap is None:
        write_benchmark_results(destination / RESULTS_FILE, [contract], [run], rows)
    else:
        (destination / RESULTS_FILE).write_text(
            json.dumps(
                {
                    "partial_task_cap": task_cap,
                    "runs": [asdict(run)],
                    "events": [asdict(event) for event in rows],
                },
                indent=2,
            )
            + "\n"
        )
    (destination / SECONDARY_FILE).write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return destination


__all__ = [
    "ALL_HISTORY_MODES",
    "HISTORY_MODES",
    "KEY_TO_DOOR_MOVES",
    "RESULTS_FILE",
    "SECONDARY_FILE",
    "SUMMARY_CLEARED",
    "WRITES_BANDS",
    "AttemptTaskResult",
    "ConcentrationLedger",
    "ConcentrationResult",
    "CountRecallStreamResult",
    "HistoryMode",
    "MazeRunnerEpisodeResult",
    "ReferenceRollout",
    "TMazeEpisodeResult",
    "attempt_secondary",
    "attempt_write_fields",
    "concentration_secondary",
    "continued_history_modes",
    "count_prior",
    "count_recall_actions",
    "count_recall_secondary",
    "cue_blind_tmaze_success",
    "evaluation_directory",
    "evaluation_environment",
    "evaluation_replicates",
    "events",
    "history_modes",
    "mazerunner_secondary",
    "prior_accuracy",
    "query_write_fields",
    "random_policy_attempt_success",
    "random_policy_door_counts",
    "random_policy_goal_fraction",
    "random_policy_pair_fraction",
    "random_reference_accuracy",
    "retained_attempts",
    "run_record",
    "secondary",
    "sweep_policy_action",
    "sweep_policy_door_counts",
    "task_intervention",
    "tmaze_secondary",
    "write_evaluation",
    "xland_one_rule_references",
    "xland_reactive_action",
]
