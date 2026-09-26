"""Native POPGym 1.0.7 CountRecall as a public-packet policy environment.

Wraps the pinned ``popgym.envs.count_recall.CountRecall`` through AMAGO's own
POPGym wrapper. Neither upstream package is patched.

The native lifecycle is kept exactly as it is and has no meta-attempt structure:
one reset deals the first value/query pair, each of the following 51 (Easy),
103 (Medium) or 207 (Hard) actions answers the query that was **already
visible** in the observation before it, and the episode terminates when the
value deck empties. An Easy stream therefore has 52 observation tokens and 51
scored decisions; Hard deals four decks of ranks (13 categories, 208 tokens,
207 decisions) and answers from 17 counts, since every rank occurs 16 times.

Only the public projection reaches the policy: two one-hot category fields and
AMAGO's ``timer / 1000``. ``info['counts']``, ``get_state()`` and the deck
contents are never read for an observation. :meth:`CountRecallEnv.render`
draws the dealt stream, the public counts and the scored answers for a human
observer, all reconstructed from the public tokens.

The scored quantity for the action taken after observing token ``t`` is

    n_t(q_t) = |{ i <= t : x_i == q_t }|

which is exactly the native ``prev_count``: the running count of dealt values
*including* the value shown alongside the query. :class:`PublicStreamCounter`
reconstructs it from public observations alone. It is a diagnostic used to build
evaluation records and to cross-check the native reward; its state is never an
actor input, a packet field, or a training target.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from reasoned_icrl.environments.base import (
    BaseEnv,
    PublicDecision,
    PublicField,
    benchmark_task_sources,
    unit_fields,
)
from reasoned_icrl.environments.rendering import (
    CATEGORY_COLORS,
    PALETTE,
    ansi_frame,
    color_grid,
    paint_cells,
    stack_frames,
)
from reasoned_icrl.environments.utils import NativeSource, project_unit_interval
from reasoned_icrl.experiments.contracts import ContractError

COUNT_RECALL_STATE_SCHEMA = "count-recall-state.v1"
COUNT_RECALL_PROTOCOLS = {
    "easy": "count-recall-easy",
    "medium": "count-recall-medium",
    "hard": "count-recall-hard",
}
"""Difficulty is a separate protocol identity; results are never pooled."""

COUNT_RECALL_CATEGORIES = {"easy": 2, "medium": 4, "hard": 13}
COUNT_RECALL_HORIZONS = {"easy": 51, "medium": 103, "hard": 207}
COUNT_RECALL_ACTIONS: dict[str, int] = {"easy": 27, "medium": 27, "hard": 17}
COUNT_RECALL_BLANK = -1
"""The decoded value of a record whose value slot is all zero: the query-only
tail of the extended-horizon adapter deals no card, so its records carry a blank value
and a query drawn
from :func:`tail_query_generator`. A blank value never occurs inside the native
stream; the adapter refuses one there."""


def continued_stream_seed(source: int, stream: int) -> int:
    """Native seed of the ``stream``-th deck pair (1-based) of task ``source``
    under the continued-stream axis (C3).
    Stream 1 is the task's own deck order; every later pair comes from a hash
    of the identity and the stream index, offset above every roster range so
    no evaluation deck pair coincides with a training identity's."""
    if isinstance(stream, bool) or int(stream) < 1:
        raise ContractError("The continued stream index is a positive integer.")
    if int(stream) == 1:
        return int(source)
    digest = hashlib.sha256(
        f"count-recall-continued:{int(source)}:{int(stream)}".encode()
    ).digest()
    return 2**31 + int.from_bytes(digest[:4], "big") % 2**31


def tail_query_generator(source: int) -> random.Random:
    """The query sequence of task ``source``'s query-only tail: a generator
    seeded from the task identity alone (a string seed, so the tail is the
    same in every process and independent of the native decks)."""
    if type(source) is not int or source < 0:
        raise ContractError("A CountRecall task identity is a nonnegative integer.")
    return random.Random(f"count-recall-tail:{source}")


"""Answers ``0 .. actions - 1``: the native deck deals every category
``actions - 1`` times (26 for one deck of colours or two of suits, 16 for four
decks of ranks), so a count outside the answer space cannot occur."""
COUNT_RECALL_SYMBOLS: dict[str, tuple[str, ...]] = {
    "easy": ("black", "red"),
    "medium": ("\u2660", "\u2666", "\u2663", "\u2665"),
    "hard": ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"),
}
"""Observer names for the category indices, in POPGym's own order
(``Deck.colors`` ``b, r``; ``Deck.suits`` ``s, d, c, h``; ``Deck.ranks``
``a .. k``). Rendering only; the packet carries the one-hot index."""
RENDER_STREAM_WIDTH = 52
"""Tokens per line when a stream is drawn: one native deck."""


def count_recall_protocol(variant: str) -> str:
    """Return the protocol identity of one declared difficulty."""
    try:
        return COUNT_RECALL_PROTOCOLS[variant]
    except KeyError as error:
        raise ContractError(f"Unknown CountRecall difficulty: {variant!r}.") from error


def count_recall_variant(protocol: str) -> str:
    """Return the difficulty behind one protocol identity."""
    for variant, name in COUNT_RECALL_PROTOCOLS.items():
        if name == protocol:
            return variant
    raise ContractError(f"Unknown CountRecall protocol: {protocol!r}.")


def public_fields(categories: int) -> tuple[PublicField, ...]:
    """Name the pinned public projection for provenance and leakage review."""
    names = [f"value_is_{index}" for index in range(categories)]
    names += [f"query_is_{index}" for index in range(categories)]
    names.append("timer")
    return unit_fields(names)


class PublicStreamCounter:
    """Exact `n_t(q_t)` from public observations only.

    Feed every observation in order, including the one returned by reset. Each
    call records the dealt value and returns the count that the *next* action
    must answer. Nothing here is derived from privileged state.
    """

    def __init__(self, categories: int) -> None:
        if type(categories) is not int or categories < 2:
            raise ContractError("CountRecall needs at least two categories.")
        self.categories = categories
        self.counts = np.zeros(categories, dtype=np.int64)
        self.queries = 0

    def observe(self, value: int, query: int) -> int:
        """Record the dealt value, then return the current query's true count."""
        for index in (value, query):
            if type(index) is not int or not 0 <= index < self.categories:
                raise ContractError("CountRecall category index is out of range.")
        self.counts[value] += 1
        self.queries += 1
        return int(self.counts[query])

    def count(self, query: int) -> int:
        """The true count of ``query`` when no value is dealt (a tail record)."""
        if type(query) is not int or not 0 <= query < self.categories:
            raise ContractError("CountRecall category index is out of range.")
        self.queries += 1
        return int(self.counts[query])

    def state(self) -> dict[str, object]:
        return {"counts": self.counts.tolist(), "queries": self.queries}

    def load(self, state: Mapping[str, object]) -> None:
        counts = np.asarray(state["counts"], dtype=np.int64)
        if counts.shape != (self.categories,):
            raise ContractError("CountRecall counter state changed shape.")
        self.counts = counts
        self.queries = int(cast(Any, state["queries"]))


@dataclass(frozen=True, slots=True)
class CountRecallQuery:
    """One scored decision, reconstructed from public observations only."""

    index: int
    query: int
    true_count: int
    answer: int
    correct: bool
    native_reward: float

    def __post_init__(self) -> None:
        if self.index < 1 or self.true_count < 0 or self.query < 0:
            raise ContractError("CountRecall query counters must be valid.")
        if self.correct != (self.answer == self.true_count):
            raise ContractError("CountRecall correctness disagrees with its answer.")


class CountRecallEnv(BaseEnv):
    """Public five-key packet over the pinned native CountRecall stream."""

    label: ClassVar[str] = "CountRecall"
    state_schema: ClassVar[str] = COUNT_RECALL_STATE_SCHEMA

    def __init__(
        self,
        *,
        variant: str = "easy",
        horizon: int | None = None,
        split: str = "train",
        source_indices: Sequence[int] | range | None = None,
        fixed_task_index: int | None = None,
        initial_seed: int = 0,
        render_mode: str | None = None,
        streams: int = 1,
    ) -> None:
        self.protocol = count_recall_protocol(variant)
        super().__init__(
            split=split,
            roster=benchmark_task_sources(split)
            if source_indices is None
            else source_indices,
            fixed_task_index=fixed_task_index,
            initial_seed=initial_seed,
            render_mode=render_mode,
        )
        self.variant = variant
        self.categories = COUNT_RECALL_CATEGORIES[variant]
        self.actions = COUNT_RECALL_ACTIONS[variant]
        self.native_horizon = COUNT_RECALL_HORIZONS[variant]
        """Scored decisions of the native stream: one fewer than its records."""
        self.horizon = self.native_horizon if horizon is None else int(horizon)
        if self.horizon < self.native_horizon:
            raise ContractError(
                "The CountRecall stream length is fixed by its native deck."
            )
        self.tail = self.horizon - self.native_horizon
        """Query-only records after the native deck (the extended-horizon
        adapter, evaluation only): each carries a blank value and a query from
        :func:`tail_query_generator`, answered against the dealt counts."""
        if isinstance(streams, bool) or int(streams) < 1:
            raise ContractError("CountRecall streams is a positive integer.")
        self.streams = int(streams)
        """Consecutive deck pairs in one outer task (the continued-stream axis,
        evaluation only): a reset-only call between streams starts the next
        pair from :func:`continued_stream_seed`, counts and the native timer
        restart, RL2 continues and the carried memory is never reset."""
        if self.streams > 1 and self.tail:
            raise ContractError(
                "A CountRecall task continues with more streams or a query-only "
                "tail, not both."
            )
        self.total_decisions = self.horizon * self.streams
        """Scored decisions of an evaluation task (``streams`` pairs)."""
        self._task_streams = self.streams
        self._task_decisions = self.total_decisions
        self._tail_rng: random.Random | None = None
        self._source = 0
        self._stream = 0
        self._stream_step = 0
        self._pending_reset = False
        self._native = NativeSource(self._make_native, seed=initial_seed)
        self._install_contract(public_fields(self.categories), self.actions)
        self._counter = PublicStreamCounter(self.categories)
        self._pending_count = 0
        self._pending_query = 0
        self._step = 0
        self._stream_return = 0.0
        self._records: list[CountRecallQuery] = []
        self._dealt: list[int] = []  # the public values in order, for rendering

    def _make_native(self) -> gym.Env[Any, Any]:
        from amago.envs.builtin.popgym_envs import POPGym

        return cast(
            gym.Env[Any, Any], POPGym(f"popgym-CountRecall{self.variant.title()}-v0")
        )

    @property
    def native(self) -> gym.Env[Any, Any]:
        """The wrapped upstream task, for inspection and audits only."""
        return self._native

    # ------------------------------------------------------------------
    # Evaluator-only accessors
    # ------------------------------------------------------------------

    @property
    def scored_queries(self) -> tuple[CountRecallQuery, ...]:
        """Every scored decision of the live stream."""
        return tuple(self._records)

    @property
    def stream_return(self) -> float:
        """Unscaled native return; a complete stream gives 2*accuracy - 1."""
        return self._stream_return

    def decode(self, packet: Mapping[str, np.ndarray]) -> tuple[int, int, float]:
        """Decode one packet's ``current`` token to (value, query, timer)."""
        current = np.asarray(packet["current"], dtype=np.float32)
        if current.shape != (len(self.fields),):
            raise ContractError("Packet width disagrees with the public contract.")
        raw = (current + 1.0) / 2.0
        value = raw[: self.categories]
        query = raw[self.categories : 2 * self.categories]
        for one_hot, blank_allowed in ((value, True), (query, False)):
            binary = np.all(np.isclose(one_hot, 0.0) | np.isclose(one_hot, 1.0))
            total = float(one_hot.sum())
            if not binary or not (
                np.isclose(total, 1.0) or (blank_allowed and np.isclose(total, 0.0))
            ):
                raise ContractError("CountRecall category field is not one-hot.")
        decoded = (
            COUNT_RECALL_BLANK
            if np.isclose(value.sum(), 0.0)
            else int(np.argmax(value))
        )
        return decoded, int(np.argmax(query)), float(raw[-1])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _accept(self, observation: object) -> np.ndarray:
        """Project one native observation and advance the public counter."""
        current = project_unit_interval(observation, len(self.fields), label=self.label)
        value, query, _ = self.decode({"current": current})
        if value == COUNT_RECALL_BLANK:
            if self._step < self.native_horizon:
                raise ContractError(
                    "A blank CountRecall value inside the native stream."
                )
            self._pending_count = self._counter.count(query)
        else:
            self._pending_count = self._counter.observe(value, query)
        if self._pending_count >= self.actions:
            # Each category is dealt exactly ``actions - 1`` times, so a larger
            # public count means the adapter has lost the native stream.
            raise ContractError(
                "Public CountRecall count exceeds the native answer space."
            )
        self._pending_query = query
        if value != COUNT_RECALL_BLANK:
            self._dealt.append(value)
        return current

    def _tail_observation(self) -> NDArray[np.float64]:
        """The next query-only record in the native ``[0, 1]`` form: a blank
        value slot, the tail generator's query and AMAGO's ``timer / 1000``
        continued past the deck."""
        if self._tail_rng is None:
            raise ContractError("The CountRecall tail generator is not initialised.")
        raw = np.zeros(2 * self.categories + 1, dtype=np.float64)
        raw[self.categories + self._tail_rng.randrange(self.categories)] = 1.0
        raw[-1] = self._step / 1000.0
        return raw

    def _begin_task(self, source: int) -> np.ndarray:
        observation, _ = self._native.reset(seed=source)
        self._tail_rng = tail_query_generator(source) if self.tail else None
        self._counter = PublicStreamCounter(self.categories)
        self._source = int(source)
        # Every task plays the environment's ``streams`` deck pairs (one, or
        # the continued-stream adapter's count).
        self._task_streams = self.streams
        self._task_decisions = self.horizon * self._task_streams
        self._stream = 0
        self._stream_step = 0
        self._pending_reset = False
        self._step = 0
        self._stream_return = 0.0
        self._records = []
        self._dealt = []
        return self._accept(observation)

    def _reset_info(self) -> dict[str, Any]:
        """Evaluator-side scalars only; native `counts` is never forwarded."""
        return {
            "evaluator_task_index": self.evaluator_task_index,
            "query_index": self._step,
            "stream_return": self._stream_return,
            "answered": False,
            "correct": False,
            "true_count": 0,
            "stream_index": self._stream + 1,
            "attempt_done": False,
        }

    def _next_stream(
        self,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """The reset-only call between two streams: the command is ignored, no
        reward is paid, the next deck pair is dealt from its own seed and its
        first record is returned as a new attempt (the Key-to-Door boundary)."""
        self._count_reset_only_step()
        self._pending_reset = False
        self._stream += 1
        self._stream_step = 0
        observation, _ = self._native.reset(
            seed=continued_stream_seed(self._source, self._stream + 1)
        )
        self._counter = PublicStreamCounter(self.categories)
        self._current = self._accept(observation)
        decision = PublicDecision(current=self._current, new_attempt=True)
        info = {
            **self._reset_info(),
            "query_index": self._step,
            "stream_return": self._stream_return,
            # The stream-cleared intervention resets the carried memory on this
            # call, after the ended pair's terminal record has been consumed and
            # before the new pair's first record is read, so the cleared context
            # starts exactly as a fresh task does (06:45 fix: raised on the last
            # decision of the pair, it let that pair's terminal card leak in).
            "attempt_done": True,
        }
        return self._packet(decision), 0.0, False, False, info

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Answer the already-visible query and advance one decision: a native
        one while the deck lasts, then a query-only tail record."""
        answer = self._validate_action(action)
        if self._pending_reset:
            return self._next_stream()
        scored_count, scored_query = self._pending_count, self._pending_query
        self._step += 1
        self._stream_step += 1
        correct = answer == scored_count
        expected = (1.0 if correct else -1.0) / self.native_horizon
        if self._stream_step <= self.native_horizon:
            observation, reward, terminated, truncated, _ = self._native.step(answer)
            # The public counter and the pinned reward must agree exactly; a
            # disagreement means the adapter has lost alignment with the stream.
            if not np.isclose(reward, expected, rtol=0, atol=1e-9):
                raise ContractError(
                    "Public CountRecall counter disagrees with the native reward."
                )
            if bool(terminated or truncated) != (
                self._stream_step >= self.native_horizon
            ):
                raise ContractError(
                    "CountRecall terminated away from its deck boundary."
                )
        else:
            # The tail: no card is dealt, the query is answered against the
            # dealt counts at the native reward scale, and the record is built
            # by the adapter.
            observation, reward = self._tail_observation(), expected
        self._stream_return += float(reward)
        self._records.append(
            CountRecallQuery(
                index=self._step,
                query=scored_query,
                true_count=scored_count,
                answer=answer,
                correct=correct,
                native_reward=float(reward),
            )
        )
        previous = self._current
        self._current = self._accept(observation)
        # The stream ends at its (extended) horizon the way the native one does:
        # terminated, never truncated (AMAGO's POPGym wrapper folds truncation).
        # Under the continued-stream axis a deck pair's end inside the outer
        # task is a physical end; the reset-only call that follows deals the
        # next pair.
        self._done = self._step >= self._task_decisions
        stream_done = self._stream_step >= self.horizon
        self._pending_reset = stream_done and not self._done
        terminated, truncated = self._done, False
        decision = PublicDecision(
            current=self._current,
            previous=previous,
            outcome=self._current,
            executed_action=answer,
            reward=float(reward),
            physical_done=stream_done and self._task_streams > 1,
            outer_terminated=terminated,
            outer_truncated=truncated,
        )
        info = {
            "evaluator_task_index": self.evaluator_task_index,
            "query_index": self._step,
            "stream_return": self._stream_return,
            "answered": True,
            "correct": correct,
            "true_count": scored_count,
            "stream_index": self._stream + 1,
            "attempt_done": False,
        }
        return self._packet(decision), float(reward), terminated, truncated, info

    def close(self) -> None:
        cast(Any, self._native).close()

    # ------------------------------------------------------------------
    # Rendering (observer only; everything here is public reconstruction)
    # ------------------------------------------------------------------

    @property
    def action_names(self) -> tuple[str, ...]:
        """Observer labels for the answers: the count each action claims."""
        return tuple(str(index) for index in range(self.action_count))

    @property
    def symbols(self) -> tuple[str, ...]:
        return COUNT_RECALL_SYMBOLS[self.variant]

    def _render_header(self) -> list[str]:
        symbols = self.symbols
        correct = sum(record.correct for record in self._records)
        lines = [
            f"{self.label} {self.variant}  task {self.evaluator_task_index}  "
            f"query {self._step}/{self.horizon}  return {self._stream_return:.3f}  "
            f"correct {correct}/{len(self._records)}",
            "counts  "
            + "  ".join(
                f"{symbol} {count}"
                for symbol, count in zip(symbols, self._counter.counts, strict=True)
            ),
        ]
        if self._done:
            lines.append("stream over")
        else:
            # A snapshot from before rendering existed restores no history.
            dealt = symbols[self._dealt[-1]] if self._dealt else "?"
            lines.append(
                f"dealt {dealt}  query {symbols[self._pending_query]}"
                f"  (true count {self._pending_count})"
            )
        if self._records:
            last = self._records[-1]
            verdict = "correct" if last.correct else f"wrong, true {last.true_count}"
            lines.append(
                f"last answer {last.answer} for {symbols[last.query]}: {verdict} "
                f"({last.native_reward:+.4f})"
            )
        return lines

    def _render_ansi(self) -> str:
        symbols = self.symbols
        width = 1 if self.variant != "hard" else 2
        rows: list[str] = []
        for start in range(0, len(self._dealt), RENDER_STREAM_WIDTH):
            chunk = self._dealt[start : start + RENDER_STREAM_WIDTH]
            rows.append(" ".join(symbols[value].rjust(width) for value in chunk))
            marks = []
            for offset in range(len(chunk)):
                # The answer to token ``index`` is record ``index``; the last
                # token of a finished stream is never answered.
                index = start + offset
                if index < len(self._records):
                    mark = "+" if self._records[index].correct else "x"
                elif not self._done:
                    mark = "?"
                else:
                    mark = " "
                marks.append(mark.rjust(width))
            rows.append(" ".join(marks))
        return ansi_frame(self._render_header(), rows)

    def _render_rgb(self) -> NDArray[np.uint8]:
        pixels = 12
        tokens = self.horizon + 1  # every token of a complete stream
        lines = -(-tokens // RENDER_STREAM_WIDTH)
        frames: list[NDArray[np.uint8]] = []
        for line in range(lines):
            start = line * RENDER_STREAM_WIDTH
            cards = color_grid(1, RENDER_STREAM_WIDTH, PALETTE["pending"])
            marks = color_grid(1, RENDER_STREAM_WIDTH, PALETTE["background"])
            for offset in range(RENDER_STREAM_WIDTH):
                index = start + offset
                if index >= tokens:
                    cards[0, offset] = PALETTE["background"]
                elif index < len(self._dealt):
                    cards[0, offset] = CATEGORY_COLORS[self._dealt[index]]
                    if index < len(self._records):
                        marks[0, offset] = PALETTE[
                            "correct" if self._records[index].correct else "wrong"
                        ]
                    elif not self._done:
                        marks[0, offset] = PALETTE["pending"]
            frames.append(paint_cells(cards, cell_pixels=pixels))
            frames.append(paint_cells(marks, cell_pixels=pixels // 2, grid=None))
        now = color_grid(1, 2, PALETTE["background"])
        if not self._done:
            if self._dealt:
                now[0, 0] = CATEGORY_COLORS[self._dealt[-1]]
            now[0, 1] = CATEGORY_COLORS[self._pending_query]
        frames.append(paint_cells(now, cell_pixels=2 * pixels))
        return stack_frames(frames)

    # ------------------------------------------------------------------
    # Restorable state
    # ------------------------------------------------------------------

    def _identity(self) -> tuple[object, ...]:
        return (self.variant, self.categories, self.horizon)

    @staticmethod
    def _deck_state(deck: Any) -> dict[str, object]:
        return {
            "idx": np.asarray(deck.idx).tolist(),
            "deck_len": int(deck.deck_len),
            "hands": {key: list(value) for key, value in deck.hands.items()},
        }

    @staticmethod
    def _load_deck(deck: Any, state: Mapping[str, object]) -> None:
        deck.idx = np.asarray(state["idx"], dtype=np.int64)
        deck.deck_len = int(cast(Any, state["deck_len"]))
        deck.hands = {
            str(key): list(cast(Any, value))
            for key, value in cast(Mapping[str, Any], state["hands"]).items()
        }

    def _state(self) -> dict[str, object]:
        native = cast(Any, self._native.unwrapped)
        popgym = cast(Any, self._native.env)
        return {
            "counter": self._counter.state(),
            "pending_count": self._pending_count,
            "pending_query": self._pending_query,
            "step": self._step,
            "stream_return": self._stream_return,
            "source": self._source,
            "stream": self._stream,
            "stream_step": self._stream_step,
            "pending_reset": self._pending_reset,
            "task_streams": self._task_streams,
            "records": [asdict(record) for record in self._records],
            "dealt": list(self._dealt),
            "native": {
                "value_deck": self._deck_state(native.value_deck),
                "query_deck": self._deck_state(native.query_deck),
                "counts": native.counts.tolist(),
                "query_counts": native.query_counts.tolist(),
                "value": int(native.value),
                "query": int(native.query),
                "last_obs": np.asarray(native.last_obs).tolist(),
                "timer": float(np.asarray(popgym.timer).reshape(-1)[0]),
            },
            "native_rng": deepcopy(native.np_random.bit_generator.state),
            "task_rng": deepcopy(self._native.generator_state()),
            "tail_rng": None if self._tail_rng is None else self._tail_rng.getstate(),
        }

    def _load_state(self, state: Mapping[str, object]) -> None:
        native_state = state.get("native")
        records = state.get("records")
        if not isinstance(native_state, Mapping) or not isinstance(records, Sequence):
            raise ContractError("CountRecall checkpoint is malformed.")
        restored = [CountRecallQuery(**cast(Any, row)) for row in records]
        native = cast(Any, self._native.unwrapped)
        popgym = cast(Any, self._native.env)
        self._load_deck(native.value_deck, cast(Any, native_state["value_deck"]))
        self._load_deck(native.query_deck, cast(Any, native_state["query_deck"]))
        native.counts = np.asarray(native_state["counts"], dtype=np.int64)
        native.query_counts = np.asarray(native_state["query_counts"], dtype=np.int64)
        native.value = int(cast(Any, native_state["value"]))
        native.query = int(cast(Any, native_state["query"]))
        native.last_obs = np.asarray(native_state["last_obs"], dtype=np.int64)
        native.prev_query = native.query
        popgym.timer = np.array([float(cast(Any, native_state["timer"]))])
        native.np_random.bit_generator.state = deepcopy(cast(Any, state["native_rng"]))
        self._native.load_generator_state(cast(Any, state["task_rng"]))
        self._counter = PublicStreamCounter(self.categories)
        self._counter.load(cast(Mapping[str, object], state["counter"]))
        self._pending_count = int(cast(Any, state["pending_count"]))
        self._pending_query = int(cast(Any, state["pending_query"]))
        self._step = int(cast(Any, state["step"]))
        self._stream_return = float(cast(Any, state["stream_return"]))
        # Snapshots written before the continued-stream axis hold one stream.
        self._source = int(cast(Any, state.get("source", self._task_index or 0)))
        self._stream = int(cast(Any, state.get("stream", 0)))
        self._stream_step = int(cast(Any, state.get("stream_step", self._step)))
        self._pending_reset = bool(state.get("pending_reset", False))
        self._task_streams = int(cast(Any, state.get("task_streams", self.streams)))
        self._task_decisions = self.horizon * self._task_streams
        self._records = restored
        # Snapshots written before rendering existed carry no dealt history.
        self._dealt = [int(value) for value in cast(Any, state.get("dealt", ()))]
        tail_state = state.get("tail_rng")
        if self.tail:
            if tail_state is None:
                raise ContractError("CountRecall checkpoint lacks the tail generator.")
            version, internal, gauss = cast(Any, tail_state)
            self._tail_rng = random.Random()
            self._tail_rng.setstate(
                (
                    int(version),
                    tuple(int(word) for word in internal),
                    None if gauss is None else float(gauss),
                )
            )
        else:
            self._tail_rng = None


__all__ = [
    "COUNT_RECALL_ACTIONS",
    "COUNT_RECALL_BLANK",
    "COUNT_RECALL_CATEGORIES",
    "COUNT_RECALL_HORIZONS",
    "COUNT_RECALL_PROTOCOLS",
    "COUNT_RECALL_SYMBOLS",
    "CountRecallEnv",
    "CountRecallQuery",
    "PublicStreamCounter",
    "continued_stream_seed",
    "count_recall_protocol",
    "count_recall_variant",
    "public_fields",
    "tail_query_generator",
]
