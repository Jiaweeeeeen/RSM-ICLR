"""Six public symbolic reveals and a single terminal equality-pattern answer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, cast

import numpy as np

from reasoned_icrl.environments.base import BaseEnv, PublicDecision, unit_fields
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.match_pattern import (
    HORIZON,
    PROTOCOL,
    SPLIT_COUNTS,
    SPLIT_STARTS,
    VOCABULARY,
    MatchPatternCorpus,
    load_corpus,
    manifest_hash,
    pattern_indices,
    task_sources,
)


@dataclass(frozen=True, slots=True)
class MatchPatternDecision:
    left_pattern: int
    right_pattern: int
    answer: int
    correct: bool
    native_reward: float

    def __post_init__(self) -> None:
        expected = int(self.left_pattern == self.right_pattern)
        if not (
            0 <= self.left_pattern < 4
            and 0 <= self.right_pattern < 4
            and self.answer in (0, 1)
        ):
            raise ContractError("Invalid match-pattern decision.")
        if self.correct != (self.answer == expected) or self.native_reward != (
            1.0 if self.correct else -1.0
        ):
            raise ContractError("Match-pattern answer, label and reward disagree.")


class MatchPatternEnv(BaseEnv):
    label: ClassVar[str] = "MatchPattern"
    protocol = PROTOCOL
    state_schema: ClassVar[str] = "match-pattern-state.v1"

    def __init__(
        self,
        *,
        split: str = "train",
        source_indices: Sequence[int] | range | None = None,
        fixed_task_index: int | None = None,
        initial_seed: int = 0,
        corpus: MatchPatternCorpus | None = None,
    ) -> None:
        super().__init__(
            split=split,
            roster=task_sources(split) if source_indices is None else source_indices,
            fixed_task_index=fixed_task_index,
            initial_seed=initial_seed,
        )
        self.horizon = HORIZON
        self._corpus = corpus
        self.corpus_sha256 = manifest_hash() if corpus is None else corpus.sha256
        names = [f"symbol_{i}" for i in range(VOCABULARY)] + [
            f"slot_{i}" for i in range(6)
        ]
        names += ["phase_reveal", "phase_query", "phase_terminal"]
        self._install_contract(unit_fields(names), 2)
        self.contract.update(
            {
                "corpus_sha256": self.corpus_sha256,
                "action_loss_protocol": "canonical-reveal-query-loss.v1",
            }
        )
        self._step = 0
        self._objects: tuple[int, ...] = ()
        self._decision: MatchPatternDecision | None = None
        count = SPLIT_COUNTS[split] if corpus is None else len(corpus.arrays[split])
        self._seen = np.zeros(count, dtype=np.bool_)

    @property
    def at_query(self) -> bool:
        return self._step == 6 and not self._done

    def canonical_action(self, action: Any) -> Any:
        """Called before AMAGO makes its action representation, after exploration."""
        return action if self.at_query else np.zeros_like(action)

    @property
    def scored_decision(self) -> MatchPatternDecision:
        if self._decision is None:
            raise ContractError("The match-pattern query has not been answered.")
        return self._decision

    def exposure_bitmap(self) -> bytes:
        """Evaluator/collector telemetry only; merge actors by bitwise OR."""
        return np.packbits(self._seen).tobytes()

    def _observation(self) -> np.ndarray:
        raw = np.zeros(73, dtype=np.float32)
        if self._step < 6:
            raw[self._objects[self._step]] = 1
            raw[64 + self._step] = 1
            raw[70] = 1
        else:
            raw[71 if self._step == 6 else 72] = 1
        return 2 * raw - 1

    def _begin_task(self, source: int) -> np.ndarray:
        if self._corpus is None:
            self._corpus = load_corpus()
        self._objects = self._corpus.example(self.split, source)
        self._step = 0
        self._decision = None
        return self._observation()

    def _reset_info(self) -> dict[str, Any]:
        return {"query_ready": False}

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        selected = self._validate_action(action)
        assert self._current is not None
        previous = self._current.copy()
        query = self.at_query
        executed = selected if query else 0
        reward = 0.0
        if query:
            left, right = pattern_indices(self._objects)
            correct = selected == int(left == right)
            reward = 1.0 if correct else -1.0
            self._decision = MatchPatternDecision(
                left, right, selected, correct, reward
            )
            self._seen[self.evaluator_task_index - SPLIT_STARTS[self.split]] = True
        self._step += 1
        self._current = self._observation()
        self._done = query
        packet = self._packet(
            PublicDecision(
                current=self._current,
                previous=previous,
                outcome=self._current,
                executed_action=executed,
                reward=reward,
                physical_done=query,
                outer_terminated=query,
            )
        )
        return packet, reward, query, False, {"query_ready": self.at_query}

    def _identity(self) -> tuple[object, ...]:
        return (self.corpus_sha256, HORIZON, "canonical-reveal-query-loss.v1")

    def _state(self) -> dict[str, object]:
        return {
            "step": self._step,
            "objects": self._objects,
            "decision": None if self._decision is None else asdict(self._decision),
            "exposure": self.exposure_bitmap(),
        }

    def _load_state(self, state: Mapping[str, object]) -> None:
        self._step = int(cast(int, state["step"]))
        self._objects = tuple(int(x) for x in cast(Sequence[int], state["objects"]))
        value = state["decision"]
        self._decision = (
            None
            if value is None
            else MatchPatternDecision(**cast(dict[str, Any], value))
        )
        packed = np.frombuffer(cast(bytes, state["exposure"]), dtype=np.uint8)
        seen = np.unpackbits(packed)[: len(self._seen)]
        if seen.shape != self._seen.shape or not 0 <= self._step <= HORIZON:
            raise ContractError("Invalid match-pattern restored state.")
        self._seen = seen.astype(np.bool_)
