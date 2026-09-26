"""Frozen public count/time controls for the official Medium study.

Fit modal answers on streams 0..1023 from the training manifest, with ties to
smaller counts. The conditional control additionally sees current value/query
equality. Neither control reads native hidden state or development labels
while fitting. These are evaluation instruments, never neural training targets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reasoned_icrl.environments.base import TRAINING_TASKS
from reasoned_icrl.environments.count_recall import CountRecallEnv, PublicStreamCounter
from reasoned_icrl.experiments.contracts import ContractError

PRIOR_SCHEMA = "count-recall-medium-training-priors.v1"
PRIOR_STREAMS = tuple(range(1024))


@dataclass(frozen=True, slots=True)
class CountTimePrior:
    training_stream_ids: tuple[int, ...]
    time_answers: tuple[int, ...]
    equality_answers: tuple[tuple[int, int], ...]
    public_stream_sha256: str
    charged_calls: int

    def predict(self, index: int, value: int, query: int, *, equality: bool) -> int:
        if not 1 <= index <= 103 or not 0 <= value < 4 or not 0 <= query < 4:
            raise ContractError("Medium prior input is outside the public contract.")
        if equality:
            return self.equality_answers[index - 1][int(value == query)]
        return self.time_answers[index - 1]

    def as_dict(self) -> dict[str, Any]:
        return {"schema": PRIOR_SCHEMA, **asdict(self)}


def fit_count_time_prior(stream_ids: tuple[int, ...] = PRIOR_STREAMS) -> CountTimePrior:
    if (
        not stream_ids
        or len(set(stream_ids)) != len(stream_ids)
        or any(stream not in TRAINING_TASKS for stream in stream_ids)
    ):
        raise ContractError("Count/time priors require distinct training-only streams.")
    frequencies = np.zeros((103, 2, 27), dtype=np.int64)
    digest = hashlib.sha256()
    env = CountRecallEnv(variant="medium", split="train")
    try:
        for stream in stream_ids:
            packet, _ = env.reset(options={"task_index": stream})
            counter = PublicStreamCounter(4)
            for index in range(103):
                value, query, _ = env.decode(packet)
                truth = counter.observe(value, query)
                frequencies[index, int(value == query), truth] += 1
                digest.update(bytes((value, query, truth)))
                packet, _, term, trunc, _ = env.step(0)
                if (term, trunc) != (index == 102, False):
                    raise ContractError(
                        "Medium prior stream ended off its native boundary."
                    )
    finally:
        env.close()
    modal = frequencies.sum(axis=1).argmax(axis=-1)
    conditional = frequencies.argmax(axis=-1)
    # A missing equality stratum falls back to the same training-only time prior.
    for index, equality in np.argwhere(frequencies.sum(axis=-1) == 0):
        conditional[index, equality] = modal[index]
    return CountTimePrior(
        stream_ids,
        tuple(int(v) for v in modal),
        tuple((int(row[0]), int(row[1])) for row in conditional),
        digest.hexdigest(),
        len(stream_ids) * 103,
    )


def frozen_count_time_prior(root: Path) -> CountTimePrior:
    """Load immutable fitted controls, or fit once before reading any fit scores."""
    path = root / "references/count-recall-medium-training-priors.json"
    if path.is_file():
        raw = json.loads(path.read_text())
        if raw.pop("schema", None) != PRIOR_SCHEMA:
            raise ContractError("Incompatible Medium reference identity.")
        prior = CountTimePrior(
            tuple(raw["training_stream_ids"]),
            tuple(raw["time_answers"]),
            tuple(tuple(row) for row in raw["equality_answers"]),
            raw["public_stream_sha256"],
            raw["charged_calls"],
        )
        if (
            prior.training_stream_ids != PRIOR_STREAMS
            or len(prior.time_answers) != 103
            or len(prior.equality_answers) != 103
        ):
            raise ContractError(
                "Frozen Medium prior has an incompatible training manifest."
            )
        return prior
    prior = fit_count_time_prior()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation refuses a concurrent replacement of a frozen instrument.
    with path.open("x") as output:
        json.dump(prior.as_dict(), output, indent=2, sort_keys=True)
        output.write("\n")
    return prior


def score_count_time_prior(
    prior: CountTimePrior, stream_ids: tuple[int, ...], *, split: str
) -> tuple[dict[int, float], dict[int, float]]:
    if set(stream_ids) & set(prior.training_stream_ids):
        raise ContractError("Reference scoring overlaps prior fitting streams.")
    scores: tuple[dict[int, float], dict[int, float]] = ({}, {})
    env = CountRecallEnv(variant="medium", split=split)
    try:
        for stream in stream_ids:
            packet, _ = env.reset(options={"task_index": stream})
            counter = PublicStreamCounter(4)
            correct = [0, 0]
            for index in range(1, 104):
                value, query, _ = env.decode(packet)
                truth = counter.observe(value, query)
                for mode in range(2):
                    answer = prior.predict(index, value, query, equality=bool(mode))
                    correct[mode] += int(answer == truth)
                packet, _, _, _, _ = env.step(0)
            for mode in range(2):
                scores[mode][stream] = correct[mode] / 103
    finally:
        env.close()
    return scores
