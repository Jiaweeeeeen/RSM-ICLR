"""Frozen symbolic match-pattern corpus and qualification constants.

Only public symbols are returned to the environment. Pattern/label helpers and
group identities are protocol/scoring instruments, never policy features.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from reasoned_icrl.experiments.contracts import ContractError

PROTOCOL = "match-pattern-symbolic-v1"
GENERATOR_SEED = 20260916
VOCABULARY = 64
HORIZON = 7
QUERY_FIELD = 71
PATTERNS = ((0, 1, 2), (0, 0, 1), (0, 1, 0), (0, 1, 1))
PATTERN_NAMES = ("ABC", "AAB", "ABA", "ABB")
SPLIT_COUNTS = {
    "train": 983040,
    "development": 1536,
    "development-bindings": 1536,
    "final": 6144,
    "final-bindings": 6144,
}
SPLIT_STARTS = {name: i * 1000000 for i, name in enumerate(SPLIT_COUNTS)}
MANIFEST_PATH = Path("configs/manifests/match-pattern-symbolic-v1.json")
CORPUS_PATH = Path("outputs/match-pattern-8m/protocol/corpus.npz")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260916
QUALIFICATION_LABELS = (900, 950, 999)
LEARNING_ACCURACY = 0.90
HISTORY_MARGIN = 0.10
PRACTICAL_EFFECT = 0.05


def equality_partition(objects: Sequence[int]) -> tuple[int, ...]:
    """Canonical equality pattern, invariant to bijective symbol renaming."""
    identities: dict[int, int] = {}
    return tuple(identities.setdefault(int(x), len(identities)) for x in objects)


def pattern_indices(objects: Sequence[int]) -> tuple[int, int]:
    if len(objects) != 6:
        raise ContractError("A match-pattern example has exactly six objects.")
    try:
        return (
            PATTERNS.index(equality_partition(objects[:3])),
            PATTERNS.index(equality_partition(objects[3:])),
        )
    except ValueError as error:
        raise ContractError(
            "Example contains an undeclared equality pattern."
        ) from error


def group_key(objects: Sequence[int]) -> bytes:
    return bytes(sorted(set(int(x) for x in objects)))


def group_bucket(objects: Sequence[int]) -> int:
    digest = hashlib.sha256(b"match-pattern-group-v1\0" + group_key(objects)).digest()
    return int.from_bytes(digest, "big") % 10


def task_sources(split: str) -> range:
    if split not in SPLIT_COUNTS:
        raise ContractError(f"Unknown match-pattern split: {split!r}.")
    start = SPLIT_STARTS[split]
    return range(start, start + SPLIT_COUNTS[split])


def protocol_definition() -> dict[str, object]:
    return {
        "schema": PROTOCOL,
        "generator": "pcg64-stratified-rejection-v1",
        "seed": GENERATOR_SEED,
        "vocabulary": VOCABULARY,
        "patterns": PATTERN_NAMES,
        "counts": SPLIT_COUNTS,
        "groups": (
            "sha256(sorted-unique-identities); buckets train/iid=0..7, dev=8, final=9"
        ),
        "packet": "symbol64-slot6-phase3.v1",
        "actions": "reveal-canonical-zero-query-binary.v1",
        "loss": "native-amago-query-only-actor-critic.v1",
        "reward": "query:+1/-1;reveal:0",
        "calls": HORIZON,
        "learning_accuracy": LEARNING_ACCURACY,
        "history_margin": HISTORY_MARGIN,
        "practical_effect": PRACTICAL_EFFECT,
        "qualification_labels": QUALIFICATION_LABELS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class MatchPatternExample:
    """One immutable public-symbol example; labels remain evaluator-side."""

    objects: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.objects) != 6 or any(
            type(x) is not int or not 0 <= x < VOCABULARY for x in self.objects
        ):
            raise ContractError(
                "Match-pattern examples require six symbols in [0, 64)."
            )
        if set(self.objects[:3]) & set(self.objects[3:]):
            raise ContractError("Triplets must use disjoint primitive identities.")
        pattern_indices(self.objects)


@dataclass(frozen=True, slots=True)
class MatchPatternCorpus:
    arrays: Mapping[str, NDArray[np.uint8]]
    manifest: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return canonical_hash(self.manifest)

    def example(self, split: str, task_id: int) -> tuple[int, ...]:
        index = task_id - SPLIT_STARTS[split]
        if not 0 <= index < len(self.arrays[split]):
            raise ContractError("Match-pattern task is outside its frozen corpus.")
        return MatchPatternExample(
            tuple(int(x) for x in self.arrays[split][index])
        ).objects


def generate_corpus(counts: Mapping[str, int] | None = None) -> MatchPatternCorpus:
    """Build all splits jointly; reserve evaluation examples before training.

    Evaluation groups are unique inside a panel. Training may repeat a group,
    but never an exact example. The RNG stream for each split is independent.
    Smaller counts are only for engineering fixtures, with distinct hashes.
    """
    counts = dict(SPLIT_COUNTS if counts is None else counts)
    if set(counts) != set(SPLIT_COUNTS) or any(
        type(n) is not int or n < 24 or n % 24 for n in counts.values()
    ):
        raise ContractError("Every corpus split needs a positive multiple of 24.")
    strata = [
        (i, j) for i in range(4) for j in range(4) for _ in range(3 if i == j else 1)
    ]
    seen: set[bytes] = set()
    arrays: dict[str, NDArray[np.uint8]] = {}
    for split in (*list(SPLIT_COUNTS)[1:], "train"):
        stream = list(SPLIT_COUNTS).index(split)
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([GENERATOR_SEED, stream]))
        )
        result = np.empty((counts[split], 6), dtype=np.uint8)
        groups: set[bytes] = set()
        allowed = (
            {8}
            if split == "development-bindings"
            else {9}
            if split == "final-bindings"
            else set(range(8))
        )
        for index in range(counts[split]):
            left, right = strata[index % 24]
            p, q = PATTERNS[left], PATTERNS[right]
            nleft, nright = max(p) + 1, max(q) + 1
            while True:
                symbols = rng.choice(VOCABULARY, size=nleft + nright, replace=False)
                objects = tuple(int(symbols[x]) for x in p) + tuple(
                    int(symbols[nleft + x]) for x in q
                )
                key = bytes(objects)
                group = group_key(objects)
                if key in seen or group_bucket(objects) not in allowed:
                    continue
                if split != "train" and group in groups:
                    continue
                break
            result[index] = objects
            seen.add(key)
            if split != "train":
                groups.add(group)
        result.flags.writeable = False
        arrays[split] = result
    definition = protocol_definition() | {"counts": counts}
    manifest = definition | {
        "array_sha256": {
            name: hashlib.sha256(arrays[name].tobytes()).hexdigest()
            for name in SPLIT_COUNTS
        },
        "training_symbol_coverage": sorted(int(x) for x in np.unique(arrays["train"])),
    }
    return MatchPatternCorpus(arrays, manifest)


def write_corpus(corpus: MatchPatternCorpus, archive: Path, manifest: Path) -> None:
    """Write the materialized corpus and a portable canonical manifest."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **cast(dict[str, Any], dict(corpus.arrays)))
    manifest.write_text(json.dumps(corpus.manifest, indent=2, sort_keys=True) + "\n")


@lru_cache(maxsize=2)
def load_corpus(
    archive: Path = CORPUS_PATH, manifest: Path = MANIFEST_PATH
) -> MatchPatternCorpus:
    if not archive.is_file() or not manifest.is_file():
        raise ContractError(
            "Build the frozen corpus with "
            "scripts/build_match_pattern_manifest.py before execution."
        )
    raw = json.loads(manifest.read_text())
    definition = json.loads(json.dumps(protocol_definition()))
    if any(raw.get(k) != v for k, v in definition.items()):
        raise ContractError("Match-pattern manifest differs from the frozen protocol.")
    with np.load(archive, allow_pickle=False) as data:
        arrays = {name: data[name] for name in SPLIT_COUNTS}
    for name, array in arrays.items():
        if array.shape != (SPLIT_COUNTS[name], 6) or array.dtype != np.uint8:
            raise ContractError("Match-pattern corpus shape/dtype changed.")
        if hashlib.sha256(array.tobytes()).hexdigest() != raw["array_sha256"][name]:
            raise ContractError("Match-pattern corpus checksum mismatch.")
        array.flags.writeable = False
    return MatchPatternCorpus(arrays, raw)


@lru_cache(maxsize=1)
def manifest_hash() -> str:
    return canonical_hash(json.loads(MANIFEST_PATH.read_text()))
