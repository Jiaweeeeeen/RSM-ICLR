"""Golden public lifecycle and shortcut tests for the symbolic RL diagnostic."""

from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np
import pytest

from reasoned_icrl.environments.match_pattern import MatchPatternEnv
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.match_pattern import (
    PATTERNS,
    SPLIT_COUNTS,
    MatchPatternCorpus,
    equality_partition,
    generate_corpus,
    group_bucket,
    group_key,
    pattern_indices,
)


@pytest.fixture(scope="module")
def corpus() -> MatchPatternCorpus:
    return generate_corpus({name: 48 for name in SPLIT_COUNTS})


def make_env(corpus: MatchPatternCorpus, index: int = 0) -> MatchPatternEnv:
    return MatchPatternEnv(
        corpus=corpus, source_indices=range(48), fixed_task_index=index
    )


def test_exhaustive_small_instance_scoring() -> None:
    strata: set[tuple[int, int]] = set()
    for objects in product(range(6), repeat=6):
        if set(objects[:3]) & set(objects[3:]):
            continue
        left, right = equality_partition(objects[:3]), equality_partition(objects[3:])
        if left not in PATTERNS or right not in PATTERNS:
            continue
        p, q = pattern_indices(objects)
        edges_left = tuple(
            objects[i] == objects[j] for i, j in ((0, 1), (0, 2), (1, 2))
        )
        edges_right = tuple(
            objects[i + 3] == objects[j + 3] for i, j in ((0, 1), (0, 2), (1, 2))
        )
        assert (p == q) == (edges_left == edges_right)
        strata.add((p, q))
    assert strata == set(product(range(4), repeat=2))


def test_frozen_balance_exclusions_and_group_invariance(
    corpus: MatchPatternCorpus,
) -> None:
    seen: set[bytes] = set()
    for split, array in corpus.arrays.items():
        pairs = Counter(pattern_indices(row) for row in array)
        assert pairs == Counter(
            {(i, j): 6 if i == j else 2 for i, j in product(range(4), repeat=2)}
        )
        assert len({bytes(row) for row in array}) == len(array)
        assert not seen.intersection(bytes(row) for row in array)
        seen.update(bytes(row) for row in array)
        for row in array:
            assert not set(row[:3]) & set(row[3:])
            assert group_key(row) == group_key(row[::-1])
            bucket = group_bucket(row)
            assert (
                bucket == (8 if split == "development-bindings" else 9)
                if split.endswith("bindings")
                else bucket < 8
            )
        if split != "train":
            assert len({group_key(row) for row in array}) == len(array)
    assert generate_corpus({name: 48 for name in SPLIT_COUNTS}).sha256 == corpus.sha256


@pytest.mark.parametrize("answer", (0, 1))
def test_golden_packet_reward_boundary(corpus: MatchPatternCorpus, answer: int) -> None:
    env = make_env(corpus)
    packet, _ = env.reset()
    objects = corpus.example("train", 0)
    assert set(packet) == {"current", "previous", "outcome", "event", "valid"}
    np.testing.assert_array_equal(packet["event"], [0, 1, 0])
    for step in range(6):
        assert np.flatnonzero(packet["current"][:64] == 1).tolist() == [objects[step]]
        assert np.flatnonzero(packet["current"][64:70] == 1).tolist() == [step]
        packet, reward, done, truncated, _ = env.step(1)
        assert (reward, done, truncated) == (0, False, False)
        np.testing.assert_array_equal(packet["event"], [1, 0, 0])
    assert packet["current"][71] == 1
    assert np.all(packet["current"][:70] == -1)
    packet, reward, done, truncated, _ = env.step(answer)
    assert (reward, done, truncated) == (1 if answer else -1, True, False)
    assert packet["current"][72] == 1
    assert env.scored_decision.correct == bool(answer)
    assert env.collection_counters() == dict(
        charged_calls=7,
        physical_actions=7,
        reset_only_steps=0,
        tasks_started=1,
        tasks_completed=1,
    )
    with pytest.raises(ContractError, match="reset"):
        env.step(0)


def test_restore_every_phase_and_no_reveal_action_channel(
    corpus: MatchPatternCorpus,
) -> None:
    for cut in range(7):
        a, b = make_env(corpus), make_env(corpus)
        a.reset()
        b.reset()
        for _ in range(cut):
            a.step(1)
        b.load_state_dict(a.state_dict())
        for _ in range(cut, 7):
            left, right = a.step(0), b.step(0)
            assert left[1:] == right[1:]
            for key in left[0]:
                np.testing.assert_array_equal(left[0][key], right[0][key])
        assert a.collection_counters() == b.collection_counters()
        assert a.exposure_bitmap() == b.exposure_bitmap()


def test_amago_records_canonical_actions(corpus: MatchPatternCorpus) -> None:
    from reasoned_icrl.runtime.environments import amago_environment

    env = amago_environment(make_env(corpus), name="MatchPattern-test")
    env.reset()
    for _ in range(6):
        timestep, reward, terminal, _, _ = env.step(np.array([1], dtype=np.int64))
        np.testing.assert_array_equal(timestep.prev_action, [[1, 0]])
        assert reward.item() == 0 and not terminal.item()
    timestep, reward, terminal, _, _ = env.step(np.array([1], dtype=np.int64))
    np.testing.assert_array_equal(timestep.prev_action, [[0, 1]])
    assert reward.item() == 1 and terminal.item()


def test_actual_query_inputs_alias_opposite_answers(corpus: MatchPatternCorpus) -> None:
    from reasoned_icrl.runtime.environments import amago_environment

    rows = []
    for index in range(48):
        env = amago_environment(make_env(corpus, index), name="alias")
        env.reset()
        for _ in range(6):
            timestep, _, _, _, _ = env.step(np.array([index % 2]))
        p, q = pattern_indices(corpus.example("train", index))
        rows.append((timestep, p == q))
    first = rows[0][0]
    assert len({label for _, label in rows}) == 2
    for row, _ in rows:
        np.testing.assert_array_equal(first.obs["current"], row.obs["current"])
        np.testing.assert_array_equal(first.prev_action, row.prev_action)
        np.testing.assert_array_equal(first.reward, row.reward)
        np.testing.assert_array_equal(first.time_idx, row.time_idx)
