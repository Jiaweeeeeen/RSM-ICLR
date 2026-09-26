"""M3 exit: CountRecall counter alignment, leakage, replay and the C1 lifecycle.

These are execution-integrity checks. Nothing here trains to convergence or
claims that the pinned stream is learnable; that is M5 work.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from amago.envs.amago_env import SequenceWrapper

from reasoned_icrl.environments.base import (
    DEVELOPMENT_TASKS,
    FINAL_TASKS,
    TRAINING_TASKS,
)
from reasoned_icrl.environments.count_recall import (
    COUNT_RECALL_ACTIONS,
    COUNT_RECALL_BLANK,
    COUNT_RECALL_CATEGORIES,
    COUNT_RECALL_HORIZONS,
    CountRecallEnv,
    PublicStreamCounter,
    count_recall_protocol,
    count_recall_variant,
    tail_query_generator,
)
from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import (
    ContractError,
    ResultValidationError,
    architecture_uses_history_packet,
)
from reasoned_icrl.experiments.evaluation import (
    RESULTS_FILE,
    CountRecallStreamResult,
    count_prior,
    count_recall_actions,
    count_recall_secondary,
    evaluation_environment,
    events,
    history_modes,
    prior_accuracy,
    random_reference_accuracy,
    run_record,
)
from reasoned_icrl.experiments.horizon import (
    check_stream_prefix,
    extended_horizon,
    horizon_kind,
    native_horizon,
    stream_window_accuracy,
    stream_window_index,
    stream_window_matrix,
)
from reasoned_icrl.experiments.records import validate_benchmark_results
from reasoned_icrl.experiments.summary_memory.configs import (
    load_retired_summary_memory_study,
)
from reasoned_icrl.runtime.environments import amago_environment
from reasoned_icrl.runtime.replay import create_replay_dataset
from reasoned_icrl.runtime.rollout import (
    evaluate,
    rollout,
)
from reasoned_icrl.runtime.training import (
    load_experiment,
    train_experiment,
)
from tests.experiments.fixtures import load_fixture_study

ROOT = Path(__file__).resolve().parents[2]
EASY_HORIZON = 51
EASY_ACTIONS = COUNT_RECALL_ACTIONS["easy"]
HARD_HORIZON = 207
HARD_ACTIONS = COUNT_RECALL_ACTIONS["hard"]
HARD_CATEGORIES = COUNT_RECALL_CATEGORIES["hard"]


def contract() -> Any:
    return load_fixture_study("dat_benchmarks").contract("count_recall")


def resolved(condition: str = "transition", **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "seed": 0,
        "repository": ROOT,
        "device": "cpu",
    }
    settings.update(overrides)
    return experiment_config(
        contract(),
        load_fixture_study("dat_benchmarks"),
        condition=condition,
        **settings,
    )


def stream_environment(**overrides: Any) -> CountRecallEnv:
    settings: dict[str, Any] = {
        "variant": "easy",
        "split": "development",
        "initial_seed": 0,
    }
    settings.update(overrides)
    return CountRecallEnv(**settings)


def run_stream(
    env: CountRecallEnv, stream: int, answers: Any
) -> tuple[list[tuple[int, int]], list[dict[str, np.ndarray]]]:
    """Drive one whole stream, returning decoded tokens and raw packets."""
    packet, info = env.reset(options={"task_index": stream})
    assert "counts" not in info
    tokens = [env.decode(packet)[:2]]
    packets = [{k: v.copy() for k, v in packet.items()}]
    done = False
    while not done:
        packet, _, terminated, truncated, info = env.step(answers())
        assert "counts" not in info
        tokens.append(env.decode(packet)[:2])
        packets.append({k: v.copy() for k, v in packet.items()})
        done = terminated or truncated
    return tokens, packets


# ----------------------------------------------------------------------
# M3.1 exact counter alignment
# ----------------------------------------------------------------------


def test_the_counter_scores_the_query_already_visible_before_the_answer() -> None:
    """`n_t(q_t)` counts dealt values through t, including the token's own."""
    counter = PublicStreamCounter(2)
    # (value, query) pairs with repeats, a zero count, and a final repeat.
    stream = [(0, 0), (1, 1), (1, 1), (0, 1), (0, 0), (1, 0)]
    assert [counter.observe(v, q) for v, q in stream] == [1, 1, 2, 2, 3, 3]
    assert counter.counts.tolist() == [3, 3] and counter.queries == 6


def test_a_first_query_for_an_unseen_category_scores_zero() -> None:
    counter = PublicStreamCounter(4)
    assert counter.observe(0, 3) == 0
    assert counter.observe(3, 3) == 1
    assert counter.observe(3, 0) == 1


def test_the_counter_rejects_categories_outside_the_declared_deck() -> None:
    counter = PublicStreamCounter(2)
    with pytest.raises(ContractError, match="category index is out of range"):
        counter.observe(2, 0)
    with pytest.raises(ContractError, match="at least two categories"):
        PublicStreamCounter(1)


@pytest.mark.parametrize("stream", [1_000_000, 1_000_007, 2_000_003])
def test_recorded_counts_equal_the_brute_force_public_definition(stream: int) -> None:
    """`n_t(q_t) = |{i <= t : x_i == q_t}|`, recomputed from the packets alone."""
    split = "development" if stream < 2_000_000 else "final"
    env = stream_environment(split=split)
    generator = np.random.default_rng(stream)
    try:
        tokens, _ = run_stream(
            env, stream, lambda: int(generator.integers(EASY_ACTIONS))
        )
        records = env.scored_queries
    finally:
        env.close()
    assert len(tokens) == EASY_HORIZON + 1
    assert len(records) == EASY_HORIZON
    values = [value for value, _ in tokens]
    for record in records:
        position = record.index - 1
        _, query = tokens[position]
        expected = sum(1 for value in values[: position + 1] if value == query)
        assert (record.query, record.true_count) == (query, expected)
    assert records[0].index == 1 and records[-1].index == EASY_HORIZON
    assert max(record.true_count for record in records) > 1
    assert min(record.true_count for record in records) >= 0


def test_the_native_return_is_exactly_two_times_accuracy_minus_one() -> None:
    env = stream_environment()
    try:
        run_stream(env, DEVELOPMENT_TASKS[3], lambda: 13)
        records = env.scored_queries
        accuracy = sum(record.correct for record in records) / len(records)
        assert env.stream_return == pytest.approx(2 * accuracy - 1, abs=1e-12)
        assert all(
            record.native_reward
            == pytest.approx((1 if record.correct else -1) / EASY_HORIZON)
            for record in records
        )
    finally:
        env.close()


def test_a_perfect_answer_policy_scores_every_query() -> None:
    """The counter is the exact answer, so following it returns +1."""
    env = stream_environment()
    try:
        packet, _ = env.reset(options={"task_index": DEVELOPMENT_TASKS[4]})
        counter = PublicStreamCounter(COUNT_RECALL_CATEGORIES["easy"])
        done = False
        while not done:
            value, query, _ = env.decode(packet)
            packet, reward, terminated, truncated, info = env.step(
                counter.observe(value, query)
            )
            assert reward == pytest.approx(1 / EASY_HORIZON) and info["correct"]
            done = terminated or truncated
        assert env.stream_return == pytest.approx(1.0)
        assert all(record.correct for record in env.scored_queries)
    finally:
        env.close()


def test_the_stream_terminates_exactly_at_its_deck_boundary() -> None:
    env = stream_environment()
    try:
        packet, _ = env.reset(options={"task_index": DEVELOPMENT_TASKS[5]})
        assert packet["event"].tolist() == [0, 1, 0]
        for step in range(1, EASY_HORIZON + 1):
            packet, _, terminated, truncated, info = env.step(0)
            assert info["query_index"] == step
            assert packet["event"].tolist() == [1, 0, 0]
            assert terminated == (step == EASY_HORIZON)
            assert not truncated
        with pytest.raises(ContractError, match="requires an outer reset"):
            env.step(0)
    finally:
        env.close()


# ----------------------------------------------------------------------
# M3.2 leakage and reproducibility
# ----------------------------------------------------------------------


def test_different_answer_policies_see_an_identical_stream() -> None:
    """Verified against the pin, not assumed from the RNG."""
    traces = []
    for actor_seed, answer in ((0, 0), (91, 26), (4096, 7)):
        env = stream_environment(initial_seed=actor_seed)
        try:
            _, packets = run_stream(env, DEVELOPMENT_TASKS[6], lambda a=answer: a)
            traces.append(
                (
                    np.stack([packet["current"] for packet in packets]),
                    [record.true_count for record in env.scored_queries],
                    [record.query for record in env.scored_queries],
                )
            )
        finally:
            env.close()
    for observations, counts, queries in traces[1:]:
        np.testing.assert_array_equal(traces[0][0], observations)
        assert traces[0][1] == counts and traces[0][2] == queries


def test_the_packet_carries_only_the_pinned_public_projection() -> None:
    env = stream_environment()
    try:
        packet, info = env.reset(options={"task_index": DEVELOPMENT_TASKS[7]})
        assert set(packet) == {"current", "previous", "outcome", "event", "valid"}
        assert set(env.observation_space.spaces) == set(packet)
        assert packet["current"].shape == (2 * COUNT_RECALL_CATEGORIES["easy"] + 1,)
        privileged = {"counts", "query_counts", "get_state", "state_space"}
        assert not privileged & set(info)
        native = env._native.unwrapped
        for _ in range(10):
            packet, _, _, _, info = env.step(0)
            assert not privileged & set(info)
            value, query, timer = env.decode(packet)
            # The public projection is the pinned wrapper's, nothing more.
            assert (value, query) == (int(native.value), int(native.query))
            # The timer decodes through a float32 packet, so compare it at
            # float32 resolution; the one-hot fields stay exact.
            assert timer == pytest.approx(env._step / 1000.0, abs=1e-6)
        # The privileged interfaces exist upstream and stay unread here.
        assert native.get_state() is not None and native.state_space is not None
    finally:
        env.close()


def test_stream_rosters_are_disjoint_and_match_the_contract() -> None:
    active = contract()
    assert active.roster("development") == tuple(DEVELOPMENT_TASKS)
    assert active.roster("final") == tuple(FINAL_TASKS)
    for left, right in (
        (TRAINING_TASKS, DEVELOPMENT_TASKS),
        (TRAINING_TASKS, FINAL_TASKS),
        (DEVELOPMENT_TASKS, FINAL_TASKS),
    ):
        assert not set(left) & set(right)
    assert active.protocol == count_recall_protocol("easy") == "count-recall-easy"
    assert count_recall_variant("count-recall-medium") == "medium"
    assert COUNT_RECALL_HORIZONS == {"easy": 51, "medium": 103, "hard": 207}
    assert COUNT_RECALL_ACTIONS == {"easy": 27, "medium": 27, "hard": 17}
    assert count_recall_variant("count-recall-hard") == "hard"
    assert count_recall_protocol("hard") == "count-recall-hard"
    with pytest.raises(ContractError, match="Unknown CountRecall protocol"):
        count_recall_variant("count-recall-expert")
    with pytest.raises(ContractError, match="Unknown CountRecall difficulty"):
        count_recall_protocol("expert")


def test_the_stream_length_cannot_be_shrunk_away_from_its_deck() -> None:
    with pytest.raises(ContractError, match="fixed by its native deck"):
        stream_environment(horizon=8)


def test_a_smoke_profile_keeps_the_native_stream_intact() -> None:
    config = resolved(smoke=True)
    assert config.environment.horizon == EASY_HORIZON
    assert config.environment.outer_length == EASY_HORIZON
    assert config.environment.has_fixed_native_horizon
    assert config.training.max_sequence_length == EASY_HORIZON


# ----------------------------------------------------------------------
# Replay alignment
# ----------------------------------------------------------------------


def collect_stream(directory: Path, *, answer: int = 3) -> tuple[Any, Any, Any, Any]:
    """Run one whole stream through the AMAGO boundary and save its replay."""
    env = stream_environment()
    wrapped = amago_environment(env, name="CountRecall-test", seed=0)
    dataset = create_replay_dataset(directory, capacity=8, full_tasks=True)
    sequence = SequenceWrapper(
        wrapped,
        save_trajs_to=dataset.save_new_trajs_to,
        save_every=None,
        save_trajs_as="npz-compressed",
    )
    env.set_task(DEVELOPMENT_TASKS[8])
    packet, _ = sequence.reset()
    observations = [{k: np.asarray(v)[0].copy() for k, v in packet.items()}]
    rewards: list[float] = []
    taken: list[int] = []
    done = False
    while not done:
        packet, reward, terminated, truncated, _ = sequence.step(
            np.array([answer], dtype=np.int64)
        )
        observations.append({k: np.asarray(v)[0].copy() for k, v in packet.items()})
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        taken.append(answer)
        done = bool(np.logical_or(terminated, truncated).reshape(-1)[0])
    sequence.save_finished_trajs()
    dataset.configure(
        items_per_epoch=1,
        max_seq_len=len(taken),
        padded_sampling="none",
        has_edit_rights=True,
    )
    dataset._refresh_files()
    return dataset, observations, rewards, taken


def test_rollout_and_replay_agree_on_every_causal_field(tmp_path: Path) -> None:
    dataset, observations, rewards, taken = collect_stream(tmp_path)
    data = dataset.sample_random_trajectory()
    length = len(data)
    assert length == len(taken) == EASY_HORIZON
    assert data.obs["current"].shape[0] == length + 1
    assert bool(data.dones[-1].item())
    assert data.time_idxs.reshape(-1).tolist() == list(range(length + 1))
    for step, expected in enumerate(observations):
        for name, value in expected.items():
            np.testing.assert_allclose(
                data.obs[name][step].numpy(), value, err_msg=f"{name}@{step}"
            )
    np.testing.assert_allclose(data.rews.reshape(-1).numpy(), rewards, rtol=1e-6)
    for step, action in enumerate(taken):
        # RL2 at t+1 is exactly [reward_t, one-hot answer_t]: the feedback that
        # scores the query shown at token t appears once, on token t+1.
        expected_rl2 = np.zeros(EASY_ACTIONS + 1, dtype=np.float32)
        expected_rl2[0] = rewards[step]
        expected_rl2[1 + action] = 1.0
        np.testing.assert_allclose(data.rl2s[step + 1].numpy(), expected_rl2, rtol=1e-6)
    np.testing.assert_allclose(
        data.rl2s[0].numpy(), np.zeros(EASY_ACTIONS + 1, dtype=np.float32)
    )


def test_every_stream_token_carries_its_evidence_without_an_allowance(
    tmp_path: Path,
) -> None:
    """CountRecall never produces a reset-only token, so the strict check holds."""
    dataset, observations, _, _ = collect_stream(tmp_path)
    assert observations[0]["event"].tolist() == [0, 1, 0]
    assert all(row["event"].tolist() == [1, 0, 0] for row in observations[1:])
    assert not dataset.reset_only_terminal
    assert dataset.sample_random_trajectory() is not None


# ----------------------------------------------------------------------
# M3.3 records, strata and reference controls
# ----------------------------------------------------------------------


def sample_results(
    streams: tuple[int, ...], answer: int = 5, *, variant: str = "easy"
) -> tuple[Any, ...]:
    results = []
    for stream in streams:
        split = "development" if stream < 2_000_000 else "final"
        env = stream_environment(split=split, variant=variant)
        try:
            run_stream(env, stream, lambda: answer)
            results.append(
                CountRecallStreamResult(
                    task_id=stream,
                    rollout_seed=0,
                    queries=env.scored_queries,
                    stream_return=env.stream_return,
                    decisions=COUNT_RECALL_HORIZONS[variant],
                )
            )
        finally:
            env.close()
    return tuple(results)


def test_events_score_every_query_and_reject_a_truncated_stream() -> None:
    active = contract()
    config = resolved()
    results = sample_results((DEVELOPMENT_TASKS[0], DEVELOPMENT_TASKS[1]))
    rows = events(
        active,
        config,
        results,
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    assert len(rows) == 2 * EASY_HORIZON
    assert all(event.kind == "query" for event in rows)
    assert all(event.step == event.event_index for event in rows)
    assert all(0 <= event.true_count < EASY_ACTIONS for event in rows)
    for event in rows:
        assert event.native_return == pytest.approx(
            (2 * event.numerator - 1) / EASY_HORIZON
        )
    short = CountRecallStreamResult(
        task_id=results[0].task_id,
        rollout_seed=0,
        queries=results[0].queries[:10],
        stream_return=0.0,
        decisions=10,
    )
    with pytest.raises(ResultValidationError, match="scored 10 queries"):
        events(
            active,
            config,
            [short],
            checkpoint="checkpoint.pt",
            split="development",
            history="retained",
        )


def test_reference_controls_are_fitted_on_development_and_reported_apart() -> None:
    config = resolved()
    development = sample_results(tuple(DEVELOPMENT_TASKS[:6]))
    final = sample_results(tuple(FINAL_TASKS[:4]))
    prior = count_prior(development, horizon=EASY_HORIZON)
    assert len(prior) == EASY_HORIZON
    assert all(0 <= value < EASY_ACTIONS for value in prior)
    fitted = prior_accuracy(prior, development)
    held_out = prior_accuracy(prior, final)
    # A position-only prior beats uniform guessing but is far from exact.
    assert random_reference_accuracy(EASY_ACTIONS) < fitted < 1.0
    assert 0.0 <= held_out < 1.0
    summary = count_recall_secondary(config.environment, final, prior=prior)
    assert summary["prior_reference_accuracy"] == pytest.approx(held_out)
    assert summary["random_reference_accuracy"] == pytest.approx(1 / 27)
    assert summary["streams"] == 4.0 and summary["queries"] == 4.0 * EASY_HORIZON
    assert summary["return_identity_gap"] == pytest.approx(0.0, abs=1e-9)
    assert summary["exact_accuracy"] == pytest.approx(
        float(np.mean([r.accuracy for r in final]))
    )
    for name in (
        "accuracy_positions_1_12",
        "accuracy_positions_39_51",
        "accuracy_counts_low",
        "accuracy_counts_high",
    ):
        assert name in summary
    assert json.dumps(summary, allow_nan=False)
    with pytest.raises(ResultValidationError, match="at least one stream"):
        count_recall_secondary(config.environment, [])


def test_the_same_query_needs_different_answers_after_different_histories() -> None:
    """The M5/C3 task diagnostic: position and query alone cannot decide."""
    results = sample_results(tuple(DEVELOPMENT_TASKS[:12]))
    by_key: dict[tuple[int, int], set[int]] = {}
    for result in results:
        for query in result.queries:
            by_key.setdefault((query.index, query.query), set()).add(query.true_count)
    ambiguous = [key for key, counts in by_key.items() if len(counts) > 1]
    assert ambiguous, "no position/query pair required two different answers"


# ----------------------------------------------------------------------
# M5.1 the Hard variant (the summary-memory study's protocol)
# ----------------------------------------------------------------------


def hard_contract() -> Any:
    return load_retired_summary_memory_study().contract("count_recall")


def test_the_hard_variant_is_four_decks_of_ranks_with_seventeen_answers() -> None:
    env = stream_environment(variant="hard")
    try:
        packet, info = env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        assert env.protocol == "count-recall-hard"
        assert (env.categories, env.horizon, env.actions) == (13, HARD_HORIZON, 17)
        assert env.action_space.n == HARD_ACTIONS == 17
        assert packet["current"].shape == (2 * HARD_CATEGORIES + 1,) == (27,)
        native = env.native.unwrapped
        assert native.num_distinct_cards == 13 and native.max_card_count == 16
        assert native.max_episode_length == HARD_HORIZON
        assert "counts" not in info
    finally:
        env.close()


@pytest.mark.parametrize("stream", [DEVELOPMENT_TASKS[1], FINAL_TASKS[1]])
def test_hard_counts_equal_the_brute_force_public_definition(stream: int) -> None:
    split = "development" if stream in DEVELOPMENT_TASKS else "final"
    env = stream_environment(variant="hard", split=split)
    generator = np.random.default_rng(stream)
    try:
        tokens, packets = run_stream(
            env, stream, lambda: int(generator.integers(HARD_ACTIONS))
        )
        records = env.scored_queries
    finally:
        env.close()
    assert len(tokens) == HARD_HORIZON + 1 and len(records) == HARD_HORIZON
    values = [value for value, _ in tokens]
    for record in records:
        position = record.index - 1
        _, query = tokens[position]
        expected = sum(1 for value in values[: position + 1] if value == query)
        assert (record.query, record.true_count) == (query, expected)
    # Four decks of ranks: every rank is dealt exactly sixteen times, so the
    # largest possible count is the top of the 17-answer space.
    assert [values.count(rank) for rank in range(HARD_CATEGORIES)] == [16] * 13
    assert 5 < max(record.true_count for record in records) <= 16
    assert all(packet["current"].shape == (27,) for packet in packets)


def test_hard_timing_scores_the_visible_query_from_seventeen_answers() -> None:
    """The action at t answers the query shown before it; 17 answers, 207 steps."""
    env = stream_environment(variant="hard")
    try:
        packet, _ = env.reset(options={"task_index": DEVELOPMENT_TASKS[2]})
        assert packet["event"].tolist() == [0, 1, 0]
        counter = PublicStreamCounter(HARD_CATEGORIES)
        for step in range(1, HARD_HORIZON + 1):
            value, query, timer = env.decode(packet)
            assert timer == pytest.approx((step - 1) / 1000.0, abs=1e-6)
            truth = counter.observe(value, query)
            # Every fourth answer is wrong on purpose, at the top of the answer
            # space, so the reward cross-check is exercised on both signs.
            wrong = step % 4 == 0 and truth != HARD_ACTIONS - 1
            answer = HARD_ACTIONS - 1 if wrong else truth
            packet, reward, terminated, truncated, info = env.step(answer)
            assert info["query_index"] == step and info["true_count"] == truth
            assert info["correct"] == (answer == truth)
            assert reward == pytest.approx((-1 if wrong else 1) / HARD_HORIZON)
            assert packet["event"].tolist() == [1, 0, 0]
            assert terminated == (step == HARD_HORIZON) and not truncated
        records = env.scored_queries
        accuracy = sum(r.correct for r in records) / len(records)
        assert env.stream_return == pytest.approx(2 * accuracy - 1, abs=1e-9)
        assert all(
            r.native_reward == pytest.approx((1 if r.correct else -1) / HARD_HORIZON)
            for r in records
        )
        with pytest.raises(ContractError, match="requires an outer reset"):
            env.step(0)
        env.reset(options={"task_index": DEVELOPMENT_TASKS[2]})
        with pytest.raises(ContractError, match="outside the action space"):
            env.step(HARD_ACTIONS)
    finally:
        env.close()


def test_hard_counter_and_reward_cross_check_on_full_streams() -> None:
    """Every decision of every stream compares the public counter to the reward."""
    results = sample_results(tuple(DEVELOPMENT_TASKS[3:6]), answer=16, variant="hard")
    for result in results:
        assert len(result.queries) == HARD_HORIZON
        assert result.stream_return == pytest.approx(2 * result.accuracy - 1)
        assert all(0 <= q.true_count <= 16 for q in result.queries)
    # A counter that drifts from the native stream is refused on the spot.
    env = stream_environment(variant="hard")
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[4]})
        env._pending_count = (env._pending_count + 1) % HARD_ACTIONS
        with pytest.raises(ContractError, match="disagrees with the native reward"):
            env.step(env._pending_count)
    finally:
        env.close()
    # A public count above the answer space cannot come from four decks: with
    # every rank already at the top of the space, the next query exceeds it.
    env = stream_environment(variant="hard")
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[5]})
        env._counter.counts[:] = HARD_ACTIONS
        with pytest.raises(ContractError, match="exceeds the native answer space"):
            env.step(0)
    finally:
        env.close()


def test_the_hard_contract_declares_the_native_deck_and_the_shared_recipe() -> None:
    active = hard_contract()
    easy = contract()
    assert active.protocol == "count-recall-hard" and active.status == "C0"
    assert (active.environment.size, active.environment.horizon) == (13, HARD_HORIZON)
    assert active.environment.has_fixed_native_horizon
    assert active.training.max_sequence_length == HARD_HORIZON
    assert active.training.trajectory_length == HARD_HORIZON + 1
    assert active.training.validation_timesteps == HARD_HORIZON
    assert active.training.exploration_rollout_horizon == HARD_HORIZON
    # The M5 training block otherwise, and the 64/256 evaluation rosters.
    for name in (
        "epochs",
        "timesteps_per_epoch",
        "batches_per_epoch",
        "batch_size",
        "learning_rate",
        "warmup_steps",
        "epsilon_anneal_steps",
        "reward_multiplier",
        "mixed_precision",
        "validation_interval",
        "checkpoint_interval",
    ):
        assert getattr(active.training, name) == getattr(easy.training, name), name
    assert active.evaluation == easy.evaluation
    assert active.roster("development") == tuple(DEVELOPMENT_TASKS)
    assert active.roster("final") == tuple(FINAL_TASKS)
    study = load_retired_summary_memory_study()
    smoke = experiment_config(
        active,
        study,
        condition="raw",
        seed=0,
        repository=ROOT,
        device="cpu",
        smoke=True,
    )
    assert smoke.environment.horizon == HARD_HORIZON  # the deck is never shrunk
    assert smoke.training.max_sequence_length == HARD_HORIZON
    reference = experiment_config(
        active, study, condition="feedforward", seed=0, repository=ROOT, device="cpu"
    )
    assert reference.training.retain_training_state
    with pytest.raises(ContractError, match="fixed by its native deck"):
        stream_environment(variant="hard", horizon=EASY_HORIZON)


def test_hard_references_records_and_strata_use_the_seventeen_answer_space() -> None:
    assert count_recall_actions("count-recall-hard") == 17
    assert count_recall_actions("count-recall-easy") == 27
    assert random_reference_accuracy(HARD_ACTIONS) == pytest.approx(1 / 17)
    with pytest.raises(TypeError):
        random_reference_accuracy()  # type: ignore[call-arg]  # no Easy default
    active = hard_contract()
    config = experiment_config(
        active,
        load_retired_summary_memory_study(),
        condition="raw",
        seed=0,
        repository=ROOT,
        device="cpu",
    )
    results = sample_results(tuple(DEVELOPMENT_TASKS[:2]), answer=3, variant="hard")
    summary = count_recall_secondary(config.environment, results)
    assert summary["random_reference_accuracy"] == pytest.approx(1 / 17)
    assert summary["queries"] == 2.0 * HARD_HORIZON
    assert (
        summary["queries_counts_low"]
        + summary["queries_counts_mid"]
        + summary["queries_counts_high"]
        == 2.0 * HARD_HORIZON
    )
    assert "accuracy_positions_1_51" in summary
    assert "accuracy_positions_156_207" in summary
    small = replace(
        active,
        evaluation=replace(
            active.evaluation,
            splits={
                name: replace(split, count=2)
                for name, split in active.evaluation.splits.items()
            },
        ),
    )
    rows = events(
        small,
        config,
        results,
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    assert len(rows) == 2 * HARD_HORIZON
    assert all(0 <= event.true_count <= 16 for event in rows)
    run = run_record(
        small,
        config,
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    validate_benchmark_results([small], [run], list(rows))
    # The record bound is the deck's: sixteen on Hard, twenty-six on Easy.
    with pytest.raises(ResultValidationError, match="valid true_count"):
        validate_benchmark_results(
            [small], [run], [replace(rows[0], true_count=17), *rows[1:]]
        )


# ----------------------------------------------------------------------
# C1 lifecycle
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("condition", ("transition", "transition_dat", "feedforward"))
def test_count_recall_cpu_train_export_evaluate_lifecycle(
    condition: str, tmp_path: Path
) -> None:
    active = contract()
    config = resolved(condition, output_root=tmp_path, smoke=True)
    assert config.environment.name == "count_recall"
    if condition == "feedforward":
        config = replace(
            config, training=replace(config.training, epochs=3, checkpoint_interval=1)
        )
    train_experiment(config)
    run = config.run_directory
    assert (run / "checkpoint.pt").is_file()
    telemetry = [
        json.loads(line)
        for line in (run / "training_metrics.jsonl").read_text().splitlines()
    ]
    assert any(
        row["panel"] == "train-update" and row["gradient_steps"] >= 1
        for row in telemetry
    )
    if condition == "feedforward":
        assert config.training.retain_training_state
        assert list((run / "ckpts/policy_weights").glob("policy_epoch_*.pt"))
    experiment = load_experiment(config, work_directory=run)
    try:
        if condition == "feedforward":
            from reasoned_icrl.runtime.rollout import (
                checkpoint_series,
            )

            epochs = sorted(
                int(p.stem.rsplit("_", 1)[1])
                for p in (run / "ckpts/policy_weights").glob("policy_epoch_*.pt")
            )
            scores = checkpoint_series(
                active, config, experiment, epochs=epochs, task_cap=1
            )
            assert len(scores) == len(epochs) and len(scores) >= 2
        run_record, rows, summary = evaluate(
            active,
            config,
            experiment,
            checkpoint="checkpoint.pt",
            split="development",
            task_cap=2,
        )
        assert run_record.status == "completed"
        assert len(rows) == 2 * EASY_HORIZON
        assert summary["streams"] == 2.0
        assert summary["return_identity_gap"] == pytest.approx(0.0, abs=1e-9)
        with pytest.raises(ResultValidationError, match="incomplete/unexpected"):
            validate_benchmark_results([active], [run_record], list(rows))

        environment = amago_environment(
            evaluation_environment(active, config, split="development", seed=0),
            name="check",
            seed=0,
        )
        try:
            forward, _ = rollout(
                experiment,
                environment,
                task_ids=[DEVELOPMENT_TASKS[0], DEVELOPMENT_TASKS[1]],
                rollout_seed=0,
            )
            reverse, _ = rollout(
                experiment,
                environment,
                task_ids=[DEVELOPMENT_TASKS[1], DEVELOPMENT_TASKS[0]],
                rollout_seed=0,
            )
            cleared, _ = rollout(
                experiment,
                environment,
                task_ids=[DEVELOPMENT_TASKS[0]],
                rollout_seed=0,
                history="current-token",
            )
        finally:
            environment.close()
        # An outer reset clears the cache, so roster order cannot change a stream.
        assert forward[0].queries == reverse[1].queries
        assert forward[1].queries == reverse[0].queries
        # The intervention leaves the scored stream itself untouched.
        assert [q.true_count for q in cleared[0].queries] == [
            q.true_count for q in forward[0].queries
        ]
        with pytest.raises(ContractError, match="history intervention"):
            rollout(
                experiment,
                environment,
                task_ids=[DEVELOPMENT_TASKS[0]],
                rollout_seed=0,
                history="attempt-cleared",
            )

        if architecture_uses_history_packet(config.model.architecture_id):
            _assert_dense_and_cached_agree(experiment, config)

        # The panel can be kept gzip-compacted in place:
        # the readers accept the twin and return the same text.
        from reasoned_icrl.experiments.evaluation import write_evaluation
        from reasoned_icrl.experiments.records import (
            compact_benchmark_results,
            read_results_text,
            results_file,
        )

        panel = write_evaluation(
            run,
            active,
            run_record,
            rows,
            summary,
            split="development",
            history="retained",
            task_cap=2,
            checkpoint_rule="selected",
        )
        plain = panel / RESULTS_FILE
        assert plain.is_file() and results_file(plain) == plain
        before = read_results_text(plain)
        assert '"partial_task_cap": 2' in before
        packed = compact_benchmark_results(plain)
        assert packed.name == "benchmark_results.json.gz" and not plain.is_file()
        assert results_file(plain) == packed
        assert compact_benchmark_results(plain) == packed  # idempotent
        assert read_results_text(plain) == before
        assert packed.stat().st_size < len(before) // 5
    finally:
        for name in ("train_envs", "val_envs"):
            closer = getattr(getattr(experiment, name, None), "close", None)
            if callable(closer):
                closer()


def test_history_modes_are_declared_per_benchmark() -> None:
    assert history_modes("count_recall") == (
        "retained",
        "current-token",
        "summary-cleared",
    )
    # The stream-cleared companion exists on a continued task only.
    from reasoned_icrl.experiments.evaluation import continued_history_modes
    from reasoned_icrl.experiments.horizon import continued_streams

    plain = contract()
    assert continued_history_modes(plain.environment) == history_modes("count_recall")
    longer, _ = continued_streams(plain, resolved(), 2)
    assert continued_history_modes(longer.environment) == (
        *history_modes("count_recall"),
        "attempt-cleared",
    )
    assert history_modes("dark_key_to_door") == (
        "retained",
        "attempt-cleared",
        "summary-cleared",
    )
    assert history_modes("darkroom") == (
        "retained",
        "attempt-cleared",
        "summary-cleared",
    )
    with pytest.raises(ContractError, match="No evaluator is implemented"):
        history_modes("multidomain_popgym")


def _assert_dense_and_cached_agree(experiment: Any, config: Any) -> None:
    """One stream, encoded once as a whole sequence and once step by step."""
    environment = evaluation_environment(
        contract(), config, split="development", seed=0
    )
    try:
        packet, _ = environment.reset(options={"task_index": DEVELOPMENT_TASKS[2]})
        packets = [packet]
        feedback = [np.zeros(EASY_ACTIONS + 1, dtype=np.float32)]
        done = False
        answer = 4
        while not done:
            packet, reward, terminated, truncated, _ = environment.step(answer)
            row = np.zeros(EASY_ACTIONS + 1, dtype=np.float32)
            row[0] = reward
            row[1 + answer] = 1.0
            packets.append(packet)
            feedback.append(row)
            done = terminated or truncated
    finally:
        environment.close()
    device = experiment.DEVICE
    observation = {
        name: torch.as_tensor(
            np.stack([step[name] for step in packets])[None], device=device
        )
        for name in packets[0]
    }
    rl2 = torch.as_tensor(np.stack(feedback)[None], device=device)
    times = torch.arange(len(packets), device=device).view(1, -1, 1)
    policy = experiment.policy
    carrier = policy.traj_encoder
    with torch.no_grad():
        tokens = policy.tstep_encoder(observation, rl2)
        dense, _ = carrier(tokens, times, None)
        hidden = carrier.init_hidden_state(1, device)
        cached = []
        for step in range(tokens.shape[1]):
            output, hidden = carrier(
                tokens[:, step : step + 1], times[:, step : step + 1], hidden
            )
            cached.append(output)
    torch.testing.assert_close(dense, torch.cat(cached, dim=1), rtol=2e-4, atol=2e-5)


# ----------------------------------------------------------------------
# C1: the query-only tail
# ----------------------------------------------------------------------


def _brute_count(tokens: list[tuple[int, int]], upto: int, query: int) -> int:
    """The count of ``query`` among the dealt values of records 1..upto."""
    return sum(1 for value, _ in tokens[:upto] if value == query)


def test_the_query_only_tail_continues_the_stream_with_blank_values() -> None:
    task = DEVELOPMENT_TASKS[3]
    plain = stream_environment()
    tail = stream_environment(horizon=2 * EASY_HORIZON + 1)  # 104 records
    assert tail.tail == EASY_HORIZON + 1 and tail.native_horizon == EASY_HORIZON
    try:
        base_tokens, _ = run_stream(plain, task, lambda: 3)
        tokens, packets = run_stream(tail, task, lambda: 3)
    finally:
        plain.close()
        tail.close()
    native_records = EASY_HORIZON + 1
    assert len(tokens) == 2 * native_records and len(base_tokens) == native_records
    # The native stream is reproduced record for record, then the tail deals
    # nothing: a blank value and a query on every record.
    assert tokens[:native_records] == base_tokens
    assert all(value == COUNT_RECALL_BLANK for value, _ in tokens[native_records:])
    assert all(0 <= query < tail.categories for _, query in tokens[native_records:])
    # AMAGO's timer continues past the deck, one per record.
    timers = [tail.decode(packet)[2] for packet in packets]
    # float32 packets: the timer is exact to the packet's precision only.
    assert timers == pytest.approx(
        [index / 1000 for index in range(len(packets))], abs=1e-6
    )
    records = tail.scored_queries
    assert len(records) == 2 * EASY_HORIZON + 1
    assert [r.index for r in records] == list(range(1, 2 * EASY_HORIZON + 2))
    for record in records:
        # Decision i answers the query visible in record i; a tail query counts
        # the queried category among the dealt values only, at the native scale.
        expected = _brute_count(tokens, record.index, record.query)
        assert record.true_count == expected
        assert record.native_reward == pytest.approx(
            (1.0 if record.answer == expected else -1.0) / EASY_HORIZON
        )
    assert tail.stream_return == pytest.approx(sum(r.native_reward for r in records))
    # The tail queries are the identity's own sequence: the same in a fresh
    # environment, different for another task.
    twin = stream_environment(horizon=2 * EASY_HORIZON + 1)
    other = stream_environment(horizon=2 * EASY_HORIZON + 1)
    try:
        twin_tokens, _ = run_stream(twin, task, lambda: 0)
        other_tokens, _ = run_stream(other, DEVELOPMENT_TASKS[4], lambda: 0)
    finally:
        twin.close()
        other.close()
    assert twin_tokens == tokens
    assert other_tokens[native_records:] != tokens[native_records:]
    generator = tail_query_generator(task)
    assert [query for _, query in tokens[native_records:]] == [
        generator.randrange(tail.categories) for _ in range(EASY_HORIZON + 1)
    ]


def test_a_blank_value_is_refused_inside_the_native_stream() -> None:
    env = stream_environment()
    try:
        packet, _ = env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        raw = (np.asarray(packet["current"]) + 1.0) / 2.0
        raw[: env.categories] = 0.0
        assert env.decode({"current": 2.0 * raw - 1.0})[0] == COUNT_RECALL_BLANK
        with pytest.raises(ContractError, match="blank CountRecall value"):
            env._accept(raw)
        bad = raw.copy()
        bad[env.categories : 2 * env.categories] = 0.0
        with pytest.raises(ContractError, match="not one-hot"):
            env.decode({"current": 2.0 * bad - 1.0})
    finally:
        env.close()
    with pytest.raises(ContractError, match="fixed by its native deck"):
        stream_environment(horizon=EASY_HORIZON - 1)


def test_tail_state_round_trips_and_keeps_the_query_sequence() -> None:
    task = DEVELOPMENT_TASKS[5]
    horizon = 2 * EASY_HORIZON + 1
    env = stream_environment(horizon=horizon)
    twin = stream_environment(horizon=horizon)
    try:
        packet, _ = env.reset(options={"task_index": task})
        for _ in range(EASY_HORIZON + 7):  # seven decisions into the tail
            packet, *_ = env.step(1)
        snapshot = env.state_dict()
        assert snapshot["tail_rng"] is not None
        twin.reset(options={"task_index": DEVELOPMENT_TASKS[6]})
        twin.load_state_dict(snapshot)
        done = False
        while not done:
            answer = 2
            packet, _, terminated, truncated, _ = env.step(answer)
            twin_packet, _, twin_terminated, twin_truncated, _ = twin.step(answer)
            assert np.array_equal(packet["current"], twin_packet["current"])
            assert (terminated, truncated) == (twin_terminated, twin_truncated)
            done = terminated or truncated
        assert env.scored_queries == twin.scored_queries
        assert len(env.scored_queries) == horizon
        lacking = dict(snapshot)
        lacking["tail_rng"] = None
        with pytest.raises(ContractError, match="lacks the tail generator"):
            stream_environment(horizon=horizon).load_state_dict(lacking)
    finally:
        env.close()
        twin.close()


def test_the_stream_extension_derives_the_records_horizon() -> None:
    from reasoned_icrl.experiments.environments import build_environment

    active = contract()
    config = resolved()
    assert horizon_kind(active) == "stream"
    assert native_horizon(active) == EASY_HORIZON + 1
    longer, extended = extended_horizon(active, config, 2 * (EASY_HORIZON + 1))
    assert longer.environment.horizon == 2 * EASY_HORIZON + 1
    assert longer.environment.outer_length == 2 * EASY_HORIZON + 1
    assert extended.training.max_sequence_length >= 2 * EASY_HORIZON + 1
    assert extended.training.trajectory_length >= 2 * EASY_HORIZON + 2
    assert (
        extended.model == config.model
        and extended.run_directory == config.run_directory
    )
    same, same_config = extended_horizon(active, config, EASY_HORIZON + 1)
    assert same.environment == active.environment and same_config == config
    with pytest.raises(ContractError, match="below the trained budget"):
        extended_horizon(active, config, EASY_HORIZON)
    env = build_environment(extended.as_runtime_mapping(), split="development", seed=0)
    assert isinstance(env, CountRecallEnv) and env.tail == EASY_HORIZON + 1
    env.close()


def test_extended_events_validate_and_the_prefix_and_windows_are_read() -> None:
    active = contract()
    config = resolved()
    records = 2 * (EASY_HORIZON + 1)
    longer, extended = extended_horizon(active, config, records)
    tasks = (DEVELOPMENT_TASKS[0], DEVELOPMENT_TASKS[1])
    base_rows = events(
        active,
        config,
        sample_results(tasks),
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    results = []
    for task in tasks:
        env = stream_environment(horizon=records - 1)
        try:
            tokens, _ = run_stream(env, task, lambda: 5)
            results.append(
                CountRecallStreamResult(
                    task_id=task,
                    rollout_seed=0,
                    queries=env.scored_queries,
                    stream_return=env.stream_return,
                    decisions=records - 1,
                    values=tuple(value for value, _ in tokens),
                )
            )
        finally:
            env.close()
    rows = events(
        longer,
        extended,
        results,
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    assert len(rows) == 2 * (records - 1)
    assert all(
        event.native_return == pytest.approx((2 * event.numerator - 1) / EASY_HORIZON)
        for event in rows
    )
    assert all(0 <= event.true_count <= (EASY_HORIZON + 1) // 2 for event in rows)
    # The tail's evidence lies entirely in the dealt records: no query of the
    # tail can disagree with its dealt count.
    assert check_stream_prefix(base_rows, rows, native=EASY_HORIZON + 1) == {
        "compared": 2 * EASY_HORIZON,
        "units": 2,
        "decision_flips": 0,
    }
    # A different answer to the same query is a bounded numerical flip;
    # a different query or count is a defect.
    flipped = [
        replace(event, numerator=1 - event.numerator)
        if (event.step, event.task_id) == (3, tasks[0])
        else event
        for event in rows
    ]
    assert (
        check_stream_prefix(base_rows, flipped, native=EASY_HORIZON + 1)[
            "decision_flips"
        ]
        == 1
    )
    twice = [
        replace(event, numerator=1 - event.numerator) if event.step == 3 else event
        for event in rows
    ]
    with pytest.raises(ContractError, match="2 of 102 decisions"):
        check_stream_prefix(base_rows, twice, native=EASY_HORIZON + 1)
    changed = [
        replace(event, true_count=event.true_count + 1) if event.step == 3 else event
        for event in rows
    ]
    with pytest.raises(ContractError, match="differs inside the native stream"):
        check_stream_prefix(base_rows, changed, native=EASY_HORIZON + 1)
    with pytest.raises(ContractError, match="not the trained panel's"):
        check_stream_prefix(
            base_rows, [e for e in rows if e.step != 2], native=EASY_HORIZON + 1
        )
    native = EASY_HORIZON + 1
    assert [
        stream_window_index(i, native=native)
        for i in (1, EASY_HORIZON, native, 2 * native - 1, 2 * native)
    ] == [0, 0, 1, 1, 2]
    matrix = stream_window_matrix(
        rows, units=[(t, 0) for t in tasks], native=native, horizon=records
    )
    assert matrix.shape == (2, 2)
    accuracy = stream_window_accuracy(rows, native=native, horizon=records)
    assert accuracy == pytest.approx(list(matrix.mean(axis=0)))
    by_window = [
        [e.numerator for e in rows if stream_window_index(e.step, native=native) == w]
        for w in (0, 1)
    ]
    assert accuracy == pytest.approx([sum(v) / len(v) for v in by_window])
    with pytest.raises(ContractError, match="fully scored"):
        stream_window_matrix(
            rows[:-1], units=[(t, 0) for t in tasks], native=native, horizon=records
        )


# ----------------------------------------------------------------------
# The continued-stream axis (C3)
# ----------------------------------------------------------------------


def test_the_continued_task_deals_a_fresh_pair_per_stream_and_repeats_the_first() -> (
    None
):
    from reasoned_icrl.environments.count_recall import continued_stream_seed

    task = DEVELOPMENT_TASKS[3]
    records = EASY_HORIZON + 1
    plain = stream_environment()
    continued = stream_environment(streams=3)
    assert continued.streams == 3 and continued.total_decisions == 3 * EASY_HORIZON
    try:
        base_tokens, base_packets = run_stream(plain, task, lambda: 3)
        tokens, packets = run_stream(continued, task, lambda: 3)
        counters = continued.collection_counters()
        queries = continued.scored_queries
    finally:
        plain.close()
        continued.close()
    # Three deck pairs of 52 records; the first pair is the trained stream,
    # packet for packet, except that its last record marks the pair's end.
    assert len(tokens) == 3 * records and tokens[:records] == base_tokens
    for index in range(records - 1):
        for key in base_packets[index]:
            assert np.array_equal(packets[index][key], base_packets[index][key]), key
    assert packets[records - 1]["event"].tolist() == [1.0, 0.0, 1.0]
    assert base_packets[records - 1]["event"].tolist() == [1.0, 0.0, 0.0]
    # The boundary record is a reset-only call: a new attempt, no executed
    # action or reward in RL2, the native timer restarted.
    for boundary in (records, 2 * records):
        assert packets[boundary]["event"].tolist() == [0.0, 1.0, 0.0]
        assert continued.decode(packets[boundary])[2] == pytest.approx(0.0, abs=1e-6)
    assert counters["charged_calls"] == 3 * EASY_HORIZON + 2
    assert counters["reset_only_steps"] == 2
    assert counters["physical_actions"] == 3 * EASY_HORIZON
    # Decisions are numbered over the whole task; every pair is scored against
    # its own dealt values only (the counts restart).
    assert [q.index for q in queries] == list(range(1, 3 * EASY_HORIZON + 1))
    for query in queries:
        stream = (query.index - 1) // EASY_HORIZON
        local = query.index - stream * EASY_HORIZON
        pair = tokens[stream * records : (stream + 1) * records]
        assert query.true_count == _brute_count(pair, local, query.query)
        assert query.native_reward == pytest.approx(
            (1.0 if query.answer == query.true_count else -1.0) / EASY_HORIZON
        )
    # The second pair is the deck order of its own seed, above every roster
    # range: a fresh environment restricted to that seed deals the same tokens.
    seed = continued_stream_seed(task, 2)
    assert seed >= 2**31 and continued_stream_seed(task, 1) == task
    assert continued_stream_seed(task, 3) not in (seed, task)
    fresh = stream_environment(source_indices=[seed])
    try:
        fresh_tokens, _ = run_stream(fresh, seed, lambda: 3)
    finally:
        fresh.close()
    assert tokens[records : 2 * records] == fresh_tokens
    assert tokens[records : 2 * records] != base_tokens
    with pytest.raises(ContractError, match="more streams or a query-only tail"):
        stream_environment(streams=2, horizon=2 * EASY_HORIZON + 1)
    with pytest.raises(ContractError, match="positive integer"):
        stream_environment(streams=0)


def test_continued_state_round_trips_across_a_stream_boundary() -> None:
    task = DEVELOPMENT_TASKS[5]
    env = stream_environment(streams=2)
    twin = stream_environment(streams=2)
    try:
        env.reset(options={"task_index": task})
        for _ in range(
            EASY_HORIZON
        ):  # the whole first pair: a reset-only call is pending
            env.step(1)
        snapshot = env.state_dict()
        assert snapshot["pending_reset"] is True and snapshot["stream"] == 0
        assert snapshot["source"] == task
        twin.reset(options={"task_index": DEVELOPMENT_TASKS[6]})
        twin.load_state_dict(snapshot)
        done = False
        while not done:
            packet, reward, terminated, truncated, _ = env.step(2)
            twin_packet, twin_reward, twin_terminated, twin_truncated, _ = twin.step(2)
            assert np.array_equal(packet["current"], twin_packet["current"])
            assert np.array_equal(packet["event"], twin_packet["event"])
            assert (reward, terminated, truncated) == (
                twin_reward,
                twin_terminated,
                twin_truncated,
            )
            done = terminated or truncated
        assert env.scored_queries == twin.scored_queries
        assert len(env.scored_queries) == 2 * EASY_HORIZON
        assert env.collection_counters()["reset_only_steps"] == 1
        assert twin.collection_counters()["reset_only_steps"] == 1
    finally:
        env.close()
        twin.close()
    # The stream-cleared intervention keys on the reset-only call between the
    # pairs (the ended pair's terminal record consumed, the new pair's first
    # record not yet read), and on nothing else.
    env = stream_environment(streams=2)
    try:
        _, info = env.reset(options={"task_index": task})
        assert info["attempt_done"] is False
        flags = []
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(1)
            flags.append(bool(info["attempt_done"]))
            done = terminated or truncated
        assert flags.index(True) == EASY_HORIZON and sum(flags) == 1
    finally:
        env.close()


def test_the_continued_task_derives_its_outer_length_and_its_events_are_read() -> None:
    from reasoned_icrl.experiments.environments import build_environment
    from reasoned_icrl.experiments.evaluation import evaluation_directory
    from reasoned_icrl.experiments.horizon import (
        continued_streams,
        streams_accuracy,
        streams_matrix,
    )

    active = contract()
    config = resolved()
    longer, extended = continued_streams(active, config, 2)
    outer = 2 * (EASY_HORIZON + 1) - 1  # two pairs and one reset-only call
    assert longer.environment.attempts == 2
    assert longer.environment.horizon == EASY_HORIZON
    assert longer.environment.outer_length == outer
    assert extended.training.max_sequence_length >= outer
    assert extended.training.trajectory_length >= outer + 1
    assert extended.model == config.model
    with pytest.raises(ContractError, match="at least two streams"):
        continued_streams(active, config, 1)
    tailed, tailed_config = extended_horizon(active, config, 2 * (EASY_HORIZON + 1))
    with pytest.raises(ContractError, match="starts from the trained stream"):
        continued_streams(tailed, tailed_config, 2)
    assert (
        evaluation_directory("development", "retained", "endpoint", streams=2)
        == "development-retained-endpoint-s2"
    )
    with pytest.raises(ContractError, match="not a horizon"):
        evaluation_directory(
            "development", "retained", "endpoint", horizon=104, streams=2
        )
    with pytest.raises(ContractError, match="at least two streams"):
        evaluation_directory("development", "retained", "endpoint", streams=1)
    env = build_environment(extended.as_runtime_mapping(), split="development", seed=0)
    assert isinstance(env, CountRecallEnv) and env.streams == 2 and env.tail == 0
    env.close()
    tasks = (DEVELOPMENT_TASKS[0], DEVELOPMENT_TASKS[1])
    base_rows = events(
        active,
        config,
        sample_results(tasks),
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    results = []
    for task in tasks:
        env = stream_environment(streams=2)
        try:
            tokens, _ = run_stream(env, task, lambda: 5)
            results.append(
                CountRecallStreamResult(
                    task_id=task,
                    rollout_seed=0,
                    queries=env.scored_queries,
                    stream_return=env.stream_return,
                    decisions=outer,
                    values=tuple(value for value, _ in tokens),
                    streams=2,
                )
            )
        finally:
            env.close()
    rows = events(
        longer,
        extended,
        results,
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    assert len(rows) == 4 * EASY_HORIZON
    assert all(event.outer_length == outer for event in rows)
    # The first pair reproduces the plain panel; the read is accuracy per pair.
    assert check_stream_prefix(base_rows, rows, native=EASY_HORIZON + 1) == {
        "compared": 2 * EASY_HORIZON,
        "units": 2,
        "decision_flips": 0,
    }
    matrix = streams_matrix(
        rows, units=[(t, 0) for t in tasks], decisions=EASY_HORIZON, streams=2
    )
    assert matrix.shape == (2, 2)
    assert streams_accuracy(rows, decisions=EASY_HORIZON, streams=2) == pytest.approx(
        list(matrix.mean(axis=0))
    )
    with pytest.raises(ContractError, match="outside the declared streams"):
        streams_matrix(
            rows, units=[(t, 0) for t in tasks], decisions=EASY_HORIZON, streams=1
        )
