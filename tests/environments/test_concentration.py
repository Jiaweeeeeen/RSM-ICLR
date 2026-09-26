"""M6 exit: Concentration board alignment, leakage, the retrieval ledger,
replay-state round trips and the registry, on all three decks.

Execution-integrity checks on the native rank deck (decision 14), on the
reduced-rank deck that fallback F7 made the study's contract and on the colour deck that
decision 19 made the study's
contract after both rank decks failed their gate. Nothing here trains to
convergence or claims a deck is learnable; that is M6.5, M6.6 and M6.10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reasoned_icrl.environments.base import (
    DEVELOPMENT_TASKS,
    FINAL_OOD_TASKS,
    FINAL_TASKS,
    benchmark_task_sources,
)
from reasoned_icrl.environments.concentration import (
    CONCENTRATION_CARDS,
    CONCENTRATION_COPIES,
    CONCENTRATION_HORIZONS,
    CONCENTRATION_OOD_COPIES,
    CONCENTRATION_PROTOCOLS,
    CONCENTRATION_RANKS,
    CONCENTRATION_SUITS,
    OOD_SPLIT,
    ConcentrationEnv,
    ConcentrationFlip,
    PublicBoard,
    concentration_protocol,
    concentration_variant,
    deck_copies,
    flip_budget,
    public_fields,
)
from reasoned_icrl.experiments.benchmarks import experiment_config, load_contract
from reasoned_icrl.experiments.contracts import ContractError, ResultValidationError
from reasoned_icrl.experiments.evaluation import (
    ConcentrationResult,
    concentration_secondary,
    events,
    history_modes,
    random_policy_pair_fraction,
)
from reasoned_icrl.experiments.records import (
    BenchmarkRun,
    validate_benchmark_results,
)
from reasoned_icrl.experiments.summary_memory.configs import (
    load_retired_summary_memory_study,
)
from reasoned_icrl.runtime.diagnostics import (
    concentration_diagnostic,
)

ROOT = Path(__file__).resolve().parents[2]
CARDS, RANKS, HORIZON = (
    CONCENTRATION_CARDS["hard"],
    CONCENTRATION_RANKS["hard"],
    CONCENTRATION_HORIZONS["hard"],
)
PAIRS = CARDS // 2
"""The native rank deck: the board-mechanics tests below run on it explicitly."""
REDUCED_VARIANT = "rank8"
REDUCED_CARDS, REDUCED_RANKS, REDUCED_HORIZON = (
    CONCENTRATION_CARDS[REDUCED_VARIANT],
    CONCENTRATION_RANKS[REDUCED_VARIANT],
    CONCENTRATION_HORIZONS[REDUCED_VARIANT],
)
REDUCED_PAIRS = REDUCED_CARDS // 2
"""The reduced-rank deck of fallback F7 (decision 16), the study's contract
until decision 18; kept for its record and its OOD split (decision 17)."""
STUDY_VARIANT = "easy"
STUDY_CARDS, STUDY_RANKS, STUDY_HORIZON = (
    CONCENTRATION_CARDS[STUDY_VARIANT],
    CONCENTRATION_RANKS[STUDY_VARIANT],
    CONCENTRATION_HORIZONS[STUDY_VARIANT],
)
STUDY_PAIRS = STUDY_CARDS // 2
"""The study's deck since decision 19 (the colour deck): what the contract, the
events, the secondary summaries and the references are checked against."""
RANK8_CONTRACT = Path(__file__).resolve().parents[2] / (
    "configs/environments/concentration_rank8.yaml"
)


def contract() -> Any:
    return load_retired_summary_memory_study().contract("concentration")


def resolved(condition: str = "raw", *, active: Any = None, **overrides: object) -> Any:
    study = load_retired_summary_memory_study()
    return experiment_config(
        contract() if active is None else active,
        study,
        condition=condition,
        seed=0,
        repository=ROOT,
        device="cpu",
        **overrides,  # type: ignore[arg-type]
    )


def board_environment(**overrides: object) -> ConcentrationEnv:
    settings: dict[str, Any] = {"split": "development", "initial_seed": 0}
    settings.update(overrides)
    return ConcentrationEnv(**settings)


def known_pairs(env: ConcentrationEnv) -> list[tuple[int, int]]:
    """Every pair of known hidden positions sharing a rank, from the public board."""
    by_rank: dict[int, list[int]] = {}
    for position, rank in env._board.known_hidden().items():
        by_rank.setdefault(rank, []).append(position)
    return [(ps[0], ps[1]) for ps in by_rank.values() if len(ps) >= 2]


def play(env: ConcentrationEnv, task: int, actions: list[int]) -> float:
    env.reset(options={"task_index": task})
    total = 0.0
    for action in actions:
        _, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return total


# --------------------------------------------------------------------------
# Identity, projection, leakage
# --------------------------------------------------------------------------


def test_the_native_deck_is_the_declared_protocol() -> None:
    env = board_environment()
    try:
        assert env.protocol == "concentration-hard" == concentration_protocol("hard")
        assert concentration_variant("concentration-hard") == "hard"
        assert (env.cards, env.ranks, env.pairs, env.horizon) == (52, 13, 26, 104)
        assert env.action_space.n == 52
        assert len(env.fields) == 52 * 14 + 1 == len(public_fields(CARDS, RANKS))
        assert env.fields[0].name == "position0_is_0"
        assert env.fields[13].name == "position0_is_13"  # the face-down sentinel
        assert env.fields[-1].name == "timer"
    finally:
        env.close()
    with pytest.raises(ContractError, match="Unknown Concentration deck"):
        concentration_protocol("medium")
    with pytest.raises(ContractError, match="fixed by its deck"):
        ConcentrationEnv(horizon=52, split="development")


def test_the_reduced_deck_is_the_declared_protocol() -> None:
    env = board_environment(variant=REDUCED_VARIANT)
    try:
        assert env.protocol == "concentration-rank8" == concentration_protocol("rank8")
        assert concentration_variant("concentration-rank8") == "rank8"
        assert (env.cards, env.ranks, env.pairs, env.horizon) == (32, 8, 16, 64)
        assert env.action_space.n == 32
        assert len(env.fields) == 32 * 9 + 1 == len(public_fields(32, 8))
        assert env.fields[8].name == "position0_is_8"  # the face-down sentinel
        assert env.fields[-1].name == "timer"
    finally:
        env.close()
    for variant, cards in CONCENTRATION_CARDS.items():
        assert cards == CONCENTRATION_RANKS[variant] * CONCENTRATION_COPIES[variant]
        assert (CONCENTRATION_COPIES[variant] == CONCENTRATION_SUITS) == (
            variant != "easy"
        )
        assert CONCENTRATION_HORIZONS[variant] == flip_budget(cards)
    assert (flip_budget(52), flip_budget(32), flip_budget(24)) == (104, 64, 48)
    with pytest.raises(ContractError, match="even number"):
        flip_budget(31)
    with pytest.raises(ContractError, match="Unknown Concentration deck"):
        concentration_protocol("rank6")
    with pytest.raises(ContractError, match="fixed by its deck"):
        ConcentrationEnv(variant="rank8", horizon=48, split="development")


def test_the_reduced_deck_plays_the_native_game_at_its_own_scale() -> None:
    first, second = (
        board_environment(variant=REDUCED_VARIANT),
        board_environment(variant=REDUCED_VARIANT, initial_seed=3),
    )
    try:
        packet, _ = first.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        board, timer = first.decode(packet)
        assert np.all(board == REDUCED_RANKS) and timer == 0.0
        assert packet["current"].shape == (REDUCED_CARDS * (REDUCED_RANKS + 1) + 1,)
        # The same task deals the same board; every rank appears once per suit.
        reveals = []
        for env in (first, second):
            env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
            for position in range(REDUCED_CARDS):
                env.step(position)
            reveals.append(list(env._board.revealed))
        assert reveals[0] == reveals[1] and None not in reveals[0]
        assert all(reveals[0].count(rank) == 4 for rank in range(REDUCED_RANKS))
        second.reset(options={"task_index": DEVELOPMENT_TASKS[1]})
        for position in range(REDUCED_CARDS):
            second.step(position)
        assert list(second._board.revealed) != reveals[0]
        # Rewards at the deck's scale: the sweep paid a mismatch (2/64) per
        # pair of flips except the lucky adjacent matches (1/16 each); a
        # match pays 1/16, an already-matched card 1/64.
        lucky = sum(flip.matched for flip in first.flips)
        assert first.episode_return == pytest.approx(
            lucky / REDUCED_PAIRS - (REDUCED_PAIRS - lucky) * 2 / REDUCED_HORIZON
        )
        pair = known_pairs(first)[0]
        _, r1, *_ = first.step(pair[0])
        _, r2, *_ = first.step(pair[1])
        assert r1 == 0.0 and r2 == pytest.approx(1 / REDUCED_PAIRS)
        _, r3, *_ = first.step(pair[0])
        assert r3 == pytest.approx(-1 / REDUCED_HORIZON)
        # A perfect second pass completes the board inside the flip budget.
        done = False
        while not done:
            a, b = known_pairs(first)[0]
            first.step(a)
            _, _, terminated, truncated, _ = first.step(b)
            done = terminated or truncated
        assert first.board_complete and first.matched_pairs == REDUCED_PAIRS
        assert terminated and len(first.flips) <= REDUCED_HORIZON
        assert first.episode_return == pytest.approx(
            sum(f.native_reward for f in first.flips)
        )
    finally:
        first.close()
        second.close()
    env = board_environment(variant=REDUCED_VARIANT)
    try:
        play(env, DEVELOPMENT_TASKS[5], list(range(20)))
        state = env.state_dict()
        twin = board_environment(variant=REDUCED_VARIANT)
        try:
            twin.load_state_dict(state)
            assert twin.state_dict() == state and twin.flips == env.flips
            actions = list(range(20, REDUCED_CARDS))
            assert [env.step(a)[1] for a in actions] == [
                twin.step(a)[1] for a in actions
            ]
        finally:
            twin.close()
    finally:
        env.close()


def test_the_ood_split_deals_an_uneven_deck_over_the_same_channels() -> None:
    """Decision 17: on ``final-ood`` four ranks hold three pairs and four hold
    one, the assignment drawn per board; every channel, the rules and the
    flip budget are the ordinary deck's."""
    assert benchmark_task_sources(OOD_SPLIT) == FINAL_OOD_TASKS
    assert not set(FINAL_OOD_TASKS) & (set(FINAL_TASKS) | set(DEVELOPMENT_TASKS))
    assert deck_copies(REDUCED_VARIANT, "final") == (4,) * REDUCED_RANKS
    assert deck_copies(REDUCED_VARIANT, OOD_SPLIT) == CONCENTRATION_OOD_COPIES["rank8"]
    with pytest.raises(ContractError, match="declares no out-of-distribution"):
        deck_copies("hard", OOD_SPLIT)
    with pytest.raises(ContractError, match="declares no out-of-distribution"):
        ConcentrationEnv(variant="hard", split=OOD_SPLIT)
    with pytest.raises(ContractError, match="declares no out-of-distribution"):
        deck_copies("easy", OOD_SPLIT)
    env = board_environment(variant=REDUCED_VARIANT, split=OOD_SPLIT)
    try:
        assert env.uneven and env.copies == (6, 6, 6, 6, 2, 2, 2, 2)
        assert (env.cards, env.pairs, env.horizon) == (32, 16, 64)
        assert len(env.fields) == 32 * 9 + 1
        assignments = []
        for task in FINAL_OOD_TASKS[:4]:
            packet, _ = env.reset(options={"task_index": task})
            board, _ = env.decode(packet)
            assert np.all(board == REDUCED_RANKS)
            for position in range(REDUCED_CARDS):
                env.step(position)
            counts = np.bincount(np.array(env._board.revealed), minlength=8)
            assert sorted(counts.tolist()) == [2, 2, 2, 2, 6, 6, 6, 6]
            assignments.append(tuple(counts.tolist()))
        assert len(set(assignments)) > 1  # which ranks are rich varies per board
        # The same seed deals the same uneven board.
        twin = board_environment(variant=REDUCED_VARIANT, split=OOD_SPLIT)
        try:
            twin.reset(options={"task_index": FINAL_OOD_TASKS[3]})
            for position in range(REDUCED_CARDS):
                twin.step(position)
            assert list(twin._board.revealed) == list(env._board.revealed)
        finally:
            twin.close()
        # A perfect second pass completes the board at the ordinary scale.
        done = False
        while not done:
            a, b = known_pairs(env)[0]
            _, r1, *_ = env.step(a)
            _, r2, terminated, truncated, _ = env.step(b)
            assert r1 == 0.0 and r2 == pytest.approx(1 / REDUCED_PAIRS)
            done = terminated or truncated
        assert env.board_complete and len(env.flips) <= REDUCED_HORIZON
        # The board's labels travel with its state.
        play(env, FINAL_OOD_TASKS[5], list(range(12)))
        state = env.state_dict()
        assert state["native"]["rank_permutation"] is not None
        twin = board_environment(variant=REDUCED_VARIANT, split=OOD_SPLIT)
        try:
            twin.load_state_dict(state)
            assert twin.state_dict() == state
            actions = list(range(12, REDUCED_CARDS))
            assert [env.step(a)[1] for a in actions] == [
                twin.step(a)[1] for a in actions
            ]
        finally:
            twin.close()
    finally:
        env.close()
    # The ordinary deck records no relabelling; its state is not an OOD
    # board's (the deck's copies are part of the environment identity).
    iid = board_environment(variant=REDUCED_VARIANT, split="final")
    try:
        play(iid, FINAL_TASKS[0], [0, 1])
        state = iid.state_dict()
        assert state["native"]["rank_permutation"] is None
        assert not iid.uneven
    finally:
        iid.close()
    ood = board_environment(variant=REDUCED_VARIANT, split=OOD_SPLIT)
    try:
        with pytest.raises(ContractError, match="contract does not match"):
            ood.load_state_dict(state)
    finally:
        ood.close()


def test_the_colour_deck_is_the_declared_protocol_and_matches_by_colour() -> None:
    """Decision 19: POPGym's ConcentrationEasy — the 52-card deck matched by
    colour, two classes of 26, the native rewards and budget, 157 fields."""
    env = board_environment(variant=STUDY_VARIANT)
    try:
        assert env.protocol == "concentration-easy" == concentration_protocol("easy")
        assert concentration_variant("concentration-easy") == "easy"
        assert (env.cards, env.ranks, env.pairs, env.horizon) == (52, 2, 26, 104)
        assert env.copies == (26, 26) and not env.uneven
        assert env.action_space.n == 52
        assert len(env.fields) == 52 * 3 + 1 == len(public_fields(52, 2))
        assert env.fields[2].name == "position0_is_2"  # the face-down sentinel
        assert env.fields[-1].name == "timer"
        packet, _ = env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        board, timer = env.decode(packet)
        assert np.all(board == STUDY_RANKS) and timer == 0.0
        assert packet["current"].shape == (STUDY_CARDS * (STUDY_RANKS + 1) + 1,)
        # A sweep reveals 26 cards of each colour; the same task deals the same board.
        twin = board_environment(variant=STUDY_VARIANT, initial_seed=5)
        try:
            reveals = []
            for board_env in (env, twin):
                board_env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
                for position in range(STUDY_CARDS):
                    board_env.step(position)
                reveals.append(list(board_env._board.revealed))
            assert reveals[0] == reveals[1] and None not in reveals[0]
            assert sorted(reveals[0]).count(0) == 26 and set(reveals[0]) == {0, 1}
            twin.reset(options={"task_index": DEVELOPMENT_TASKS[1]})
            for position in range(STUDY_CARDS):
                twin.step(position)
            assert list(twin._board.revealed) != reveals[0]
        finally:
            twin.close()
        # Rewards at the native scale: a colour match pays 1/26, a mismatch
        # 2/104, an already-matched card 1/104.
        lucky = sum(flip.matched for flip in env.flips)
        assert env.episode_return == pytest.approx(
            lucky / STUDY_PAIRS - (STUDY_PAIRS - lucky) * 2 / STUDY_HORIZON
        )
        pair = known_pairs(env)[0]
        _, r1, *_ = env.step(pair[0])
        _, r2, *_ = env.step(pair[1])
        assert r1 == 0.0 and r2 == pytest.approx(1 / STUDY_PAIRS)
        assert env.flips[-1].matched and env.flips[-1].opportunity
        _, r3, *_ = env.step(pair[0])
        assert r3 == pytest.approx(-1 / STUDY_HORIZON) and env.flips[-1].invalid
        # Two hidden cards of different colours mismatch and hide again.
        known = env._board.known_hidden()
        black = next(p for p, c in known.items() if c == 0)
        red = next(p for p, c in known.items() if c == 1)
        _, r4, *_ = env.step(black)
        packet, r5, *_ = env.step(red)
        assert r4 == 0.0 and r5 == pytest.approx(-2 / STUDY_HORIZON)
        assert not env.flips[-1].hit and env.flips[-1].opportunity
        # A perfect second pass completes the board inside the flip budget.
        done = False
        while not done:
            a, b = known_pairs(env)[0]
            env.step(a)
            _, _, terminated, truncated, _ = env.step(b)
            done = terminated or truncated
        assert env.board_complete and env.matched_pairs == STUDY_PAIRS
        assert terminated and len(env.flips) <= STUDY_HORIZON
        assert env.episode_return == pytest.approx(
            sum(f.native_reward for f in env.flips)
        )
    finally:
        env.close()
    env = board_environment(variant=STUDY_VARIANT)
    try:
        play(env, DEVELOPMENT_TASKS[5], list(range(20)))
        state = env.state_dict()
        assert state["native"]["rank_permutation"] is None
        twin = board_environment(variant=STUDY_VARIANT)
        try:
            twin.load_state_dict(state)
            assert twin.state_dict() == state and twin.flips == env.flips
            actions = list(range(20, STUDY_CARDS))
            assert [env.step(a)[1] for a in actions] == [
                twin.step(a)[1] for a in actions
            ]
        finally:
            twin.close()
    finally:
        env.close()
    with pytest.raises(ContractError, match="fixed by its deck"):
        ConcentrationEnv(variant="easy", horizon=64, split="development")


def test_a_board_starts_face_down_and_the_same_seed_deals_the_same_board() -> None:
    first, second = board_environment(), board_environment(initial_seed=7)
    try:
        packet, info = first.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        board, timer = first.decode(packet)
        assert np.all(board == RANKS) and timer == 0.0
        assert "deck" not in info and set(info) == {
            "evaluator_task_index",
            "flip_index",
            "matched_pairs",
            "episode_return",
            "board_complete",
        }
        # A full sweep reveals every rank; the same task shows the same ranks.
        reveals = []
        for env in (first, second):
            env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
            for position in range(CARDS):
                packet, *_ = env.step(position)
            reveals.append(list(env._board.revealed))
        assert reveals[0] == reveals[1] and None not in reveals[0]
        assert sorted(reveals[0]).count(0) == 4  # four cards per rank
        second.reset(options={"task_index": DEVELOPMENT_TASKS[1]})
        for position in range(CARDS):
            second.step(position)
        assert list(second._board.revealed) != reveals[0]
    finally:
        first.close()
        second.close()


def test_the_packet_carries_only_the_pinned_public_projection() -> None:
    env = board_environment()
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[2]})
        packet, *_ = env.step(5)
        board, timer = env.decode(packet)
        assert board[5] != RANKS and np.sum(board != RANKS) == 1
        assert timer == pytest.approx(1 / 1000, abs=1e-6)
        assert set(packet) == {"current", "previous", "outcome", "event", "valid"}
        assert packet["current"].shape == (CARDS * (RANKS + 1) + 1,)
        with pytest.raises(ContractError, match="not one-hot"):
            env.decode({"current": np.zeros(CARDS * (RANKS + 1) + 1, np.float32)})
    finally:
        env.close()


# --------------------------------------------------------------------------
# Rewards and the public board
# --------------------------------------------------------------------------


def test_every_reward_branch_is_reproduced_from_the_public_board() -> None:
    env = board_environment()
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[0]})
        # First flip pays nothing; the same position twice is a mismatch.
        _, first, *_ = env.step(3)
        _, again, *_ = env.step(3)
        assert first == 0.0 and again == pytest.approx(-2 / HORIZON)
        assert (
            env.flips[-1].invalid and env.flips[-1].second and not env.flips[-1].matched
        )
        # Sweep the deck, then match a known pair.
        for position in range(CARDS):
            env.step(position)
        pair = known_pairs(env)[0]
        _, r1, *_ = env.step(pair[0])
        _, r2, *_ = env.step(pair[1])
        assert r1 == 0.0 and r2 == pytest.approx(1 / PAIRS)
        assert env.flips[-1].matched and env.matched_pairs >= 1
        # Flipping an already-matched card is refused by the game, not us.
        _, r3, *_ = env.step(pair[0])
        assert r3 == pytest.approx(-1 / HORIZON)
        assert env.flips[-1].invalid and env._board.in_play == []
        # A mismatch hides both cards on the next observation.
        hidden = [p for p in range(CARDS) if p not in env._board.matched][:2]
        while env._board.revealed[hidden[0]] == env._board.revealed[hidden[1]]:
            hidden = [*hidden[1:], hidden[0] + 2]
        _, r4, *_ = env.step(hidden[0])
        packet, r5, *_ = env.step(hidden[1])
        board, _ = env.decode(packet)
        assert r4 == 0.0 and r5 == pytest.approx(-2 / HORIZON)
        assert board[hidden[0]] != RANKS and board[hidden[1]] != RANKS
        packet, *_ = env.step(hidden[0])
        board, _ = env.decode(packet)
        assert board[hidden[1]] == RANKS
        assert env.episode_return == pytest.approx(
            sum(f.native_reward for f in env.flips)
        )
    finally:
        env.close()


def test_a_perfect_second_pass_completes_the_board_and_ends_it() -> None:
    env = board_environment()
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[3]})
        for position in range(CARDS):
            env.step(position)
        done = False
        while not done:
            first, second = known_pairs(env)[0]
            env.step(first)
            _, _, terminated, truncated, info = env.step(second)
            done = terminated or truncated
        assert env.board_complete and env.matched_pairs == PAIRS
        assert info["board_complete"] and terminated
        assert len(env.flips) == CARDS + 2 * (
            PAIRS - sum(f.matched for f in env.flips[:CARDS])
        )
        with pytest.raises(ContractError, match="requires an outer reset"):
            env.step(0)
    finally:
        env.close()


def test_the_public_board_refuses_an_observation_it_cannot_explain() -> None:
    board = PublicBoard(CARDS, RANKS, HORIZON)
    board.begin(np.full(CARDS, RANKS))
    after = np.full(CARDS, RANKS)
    after[0] = 4
    after[9] = 4  # a second card visible that was never flipped
    with pytest.raises(ContractError, match="disagrees with the public board"):
        board.apply(0, after)
    with pytest.raises(ContractError, match="starts fully face down"):
        PublicBoard(CARDS, RANKS, HORIZON).begin(after)


# --------------------------------------------------------------------------
# The retrieval ledger
# --------------------------------------------------------------------------


def test_the_ledger_scores_partner_selection_from_what_was_public() -> None:
    env = board_environment()
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[4]})
        for position in range(CARDS):
            env.step(position)
        # The very first pair had nothing to retrieve; later sweep flips may.
        assert not env.flips[1].opportunity and env.flips[1].second
        (first, second), *_ = known_pairs(env)
        rank = env._board.revealed[first]
        eligible = sum(
            1 for p, r in env._board.known_hidden().items() if r == rank and p != first
        )
        env.step(first)
        wrong = next(p for p, r in env._board.known_hidden().items() if r != rank)
        env.step(wrong)
        miss = env.flips[-1]
        assert miss.opportunity and miss.eligible == eligible and not miss.hit
        assert miss.partner_last_reveal is not None
        assert miss.partner_last_reveal <= CARDS  # revealed during the sweep
        env.step(first)
        env.step(second)
        hit = env.flips[-1]
        assert hit.opportunity and hit.hit and hit.matched
    finally:
        env.close()
    # Re-flipping a known card whose partner is not known is a wasted flip.
    env = board_environment()
    try:
        env.reset(options={"task_index": DEVELOPMENT_TASKS[6]})
        env.step(0)
        env.step(1)
        while env._board.revealed[0] == env._board.revealed[1]:  # a lucky match
            env.step(2)
            env.step(3)
        first_known = next(p for p in (0, 1, 2, 3) if p in env._board.known_hidden())
        env.step(first_known)
        assert env.flips[-1].redundant and not env.flips[-1].opportunity
    finally:
        env.close()


def _flip(index: int, **overrides: Any) -> ConcentrationFlip:
    fields: dict[str, Any] = {
        "index": index,
        "position": index % CARDS,
        "rank": 0,
        "second": False,
        "matched": False,
        "invalid": False,
        "redundant": False,
        "opportunity": False,
        "eligible": 0,
        "hit": False,
        "partner_last_reveal": None,
        "native_reward": 0.0,
    }
    fields.update(overrides)
    return ConcentrationFlip(**fields)


def test_eviction_is_read_from_the_partner_reveal_and_the_write_counters() -> None:
    """Observation k feeds decision k + 1; an opportunity is evicted when the
    partner's most recent reveal was consumed under an earlier segment."""
    flips = [_flip(i) for i in range(1, 41)]
    flips[19] = _flip(
        20,
        second=True,
        opportunity=True,
        eligible=2,
        hit=True,
        matched=True,
        partner_last_reveal=5,
        native_reward=1 / PAIRS,
    )
    flips[39] = _flip(
        40,
        second=True,
        opportunity=True,
        eligible=1,
        hit=False,
        partner_last_reveal=30,
        native_reward=-2 / HORIZON,
    )
    writes = tuple((i - 1) // 16 for i in range(1, 41))  # C = 16
    result = ConcentrationResult(
        DEVELOPMENT_TASKS[0],
        0,
        tuple(flips),
        1,
        PAIRS,
        flips[19].native_reward + flips[39].native_reward,
        40,
        writes,
    )
    ledger = result.ledger()
    assert (ledger.opportunities, ledger.successes) == (2, 1)
    # Partner seen at observation 5 (segment 0), flip 20 sits in segment 1:
    # evicted and hit. Partner seen at 30 (segment 1), flip 40 in segment 2:
    # evicted and missed.
    assert (ledger.evicted_opportunities, ledger.evicted_successes) == (2, 1)
    untracked = ConcentrationResult(
        DEVELOPMENT_TASKS[0], 0, tuple(flips), 1, PAIRS, result.episode_return, 40
    )
    assert untracked.ledger().evicted_opportunities is None
    with pytest.raises(ResultValidationError, match="one flip per decision"):
        ConcentrationResult(
            DEVELOPMENT_TASKS[0], 0, tuple(flips[:10]), 1, PAIRS, 0.0, 40
        )
    with pytest.raises(ContractError, match="names its partners"):
        _flip(1, opportunity=True, eligible=1)


# --------------------------------------------------------------------------
# Events, records, secondary summaries, references
# --------------------------------------------------------------------------


def sample_results(
    tasks: tuple[int, ...], seed: int = 0, variant: str = STUDY_VARIANT
) -> tuple[ConcentrationResult, ...]:
    generator = np.random.default_rng(seed)
    results = []
    for task in tasks:
        env = board_environment(
            split="development" if task < 2_000_000 else "final", variant=variant
        )
        try:
            env.reset(options={"task_index": task})
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(
                    int(generator.integers(env.cards))
                )
                done = terminated or truncated
            results.append(
                ConcentrationResult(
                    task,
                    0,
                    env.flips,
                    env.matched_pairs,
                    env.pairs,
                    env.episode_return,
                    len(env.flips),
                )
            )
        finally:
            env.close()
    return tuple(results)


def test_events_score_one_episode_per_board_and_validate() -> None:
    active = contract()
    config = resolved()
    results = sample_results(active.roster("development"))
    rows = events(
        active,
        config,
        results,
        checkpoint="checkpoint.pt",
        split="development",
        history="retained",
    )
    assert [row.kind for row in rows] == ["episode"] * 64
    for row, result in zip(rows, results, strict=True):
        assert (row.numerator, row.denominator) == (result.matched_pairs, STUDY_PAIRS)
        assert row.native_return == pytest.approx(result.episode_return)
        assert row.start_step == 1 and row.end_step == row.step == result.decisions
        assert row.retrieval_opportunities is not None
        assert row.retrieval_successes is not None
        assert row.retrieval_successes <= row.retrieval_opportunities
        assert row.evicted_opportunities is None  # no summary counters
        assert row.board_complete is False
        assert row.writes_before_decision is None
    run = BenchmarkRun(
        active.protocol,
        "concentration",
        "raw",
        0,
        "checkpoint.pt",
        "development",
        "retained",
        "completed",
    )
    validate_benchmark_results([active], [run], rows)
    with pytest.raises(ResultValidationError, match="incomplete/unexpected"):
        validate_benchmark_results([active], [run], rows[:10])
    from dataclasses import replace

    rest = rows[1:]
    with pytest.raises(ResultValidationError, match="successes exceed"):
        validate_benchmark_results(
            [active], [run], [replace(rows[0], retrieval_successes=99), *rest]
        )
    with pytest.raises(ResultValidationError, match="board_complete"):
        validate_benchmark_results(
            [active], [run], [replace(rows[0], board_complete=True), *rest]
        )
    with pytest.raises(ResultValidationError, match="summary write counters"):
        validate_benchmark_results(
            [active],
            [run],
            [replace(rows[0], evicted_opportunities=1, evicted_successes=0), *rest],
        )


def test_ood_events_validate_and_the_ood_reference_is_measured() -> None:
    """Decision 17's split lives on the rank8 contract, kept outside the study
    since decision 19; its events and reference still validate."""
    active = load_contract(RANK8_CONTRACT)
    config = resolved(active=active)
    tasks = active.roster("final_ood")
    results = []
    for task in tasks:
        env = board_environment(variant=REDUCED_VARIANT, split=OOD_SPLIT)
        try:
            generator = np.random.default_rng(int(task))
            env.reset(options={"task_index": task})
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(
                    int(generator.integers(env.cards))
                )
                done = terminated or truncated
            results.append(
                ConcentrationResult(
                    task,
                    0,
                    env.flips,
                    env.matched_pairs,
                    env.pairs,
                    env.episode_return,
                    len(env.flips),
                )
            )
        finally:
            env.close()
    rows = events(
        active,
        config,
        results,
        checkpoint="checkpoint.pt",
        split="final_ood",
        history="retained",
    )
    assert [row.denominator for row in rows] == [REDUCED_PAIRS] * len(tasks)
    assert {row.split for row in rows} == {"final_ood"}
    run = BenchmarkRun(
        active.protocol,
        "concentration",
        "raw",
        0,
        "checkpoint.pt",
        "final_ood",
        "retained",
        "completed",
    )
    validate_benchmark_results([active], [run], rows)
    with pytest.raises(ResultValidationError, match="incomplete/unexpected"):
        validate_benchmark_results([active], [run], rows[:10])
    env = board_environment(variant=REDUCED_VARIANT, split=OOD_SPLIT)
    try:
        measured = random_policy_pair_fraction(env, task_ids=tasks[:8])
    finally:
        env.close()
    assert 0.0 < measured["random_reference_pair_fraction"] < 0.5


def test_the_secondary_summary_and_the_random_reference() -> None:
    config = resolved()
    results = sample_results(tuple(DEVELOPMENT_TASKS[:3]))
    summary = concentration_secondary(config.environment, results)
    assert summary["boards"] == 3.0 and summary["flips"] == pytest.approx(
        sum(r.decisions for r in results)
    )
    assert 0.0 <= summary["pair_fraction"] <= 1.0
    assert summary["retrieval_rate"] is None or 0.0 <= summary["retrieval_rate"] <= 1.0
    assert summary["evicted_retrieval_rate"] is None  # untracked carriers
    env = board_environment(variant=STUDY_VARIANT)
    try:
        measured = random_policy_pair_fraction(env, task_ids=DEVELOPMENT_TASKS[:3])
    finally:
        env.close()
    assert measured["random_reference_pair_fraction"] == pytest.approx(
        float(np.mean([r.pair_fraction for r in results]))
    )
    # Half the deck matches any card on the colour deck, so uniform flipping
    # already matches about half the pairs (decision 19 records ≈ 0.54).
    assert 0.3 < measured["random_reference_pair_fraction"] < 0.8


@pytest.mark.parametrize("variant", sorted(CONCENTRATION_PROTOCOLS))
def test_the_task_diagnostic_finds_identical_inputs_with_different_partners(
    variant: str,
) -> None:
    env = board_environment(variant=variant)
    try:
        diagnostic = concentration_diagnostic(env, board_ids=DEVELOPMENT_TASKS[:64])
    finally:
        env.close()
    assert diagnostic.benchmark == "concentration"
    assert diagnostic.measurements["shared_packets"] >= 5
    fraction = diagnostic.measurements["ambiguous_packet_fraction"]
    if variant == "easy":
        # Decision 19 records this before any fit: on the colour deck half the
        # known hidden cards are eligible partners, so identical inputs across
        # boards usually share an eligible flip and the sweep's incompatible
        # groups sit just under the declared 0.5 minimum (0.485 on the 64
        # development boards). The gate's C3 blocks on it.
        assert 0.4 < fraction < 0.5 and not diagnostic.passed
    else:
        assert fraction >= 0.5 and diagnostic.passed


# --------------------------------------------------------------------------
# Contract, rosters, registry, state
# --------------------------------------------------------------------------


def test_the_contract_declares_the_colour_deck_and_its_memory() -> None:
    active = contract()
    assert active.protocol == "concentration-easy" and active.status == "C0"
    env = active.environment
    assert (env.size, env.horizon, env.attempts, env.outer_length) == (52, 104, 1, 104)
    assert active.evaluation.primary_metric == "pair_fraction"
    assert active.event_kind == "episode"
    assert active.memory == {
        "summary": {"segment_length": 16, "memory_tokens": 4},
        "window": {"segment_length": 24},
    }
    assert active.training.max_sequence_length == 104
    assert active.training.trajectory_length == 105
    assert len(active.roster("development")) == 64
    assert len(active.roster("final")) == 256
    assert not set(active.roster("development")) & set(active.roster("final"))
    assert set(active.roster("final")) <= set(FINAL_TASKS)
    # The colour deck declares no OOD split; decision 17's stays on rank8.
    assert "final_ood" not in active.evaluation.splits
    summary = resolved("raw_summary").model.summary
    assert summary is not None and summary.capacity == 24
    dat = resolved("raw_dat_summary").model.dat
    assert dat is not None and dat.max_relative_distance == 24
    window = resolved("raw_window").model.window
    assert window is not None and window.segment_length == 24
    assert history_modes("concentration") == (
        "retained",
        "current-token",
        "summary-cleared",
    )
    reloaded = load_contract(ROOT / "configs/environments/concentration.yaml")
    assert reloaded.protocol == active.protocol
    # The rank decks' contracts stay on disk for their gate records (F7,
    # decisions 16 and 18) but are no longer study contracts.
    native = load_contract(ROOT / "configs/environments/concentration_hard.yaml")
    assert native.protocol == "concentration-hard"
    assert (native.environment.size, native.environment.horizon) == (52, 104)
    assert native.memory == active.memory
    reduced = load_contract(RANK8_CONTRACT)
    assert reduced.protocol == "concentration-rank8"
    assert (reduced.environment.size, reduced.environment.horizon) == (32, 64)
    assert reduced.memory == active.memory
    # Decision 17: the OOD final split, its own band, the uneven deck.
    assert reduced.evaluation.splits["final_ood"].source == OOD_SPLIT
    assert len(reduced.roster("final_ood")) == 256
    assert set(reduced.roster("final_ood")) <= set(FINAL_OOD_TASKS)
    assert not set(reduced.roster("final_ood")) & set(reduced.roster("final"))
    for retired in ("concentration-hard", "concentration-rank8"):
        with pytest.raises(ContractError, match="Unknown environment"):
            load_retired_summary_memory_study().contract(retired)


def test_the_smoke_profile_keeps_the_deck_intact() -> None:
    config = resolved(smoke=True)
    assert config.environment.horizon == STUDY_HORIZON
    assert config.training.max_sequence_length == STUDY_HORIZON


def test_the_board_state_round_trips_mid_game() -> None:
    env = board_environment()
    try:
        play(env, DEVELOPMENT_TASKS[5], list(range(30)))
        state = env.state_dict()
        twin = board_environment()
        try:
            twin.load_state_dict(state)
            assert twin.state_dict() == state
            assert twin.flips == env.flips and twin.matched_pairs == env.matched_pairs
            actions = list(range(30, 52))
            mine = [env.step(a)[1] for a in actions]
            theirs = [twin.step(a)[1] for a in actions]
            assert mine == theirs
            assert twin._board.state() == env._board.state()
        finally:
            twin.close()
    finally:
        env.close()
