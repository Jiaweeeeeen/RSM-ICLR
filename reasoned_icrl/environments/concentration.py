"""POPGym 1.0.7 Concentration as a public-packet policy environment.

Three decks, three protocol identities. ``hard`` wraps the pinned
``popgym.envs.concentration.ConcentrationHard`` (52 cards matched by rank)
through AMAGO's own POPGym wrapper. ``rank8`` is the project-owned reduced-rank
deck of fallback F7: the first eight of the
thirteen ranks, four suits each, under the same rules, reward scales and
wrapper chain — POPGym has no reduced-rank Concentration, and neither upstream
package is patched: the deck subclasses POPGym's ``Deck`` and the game
subclasses its ``Concentration``. ``easy`` wraps the pinned
``ConcentrationEasy`` (chosen after both rank decks failed their gate): the same 52
cards matched by **colour**,
two classes of 26 cards, under the same rules, budget and wrapper.

The game is kept exactly as it is: ``cards`` cards of ``ranks`` matching
classes (ranks on the rank decks, colours on ``easy``) lie face down in an
order shuffled by the task seed; one action flips one position; at most two
cards are in play; a second flip of the same class (at a different position)
pays ``+1/pairs`` and both stay face up for good; a mismatch — or the same
position twice — pays ``-2/horizon`` and both are hidden again on the next
observation; a first flip pays 0; flipping an already-matched card pays
``-1/horizon`` per card in play and discards the pending card. The board ends
when every pair is matched or after ``horizon`` flips, POPGym's
``ceil(2n - n / (2n - 1))`` for ``n`` cards (104 for 52 cards, 64 for 32).

Only the public projection reaches the policy: the board as ``cards``
position x ``ranks + 1`` one-hots (the classes plus the face-down sentinel,
exactly the native observation) and AMAGO's ``timer / 1000``. ``get_state()``
and the deck are never read for an observation; native ``info`` is empty.

:class:`PublicBoard` reconstructs, from observations and the agent's own
actions only, what a perfect public memory would know: the last revealed rank
of every position, the permanently matched positions and the pair in play. It
cross-checks every native reward, scores the task (matched pairs of ``pairs``) and
builds the *retrieval ledger* of the paper's tier-1 diagnostics: a second flip
is a retrieval opportunity when a same-rank card was revealed earlier, is now
hidden and unmatched; any of those eligible partner positions counts as a hit;
the observation index of the partners' most recent reveal lets the evaluator
say whether that evidence was already evicted from the current segment. Its
state is never an actor input, a packet field or a training target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np

from reasoned_icrl.environments.base import (
    BaseEnv,
    PublicDecision,
    PublicField,
    benchmark_task_sources,
    unit_fields,
)
from reasoned_icrl.environments.utils import NativeSource, project_unit_interval
from reasoned_icrl.experiments.contracts import ContractError

CONCENTRATION_STATE_SCHEMA = "concentration-state.v1"
CONCENTRATION_PROTOCOLS = {
    "hard": "concentration-hard",
    "rank8": "concentration-rank8",
    "easy": "concentration-easy",
}
"""The native rank-matching deck (decision 14), the reduced-rank deck that
fallback F7 added once its ``raw`` reference failed C2 (decision 16: eight
ranks x 4 suits, the same rank-matching structure) and the native
colour-matching deck chosen after both failed (decision 19). Each is
a separate protocol identity."""

CONCENTRATION_CARDS = {"hard": 52, "rank8": 32, "easy": 52}
CONCENTRATION_RANKS = {"hard": 13, "rank8": 8, "easy": 2}
"""Matching classes per deck: ranks on the rank decks, the two colours on
``easy``. The name is kept from the rank decks; every ``ranks`` below is the
class count."""
CONCENTRATION_SUITS = 4
"""The rank decks deal each rank once per suit."""
CONCENTRATION_COPIES = {"hard": 4, "rank8": 4, "easy": 26}
"""Cards per matching class on the ordinary deck: one per suit on the rank
decks, half the deck per colour on ``easy``."""


def flip_budget(cards: int) -> int:
    """POPGym's flip budget for a deck of ``cards``: ``ceil(2n - n / (2n - 1))``."""
    if cards < 2 or cards % 2:
        raise ContractError("A Concentration deck holds an even number of cards.")
    return math.ceil(2 * cards - cards / (2 * cards - 1))


CONCENTRATION_HORIZONS = {"hard": 104, "rank8": 64, "easy": 104}
"""At most ``horizon`` flips per board, :func:`flip_budget` of the deck."""
CONCENTRATION_NATIVE_IDS = {
    "hard": "popgym-ConcentrationHard-v0",
    "rank8": "reasoned-icrl-ConcentrationRank8-v0",
    "easy": "popgym-ConcentrationEasy-v0",
}
"""The upstream POPGym ids of the native decks; the reduced deck is
project-owned (POPGym registers none) and is built by
:func:`reduced_rank_native`."""

OOD_SPLIT = "final-ood"
CONCENTRATION_OOD_COPIES = {"rank8": (6, 6, 6, 6, 2, 2, 2, 2)}
"""The out-of-distribution deck of the ``final-ood`` split (decision 17): the
same 32 positions and eight rank channels, but per board four ranks hold three
pairs and four hold one, the assignment of copies to rank ids drawn from the
task seed. Every channel stays trained and the equality rule is unchanged;
what shifts is the binding structure (four copies per rank) a policy learned
to expect. Only the reduced deck declares one; the native deck refuses it."""


def deck_copies(variant: str, split: str) -> tuple[int, ...]:
    """Cards per class of one deck on one split: ordinary, or the OOD shift."""
    ranks = CONCENTRATION_RANKS[variant]
    if split != OOD_SPLIT:
        return (CONCENTRATION_COPIES[variant],) * ranks
    try:
        copies = CONCENTRATION_OOD_COPIES[variant]
    except KeyError as error:
        raise ContractError(
            f"The {variant!r} deck declares no out-of-distribution split."
        ) from error
    if len(copies) != ranks or sum(copies) != CONCENTRATION_CARDS[variant]:
        raise ContractError("An OOD deck keeps its rank count and card count.")
    return copies


def concentration_protocol(variant: str) -> str:
    """Return the protocol identity of one declared deck."""
    try:
        return CONCENTRATION_PROTOCOLS[variant]
    except KeyError as error:
        raise ContractError(f"Unknown Concentration deck: {variant!r}.") from error


def concentration_variant(protocol: str) -> str:
    """Return the deck behind one protocol identity."""
    for variant, name in CONCENTRATION_PROTOCOLS.items():
        if name == protocol:
            return variant
    raise ContractError(f"Unknown Concentration protocol: {protocol!r}.")


@cache
def _reduced_rank_classes() -> tuple[type[Any], type[Any]]:
    """The project-owned reduced-rank game and the POPGym wrapper over a game.

    Built once, lazily: the upstream packages are imported the way the native
    deck imports them, inside the factory, never at module import.
    """
    from amago.envs.builtin.popgym_envs import POPGym, _MultiDiscreteToBox
    from amago.envs.env_utils import extend_box_obs_space_by
    from popgym.core.deck import COLORS, RANKS, SUITS, Deck
    from popgym.envs.concentration import Concentration
    from popgym.wrappers import DiscreteAction, Flatten

    class ReducedRankDeck(Deck):  # type: ignore[misc]
        """POPGym's deck restricted to its first ``ranks`` ranks, ``copies`` each.

        Four copies per rank is the ordinary deck (one per suit); an uneven
        ``copies`` is the OOD deck, whose extra cards cycle through the suits.
        """

        def __init__(self, ranks: int, copies: tuple[int, ...] | None = None) -> None:
            if not 1 <= ranks <= RANKS.size or SUITS.size != CONCENTRATION_SUITS:
                raise ContractError("A reduced deck keeps 1..13 ranks of four suits.")
            counts = (SUITS.size,) * ranks if copies is None else tuple(copies)
            if len(counts) != ranks or any(c < 2 or c % 2 for c in counts):
                raise ContractError(
                    "Every rank of a deck holds an even, positive count."
                )
            self.num_decks = 1
            self.num_cards = int(sum(counts))
            self.idx = np.arange(self.num_cards)
            self.ranks_idx = np.repeat(np.arange(ranks), counts)
            self.ranks = RANKS[self.ranks_idx]
            self.suits_idx = np.concatenate(
                [np.arange(count) % SUITS.size for count in counts]
            )
            self.suits = SUITS[self.suits_idx]
            self.colors = np.tile(COLORS, self.num_cards // 2)
            self.colors_idx = np.tile(np.arange(COLORS.size), self.num_cards // 2)
            self.hands: dict[str, list[int]] = {}
            self.deck_len = self.num_cards
            self.keys = ["idx", "ranks", "suits", "colors"]
            self.idx_keys = ["ranks_idx", "suits_idx", "colors_idx", "idx"]

    class ReducedRankConcentration(Concentration):  # type: ignore[misc]
        """POPGym's rank-matching game over a :class:`ReducedRankDeck`.

        The rules, reward scales and observation layout are the upstream
        class's own, recomputed for the smaller deck exactly as its
        constructor computes them for 52 cards.
        """

        def __init__(self, ranks: int, copies: tuple[int, ...] | None = None) -> None:
            super().__init__(num_decks=1, deck_type="ranks")
            deck = ReducedRankDeck(ranks, copies)
            cards = deck.num_cards
            self.deck = deck
            self.rank_count = ranks
            self.uneven = copies is not None and len(set(copies)) > 1
            self.base_ranks_idx = deck.ranks_idx.copy()
            self.rank_permutation: np.ndarray | None = None
            self.episode_length = flip_budget(cards)
            self.success_reward_scale = 1 / (cards // 2)
            self.failure_reward_scale = -1 / self.episode_length
            self.deck_type = deck.ranks
            self.deck_idx_type = deck.ranks_idx
            self.facedown_card = ranks
            values = np.full(cards, 1 + ranks, dtype=np.int64)
            self.observation_space = gym.spaces.MultiDiscrete(values)
            self.state_space = gym.spaces.Tuple(
                (
                    gym.spaces.MultiDiscrete(values - 1),
                    gym.spaces.MultiBinary(cards),
                    gym.spaces.Discrete(cards),
                    gym.spaces.Discrete(cards),
                )
            )
            self.action_space = gym.spaces.Discrete(cards)
            deck.add_players("face_up", "face_up_idx", "in_play", "in_play_idx")

        def relabel(self, permutation: np.ndarray | None) -> None:
            """Assign the deck's copy counts to rank ids through ``permutation``.

            ``None`` keeps the deck's own labels (the ordinary deck). Matching
            compares the relabelled ids, so equality is preserved exactly.
            """
            self.rank_permutation = None if permutation is None else permutation
            labels = (
                self.base_ranks_idx
                if permutation is None
                else permutation[self.base_ranks_idx]
            )
            self.deck_idx_type = labels
            self.deck_type = labels
            self.state = self.deck_idx_type[self.deck.idx].copy().astype(np.int64)
            self.obs = self.get_obs()

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[Any, dict[str, Any]]:
            obs, info = super().reset(seed=seed, options=options)
            if not self.uneven:
                return obs, info
            # The OOD deck: which ranks hold three pairs is drawn per board.
            self.relabel(self.np_random.permutation(self.rank_count))
            return self.obs, info

    class GamePOPGym(POPGym):  # type: ignore[misc]
        """AMAGO's POPGym wrapper chain over a constructed game rather than an id."""

        def __init__(
            self, game: gym.Env[Any, Any], truncated_is_done: bool = True
        ) -> None:
            env: Any = Flatten(game)
            discrete = gym.spaces.Discrete | gym.spaces.MultiDiscrete
            if isinstance(env.action_space, discrete):
                env = DiscreteAction(env)
            if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
                env = _MultiDiscreteToBox(env)
            self.truncated_is_done = truncated_is_done
            gym.Wrapper.__init__(self, env)
            self.observation_space = extend_box_obs_space_by(
                env.observation_space, by=1, low=0.0, high=1.0
            )

    return ReducedRankConcentration, GamePOPGym


def reduced_rank_native(
    ranks: int, copies: tuple[int, ...] | None = None
) -> gym.Env[Any, Any]:
    """AMAGO's POPGym wrapper over the project-owned reduced-rank deck (F7).

    ``copies`` per rank selects the OOD deck of the ``final-ood`` split.
    """
    game_class, wrapper_class = _reduced_rank_classes()
    return cast(gym.Env[Any, Any], wrapper_class(game_class(ranks, copies)))


def public_fields(cards: int, ranks: int) -> tuple[PublicField, ...]:
    """Name the pinned public projection: position-major one-hots plus the timer.

    AMAGO's POPGym wrapper one-hots every position over ``ranks + 1`` values,
    the last being the face-down sentinel, and appends ``timer / 1000``.
    """
    names = [
        f"position{position}_is_{category}"
        for position in range(cards)
        for category in range(ranks + 1)
    ]
    names.append("timer")
    return unit_fields(names)


@dataclass(frozen=True, slots=True)
class FlipOutcome:
    """What one flip did, as the public board sees it (before the flip is
    applied for the ledger fields, after it for the outcome fields)."""

    second: bool
    matched: bool
    invalid: bool
    redundant: bool
    opportunity: bool
    eligible: int
    hit: bool
    partner_last_reveal: int | None
    expected_reward: float
    revealed_rank: int


class PublicBoard:
    """Exact public memory of one board, from observations and actions only.

    Feed the reset observation through :meth:`begin`, then every action with
    the observation it produced through :meth:`apply`. Observation ``k`` (the
    reset observation is ``0``) is the input of decision ``k + 1``, which is the
    index the evaluator's per-decision write counters use.
    """

    def __init__(self, cards: int, ranks: int, horizon: int) -> None:
        if cards < 2 or cards % 2 or ranks < 1 or horizon < 1:
            raise ContractError("Concentration needs an even deck and a budget.")
        self.cards, self.ranks, self.horizon = cards, ranks, horizon
        self.sentinel = ranks
        self.pairs = cards // 2
        self.revealed: list[int | None] = [None] * cards
        self.last_reveal: list[int | None] = [None] * cards
        self.matched: set[int] = set()
        self.in_play: list[int] = []
        self.observations = 0

    def begin(self, board: np.ndarray) -> None:
        """Consume the reset observation: every card face down."""
        if board.shape != (self.cards,) or np.any(board != self.sentinel):
            raise ContractError("A Concentration board starts fully face down.")
        self.revealed = [None] * self.cards
        self.last_reveal = [None] * self.cards
        self.matched = set()
        self.in_play = []
        self.observations = 1

    @property
    def matched_pairs(self) -> int:
        return len(self.matched) // 2

    def known_hidden(self) -> dict[int, int]:
        """Positions whose rank is known and that are hidden and unmatched now."""
        return {
            position: rank
            for position, rank in enumerate(self.revealed)
            if rank is not None
            and position not in self.matched
            and position not in self.in_play
        }

    def apply(self, action: int, board_after: np.ndarray) -> FlipOutcome:
        """Apply one flip, cross-check the observation and score the ledger."""
        if not 0 <= action < self.cards:
            raise ContractError("Concentration position is out of range.")
        if board_after.shape != (self.cards,):
            raise ContractError("Concentration board width changed.")
        in_play_before = list(self.in_play)
        matched_before = set(self.matched)
        known = self.known_hidden()
        # Retrieval ledger: read from what was known before this flip.
        opportunity, eligible, hit, partner_last = False, 0, False, None
        redundant = False
        if in_play_before:
            first = in_play_before[0]
            rank = self.revealed[first]
            partners = [p for p, r in known.items() if r == rank and p != first]
            if partners:
                opportunity, eligible, hit = True, len(partners), action in partners
                partner_last = max(cast(int, self.last_reveal[p]) for p in partners)
        elif action in known:
            rank = known[action]
            redundant = not any(r == rank and p != action for p, r in known.items())
        invalid = action in matched_before or (
            bool(in_play_before) and action == in_play_before[0]
        )
        # Outcome: the native rules, decided from the public state.
        in_play = [*in_play_before, action]
        matched = False
        if any(position in matched_before for position in in_play):
            expected = -len(in_play) / self.horizon
            self.in_play = []
        elif len(in_play) == 2:
            first, second = in_play
            if first != second and board_after[first] == board_after[second]:
                expected = 1.0 / self.pairs
                self.matched |= {first, second}
                matched = True
            else:
                expected = -2.0 / self.horizon
            self.in_play = []
        else:
            expected = 0.0
            self.in_play = in_play
        visible = {int(p) for p in np.flatnonzero(board_after != self.sentinel)}
        if visible != matched_before | set(in_play):
            raise ContractError(
                "Concentration observation disagrees with the public board."
            )
        for position in visible:
            self.revealed[position] = int(board_after[position])
            self.last_reveal[position] = self.observations
        self.observations += 1
        return FlipOutcome(
            second=len(in_play) == 2,
            matched=matched,
            invalid=invalid,
            redundant=redundant,
            opportunity=opportunity,
            eligible=eligible,
            hit=hit,
            partner_last_reveal=partner_last,
            expected_reward=expected,
            revealed_rank=int(board_after[action]),
        )

    def state(self) -> dict[str, object]:
        return {
            "revealed": list(self.revealed),
            "last_reveal": list(self.last_reveal),
            "matched": sorted(self.matched),
            "in_play": list(self.in_play),
            "observations": self.observations,
        }

    def load(self, state: Mapping[str, object]) -> None:
        revealed = list(cast(Any, state["revealed"]))
        last = list(cast(Any, state["last_reveal"]))
        if len(revealed) != self.cards or len(last) != self.cards:
            raise ContractError("Concentration board state changed shape.")
        self.revealed = [None if v is None else int(v) for v in revealed]
        self.last_reveal = [None if v is None else int(v) for v in last]
        self.matched = {int(v) for v in cast(Any, state["matched"])}
        self.in_play = [int(v) for v in cast(Any, state["in_play"])]
        self.observations = int(cast(Any, state["observations"]))


@dataclass(frozen=True, slots=True)
class ConcentrationFlip:
    """One flip, reconstructed from public observations only.

    ``partner_last_reveal`` is the observation index (reset = 0) of the most
    recent reveal among the eligible partner positions, or None when the flip
    was not a retrieval opportunity.
    """

    index: int
    position: int
    rank: int
    second: bool
    matched: bool
    invalid: bool
    redundant: bool
    opportunity: bool
    eligible: int
    hit: bool
    partner_last_reveal: int | None
    native_reward: float

    def __post_init__(self) -> None:
        if self.index < 1 or self.position < 0 or self.rank < 0:
            raise ContractError("Concentration flip counters must be valid.")
        if self.matched and not self.second:
            raise ContractError("Only a second flip can match.")
        if self.opportunity != (self.eligible > 0) or (
            self.hit and not self.opportunity
        ):
            raise ContractError("Concentration ledger fields disagree.")
        if (self.partner_last_reveal is None) == self.opportunity:
            raise ContractError("An opportunity names its partners' last reveal.")


class ConcentrationEnv(BaseEnv):
    """Public five-key packet over the pinned native Concentration board."""

    label: ClassVar[str] = "Concentration"
    state_schema: ClassVar[str] = CONCENTRATION_STATE_SCHEMA

    def __init__(
        self,
        *,
        variant: str = "hard",
        horizon: int | None = None,
        split: str = "train",
        source_indices: Sequence[int] | range | None = None,
        fixed_task_index: int | None = None,
        initial_seed: int = 0,
    ) -> None:
        self.protocol = concentration_protocol(variant)
        super().__init__(
            split=split,
            roster=benchmark_task_sources(split)
            if source_indices is None
            else source_indices,
            fixed_task_index=fixed_task_index,
            initial_seed=initial_seed,
        )
        self.variant = variant
        self.cards = CONCENTRATION_CARDS[variant]
        self.ranks = CONCENTRATION_RANKS[variant]
        self.pairs = self.cards // 2
        self.copies = deck_copies(variant, split)
        self.uneven = len(set(self.copies)) > 1
        self.horizon = CONCENTRATION_HORIZONS[variant] if horizon is None else horizon
        if self.horizon != CONCENTRATION_HORIZONS[variant]:
            raise ContractError("The Concentration flip budget is fixed by its deck.")
        self._native = NativeSource(self._make_native, seed=initial_seed)
        self._install_contract(public_fields(self.cards, self.ranks), self.cards)
        self._board = PublicBoard(self.cards, self.ranks, self.horizon)
        self._step = 0
        self._episode_return = 0.0
        self._records: list[ConcentrationFlip] = []

    def _make_native(self) -> gym.Env[Any, Any]:
        if self.variant in ("hard", "easy"):
            from amago.envs.builtin.popgym_envs import POPGym

            native_id = CONCENTRATION_NATIVE_IDS[self.variant]
            return cast(gym.Env[Any, Any], POPGym(native_id))
        return reduced_rank_native(self.ranks, self.copies if self.uneven else None)

    @property
    def native(self) -> gym.Env[Any, Any]:
        """The wrapped upstream task, for inspection and audits only."""
        return self._native

    # ------------------------------------------------------------------
    # Evaluator-only accessors
    # ------------------------------------------------------------------

    @property
    def flips(self) -> tuple[ConcentrationFlip, ...]:
        """Every flip of the live board, in order."""
        return tuple(self._records)

    def retrieval_targets(self) -> tuple[tuple[int, ...], int | None]:
        """Evaluator-only eligible partners and latest public reveal, before a flip."""
        if not self._board.in_play:
            return (), None
        first = self._board.in_play[0]
        rank = self._board.revealed[first]
        targets = tuple(
            p for p, r in self._board.known_hidden().items() if r == rank and p != first
        )
        latest = (
            max(cast(int, self._board.last_reveal[p]) for p in targets)
            if targets
            else None
        )
        return targets, latest

    @property
    def matched_pairs(self) -> int:
        return self._board.matched_pairs

    @property
    def board_complete(self) -> bool:
        return self._board.matched_pairs == self.pairs

    @property
    def episode_return(self) -> float:
        """Unscaled native return: matches paid at ``1/pairs`` minus the penalties."""
        return self._episode_return

    def decode(self, packet: Mapping[str, np.ndarray]) -> tuple[np.ndarray, float]:
        """Decode one packet's ``current`` token to (board, timer).

        The board holds the visible class per position, ``ranks`` for face down.
        """
        current = np.asarray(packet["current"], dtype=np.float32)
        if current.shape != (len(self.fields),):
            raise ContractError("Packet width disagrees with the public contract.")
        raw = (current + 1.0) / 2.0
        blocks = raw[:-1].reshape(self.cards, self.ranks + 1)
        if not np.allclose(blocks.sum(axis=1), 1.0) or not np.all(
            np.isclose(blocks, 0.0) | np.isclose(blocks, 1.0)
        ):
            raise ContractError("Concentration position field is not one-hot.")
        return np.argmax(blocks, axis=1).astype(np.int64), float(raw[-1])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _project(self, observation: object) -> np.ndarray:
        return project_unit_interval(observation, len(self.fields), label=self.label)

    def _begin_task(self, source: int) -> np.ndarray:
        observation, _ = self._native.reset(seed=source)
        current = self._project(observation)
        board, _ = self.decode({"current": current})
        self._board = PublicBoard(self.cards, self.ranks, self.horizon)
        self._board.begin(board)
        self._step = 0
        self._episode_return = 0.0
        self._records = []
        return current

    def _reset_info(self) -> dict[str, Any]:
        """Evaluator-side scalars only; the deck is never forwarded."""
        return {
            "evaluator_task_index": self.evaluator_task_index,
            "flip_index": 0,
            "matched_pairs": 0,
            "episode_return": 0.0,
            "board_complete": False,
        }

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Flip one position and score it against the public board."""
        position = self._validate_action(action)
        observation, reward, terminated, truncated, _ = self._native.step(position)
        self._step += 1
        current = self._project(observation)
        board, _ = self.decode({"current": current})
        outcome = self._board.apply(position, board)
        # The public board and the pinned reward must agree exactly; a
        # disagreement means the adapter has lost alignment with the game.
        if not np.isclose(reward, outcome.expected_reward, rtol=0, atol=1e-9):
            raise ContractError(
                "Public Concentration board disagrees with the native reward."
            )
        self._episode_return += float(reward)
        self._records.append(
            ConcentrationFlip(
                index=self._step,
                position=position,
                rank=outcome.revealed_rank,
                second=outcome.second,
                matched=outcome.matched,
                invalid=outcome.invalid,
                redundant=outcome.redundant,
                opportunity=outcome.opportunity,
                eligible=outcome.eligible,
                hit=outcome.hit,
                partner_last_reveal=outcome.partner_last_reveal,
                native_reward=float(reward),
            )
        )
        previous = self._current
        self._current = current
        self._done = bool(terminated or truncated)
        expected_done = self.board_complete or self._step >= self.horizon
        if self._done != expected_done:
            raise ContractError("Concentration ended away from its declared boundary.")
        decision = PublicDecision(
            current=self._current,
            previous=previous,
            outcome=self._current,
            executed_action=position,
            reward=float(reward),
            outer_terminated=terminated,
            outer_truncated=truncated,
        )
        info = {
            "evaluator_task_index": self.evaluator_task_index,
            "flip_index": self._step,
            "matched_pairs": self.matched_pairs,
            "episode_return": self._episode_return,
            "board_complete": self.board_complete,
        }
        return self._packet(decision), float(reward), terminated, truncated, info

    def close(self) -> None:
        cast(Any, self._native).close()

    # ------------------------------------------------------------------
    # Restorable state
    # ------------------------------------------------------------------

    def _identity(self) -> tuple[object, ...]:
        return (self.variant, self.cards, self.ranks, self.horizon, self.copies)

    def _state(self) -> dict[str, object]:
        native = cast(Any, self._native.unwrapped)
        popgym = cast(Any, self._native.env)
        deck = native.deck
        return {
            "board": self._board.state(),
            "step": self._step,
            "episode_return": self._episode_return,
            "records": [asdict(record) for record in self._records],
            "native": {
                "deck_idx": np.asarray(deck.idx).tolist(),
                "deck_len": int(deck.deck_len),
                "hands": {key: list(value) for key, value in deck.hands.items()},
                "state": np.asarray(native.state).tolist(),
                "curr_step": int(native.curr_step),
                "last_in_play_idx": list(native.last_in_play_idx),
                "obs": np.asarray(native.obs).tolist(),
                "timer": float(np.asarray(popgym.timer).reshape(-1)[0]),
                "rank_permutation": (
                    None
                    if getattr(native, "rank_permutation", None) is None
                    else np.asarray(native.rank_permutation).tolist()
                ),
            },
            "native_rng": deepcopy(native.np_random.bit_generator.state),
            "task_rng": deepcopy(self._native.generator_state()),
        }

    def _load_state(self, state: Mapping[str, object]) -> None:
        native_state = state.get("native")
        records = state.get("records")
        if not isinstance(native_state, Mapping) or not isinstance(records, Sequence):
            raise ContractError("Concentration checkpoint is malformed.")
        restored = [ConcentrationFlip(**cast(Any, row)) for row in records]
        native = cast(Any, self._native.unwrapped)
        popgym = cast(Any, self._native.env)
        deck = native.deck
        deck.idx = np.asarray(native_state["deck_idx"], dtype=np.int64)
        deck.deck_len = int(cast(Any, native_state["deck_len"]))
        deck.hands = {
            str(key): list(cast(Any, value))
            for key, value in cast(Mapping[str, Any], native_state["hands"]).items()
        }
        native.state = np.asarray(native_state["state"], dtype=np.int64)
        native.curr_step = int(cast(Any, native_state["curr_step"]))
        native.last_in_play_idx = list(cast(Any, native_state["last_in_play_idx"]))
        native.obs = np.asarray(native_state["obs"], dtype=np.int64)
        permutation = native_state.get("rank_permutation")
        if permutation is not None or self.uneven:
            if permutation is None:
                raise ContractError("An OOD Concentration board records its labels.")
            native.relabel(np.asarray(permutation, dtype=np.int64))
            native.obs = np.asarray(native_state["obs"], dtype=np.int64)
        popgym.timer = np.array([float(cast(Any, native_state["timer"]))])
        native.np_random.bit_generator.state = deepcopy(cast(Any, state["native_rng"]))
        self._native.load_generator_state(cast(Any, state["task_rng"]))
        self._board = PublicBoard(self.cards, self.ranks, self.horizon)
        self._board.load(cast(Mapping[str, object], state["board"]))
        self._step = int(cast(Any, state["step"]))
        self._episode_return = float(cast(Any, state["episode_return"]))
        self._records = restored


__all__ = [
    "CONCENTRATION_CARDS",
    "CONCENTRATION_COPIES",
    "CONCENTRATION_HORIZONS",
    "CONCENTRATION_NATIVE_IDS",
    "CONCENTRATION_OOD_COPIES",
    "CONCENTRATION_PROTOCOLS",
    "CONCENTRATION_RANKS",
    "CONCENTRATION_SUITS",
    "OOD_SPLIT",
    "ConcentrationEnv",
    "ConcentrationFlip",
    "FlipOutcome",
    "PublicBoard",
    "concentration_protocol",
    "concentration_variant",
    "deck_copies",
    "flip_budget",
    "public_fields",
    "reduced_rank_native",
]
