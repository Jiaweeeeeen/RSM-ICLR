"""Task-side information diagnostics for the C3 gate.

Each benchmark's C3 gate has two columns: a task-specific information
diagnostic and a neural evidence-use comparison. The diagnostics here are the
first column. They are **evaluation-only**: witness search and optimal labels
may use evaluator state, but input equality and history-based resolution use
only public streams. No learner weights, training targets or fit are involved.
They drive an AMAGO sequence wrapper over the public stream, which is why they
live in the runtime layer; the record they produce is the pure
:class:`~reasoned_icrl.experiments.qualification.TaskDiagnostic`.

That separation matters. A diagnostic that passes says the task can reward
history; it does not say any trained agent uses it. The neural comparison in
the study's revised gate is what tests the learner, and both must pass.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from amago.envs.amago_env import SequenceWrapper

from reasoned_icrl.environments.concentration import ConcentrationEnv, PublicBoard
from reasoned_icrl.environments.count_recall import CountRecallEnv, PublicStreamCounter
from reasoned_icrl.environments.dark_key_to_door import DarkKeyToDoorEnv
from reasoned_icrl.environments.darkroom import DarkRoomEnv
from reasoned_icrl.environments.mazerunner import (
    MAZERUNNER_ACTIONS,
    MazeRunnerEnv,
    decode_grid,
)
from reasoned_icrl.environments.tmaze import DOWN, FORWARD, UP, TMazeEnv
from reasoned_icrl.experiments.benchmarks import BenchmarkContract
from reasoned_icrl.experiments.config import ExperimentConfig
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import evaluation_environment
from reasoned_icrl.experiments.qualification import TaskDiagnostic
from reasoned_icrl.runtime.environments import amago_environment


# Equality includes every actual public packet field, AMAGO action/reward
# feedback and the absolute time index. No source/content hashes are computed.
def public_signature(sequence: Any) -> tuple[bytes, ...]:
    observation, rl2, time_index = sequence.current_timestep
    arrays = [observation[key] for key in sorted(observation)] + [rl2, time_index]
    return tuple(np.asarray(value).tobytes() for value in arrays)


def incompatible_actions(choices: Sequence[set[int]]) -> bool:
    """Tied actions in one state, or a shared optimal action, are not ambiguity."""
    return len(choices) > 1 and all(choices) and not set.intersection(*choices)


def _packet(sequence: Any) -> dict[str, np.ndarray]:
    observation, _, _ = sequence.current_timestep
    return {key: np.asarray(value)[0] for key, value in observation.items()}


def _step(sequence: Any, action: int) -> bool:
    _, _, terminated, truncated, _ = sequence.step(np.asarray([action]))
    return bool(np.logical_or(terminated, truncated).reshape(-1)[0])


def count_recall_diagnostic(
    environment: CountRecallEnv,
    *,
    stream_ids: Sequence[int],
    minimum_fraction: float = 0.5,
) -> TaskDiagnostic:
    """Compare complete decision inputs from valid, public-counted streams.

    Fixed answer zero creates reproducible, valid feedback histories. Answers
    are reconstructed from the public dealt values, never native hidden counts.
    Failure to find enough witnesses is unresolved evidence, not a memoryless
    solution. Easy/Hard retain the legacy fraction threshold; Medium uses the
    predeclared verified-pair/task coverage rule, with the fraction descriptive.
    """
    if not stream_ids:
        raise ContractError("The CountRecall diagnostic needs streams to run.")
    sequence = SequenceWrapper(
        amago_environment(environment, name="CountDiagnostic"),
        save_trajs_to=None,
        save_every=None,
    )
    counts: dict[tuple[bytes, ...], set[int]] = {}
    witnesses: dict[tuple[bytes, ...], list[tuple[int, int]]] = {}
    decisions = 0
    for stream_id in stream_ids:
        environment.set_task(int(stream_id))
        sequence.reset()
        counter = PublicStreamCounter(environment.categories)
        done = False
        while not done:
            value, query, _ = environment.decode(_packet(sequence))
            truth = counter.observe(value, query)
            signature = public_signature(sequence)
            counts.setdefault(signature, set()).add(truth)
            witnesses.setdefault(signature, []).append((int(stream_id), truth))
            decisions += 1
            done = _step(sequence, 0)
    ambiguous = sum(len(answers) > 1 for answers in counts.values())
    fraction = ambiguous / max(len(counts), 1)
    pairs = 0
    paired_tasks: set[int] = set()
    for group in witnesses.values():
        for index, (task, truth) in enumerate(group):
            for other_task, other_truth in group[index + 1 :]:
                if task != other_task and truth != other_truth:
                    pairs += 1
                    paired_tasks.update((task, other_task))
    return TaskDiagnostic(
        benchmark="count_recall",
        question=(
            "Can identical complete current inputs require different public counts?"
        ),
        passed=(pairs >= 32 and len(paired_tasks) >= 16)
        if environment.variant == "medium"
        else fraction >= minimum_fraction,
        measurements={
            "verified_pairs": float(pairs),
            "tasks_in_verified_pairs": float(len(paired_tasks)),
            "streams": float(len(stream_ids)),
            "scored_decisions": float(decisions),
            "distinct_current_packets": float(len(counts)),
            "ambiguous_packets": float(ambiguous),
            "ambiguous_packet_fraction": fraction,
            (
                "legacy_fraction_threshold_not_applied"
                if environment.variant == "medium"
                else "minimum_fraction"
            ): minimum_fraction,
        },
        statement=(
            f"{ambiguous}/{len(counts)} complete input groups contain different "
            "public-history answers, including equality of current/previous/outcome, "
            "event/valid, RL2 and time. This establishes ambiguity only at witnessed "
            "decisions; the public counter resolves their answers. A failed threshold "
            "means insufficient diagnostic evidence, not a memoryless solution."
            + (
                " Medium uses at least 32 verified pairs over 16 streams; the "
                "legacy fraction threshold is not applied."
                if environment.variant == "medium"
                else ""
            )
        ),
    )


def concentration_diagnostic(
    environment: ConcentrationEnv,
    *,
    board_ids: Sequence[int],
    minimum_fraction: float = 0.5,
    minimum_witnesses: int = 5,
) -> TaskDiagnostic:
    """Compare complete decision inputs across boards with different histories.

    A fixed exploration policy flips positions 0..25 twice in a row, then again
    (104 flips): the same position twice never matches, so no board's matched
    set diverges, and every second flip of the second pass is a partner
    selection whose eligible positions were revealed in the first pass. Across
    boards the visible board, timer and RL2 feedback at such a decision are
    identical whenever the two most recently shown ranks coincide, while the
    public history — where the partners of that rank lie — is not. A decision is
    a witness when a card is in play and a partner of its rank is known and
    hidden: its *determined* correct set is those eligible positions (the
    review's partner-selection diagnostic). Identical inputs whose determined
    sets are incompatible across boards are the ambiguity the task must resolve
    from history. Failure to find enough witnesses is unresolved evidence, not
    a memoryless solution.
    """
    if not board_ids:
        raise ContractError("The Concentration diagnostic needs boards to run.")
    sequence = SequenceWrapper(
        amago_environment(environment, name="ConcentrationDiagnostic"),
        save_trajs_to=None,
        save_every=None,
    )
    cards = environment.cards
    groups: dict[tuple[bytes, ...], list[set[int]]] = {}
    paired_groups: dict[tuple[bytes, ...], list[tuple[int, set[int]]]] = {}
    decisions = witnesses = 0
    for board_id in board_ids:
        environment.set_task(int(board_id))
        sequence.reset()
        board = PublicBoard(cards, environment.ranks, environment.horizon)
        board.begin(environment.decode(_packet(sequence))[0])
        done = False
        flip = 0
        while not done:
            known = board.known_hidden()
            correct: set[int] = set()
            if board.in_play:
                # A partner selection: the eligible positions of the card in play.
                rank = board.revealed[board.in_play[0]]
                correct = {
                    p for p, r in known.items() if r == rank and p != board.in_play[0]
                }
            if correct:
                signature = public_signature(sequence)
                groups.setdefault(signature, []).append(correct)
                paired_groups.setdefault(signature, []).append((int(board_id), correct))
                witnesses += 1
            action = (flip // 2) % (cards // 2)
            decisions += 1
            done = _step(sequence, action)
            board.apply(action, environment.decode(_packet(sequence))[0])
            flip += 1
    shared = {key: sets for key, sets in groups.items() if len(sets) > 1}
    ambiguous = sum(incompatible_actions(sets) for sets in shared.values())
    fraction = ambiguous / max(len(shared), 1)
    verified_pairs = 0
    paired_tasks: set[int] = set()
    for values in paired_groups.values():
        for i, (task, actions) in enumerate(values):
            for other_task, other_actions in values[i + 1 :]:
                if task != other_task and actions.isdisjoint(other_actions):
                    verified_pairs += 1
                    paired_tasks.update((task, other_task))
    return TaskDiagnostic(
        benchmark="concentration",
        question=(
            "Can identical complete current inputs require different partner flips?"
        ),
        passed=fraction >= minimum_fraction and len(shared) >= minimum_witnesses,
        measurements={
            "boards": float(len(board_ids)),
            "verified_pairs": float(verified_pairs),
            "tasks_in_verified_pairs": float(len(paired_tasks)),
            "decisions": float(decisions),
            "determined_decisions": float(witnesses),
            "distinct_current_packets": float(len(groups)),
            "shared_packets": float(len(shared)),
            "ambiguous_packets": float(ambiguous),
            "ambiguous_packet_fraction": fraction,
            "minimum_fraction": minimum_fraction,
            "minimum_witnesses": float(minimum_witnesses),
        },
        statement=(
            f"{ambiguous}/{len(shared)} complete input groups seen on more than one "
            "board need incompatible flips, including equality of current/previous/"
            "outcome, event/valid, RL2 and time. This establishes ambiguity only at "
            "witnessed decisions; the public board resolves their answers. A failed "
            "threshold means insufficient diagnostic evidence, not a memoryless "
            "solution."
        ),
    )


_DIRECTIONS = np.asarray([(0, -1), (-1, 0), (0, 1), (1, 0), (0, 0)])


def xland_diagnostic(
    environment: Any,
    *,
    task_ids: Sequence[int],
    minimum_fraction: float = 0.5,
    minimum_witnesses: int = 5,
) -> TaskDiagnostic:
    """Does the public goal determine the hidden rules? Measured on the benchmark.

    XLand-MiniGrid's packet carries the goal encoding and never the rule
    encoding (decision 15). For every development ruleset, count the rulesets
    of the whole pinned benchmark that share its goal encoding exactly and
    differ in their rule encoding. A development ruleset is a witness when its
    goal admits at least one other rule set: no function of the current input
    alone can then know which object interactions produce the goal's objects,
    so that knowledge has to come from the task's own history. This is
    distribution-level ambiguity, deterministic and evaluator-side; it is not
    an input-level witness (identical complete inputs needing different
    actions), which on this task is the post-qualification interaction ledger
    of the plan (section 5.5). Unresolved rulesets are reported, never assumed.
    """
    from reasoned_icrl.environments.xland_minigrid import (
        benchmark_arrays,
        ruleset_index,
    )

    if not task_ids:
        raise ContractError("The XLand diagnostic needs tasks to run.")
    goals, rules = benchmark_arrays()
    goal_keys = (
        np.ascontiguousarray(goals)
        .view(np.dtype((np.void, goals.dtype.itemsize * goals.shape[1])))
        .reshape(-1)
    )
    rule_keys = (
        np.ascontiguousarray(rules)
        .view(np.dtype((np.void, rules.dtype.itemsize * rules.shape[1])))
        .reshape(-1)
    )
    _, goal_groups = np.unique(goal_keys, return_inverse=True)
    goal_groups = goal_groups.reshape(-1)
    # Distinct rule encodings per goal group, over the whole benchmark.
    pairs = np.unique(
        np.stack(
            [goal_groups, np.unique(rule_keys, return_inverse=True)[1].reshape(-1)], 1
        ),
        axis=0,
    )
    distinct_rules = np.bincount(pairs[:, 0], minlength=int(goal_groups.max()) + 1)
    rulesets_per_goal = np.bincount(goal_groups, minlength=int(goal_groups.max()) + 1)
    witnesses = 0
    admitted: list[float] = []
    for task_id in task_ids:
        group = int(goal_groups[ruleset_index(int(task_id))])
        admitted.append(float(distinct_rules[group]))
        if distinct_rules[group] > 1:
            witnesses += 1
    fraction = witnesses / len(task_ids)
    return TaskDiagnostic(
        benchmark="xland_minigrid",
        question="Does the public goal encoding determine the hidden rules?",
        passed=fraction >= minimum_fraction and witnesses >= minimum_witnesses,
        measurements={
            "tasks": float(len(task_ids)),
            "witnesses": float(witnesses),
            "witness_fraction": fraction,
            "distinct_rule_sets_per_goal_mean": float(np.mean(admitted)),
            "distinct_rule_sets_per_goal_min": float(np.min(admitted)),
            "benchmark_rulesets": float(len(goals)),
            "benchmark_distinct_goals": float(len(distinct_rules)),
            "benchmark_rulesets_per_goal_mean": float(np.mean(rulesets_per_goal)),
            "minimum_fraction": minimum_fraction,
            "minimum_witnesses": float(minimum_witnesses),
        },
        statement=(
            f"{witnesses}/{len(task_ids)} development rulesets have a goal encoding "
            "that the pinned benchmark pairs with more than one rule encoding "
            f"(mean {float(np.mean(admitted)):.1f} distinct rule sets per goal); the "
            "rule encoding is in no packet field, so the current input alone cannot "
            "know the interaction consequences the goal depends on. Distribution-level "
            "ambiguity only: input-level witnesses are the interaction ledger."
        ),
    )


def _towards(position: tuple[int, int], goal: tuple[int, int]) -> set[int]:
    if position == goal:
        return {4}  # Native key/door checks occur on a step, never at reset.
    return {
        a
        for a, d in enumerate(_DIRECTIONS)
        if _manhattan((position[0] + int(d[0]), position[1] + int(d[1])), goal)
        < _manhattan(position, goal)
    }


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _discovery_path(
    start: tuple[int, int],
    key: tuple[int, int],
    door: tuple[int, int],
    size: int,
    horizon: int,
) -> list[int]:
    """Evaluator-only witness search: first door completion exactly at horizon.

    Search chooses a valid action trace, not a policy. Nothing from this search
    is passed to the subsequent public-history recall controller.
    """
    initial = (*start, False)
    parents: list[dict[tuple[int, int, bool], tuple[tuple[int, int, bool], int]]] = []
    frontier = {initial}
    terminal: tuple[int, int, bool] | None = None
    for step in range(1, horizon + 1):
        next_states: dict[tuple[int, int, bool], tuple[tuple[int, int, bool], int]] = {}
        for state in sorted(frontier):
            for action, move in enumerate(_DIRECTIONS):
                position = (
                    max(0, min(size - 1, state[0] + int(move[0]))),
                    max(0, min(size - 1, state[1] + int(move[1]))),
                )
                success = state[2] and position == door
                if success != (step == horizon):
                    continue
                after = (*position, state[2] or position == key)
                next_states.setdefault(after, (state, action))
                if success:
                    terminal = after
        parents.append(next_states)
        frontier = set(next_states)
    if terminal is None:
        return []
    actions: list[int] = []
    state = terminal
    for previous in reversed(parents):
        state, action = previous[state]
        actions.append(action)
    return actions[::-1]


def key_to_door_diagnostic(
    environment: DarkKeyToDoorEnv,
    *,
    task_ids: Sequence[int],
    generator_seed: int = 0,
) -> TaskDiagnostic:
    """Elicit valid discovery traces, then measure public-only recall.

    An evaluator-only search schedules discovery success at physical time H,
    so all subsequent reset decisions occur at H+1. Equal starts can then be
    compared with equality of the *entire* packet, RL2 and global time.
    Hidden coordinates are used only to construct the witness prefix; the
    subsequent controller reconstructs both locations from public events.
    This is conditional trace sufficiency/efficient-route evidence, never an
    exploration-policy result or a universal memoryless success bound.
    """
    if not task_ids:
        raise ContractError("The Key-to-Door diagnostic needs tasks to run.")
    sequence = SequenceWrapper(
        amago_environment(environment, name="DoorDiagnostic"),
        save_trajs_to=None,
        save_every=None,
    )
    grouped: dict[tuple[bytes, ...], list[set[int]]] = {}
    group_tasks: dict[tuple[bytes, ...], list[int]] = {}
    discovered = solved = 0
    fits: list[bool] = []
    for task_id in task_ids:
        environment.set_task(int(task_id))
        sequence.reset(seed=generator_seed)
        native = cast(Any, environment.native.unwrapped)
        path = _discovery_path(
            (int(native.start[0]), int(native.start[1])),
            (int(native.key[0]), int(native.key[1])),
            (int(native.goal[0]), int(native.goal[1])),
            environment.size,
            environment.horizon,
        )
        if not path:
            continue
        key: tuple[int, int] | None = None
        door: tuple[int, int] | None = None
        previous_has_key = False
        for action in path:
            if _step(sequence, action):
                raise ContractError("Discovery witness exceeded the outer budget.")
            public = environment.public_fields(_packet(sequence))
            position = (
                round(public["position_x"] * environment.size),
                round(public["position_y"] * environment.size),
            )
            has_key = public["has_key"] > 0.5
            _, rl2, _ = sequence.current_timestep
            reward = float(np.asarray(rl2).reshape(-1)[0])
            if has_key and not previous_has_key:
                key = position
            if reward > 0 and previous_has_key and has_key:
                door = position
            previous_has_key = has_key
        if key is None or door is None:
            continue
        discovered += 1
        _step(sequence, 4)  # The native reset-only decision executes no movement.
        public = environment.public_fields(_packet(sequence))
        start = (
            round(public["position_x"] * environment.size),
            round(public["position_y"] * environment.size),
        )
        choices = _towards(start, key)
        signature = public_signature(sequence)
        grouped.setdefault(signature, []).append(choices)
        group_tasks.setdefault(signature, []).append(int(task_id))
        length = max(1, _manhattan(start, key)) + max(1, _manhattan(key, door))
        fits.append(length <= environment.horizon)
        # No native state access from this point: execute only remembered public
        # locations, the current public position and possession bit.
        for _ in range(environment.horizon):
            public = environment.public_fields(_packet(sequence))
            position = (
                round(public["position_x"] * environment.size),
                round(public["position_y"] * environment.size),
            )
            has_key = public["has_key"] > 0.5
            done = _step(sequence, min(_towards(position, door if has_key else key)))
            _, rl2, _ = sequence.current_timestep
            if has_key and float(np.asarray(rl2).reshape(-1)[0]) > 0:
                solved += 1
                break
            if done:
                break
    contested = sum(incompatible_actions(choices) for choices in grouped.values())
    # R5 coverage: verified pairs are two tasks with identical complete inputs
    # whose efficient-route action sets are disjoint; the tasks they involve
    # are counted once each.
    pairs = 0
    paired_tasks: set[int] = set()
    for signature, action_sets in grouped.items():
        members = group_tasks[signature]
        for i in range(len(action_sets)):
            for j in range(i + 1, len(action_sets)):
                left, right = action_sets[i], action_sets[j]
                if left and right and not (left & right):
                    pairs += 1
                    paired_tasks.update((members[i], members[j]))
    fit_fraction = float(np.mean(fits)) if fits else 0.0
    return TaskDiagnostic(
        benchmark="dark_key_to_door",
        question="Do public discoveries resolve identical-input later route decisions?",
        passed=bool(solved > 0 and fit_fraction >= 0.99 and contested > 0),
        measurements={
            "tasks": float(len(task_ids)),
            "public_discovery_tasks": float(discovered),
            "public_recall_success_tasks": float(solved),
            "route_within_limit_fraction": fit_fraction,
            "distinct_current_packets": float(len(grouped)),
            "incompatible_packet_groups": float(contested),
            "verified_pairs": float(pairs),
            "tasks_in_verified_pairs": float(len(paired_tasks)),
        },
        statement=(
            f"Evaluator-elicited valid discovery traces revealed both locations in "
            f"{discovered} tasks; the public-only recall controller solved {solved} "
            f"later attempts. {contested} full-input groups at aligned reset time "
            "require incompatible efficient-route actions. This is conditional "
            "trace evidence, not proof that every memoryless policy fails or that "
            "a learner discovers these traces. Hidden witness-search labels never "
            "enter the recall controller or any learner input."
        ),
    )


def darkroom_diagnostic(
    environment: DarkRoomEnv, *, task_ids: Sequence[int]
) -> TaskDiagnostic:
    """Does the first attempt's public outcome resolve the second attempt's route?

    The evaluator walks a shortest path to the hidden goal in attempt one, using
    the goal only to construct that witness. At the start of attempt two the
    current token is identical for every goal at the same distance, yet the
    optimal first move differs; a controller that remembers the public
    ``outcome`` packet of the discovery step solves attempt two from public
    history alone. RL2 feedback is deliberately left out of the signature: it
    repeats the final discovery move, which is always one valid first move, so
    including it can never produce an incompatible group.
    """
    if not task_ids:
        raise ContractError("The DarkRoom diagnostic needs tasks to run.")
    sequence = SequenceWrapper(
        amago_environment(environment, name="DarkRoomDiagnostic"),
        save_trajs_to=None,
        save_every=None,
    )
    grouped: dict[tuple[bytes, ...], list[set[int]]] = {}
    solved = 0
    for task_id in task_ids:
        environment.set_task(int(task_id))
        sequence.reset(seed=0)
        goal = environment.goal
        position = environment.public_fields(_packet(sequence))["position"]
        remembered: tuple[int, int] | None = None
        done = False
        while position != goal and not done:
            done = _step(sequence, min(_towards(position, goal)))
            fields = environment.public_fields(_packet(sequence))
            position = fields["position"]
            if fields["attempt_boundary"]:
                # The discovery step's public endpoint is the goal cell itself.
                outcome = _packet(sequence)["outcome"]
                remembered = environment.public_fields({"current": outcome})["position"]
                position = remembered
                break
        if remembered is None:
            continue
        observation, _, _ = sequence.current_timestep
        signature = (np.asarray(observation["current"]).tobytes(),)
        start = environment.public_fields(_packet(sequence))["position"]
        grouped.setdefault(signature, []).append(_towards(start, remembered))
        # Public-only recall: steer towards the remembered outcome cell.
        for _ in range(environment.horizon):
            here = environment.public_fields(_packet(sequence))["position"]
            if here == remembered:
                solved += 1
                break
            if _step(sequence, min(_towards(here, remembered))):
                break
            fields = environment.public_fields(_packet(sequence))
            if fields["attempt_boundary"]:
                solved += 1
                break
    contested = sum(incompatible_actions(choices) for choices in grouped.values())
    return TaskDiagnostic(
        benchmark="darkroom",
        question="Does the discovery outcome resolve identical-input later moves?",
        passed=bool(solved > 0 and contested > 0),
        measurements={
            "tasks": float(len(task_ids)),
            "public_recall_success_tasks": float(solved),
            "distinct_current_inputs": float(len(grouped)),
            "incompatible_input_groups": float(contested),
        },
        statement=(
            f"{solved} of {len(task_ids)} tasks were solved in attempt two from the "
            f"remembered public outcome alone; {contested} current-token groups "
            "required incompatible first moves."
        ),
    )


def mazerunner_diagnostic(
    environment: MazeRunnerEnv,
    *,
    map_ids: Sequence[int],
    minimum_fraction: float = 0.05,
    generator_seed: int = 0,
) -> TaskDiagnostic:
    """Seek ambiguity on actual bounded trajectories, with public resolution.

    Full current inputs include the timer and preceding transition. Native maps
    supply evaluator-only optimal-action labels. Public wall rays and observed
    action displacements must independently reconstruct an optimal route before
    a history is credited with resolving it. No invented unreachable states.
    """
    if not map_ids:
        raise ContractError("The MazeRunner diagnostic needs maps to run.")
    rng = np.random.default_rng(generator_seed)
    sequence = SequenceWrapper(
        amago_environment(environment, name="MazeDiagnostic"),
        save_trajs_to=None,
        save_every=None,
    )
    grouped: dict[tuple[bytes, ...], list[tuple[int, set[int], bool]]] = {}
    states = 0
    for map_id in map_ids:
        environment.set_task(int(map_id))
        sequence.reset()
        native = cast(Any, environment.native.unwrapped)
        maze = np.asarray(native.maze)
        dirs = np.asarray(native.action_dirs)
        known = np.ones_like(maze)  # Unseen cells are not certified traversable.
        moves: dict[int, tuple[int, int]] = {}
        previous_position: tuple[int, int] | None = None
        previous_action = 0
        done = False
        while not done:
            packet = _packet(sequence)
            current = np.asarray(packet["current"])
            values = decode_grid(current[:6], environment.size)
            position = (int(values[0]), int(values[1]))
            known[position] = 0
            if previous_position is not None and position != previous_position:
                moves[previous_action] = (
                    position[0] - previous_position[0],
                    position[1] - previous_position[1],
                )
            for distance, move in zip(values[2:6], _DIRECTIONS[:4], strict=True):
                for offset in range(1, int(distance) + 1):
                    cell = (
                        position[0] + offset * int(move[0]),
                        position[1] + offset * int(move[1]),
                    )
                    known[cell] = 0
            goals = decode_grid(current[7:], environment.size).reshape(-1, 2)
            remaining = [tuple(int(x) for x in goal) for goal in goals if goal[0] >= 0]
            if remaining:
                goal = cast(tuple[int, int], remaining[0])
                distance = _flood(maze, goal, environment.size)
                best = _optimal_actions(
                    maze, distance, position, dirs, environment.size
                )
                public_distance = _flood(known, goal, environment.size)
                public_best = {
                    action
                    for action, move in moves.items()
                    if public_distance[position] > 0
                    and public_distance[(position[0] + move[0], position[1] + move[1])]
                    == public_distance[position] - 1
                }
                resolved = bool(
                    public_best
                    and public_best <= best
                    and public_distance[position] == distance[position]
                )
                grouped.setdefault(public_signature(sequence), []).append(
                    (int(map_id), best, resolved)
                )
                states += 1
            previous_position = position
            previous_action = int(rng.integers(MAZERUNNER_ACTIONS))
            done = _step(sequence, previous_action)
    contested = resolved_groups = repeated = 0
    for records in grouped.values():
        different_maps = len({record[0] for record in records}) > 1
        repeated += int(different_maps)
        conflict = different_maps and incompatible_actions([r[1] for r in records])
        contested += int(conflict)
        resolved_groups += int(conflict and all(r[2] for r in records))
    fraction = resolved_groups / max(len(grouped), 1)
    return TaskDiagnostic(
        benchmark="mazerunner",
        question=(
            "Do observed public histories resolve incompatible identical-input "
            "route decisions?"
        ),
        passed=fraction >= minimum_fraction,
        measurements={
            "maps": float(len(map_ids)),
            "reachable_states": float(states),
            "distinct_current_packets": float(len(grouped)),
            "cross_map_repeated_packets": float(repeated),
            "ambiguous_packets": float(contested),
            "publicly_resolved_ambiguous_packets": float(resolved_groups),
            "ambiguous_packet_fraction": fraction,
            "minimum_fraction": minimum_fraction,
        },
        statement=(
            f"On valid bounded random trajectories, {contested}/{len(grouped)} "
            f"complete-input groups have incompatible optimal actions across maps; "
            f"{resolved_groups} are also resolved by accumulated public wall rays and "
            "observed controls. Tied optimal routes alone do not count. Insufficient "
            "witnesses leave C3 unresolved; they do not prove history is unnecessary."
        ),
    )


def _flood(maze: np.ndarray, goal: tuple[int, int], size: int) -> np.ndarray:
    """Breadth-first distance to the goal over open squares; -1 is unreachable."""
    distance = np.full((size, size), -1, dtype=np.int64)
    if maze[goal] != 0:
        return distance
    distance[goal] = 0
    queue: deque[tuple[int, int]] = deque([goal])
    while queue:
        row, column = queue.popleft()
        for step_row, step_column in ((0, -1), (-1, 0), (0, 1), (1, 0)):
            near_row, near_column = row + step_row, column + step_column
            if not (0 <= near_row < size and 0 <= near_column < size):
                continue
            if maze[near_row, near_column] != 0:
                continue
            if distance[near_row, near_column] >= 0:
                continue
            distance[near_row, near_column] = distance[row, column] + 1
            queue.append((near_row, near_column))
    return distance


def _optimal_actions(
    maze: np.ndarray,
    distance: np.ndarray,
    position: tuple[int, int],
    dirs: np.ndarray,
    size: int,
) -> set[int]:
    """Native action indices that strictly reduce the distance to the goal."""
    best: set[int] = set()
    here = distance[position]
    for action in range(MAZERUNNER_ACTIONS):
        move = dirs[action]
        row, column = position[0] + int(move[0]), position[1] + int(move[1])
        if not (0 <= row < size and 0 <= column < size) or maze[row, column] != 0:
            continue
        if distance[row, column] >= 0 and distance[row, column] == here - 1:
            best.add(action)
    return best


def match_pattern_diagnostic(
    environment: Any, *, task_ids: Sequence[int]
) -> TaskDiagnostic:
    """Verify aliases of every packet field, RL2 and time; resolve from symbols."""
    from reasoned_icrl.experiments.match_pattern import pattern_indices

    sequence = SequenceWrapper(
        amago_environment(environment, name="MatchDiagnostic"),
        save_trajs_to=None,
        save_every=None,
    )
    groups: dict[tuple[bytes, ...], list[tuple[int, int]]] = {}
    for task in task_ids:
        environment.set_task(int(task))
        sequence.reset()
        objects: list[int] = []
        for _ in range(6):
            symbol = np.flatnonzero(_packet(sequence)["current"][:64] == 1)
            if symbol.size != 1:
                raise ContractError("Alias witness has no single revealed symbol.")
            objects.append(int(symbol[0]))
            assert not _step(sequence, 0)
        left, right = pattern_indices(objects)
        truth = int(left == right)
        groups.setdefault(public_signature(sequence), []).append((int(task), truth))
        assert _step(sequence, truth)
        assert environment.scored_decision.correct
    pairs = 0
    covered: set[int] = set()
    for group in groups.values():
        positives = [task for task, label in group if label]
        negatives = [task for task, label in group if not label]
        pairs += len(positives) * len(negatives)
        if positives and negatives:
            covered.update(positives + negatives)
    return TaskDiagnostic(
        "match_pattern",
        "Do identical complete inputs require opposite answers?",
        pairs >= 32 and len(covered) >= 16,
        {
            "verified_pairs": float(pairs),
            "tasks_in_verified_pairs": float(len(covered)),
            "examples": float(len(task_ids)),
            "distinct_current_packets": float(len(groups)),
        },
        "Verified all five packet fields, canonical RL2 and absolute time; "
        "opposite labels reconstructed from the six public symbols.",
    )


def tmaze_diagnostic(
    environment: TMazeEnv, *, task_ids: Sequence[int]
) -> TaskDiagnostic:
    """Does the first observation resolve the identical-input junction decision?

    The evaluator walks the corridor with the forward move, so every task
    arrives at the junction with the same complete policy input: the same
    packet ([1, 0] with the forward move as its physical evidence), the same
    RL2 feedback (reward 0, action forward) and the same time index. The
    correct turn is the cue shown at reset, read from the public reset
    observation alone. A verified pair is an up-cue task and a down-cue task
    with identical junction inputs; the tasks it involves are counted once.
    """
    if not task_ids:
        raise ContractError("The T-Maze diagnostic needs tasks to run.")
    sequence = SequenceWrapper(
        amago_environment(environment, name="TMazeDiagnostic"),
        save_trajs_to=None,
        save_every=None,
    )
    groups: dict[tuple[bytes, ...], list[tuple[int, int]]] = {}
    solved = 0
    for task in task_ids:
        environment.set_task(int(task))
        sequence.reset(seed=0)
        cue = environment.public_fields(_packet(sequence))["cue_or_lateral"]
        if cue not in (-1, 1):
            raise ContractError("The reset observation must show the cue.")
        done = False
        while (
            not done and not environment.public_fields(_packet(sequence))["at_junction"]
        ):
            done = _step(sequence, FORWARD)
        if done:
            raise ContractError("The corridor cannot exhaust the budget.")
        groups.setdefault(public_signature(sequence), []).append((int(task), cue))
        done = _step(sequence, UP if cue == 1 else DOWN)
        assert done and environment.episode is not None
        solved += int(environment.episode.success)
    pairs = 0
    covered: set[int] = set()
    for group in groups.values():
        ups = [task for task, cue in group if cue == 1]
        downs = [task for task, cue in group if cue == -1]
        pairs += len(ups) * len(downs)
        if ups and downs:
            covered.update(ups + downs)
    return TaskDiagnostic(
        benchmark="tmaze",
        question="Does the first observation resolve identical junction inputs?",
        passed=bool(solved == len(task_ids) and pairs >= 32 and len(covered) >= 16),
        measurements={
            "tasks": float(len(task_ids)),
            "public_recall_success_tasks": float(solved),
            "distinct_junction_inputs": float(len(groups)),
            "verified_pairs": float(pairs),
            "tasks_in_verified_pairs": float(len(covered)),
        },
        statement=(
            f"After {environment.corridor_length} forward moves every task presents "
            f"{len(groups)} distinct complete junction input(s) (packet, RL2, time); "
            f"{pairs} up/down pairs over {len(covered)} tasks require opposite turns "
            f"and the public-cue controller solved {solved} of {len(task_ids)}. "
            "The budget has no spare step, so no corridor action can encode the "
            "cue; only carried history separates the pairs."
        ),
    )


def task_diagnostic(
    contract: BenchmarkContract,
    config: ExperimentConfig,
    *,
    task_cap: int | None = None,
    generator_seed: int = 0,
) -> TaskDiagnostic:
    """Run the benchmark's declared task-side information diagnostic."""
    roster = contract.roster("development")
    if task_cap is not None:
        roster = roster[:task_cap]
    name = contract.environment.name
    environment = evaluation_environment(contract, config, split="development", seed=0)
    try:
        if name == "match_pattern":
            return match_pattern_diagnostic(environment, task_ids=roster)
        if name == "count_recall":
            return count_recall_diagnostic(cast(Any, environment), stream_ids=roster)
        if name == "concentration":
            return concentration_diagnostic(cast(Any, environment), board_ids=roster)
        if name == "mazerunner":
            return mazerunner_diagnostic(cast(Any, environment), map_ids=roster)
        if name == "darkroom":
            return darkroom_diagnostic(cast(Any, environment), task_ids=roster)
        if name == "xland_minigrid":
            return xland_diagnostic(cast(Any, environment), task_ids=roster)
        if name == "tmaze":
            return tmaze_diagnostic(cast(Any, environment), task_ids=roster)
        if name == "xland_one_rule":
            raise ContractError(
                "The one-rule controlled-history diagnostic belongs to its "
                "evaluator stage and is not implemented yet."
            )
        return key_to_door_diagnostic(
            cast(Any, environment), task_ids=roster, generator_seed=generator_seed
        )
    finally:
        environment.close()


__all__ = [
    "concentration_diagnostic",
    "count_recall_diagnostic",
    "darkroom_diagnostic",
    "incompatible_actions",
    "key_to_door_diagnostic",
    "mazerunner_diagnostic",
    "public_signature",
    "task_diagnostic",
    "tmaze_diagnostic",
    "xland_diagnostic",
]
