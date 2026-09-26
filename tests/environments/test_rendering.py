"""The observer frames of the three paper benchmarks and the shared helpers.

A frame is checked against the hidden state it draws (the native layout, the
public counts, the native position), never the other way round: rendering
reads state, it never writes any, and the packet the policy receives is the
same with or without a ``render_mode``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from reasoned_icrl.environments.base import DEVELOPMENT_TASKS
from reasoned_icrl.environments.concentration import ConcentrationEnv
from reasoned_icrl.environments.count_recall import (
    COUNT_RECALL_SYMBOLS,
    RENDER_STREAM_WIDTH,
    CountRecallEnv,
)
from reasoned_icrl.environments.dark_key_to_door import (
    KEY_TO_DOOR_ACTION_NAMES,
    DarkKeyToDoorEnv,
)
from reasoned_icrl.environments.rendering import (
    CELL_PIXELS,
    PALETTE,
    RENDER_MODES,
    color_grid,
    paint_cells,
    stack_frames,
)
from reasoned_icrl.environments.tmaze import (
    DOWN,
    FORWARD,
    RENDER_COLUMNS,
    TMAZE_ACTION_NAMES,
    UP,
    TMazeEnv,
)
from reasoned_icrl.experiments.contracts import ContractError

TASK = DEVELOPMENT_TASKS[0]


def grid(frame: str, header_lines: int) -> list[list[str]]:
    return [row.split() for row in frame.splitlines()[header_lines:]]


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def test_paint_cells_scales_and_frames_the_board() -> None:
    board = color_grid(2, 3, PALETTE["floor"])
    board[1, 2] = PALETTE["agent"]
    framed = paint_cells(board, cell_pixels=4)
    assert framed.shape == (2 * 4 + 1, 3 * 4 + 1, 3) and framed.dtype == np.uint8
    assert tuple(framed[1 * 4 + 2, 2 * 4 + 2]) == PALETTE["agent"]
    assert tuple(framed[0, 0]) == PALETTE["grid"]
    bare = paint_cells(board, cell_pixels=4, grid=None)
    assert bare.shape == (8, 12, 3)
    with pytest.raises(ValueError):
        paint_cells(board.astype(np.float32), cell_pixels=4)
    with pytest.raises(ValueError):
        paint_cells(board, cell_pixels=1)


def test_stack_frames_pads_to_the_widest() -> None:
    narrow = color_grid(2, 3, PALETTE["agent"])
    wide = color_grid(1, 5, PALETTE["key"])
    stacked = stack_frames([narrow, wide], gap=2)
    assert stacked.shape == (2 + 2 + 1, 5, 3)
    assert tuple(stacked[0, 4]) == PALETTE["background"]
    assert tuple(stacked[4, 4]) == PALETTE["key"]


# ----------------------------------------------------------------------
# The dispatcher
# ----------------------------------------------------------------------


def test_render_modes_are_declared_and_checked() -> None:
    assert DarkKeyToDoorEnv.metadata["render_modes"] == list(RENDER_MODES)
    with pytest.raises(ContractError, match="render modes"):
        DarkKeyToDoorEnv(split="development", render_mode="svg")
    env = DarkKeyToDoorEnv(split="development")
    with pytest.raises(ContractError, match="reset before rendering"):
        env.render()
    env.reset(options={"task_index": TASK})
    with pytest.raises(ContractError, match="render modes"):
        env.render("svg")


def test_human_mode_prints_the_text_frame(capsys: pytest.CaptureFixture[str]) -> None:
    env = DarkKeyToDoorEnv(split="development", render_mode="human")
    env.reset(options={"task_index": TASK})
    assert env.render() is None
    assert capsys.readouterr().out.strip() == env.render("ansi").strip()


def test_an_environment_without_frames_refuses_cleanly() -> None:
    env = ConcentrationEnv(variant="easy", split="development")
    env.reset(options={"task_index": TASK})
    with pytest.raises(ContractError, match="does not render"):
        env.render()


def test_rendering_never_changes_the_packet() -> None:
    plain = DarkKeyToDoorEnv(split="development", initial_seed=0)
    drawn = DarkKeyToDoorEnv(split="development", initial_seed=0, render_mode="ansi")
    first, _ = plain.reset(options={"task_index": TASK})
    second, _ = drawn.reset(options={"task_index": TASK})
    drawn.render()
    for action in (0, 1, 2, 3, 4, 2):
        first = plain.step(action)[0]
        second = drawn.step(action)[0]
        drawn.render()
        drawn.render("rgb_array")
        for key in first:
            assert np.array_equal(first[key], second[key])


# ----------------------------------------------------------------------
# Dark Key-to-Door
# ----------------------------------------------------------------------


def test_key_to_door_frame_draws_the_hidden_layout() -> None:
    env = DarkKeyToDoorEnv(split="development", initial_seed=0)
    env.reset(options={"task_index": TASK})
    native = cast(Any, env.native.unwrapped)
    rows = grid(env.render(), header_lines=2)
    assert np.asarray(rows).shape == (env.size, env.size)
    agent, key, door = (
        tuple(int(v) for v in c) for c in (native.pos, native.key, native.goal)
    )
    assert rows[agent[0]][agent[1]] == "A"
    assert rows[door[0]][door[1]] == "D" or door == agent
    assert rows[key[0]][key[1]] == "K" or key in (agent, door)
    header = env.render().splitlines()[0]
    assert f"task {TASK}" in header and "call 0/500" in header
    assert env.action_names == KEY_TO_DOOR_ACTION_NAMES
    image = env.render("rgb_array")
    assert image.shape == (env.size * CELL_PIXELS + 1,) * 2 + (3,)
    centre = CELL_PIXELS // 2 + 1
    assert (
        tuple(image[agent[0] * CELL_PIXELS + centre, agent[1] * CELL_PIXELS + centre])
        == PALETTE["agent"]
    )


def test_key_to_door_frame_dims_the_key_once_held() -> None:
    env = DarkKeyToDoorEnv(split="development", initial_seed=0)
    env.reset(options={"task_index": TASK})
    native = cast(Any, env.native.unwrapped)
    door = tuple(int(v) for v in native.goal)
    row, column = (int(v) for v in native.pos)
    key_row, key_column = (int(v) for v in native.key)
    # Walk onto the key with the fixed action table (left/up/right/down/stay).
    while (row, column) != (key_row, key_column):
        if row != key_row:
            action, row = (3, row + 1) if row < key_row else (1, row - 1)
        else:
            action, column = (2, column + 1) if column < key_column else (0, column - 1)
        env.step(action)
    frame = env.render()
    assert "key held" in frame.splitlines()[1]
    assert grid(frame, header_lines=2)[key_row][key_column] == "A"
    # Step off the key onto a plain cell, so the taken key shows on its own.
    for action, target in (
        (0, (key_row, key_column - 1)),
        (2, (key_row, key_column + 1)),
    ):
        if 0 <= target[1] < env.size and target != door:
            env.step(action)
            break
    else:
        pytest.skip("both neighbours of the key are the wall or the door")
    assert grid(env.render(), header_lines=2)[key_row][key_column] == "k"
    centre = CELL_PIXELS // 2 + 1
    image = env.render("rgb_array")
    assert (
        tuple(image[key_row * CELL_PIXELS + centre, key_column * CELL_PIXELS + centre])
        == PALETTE["key_taken"]
    )


def test_key_to_door_labels_are_opaque_under_randomised_actions() -> None:
    env = DarkKeyToDoorEnv(split="development", randomized_actions=True)
    assert env.action_names == tuple(f"action {i}" for i in range(5))


# ----------------------------------------------------------------------
# CountRecall
# ----------------------------------------------------------------------


def test_count_recall_frame_follows_the_public_stream() -> None:
    env = CountRecallEnv(variant="medium", split="development", initial_seed=0)
    packet, _ = env.reset(options={"task_index": TASK})
    symbols = COUNT_RECALL_SYMBOLS["medium"]
    value, query, _ = env.decode(packet)
    frame = env.render()
    lines = frame.splitlines()
    assert f"dealt {symbols[value]}  query {symbols[query]}" in lines[2]
    assert lines[-2] == symbols[value] and lines[-1] == "?"
    dealt = [value]
    answers = []
    for answer in (0, 1, 1, 2):
        packet = env.step(answer)[0]
        dealt.append(env.decode(packet)[0])
        answers.append(env.scored_queries[-1].correct)
    lines = env.render().splitlines()
    assert lines[-2] == " ".join(symbols[v] for v in dealt)
    assert lines[-1] == " ".join(["+" if c else "x" for c in answers] + ["?"])
    assert (
        "counts  "
        + "  ".join(
            f"{s} {n}"
            for s, n in zip(symbols, np.bincount(dealt, minlength=4), strict=True)
        )
        == lines[1]
    )
    assert env.action_names == tuple(str(i) for i in range(27))


def test_count_recall_rgb_lays_out_one_deck_per_line() -> None:
    env = CountRecallEnv(variant="medium", split="development", initial_seed=0)
    env.reset(options={"task_index": TASK})
    image = env.render("rgb_array")
    lines = -(-(env.horizon + 1) // RENDER_STREAM_WIDTH)
    assert lines == 2 and image.dtype == np.uint8
    assert image.shape[1] == RENDER_STREAM_WIDTH * 12 + 1
    # Per line: cards (12 px + grid) and marks (6 px), a 4 px gap after each;
    # then the two-cell "now" row.
    assert image.shape[0] == lines * ((12 + 1) + 6 + 2 * 4) + (2 * 12 + 1)
    for _ in range(env.horizon):
        env.step(0)
    assert env.render().splitlines()[2] == "stream over"
    assert "?" not in env.render().splitlines()[-1]


def test_count_recall_state_carries_the_dealt_history() -> None:
    env = CountRecallEnv(variant="easy", split="development", initial_seed=0)
    env.reset(options={"task_index": TASK})
    for _ in range(5):
        env.step(1)
    state = env.state_dict()
    assert len(cast(list[int], state["dealt"])) == 6
    other = CountRecallEnv(variant="easy", split="development", initial_seed=0)
    other.load_state_dict(state)
    assert other.render() == env.render()
    legacy = dict(state)
    del legacy["dealt"]
    other.load_state_dict(legacy)
    assert other.render().splitlines()[:2] == env.render().splitlines()[:2]


# ----------------------------------------------------------------------
# Passive T-Maze
# ----------------------------------------------------------------------


def test_tmaze_frame_draws_the_corridor_the_cue_side_and_the_agent() -> None:
    env = TMazeEnv(corridor_length=8, split="development", initial_seed=0)
    env.reset(options={"task_index": TASK})
    header, above, corridor, below = env.render().splitlines()
    assert f"cue {'up' if env.cue == 1 else 'down'}" in header and "x 0/8" in header
    assert corridor == "A" + "." * 7 + "+"
    goal_row, other_row = (above, below) if env.cue == 1 else (below, above)
    assert goal_row == " " * 8 + "G" and other_row == " " * 8 + "."
    for _ in range(8):
        env.step(FORWARD)
    assert env.render().splitlines()[2] == "." * 8 + "A"
    env.step(UP if env.cue == 1 else DOWN)
    header, above, corridor, below = env.render().splitlines()
    assert header.endswith("success") and corridor == "." * 8 + "+"
    assert (above if env.cue == 1 else below) == " " * 8 + "A"
    assert env.action_names == TMAZE_ACTION_NAMES
    image = env.render("rgb_array")
    assert image.shape == (3 * CELL_PIXELS + 1, 9 * CELL_PIXELS + 1, 3)


def test_tmaze_frame_scales_a_long_corridor_to_the_column_budget() -> None:
    env = TMazeEnv(corridor_length=128, split="development", initial_seed=0)
    env.reset(options={"task_index": TASK})
    for _ in range(40):
        env.step(FORWARD)
    header, _, corridor, _ = env.render().splitlines()
    scale = -(-129 // RENDER_COLUMNS)
    assert scale == 3 and len(corridor) == -(-129 // scale) == 43
    assert corridor[40 // scale] == "A" and corridor[-1] == "+"
    assert "x 40/128" in header
    image = env.render("rgb_array")
    pixels = max(6, min(CELL_PIXELS, 1024 // 129))
    assert pixels == 7 and image.shape == (3 * pixels, 129 * pixels, 3)
