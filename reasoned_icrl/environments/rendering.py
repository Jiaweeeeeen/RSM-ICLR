"""Frame helpers shared by the environments' ``render`` implementations.

Every rendering environment draws two views of the same observer state: a
text frame (``ansi``) for the terminal and an RGB frame (``rgb_array``) for
GIFs and figures. The helpers here turn a small grid of glyphs or colours
into those frames; nothing here reads an environment, so the environments
keep the only view of their own state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

RENDER_MODES: tuple[str, ...] = ("ansi", "rgb_array", "human")
"""The modes every rendering environment accepts; ``human`` prints ``ansi``."""

CELL_PIXELS = 24
"""Side of one grid cell in an RGB frame, before any per-environment shrink."""

Color = tuple[int, int, int]

PALETTE: Mapping[str, Color] = {
    "background": (24, 24, 28),
    "floor": (52, 52, 60),
    "wall": (12, 12, 14),
    "grid": (36, 36, 42),
    "agent": (66, 135, 245),
    "key": (245, 200, 66),
    "key_taken": (110, 96, 48),
    "door": (76, 175, 80),
    "goal": (76, 175, 80),
    "goal_other": (96, 96, 104),
    "correct": (76, 175, 80),
    "wrong": (211, 67, 67),
    "pending": (40, 40, 46),
    "highlight": (245, 245, 245),
}
"""One fixed observer palette so the three benchmarks read as one set."""

CATEGORY_COLORS: tuple[Color, ...] = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (255, 187, 120),
    (152, 223, 138),
    (255, 152, 150),
)
"""Thirteen distinguishable colours, one per CountRecall category at most."""


def ansi_frame(header: Sequence[str], rows: Sequence[str]) -> str:
    """Join header lines and grid rows into one text frame."""
    return "\n".join([*header, *rows])


def paint_cells(
    colors: NDArray[np.uint8],
    *,
    cell_pixels: int = CELL_PIXELS,
    grid: Color | None = PALETTE["grid"],
) -> NDArray[np.uint8]:
    """Scale an ``(H, W, 3)`` array of cell colours to an RGB frame.

    Every cell becomes a ``cell_pixels`` square; with ``grid`` set, a one-pixel
    line in that colour separates the cells and frames the board.
    """
    if colors.ndim != 3 or colors.shape[2] != 3 or colors.dtype != np.uint8:
        raise ValueError("Cell colours must be a uint8 (H, W, 3) array.")
    if cell_pixels < 2:
        raise ValueError("A cell needs at least two pixels.")
    frame = np.repeat(np.repeat(colors, cell_pixels, axis=0), cell_pixels, axis=1)
    if grid is not None:
        height, width = colors.shape[:2]
        line = np.asarray(grid, dtype=np.uint8)
        frame[::cell_pixels, :] = line
        frame[:, ::cell_pixels] = line
        frame = np.concatenate([frame, np.tile(line, (1, width * cell_pixels, 1))], 0)
        frame = np.concatenate(
            [frame, np.tile(line, (height * cell_pixels + 1, 1, 1))], 1
        )
    return frame


def color_grid(height: int, width: int, fill: Color) -> NDArray[np.uint8]:
    """An ``(height, width, 3)`` uint8 array filled with one colour."""
    grid = np.empty((height, width, 3), dtype=np.uint8)
    grid[:] = np.asarray(fill, dtype=np.uint8)
    return grid


def stack_frames(
    frames: Sequence[NDArray[np.uint8]],
    *,
    gap: int = 4,
    fill: Color = PALETTE["background"],
) -> NDArray[np.uint8]:
    """Stack RGB frames vertically, left-aligned, padded to the widest one."""
    width = max(frame.shape[1] for frame in frames)
    padded: list[NDArray[np.uint8]] = []
    for index, frame in enumerate(frames):
        if index:
            padded.append(color_grid(gap, width, fill))
        if frame.shape[1] < width:
            frame = np.concatenate(
                [frame, color_grid(frame.shape[0], width - frame.shape[1], fill)], 1
            )
        padded.append(frame)
    return np.concatenate(padded, 0)


__all__ = [
    "CATEGORY_COLORS",
    "CELL_PIXELS",
    "PALETTE",
    "RENDER_MODES",
    "Color",
    "ansi_frame",
    "color_grid",
    "paint_cells",
    "stack_frames",
]
