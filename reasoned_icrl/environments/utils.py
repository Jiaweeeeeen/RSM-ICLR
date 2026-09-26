"""Helpers shared by the environment adapters."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from reasoned_icrl.experiments.contracts import ContractError


def validate_roster(
    roster: Sequence[int], allowed: Sequence[int], *, label: str
) -> tuple[int, ...]:
    """Return a checked, ordered task roster drawn from the allowed partition."""
    values = tuple(int(value) for value in roster)
    if not values:
        raise ContractError(f"{label} source_indices cannot be empty.")
    if len(set(values)) != len(values):
        raise ContractError(f"{label} source_indices must be unique.")
    if not set(values).issubset({int(value) for value in allowed}):
        raise ContractError(f"{label} source_indices lie outside the active partition.")
    return values


def project_unit_interval(
    observation: object, width: int, *, label: str
) -> NDArray[np.float32]:
    """Map a native ``[0, 1]`` Box observation to the shared ``[-1, 1]`` packet."""
    raw = np.asarray(observation, dtype=np.float32)
    if raw.shape != (width,) or not np.isfinite(raw).all():
        raise ContractError(f"{label} observation has an unexpected shape.")
    if np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ContractError(f"{label} observation left its declared bounds.")
    return (2.0 * raw - 1.0).astype(np.float32)


def normalize_fields(
    values: NDArray[np.float64], low: NDArray[np.float64], high: NDArray[np.float64]
) -> NDArray[np.float32]:
    """Rescale declared ``[low, high]`` values to ``[-1, 1]`` exactly as recorded."""
    if not np.isfinite(values).all() or (values < low).any() or (values > high).any():
        raise ContractError("Public value outside its declared bounds.")
    return (2 * (values - low) / np.maximum(high - low, 1) - 1).astype(np.float32)


@contextmanager
def random_scope(generator: random.Random) -> Iterator[None]:
    """Isolate upstream global RNG in sequential scalar/process actor calls.

    Native reset calls must not run concurrently in threads. Use separate
    processes or sequential sync composition, as the runtime does.
    """
    caller = random.getstate()
    random.setstate(generator.getstate())
    try:
        yield
    finally:
        generator.setstate(random.getstate())
        random.setstate(caller)


class NativeSource(gym.Wrapper[Any, Any, Any, Any]):
    """An upstream environment with actor-isolated, reproducible task sampling.

    ``factory`` constructs the pinned upstream task under the isolated RNG.
    ``reseed`` receives the unwrapped task and the reset seed for upstream
    environments that keep their own NumPy generator.
    """

    def __init__(
        self,
        factory: Callable[[], gym.Env[Any, Any]],
        *,
        seed: int,
        reseed: Callable[[Any, int], None] | None = None,
    ) -> None:
        if type(seed) is not int or seed < 0:
            raise ContractError("Native environment seed must be nonnegative.")
        self._seed = seed
        self._reseed = reseed
        self._generator = random.Random(seed)
        with random_scope(self._generator):
            env = factory()
        super().__init__(env)
        self._first_reset = True
        self.action_space.seed(seed)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        selected = self._seed if self._first_reset and seed is None else seed
        if selected is not None:
            if type(selected) is not int or selected < 0:
                raise ContractError("Native reset seed must be nonnegative.")
            self._generator.seed(selected)
            if self._reseed is not None:
                self._reseed(cast(Any, self.env.unwrapped), selected)
        self._first_reset = False
        with random_scope(self._generator):
            return self.env.reset(seed=selected, options=options)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        with random_scope(self._generator):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return obs, float(reward), terminated, truncated, info

    def generator_state(self) -> tuple[Any, ...]:
        """Expose the actor-local task RNG so adapters can checkpoint it."""
        return self._generator.getstate()

    def load_generator_state(self, state: Any) -> None:
        """Restore a snapshot of the actor-local task RNG."""
        self._generator.setstate(tuple(state) if isinstance(state, list) else state)


__all__ = [
    "NativeSource",
    "normalize_fields",
    "project_unit_interval",
    "random_scope",
    "validate_roster",
]
