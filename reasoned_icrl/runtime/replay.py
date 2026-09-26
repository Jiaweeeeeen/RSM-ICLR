"""Deterministic online replay construction for AMAGO."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from amago.hindsight import FrozenTraj, Relabeler, Trajectory
from amago.loading import DiskTrajDataset, RLData, load_traj_from_disk

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.xland_one_rule import POOLS, CurriculumSchedule


def replay_pool(filename: str) -> str | None:
    """The pool tag AMAGO's file name carries: ``<env>-<pool>_<id>_<time>``."""
    stem = Path(filename).name.split("_", 1)[0]
    pool = stem.rsplit("-", 1)[-1]
    return pool if pool in POOLS else None


class OrderedDiskTrajDataset(DiskTrajDataset):
    """Disk replay whose random indices retain a stable filename ordering.

    With a ``curriculum`` (the one-rule XLand task) every file carries the
    pool of the lifetime that wrote it, and sampling is restricted to the
    pools the learner's measured charged calls make eligible: warmup only,
    both, then primary only. Ineligible files stay on disk until the FIFO
    evicts them; nothing is relabelled.
    """

    def __init__(
        self,
        *,
        dset_root: str,
        dset_name: str,
        dset_max_size: int,
        full_tasks: bool = False,
        reset_only_terminal: bool = False,
        relabeler: Relabeler | None = None,
        curriculum: CurriculumSchedule | None = None,
    ) -> None:
        self.full_tasks = full_tasks
        self.reset_only_terminal = reset_only_terminal
        self.curriculum = curriculum
        self.learner_calls = 0
        self._eligible: list[str] | None = None
        self._ordered_filenames: list[str] = []
        self.resume_deviation: dict[str, object] | None = None
        super().__init__(
            dset_root=dset_root,
            dset_name=dset_name,
            dset_max_size=dset_max_size,
            relabeler=relabeler,
        )

    def eligible_filenames(self) -> list[str]:
        """The files the learner may sample under the current phase."""
        if self.curriculum is None:
            return list(self.all_filenames)
        pools = self.curriculum.eligible(self.learner_calls)
        eligible: list[str] = []
        for filename in self.all_filenames:
            pool = replay_pool(filename)
            if pool is None:
                raise ContractError(
                    f"Replay file {Path(filename).name} carries no curriculum pool."
                )
            if pool in pools:
                eligible.append(filename)
        return eligible

    def on_end_of_collection(self, experiment: Any) -> dict[str, Any]:
        log = dict(super().on_end_of_collection(experiment))
        if self.curriculum is not None:
            counters = getattr(experiment, "collection_counters", {})
            self.learner_calls = int(counters.get("charged_calls", 0))
            pools = sorted(self.curriculum.eligible(self.learner_calls))
            # The file set and the phase change only here, between epochs,
            # so the eligible list is fixed for the whole update pass.
            eligible = self._eligible = self.eligible_filenames()
            log["Curriculum Learner Charged Calls"] = self.learner_calls
            log["Curriculum Eligible Pools"] = " ".join(pools)
            log["Curriculum Eligible Trajectory Files"] = len(eligible)
            for pool in POOLS:
                log[f"Trajectory Files In Pool {pool}"] = sum(
                    1 for f in self.all_filenames if replay_pool(f) == pool
                )
        return log

    def sample_random_trajectory(self) -> RLData:
        if self.curriculum is not None:
            self.check_configured()
            eligible = (
                self.eligible_filenames() if self._eligible is None else self._eligible
            )
            if not eligible:
                raise ContractError(
                    "No replay trajectory is eligible under the curriculum phase."
                )
            traj = load_traj_from_disk(random.choice(eligible))
            data = self._traj_to_rl_data(self.relabeler(traj))
        else:
            data = super().sample_random_trajectory()
        if self.full_tasks:
            if int(data.time_idxs[0].item()) != 0 or not bool(data.dones[-1].item()):
                raise ContractError(
                    "History replay must contain a complete outer task."
                )
            if (
                len(data) > self.max_seq_len
                or data.obs["current"].shape[0] != len(data) + 1
            ):
                raise ContractError("History replay would lose a causal task prefix.")
            if data.obs["event"][0, 0] != 0:
                raise ContractError("History replay lost its initial event.")
            if data.obs["event"][-1, 0] != 1:
                # A task whose outer budget can expire on a reset-only step ends
                # on a token that executed no physical action. That token is
                # still preceded by the physical step that closed the attempt,
                # so the terminal evidence is present, one position earlier.
                if not self.reset_only_terminal:
                    raise ContractError("History replay lost its terminal event.")
                if (
                    data.obs["event"][-1, 1] != 1
                    or len(data) < 2
                    or data.obs["event"][-2, 0] != 1
                ):
                    raise ContractError(
                        "A reset-only terminal token must follow a physical step."
                    )
        return data

    @staticmethod
    def _sort_key(filename: str) -> tuple[float, str]:
        stem = Path(filename).stem
        try:
            timestamp = float(stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            timestamp = Path(filename).stat().st_mtime
        return timestamp, Path(filename).name

    def _refresh_files(self) -> None:
        self.fifo_filenames = self._list_abs_path_to_files(self.fifo_path)
        self.protected_filenames = self._list_abs_path_to_files(self.protected_path)
        discovered = self.fifo_filenames | self.protected_filenames
        retained = [name for name in self._ordered_filenames if name in discovered]
        new = sorted(discovered.difference(retained), key=self._sort_key)
        self._ordered_filenames = retained + new
        self.all_filenames = list(self._ordered_filenames)

    def _filter(self) -> None:
        super()._filter()
        self._refresh_files()

    def state_dict(self) -> dict[str, object]:
        self._refresh_files()
        ordered: list[dict[str, str]] = []
        fifo = Path(self.fifo_path)
        protected = Path(self.protected_path)
        for raw in self._ordered_filenames:
            path = Path(raw)
            if path.parent == fifo:
                area = "fifo"
            elif path.parent == protected:
                area = "protected"
            else:
                raise ContractError("Replay file escaped the configured buffer.")
            ordered.append({"area": area, "name": path.name})
        return {"schema": "ordered-disk-replay.v1", "ordered": ordered}

    def load_state_dict(
        self, state: Mapping[str, object], *, allow_missing: bool = False
    ) -> None:
        """Restore the checkpoint's replay order.

        Every file the checkpoint lists must be present; the FIFO keeps
        collecting after a checkpoint is saved, so a run that died later has
        usually evicted the checkpoint's oldest trajectories, and an exact
        resume is refused. With ``allow_missing`` (R6, explicit and recorded)
        the evicted entries are dropped from the order, the surviving files
        keep their order and the deviation is kept in ``resume_deviation`` for
        the run's records; the buffer refills as the resumed run collects.
        """
        self.resume_deviation = None
        if state.get("schema") != "ordered-disk-replay.v1":
            raise ContractError("Unsupported replay-order checkpoint state.")
        rows = state.get("ordered")
        if not isinstance(rows, Sequence) or isinstance(rows, str):
            raise ContractError("Replay-order checkpoint is malformed.")
        expected: list[str] = []
        for value in rows:
            if not isinstance(value, Mapping):
                raise ContractError("Replay-order checkpoint is malformed.")
            area = str(value.get("area"))
            name = str(value.get("name"))
            if area == "fifo":
                root = Path(self.fifo_path)
            elif area == "protected":
                root = Path(self.protected_path)
            else:
                raise ContractError(
                    "Replay checkpoint contains an unknown buffer area."
                )
            if Path(name).name != name:
                raise ContractError("Replay checkpoint contains an unsafe filename.")
            expected.append(str(root / name))

        discovered = self._list_abs_path_to_files(
            self.fifo_path
        ) | self._list_abs_path_to_files(self.protected_path)
        missing = set(expected).difference(discovered)
        if missing and not allow_missing:
            raise ContractError(
                "Replay files required by the selected checkpoint are missing "
                f"({len(missing)} of {len(expected)}): the FIFO evicted them after "
                "the checkpoint was saved, so an exact resume is impossible."
            )
        if missing:
            self.resume_deviation = {
                "schema": "replay-resume-deviation.v1",
                "expected_files": len(expected),
                "missing_files": len(missing),
                "missing": sorted(Path(name).name for name in missing),
            }
            expected = [name for name in expected if name not in missing]
        extras = discovered.difference(expected)
        if extras:
            quarantine = Path(self.fifo_path).parents[1] / "orphaned-after-checkpoint"
            quarantine.mkdir(parents=True, exist_ok=True)
            for raw in sorted(extras):
                source = Path(raw)
                destination = quarantine / source.name
                suffix = 1
                while destination.exists():
                    destination = quarantine / f"{source.stem}-{suffix}{source.suffix}"
                    suffix += 1
                source.replace(destination)

        self._ordered_filenames = list(expected)
        self._refresh_files()
        if self._ordered_filenames != expected:
            raise ContractError(
                "Replay ordering changed while restoring the checkpoint."
            )


def create_replay_dataset(
    run_directory: str | Path,
    *,
    capacity: int,
    full_tasks: bool = False,
    reset_only_terminal: bool = False,
    relabeler: Relabeler | None = None,
    curriculum: CurriculumSchedule | None = None,
) -> OrderedDiskTrajDataset:
    """Create the online replay dataset documented by AMAGO.

    ``relabeler`` runs on every sampled trajectory before it becomes training
    data. Only a task that declares hindsight relabeling supplies one; the
    default keeps AMAGO's no-op behavior.
    """
    if capacity <= 0:
        raise ContractError("Replay capacity must be positive.")
    return OrderedDiskTrajDataset(
        dset_root=str(Path(run_directory)),
        dset_name="replay",
        dset_max_size=capacity,
        full_tasks=full_tasks,
        reset_only_terminal=reset_only_terminal,
        relabeler=relabeler,
        curriculum=curriculum,
    )


class ReconstructingRelabeler(Relabeler):
    """Relabel raw native trajectories before rebuilding causal policy fields.

    The callback belongs to each environment adapter (M2--M4). No precomputed
    transition field is accepted as authoritative after relabeling. Existing
    replay defaults and full-task checks are unchanged.
    """

    def __init__(
        self, native: Relabeler, reconstruct: Callable[[FrozenTraj], FrozenTraj]
    ) -> None:
        self.native = native
        self.reconstruct = reconstruct

    def relabel(self, traj: Trajectory | FrozenTraj) -> FrozenTraj:
        result = self.reconstruct(self.native(deepcopy(traj)))
        length = len(result.rews)
        if length < 1 or set(result.obs) != {
            "current",
            "previous",
            "outcome",
            "event",
            "valid",
        }:
            raise ContractError(
                "Reconstructed replay requires only public packet fields."
            )
        if any(len(v) != length + 1 for v in result.obs.values()):
            raise ContractError("Reconstructed replay lost an observation endpoint.")
        if (
            len(result.rl2s) != length + 1
            or len(result.time_idxs) != length + 1
            or len(result.actions) != length
            or len(result.dones) != length
        ):
            raise ContractError(
                "Reconstructed replay has inconsistent sequence lengths."
            )
        if (
            result.rl2s.ndim != 2
            or result.rl2s.shape[1] < 2
            or result.rews.shape != (length, 1)
            or not np.array_equal(result.rl2s[1:, :1], result.rews)
        ):
            raise ContractError("Reconstructed replay reward/RL2 alignment failed.")
        if any(
            not np.isfinite(v).all()
            for v in (
                *result.obs.values(),
                result.rl2s,
                result.rews,
                result.actions,
            )
        ):
            raise ContractError("Reconstructed replay contains nonfinite data.")
        return result
