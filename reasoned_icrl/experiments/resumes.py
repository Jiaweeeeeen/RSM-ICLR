"""Resume records, reconciled counters and per-session hours of resumed runs (R6).

A run that a died pack resumed keeps its run directory and identities, so
its artefacts come from several training sessions. Every resume appends one
line to ``resumes.jsonl``; the trainer and the reports use these records to
say from which label each session continued, whether evicted replay files
were dropped, which hardware ran each session, and to reconcile the measured
interaction counters, which the environment actors count per process:
a session that started from a restored state whose environment snapshot did
not carry the counters reports that session only. Charged calls are exact by
construction (every collected decision is charged, so the count at label
``N`` is ``(N + 1) x timesteps_per_epoch x actors``); physical, reset-only and
task counts before such a resume were not recorded and stay ``None``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

RESUMES_FILE = "resumes.jsonl"
SESSION_ONLY_BACKUP = "systems.session-only.json"
COUNTER_KEYS = (
    "charged_calls",
    "physical_actions",
    "reset_only_steps",
    "tasks_started",
    "tasks_completed",
)
CARRIED = "carried across resume"
RECONCILED = (
    "session-only counters reconciled: charged calls are exact by construction; "
    "physical, reset-only and task counts before the resume were not recorded"
)


def read_resumes(run: Path) -> list[dict[str, object]]:
    """The run's resume records, oldest first (empty for an uninterrupted run)."""
    path = run / RESUMES_FILE
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_resume(run: Path, entry: Mapping[str, object]) -> None:
    with (run / RESUMES_FILE).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(entry), sort_keys=True) + "\n")


def resumes_for_systems(
    resumes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """The compact copy ``systems.json`` carries: no command line, no evicted
    file names and no per-file source hashes (all of which stay in the record)."""
    compact: list[dict[str, object]] = []
    for entry in resumes:
        row: dict[str, object] = {}
        for key, value in entry.items():
            if key in ("command", "source_sha256"):
                continue
            if key == "replay_deviation" and isinstance(value, Mapping):
                row[key] = {k: v for k, v in value.items() if k != "missing"}
            else:
                row[key] = value
        compact.append(row)
    return compact


def reconcile_measured(
    measured: Mapping[str, object],
    *,
    resumes: Sequence[Mapping[str, object]],
    epochs: int,
    timesteps_per_epoch: int,
    actors: int,
    reset_capable: bool,
) -> dict[str, object]:
    """Turn a resumed run's live counters into exact totals or ``None``.

    ``reset_capable`` says whether the environment can take reset-only steps
    (attempt tasks); when it cannot and the session took none, physical
    actions equal charged calls exactly. Uninterrupted runs, runs whose
    counters were carried by the environment snapshot, and already reconciled
    blocks are returned unchanged apart from the ``counters`` note.
    """
    out = dict(measured)
    if not resumes:
        return out
    if out.get("counters") in (CARRIED, RECONCILED) or "session" in out:
        return out
    per_epoch = int(timesteps_per_epoch) * int(actors)
    total = int(epochs) * per_epoch
    last_label = int(cast(int, resumes[-1]["resumed_label"]))
    session_epochs = int(epochs) - (last_label + 1)
    live = int(cast(int, measured["charged_calls"]))
    if live == total:
        out["counters"] = CARRIED
        return out
    if live != session_epochs * per_epoch:
        out["counters"] = (
            f"inconsistent: live charged calls {live} match neither the full "
            f"budget {total} nor the resumed session {session_epochs * per_epoch}"
        )
        return out
    session = {key: measured.get(key) for key in COUNTER_KEYS}
    exact_physical = (
        not reset_capable and int(cast(int, session["reset_only_steps"])) == 0
    )
    out.update(
        {
            "charged_calls": total,
            "physical_actions": total if exact_physical else None,
            "reset_only_steps": 0 if exact_physical else None,
            "tasks_started": None,
            "tasks_completed": None,
            "session": session,
            "session_epochs": session_epochs,
            "before_resume": {
                "epochs": last_label + 1,
                "charged_calls": total - live,
                "source": "label arithmetic: every collected decision is charged",
            },
            "counters": RECONCILED,
        }
    )
    return out


def _mtime(path: Path) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def sessions_from_disk(
    run: Path,
    resumes: Sequence[Mapping[str, object]],
    *,
    epochs: int,
    first_gpu: object,
    first_job: object,
    runtime_seconds: float | None,
) -> list[dict[str, object]]:
    """One entry per training session: label range, hardware, job and hours.

    A died session's hours run from its start (the first start's
    ``config.yaml``, later the resume record's timestamp) to the save of the
    last label it left behind, so the epochs it collected after that label
    and recollected later are excluded; the final session reports the
    trainer's measured runtime.
    """
    sessions: list[dict[str, object]] = []
    start = _mtime(run / "config.yaml")
    gpu, job, first_label = first_gpu, first_job, 0
    for index, entry in enumerate(resumes, start=1):
        label = int(cast(int, entry["resumed_label"]))
        end = _mtime(run / "ckpts" / "policy_weights" / f"policy_epoch_{label}.pt")
        sessions.append(
            {
                "session": index,
                "labels": [first_label, label],
                "gpu": gpu,
                "slurm_job_id": job,
                "hours_to_last_saved_label": (
                    None if start is None or end is None else (end - start) / 3600.0
                ),
                "ended": f"died; epochs after label {label} were recollected",
            }
        )
        stamp = entry.get("timestamp")
        try:
            start = time.mktime(time.strptime(str(stamp)[:19], "%Y-%m-%dT%H:%M:%S"))
        except (TypeError, ValueError):
            start = None
        gpu, job, first_label = entry.get("gpu"), entry.get("slurm_job_id"), label + 1
    sessions.append(
        {
            "session": len(resumes) + 1,
            "labels": [first_label, int(epochs) - 1],
            "gpu": gpu,
            "slurm_job_id": job,
            "hours": None
            if runtime_seconds is None
            else float(runtime_seconds) / 3600.0,
            "ended": "completed",
        }
    )
    return sessions


def session_hours(sessions: Sequence[Mapping[str, object]]) -> float | None:
    """Total wall hours over the sessions, or ``None`` when one is unknown."""
    total = 0.0
    for entry in sessions:
        value = entry.get("hours", entry.get("hours_to_last_saved_label"))
        if value is None:
            return None
        total += float(cast(float, value))
    return total


def gpu_chain(sessions: Sequence[Mapping[str, object]]) -> str:
    """The GPU models in session order, e.g. ``A -> B``; repeats collapsed."""
    names: list[str] = []
    for entry in sessions:
        name = str(entry.get("gpu") or "unknown")
        if not names or names[-1] != name:
            names.append(name)
    return " -> ".join(names)


def reconcile_systems_file(
    run: Path,
    *,
    epochs: int,
    timesteps_per_epoch: int,
    actors: int,
    reset_capable: bool,
) -> bool:
    """Reconcile a finished resumed run's ``systems.json`` in place (idempotent).

    Keeps the original as ``systems.session-only.json`` the first time and
    returns whether the file changed.
    """
    resumes = read_resumes(run)
    path = run / "systems.json"
    if not resumes or not path.is_file():
        return False
    systems = json.loads(path.read_text(encoding="utf-8"))
    measured = cast(Mapping[str, object], systems.get("measured") or {})
    if "sessions" in systems and measured.get("counters") in (CARRIED, RECONCILED):
        return False
    provenance_path = run / "provenance.json"
    provenance: dict[str, Any] = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {}
    )
    backup = run / SESSION_ONLY_BACKUP
    if not backup.is_file():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    systems["measured"] = reconcile_measured(
        measured,
        resumes=resumes,
        epochs=epochs,
        timesteps_per_epoch=timesteps_per_epoch,
        actors=actors,
        reset_capable=reset_capable,
    )
    systems["sessions"] = sessions_from_disk(
        run,
        resumes,
        epochs=epochs,
        first_gpu=provenance.get("gpu"),
        first_job=provenance.get("slurm_job_id"),
        runtime_seconds=systems.get("runtime_seconds"),
    )
    systems["resumes"] = resumes_for_systems(resumes)
    systems["reconciled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(systems, indent=2) + "\n", encoding="utf-8")
    return True


__all__ = [
    "CARRIED",
    "COUNTER_KEYS",
    "RECONCILED",
    "RESUMES_FILE",
    "SESSION_ONLY_BACKUP",
    "append_resume",
    "gpu_chain",
    "read_resumes",
    "reconcile_measured",
    "reconcile_systems_file",
    "resumes_for_systems",
    "session_hours",
    "sessions_from_disk",
]
