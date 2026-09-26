"""Resume records, reconciled counters and per-session hours (R6)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from reasoned_icrl.experiments.resumes import (
    CARRIED,
    RECONCILED,
    gpu_chain,
    read_resumes,
    reconcile_measured,
    reconcile_systems_file,
    resumes_for_systems,
    session_hours,
    sessions_from_disk,
)

EPOCHS, TPS, ACTORS = 1000, 500, 16
RESUME = {
    "schema": "reasoned-icrl-resume.v1",
    "timestamp": "2000-01-01T00:00:00+0000",
    "resumed_label": 800,
    "next_epoch": 801,
    "allow_missing_replay": True,
    "replay_deviation": {
        "schema": "replay-resume-deviation.v1",
        "expected_files": 10000,
        "missing_files": 1616,
        "missing": ["a.npz", "b.npz"],
    },
    "gpu": "NVIDIA GeForce RTX 3090",
    "slurm_job_id": "140809",
    "git_commit": "abc",
    "source_sha256": {"x.py": "0" * 64},
    "command": ["train.py"],
}


def _session_only(charged: int) -> dict[str, object]:
    return {
        "scalar_training_transitions": EPOCHS * TPS * ACTORS,
        "charged_calls": charged,
        "physical_actions": charged,
        "reset_only_steps": 0,
        "tasks_started": 15328,
        "tasks_completed": 15312,
        "validation": {"charged_calls": 4992},
    }


def test_session_only_counters_become_exact_totals_or_none() -> None:
    session = 199 * TPS * ACTORS  # epochs 801..999 of a run resumed at label 800
    out = reconcile_measured(
        _session_only(session),
        resumes=[RESUME],
        epochs=EPOCHS,
        timesteps_per_epoch=TPS,
        actors=ACTORS,
        reset_capable=False,
    )
    assert out["charged_calls"] == EPOCHS * TPS * ACTORS
    assert out["physical_actions"] == EPOCHS * TPS * ACTORS  # no reset-only steps here
    assert out["reset_only_steps"] == 0
    assert out["tasks_started"] is None and out["tasks_completed"] is None
    assert out["session"]["charged_calls"] == session
    assert out["session_epochs"] == 199
    assert out["before_resume"] == {
        "epochs": 801,
        "charged_calls": 801 * TPS * ACTORS,
        "source": "label arithmetic: every collected decision is charged",
    }
    assert out["counters"] == RECONCILED
    assert out["validation"] == {"charged_calls": 4992}  # untouched, session-only
    # An attempt task can take reset-only steps before the resume: unknown.
    attempt = reconcile_measured(
        _session_only(session),
        resumes=[RESUME],
        epochs=EPOCHS,
        timesteps_per_epoch=TPS,
        actors=ACTORS,
        reset_capable=True,
    )
    assert attempt["charged_calls"] == EPOCHS * TPS * ACTORS
    assert attempt["physical_actions"] is None and attempt["reset_only_steps"] is None
    # Reconciling twice changes nothing.
    assert (
        reconcile_measured(
            out,
            resumes=[RESUME],
            epochs=EPOCHS,
            timesteps_per_epoch=TPS,
            actors=ACTORS,
            reset_capable=False,
        )
        == out
    )


def test_carried_uninterrupted_and_inconsistent_counters_are_left_alone() -> None:
    full = _session_only(EPOCHS * TPS * ACTORS)
    carried = reconcile_measured(
        full,
        resumes=[RESUME],
        epochs=EPOCHS,
        timesteps_per_epoch=TPS,
        actors=ACTORS,
        reset_capable=True,
    )
    assert carried["charged_calls"] == EPOCHS * TPS * ACTORS
    assert carried["counters"] == CARRIED and "session" not in carried
    untouched = reconcile_measured(
        full,
        resumes=[],
        epochs=EPOCHS,
        timesteps_per_epoch=TPS,
        actors=ACTORS,
        reset_capable=True,
    )
    assert untouched == full
    odd = reconcile_measured(
        _session_only(123),
        resumes=[RESUME],
        epochs=EPOCHS,
        timesteps_per_epoch=TPS,
        actors=ACTORS,
        reset_capable=False,
    )
    assert odd["charged_calls"] == 123 and str(odd["counters"]).startswith(
        "inconsistent"
    )


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "proto" / "cell" / "seed-42"
    (run / "ckpts" / "policy_weights").mkdir(parents=True)
    (run / "config.yaml").write_text("x: 1\n")
    os.utime(run / "config.yaml", (1_000_000, 1_000_000))
    weights = run / "ckpts" / "policy_weights" / "policy_epoch_800.pt"
    weights.write_bytes(b"w")
    os.utime(weights, (1_000_000 + 21 * 3600, 1_000_000 + 21 * 3600))
    (run / "provenance.json").write_text(
        json.dumps({"gpu": "NVIDIA RTX PRO 6000", "slurm_job_id": "140002"})
    )
    (run / "resumes.jsonl").write_text(json.dumps(RESUME) + "\n")
    return run


def test_sessions_come_from_disk_timestamps_and_the_final_runtime(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    resumes = read_resumes(run)
    sessions = sessions_from_disk(
        run,
        resumes,
        epochs=EPOCHS,
        first_gpu="NVIDIA RTX PRO 6000",
        first_job="140002",
        runtime_seconds=2.5 * 3600,
    )
    assert [s["labels"] for s in sessions] == [[0, 800], [801, 999]]
    assert (
        sessions[0]["gpu"] == "NVIDIA RTX PRO 6000"
        and sessions[0]["slurm_job_id"] == "140002"
    )
    assert sessions[0]["hours_to_last_saved_label"] == 21.0
    assert (
        sessions[1]["gpu"] == "NVIDIA GeForce RTX 3090" and sessions[1]["hours"] == 2.5
    )
    assert session_hours(sessions) == 23.5
    assert gpu_chain(sessions) == "NVIDIA RTX PRO 6000 -> NVIDIA GeForce RTX 3090"
    assert gpu_chain([{"gpu": "A"}, {"gpu": "A"}]) == "A"
    compact = resumes_for_systems(resumes)
    assert "command" not in compact[0] and "source_sha256" not in compact[0]
    assert "missing" not in compact[0]["replay_deviation"]
    assert compact[0]["replay_deviation"]["missing_files"] == 1616


def test_a_finished_resumed_systems_file_is_reconciled_once_with_a_backup(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    systems = {"runtime_seconds": 9000.0, "measured": _session_only(199 * TPS * ACTORS)}
    (run / "systems.json").write_text(json.dumps(systems))
    assert reconcile_systems_file(
        run, epochs=EPOCHS, timesteps_per_epoch=TPS, actors=ACTORS, reset_capable=False
    )
    backup = json.loads((run / "systems.session-only.json").read_text())
    assert backup["measured"]["charged_calls"] == 199 * TPS * ACTORS
    written = json.loads((run / "systems.json").read_text())
    assert written["measured"]["charged_calls"] == EPOCHS * TPS * ACTORS
    assert written["measured"]["counters"] == RECONCILED
    assert [s["labels"] for s in written["sessions"]] == [[0, 800], [801, 999]]
    assert (
        written["resumes"][0]["resumed_label"] == 800
        and "command" not in written["resumes"][0]
    )
    assert written["reconciled_at"]
    # Idempotent: a second call changes nothing and keeps the first backup.
    assert not reconcile_systems_file(
        run, epochs=EPOCHS, timesteps_per_epoch=TPS, actors=ACTORS, reset_capable=False
    )
    assert json.loads((run / "systems.session-only.json").read_text()) == backup
    # An uninterrupted run is never touched.
    plain = tmp_path / "proto" / "cell" / "seed-7"
    plain.mkdir(parents=True)
    (plain / "systems.json").write_text(json.dumps(systems))
    assert not reconcile_systems_file(
        plain,
        epochs=EPOCHS,
        timesteps_per_epoch=TPS,
        actors=ACTORS,
        reset_capable=False,
    )
