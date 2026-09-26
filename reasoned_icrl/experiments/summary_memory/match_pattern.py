"""Match-pattern's frozen preparation and admission, using shared runtime owners."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reasoned_icrl.analysis.match_pattern import learning_summary
from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.match_pattern import MANIFEST_PATH, PROTOCOL, load_corpus
from reasoned_icrl.experiments.records import BenchmarkEvent
from reasoned_icrl.experiments.summary_memory.configs import load_summary_memory_study
from reasoned_icrl.experiments.summary_memory.revised import (
    decide_revised_c2,
    decide_revised_c3,
    measure_references,
    panel_events,
    read_references,
    read_run_development,
    write_references,
)
from reasoned_icrl.runtime.diagnostics import task_diagnostic
from reasoned_icrl.runtime.experiment import preflight_config
from reasoned_icrl.utils import repository_root

STUDY_FILE = Path("configs/match_pattern_8m.yaml")
CELLS = ("full_context", "full_dual_relational", "full_dual_content", "full_gru")


def write_record(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def prepare_match_pattern(root: Path) -> dict[str, Any]:
    """Freeze corpus, references, gates, aliases and actual capacity before fit."""
    repository = repository_root()
    study = load_summary_memory_study(repository / STUDY_FILE)
    contract = study.contract("match_pattern")
    if study.cells(contract) != CELLS or study.training_seeds != (42, 100, 2026):
        raise ContractError(
            "Match-pattern requires its exact four-cell/three-seed roster."
        )
    corpus = load_corpus()
    audit = {}
    configs = {}
    for condition in CELLS:
        config = experiment_config(
            contract,
            study,
            condition=condition,
            seed=42,
            repository=repository,
            device="cpu",
            output_root=root,
        )
        configs[condition] = config
        audit[condition] = preflight_config(config)
    dat_count = audit["full_dual_relational"]["dat"]["active_attention_parameters"]
    control_count = audit["full_dual_content"]["dat"]["active_attention_parameters"]
    difference = abs(control_count - dat_count) / dat_count
    if difference > 0.05:
        raise ContractError("Constructed dual-content active attention differs by >5%.")
    for module in ("timestep", "actor", "critics"):
        hashes = {
            row["initial_state_sha256_by_module"][module] for row in audit.values()
        }
        if len(hashes) != 1:
            raise ContractError(
                f"Match-pattern common initialization mismatch: {module}."
            )
    config = configs["full_context"]
    if read_references(root, PROTOCOL, "development") is None:
        write_references(root, measure_references(contract, config))
    references = read_references(root, PROTOCOL, "development")
    assert references is not None
    if references.level != 0.5 or references.task_ids != tuple(
        contract.roster("development")
    ):
        raise ContractError(
            "Match-pattern reference panel differs from the balanced frozen roster."
        )
    diagnostic = task_diagnostic(contract, config)
    if not diagnostic.passed:
        raise ContractError("Match-pattern complete-input alias coverage failed.")
    manifest = {
        "protocol": PROTOCOL,
        "corpus_sha256": corpus.sha256,
        "cells": CELLS,
        "seeds": study.training_seeds,
        "charged_calls_per_fit": 8_000_000,
        "training": asdict(config.training),
        "model": asdict(config.model),
        "split_counts": dict(corpus.manifest["counts"]),
        "c2": {"accuracy": 0.90, "margin": 0.40, "labels": [900, 950, 999]},
        "c3": {
            "margin": 0.10,
            "positive_seeds": 2,
            "conditional_lower_above": 0,
            "alias_pairs": 32,
            "alias_examples": 16,
        },
        "comparison_margin": 0.05,
        "bootstrap": {"draws": 10000, "seed": 20260916},
        "status": (
            "pilot selected; full comparison conditional on qualification and forecast"
        ),
    }
    path = root / "protocol" / "execution.json"
    normalized = json.loads(json.dumps(manifest))
    if path.is_file() and json.loads(path.read_text()) != normalized:
        raise ContractError(
            "Existing match-pattern protocol differs; do not overwrite runs."
        )
    write_record(path, manifest)
    write_record(
        root / "protocol" / "corpus-manifest.json",
        json.loads((repository / MANIFEST_PATH).read_text()),
    )
    write_record(root / "qualification" / "aliases.json", diagnostic.as_dict())
    write_record(
        root / "qualification" / "construction.json",
        {
            "conditions": audit,
            "dat_active_attention_parameters": dat_count,
            "control_active_attention_parameters": control_count,
            "relative_difference": difference,
        },
    )
    return manifest


def qualify_match_pattern(root: Path, *, provisional: bool = False) -> dict[str, Any]:
    """Decide admission from full development panels; seed 42 stays provisional."""
    study = load_summary_memory_study(repository_root() / STUDY_FILE)
    contract = study.contract("match_pattern")
    seeds = (42,) if provisional else study.training_seeds
    references = read_references(root, PROTOCOL, "development")
    if references is None:
        raise ContractError("References must be frozen before fitting.")
    diagnostic_path = root / "qualification" / "aliases.json"
    from reasoned_icrl.experiments.qualification import TaskDiagnostic

    diagnostic = TaskDiagnostic(**json.loads(diagnostic_path.read_text()))
    series = {}
    retained: list[BenchmarkEvent] = []
    cleared: list[BenchmarkEvent] = []
    complete_grid = True
    for seed in seeds:
        directory = root / PROTOCOL / "full_context" / f"seed-{seed}"
        development = read_run_development(directory)
        if development is None or development.task_cap is not None:
            complete_grid = False
            continue
        series[seed] = development.by_epoch
        points = {
            s.charged_calls: s.primary
            for s in development.series
            if s.charged_calls is not None
        }
        complete_grid &= bool(learning_summary(points)["complete"])
        retained.extend(
            panel_events(
                directory,
                contract,
                split="development",
                history="retained",
                rule="endpoint",
            )
        )
        cleared.extend(
            panel_events(
                directory,
                contract,
                split="development",
                history="current-token",
                rule="endpoint",
            )
        )
    c2 = decide_revised_c2(
        contract,
        "full_context",
        series,
        references=references,
        delta=0.40,
        expected_seeds=len(seeds),
    )
    c3 = (
        decide_revised_c3(
            contract,
            "full_context",
            retained=retained,
            intervened=cleared,
            diagnostic=diagnostic,
            delta=0.05,
            expected_seeds=len(seeds),
        )
        if retained and cleared
        else None
    )
    passed = complete_grid and c2.passed and c3 is not None and c3.passed
    result = {
        "protocol": PROTOCOL,
        "corpus_sha256": load_corpus().sha256,
        "models": {
            str(seed): hashlib.sha256(
                (
                    root
                    / PROTOCOL
                    / "full_context"
                    / f"seed-{seed}"
                    / "ckpts/policy_weights/policy_epoch_999.pt"
                ).read_bytes()
            ).hexdigest()
            for seed in seeds
            if (
                root
                / PROTOCOL
                / "full_context"
                / f"seed-{seed}"
                / "ckpts/policy_weights/policy_epoch_999.pt"
            ).is_file()
        },
        "provisional": provisional,
        "seeds": list(seeds),
        "complete_learning_grid": complete_grid,
        "passed": passed,
        "c2": c2.as_dict(),
        "c3": None if c3 is None else c3.as_dict(),
        "next_stage": (
            "ordinary confirmation seeds 100/2026"
            if provisional
            else "comparison forecast required"
        )
        if passed
        else "retain failed/incomplete qualification; no automatic expansion",
    }
    write_record(
        root
        / "qualification"
        / ("pilot-seed42.json" if provisional else "ordinary-three-seed.json"),
        result,
    )
    return result


def require_fit_admission(root: Path, condition: str, seed: int) -> None:
    """Guard official fits; engineering smokes are costed separately by the caller."""
    engineering = root / "qualification" / "engineering.json"
    if not engineering.is_file() or not json.loads(engineering.read_text()).get(
        "passed"
    ):
        raise ContractError(
            "Match-pattern official fitting requires recorded "
            "CPU/GPU engineering acceptance."
        )
    if condition not in CELLS or seed not in (42, 100, 2026):
        raise ContractError(
            "Official match-pattern fits require the locked cells/seeds."
        )
    if not (root / "protocol" / "execution.json").is_file():
        raise ContractError(
            "Match-pattern protocol/references must be frozen before fitting."
        )
    if condition == "full_context" and seed == 42:
        return
    gate_name = (
        "pilot-seed42.json"
        if condition == "full_context"
        else "ordinary-three-seed.json"
    )
    gate = root / "qualification" / gate_name
    if not gate.is_file() or not json.loads(gate.read_text()).get("passed"):
        raise ContractError(f"Match-pattern fit is gated by {gate_name}.")
    evidence = json.loads(gate.read_text())
    if evidence.get("corpus_sha256") != load_corpus().sha256:
        raise ContractError(
            "Qualification belongs to a different match-pattern corpus."
        )
    for gate_seed, digest in evidence["models"].items():
        path = (
            root
            / PROTOCOL
            / "full_context"
            / f"seed-{gate_seed}"
            / "ckpts/policy_weights/policy_epoch_999.pt"
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ContractError(
                "Qualification endpoint changed; no expansion admitted."
            )
    if condition != "full_context":
        forecast = root / "execution" / "forecast.json"
        if not forecast.is_file() or not json.loads(forecast.read_text()).get(
            "feasible"
        ):
            raise ContractError(
                "Match-pattern comparisons require a feasible paper-freeze forecast."
            )
