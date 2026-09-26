"""The paper figure notebooks build from saved records and never invent a roster.

The estimators and builders are defined inside the notebooks (cells tagged
``definitions``) on top of ``notebooks/figures_common.py``; this module loads
them the way a fresh kernel would and checks them on a temporary study layout
holding synthetic Key-to-Door endpoint panels for two cells on the development
split only. The figure must draw them on that split, label it provisional,
name the cells without records in its legend (never on a panel), refuse to
fill a column from a second split, and write every plotted table beside the
image.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib
import nbformat
import numpy as np
import pytest

from reasoned_icrl.analysis.statistics import cumulative_curves
from reasoned_icrl.experiments.benchmarks import EvaluationSplit
from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.evaluation import evaluation_directory
from reasoned_icrl.experiments.records import (
    BenchmarkEvent,
    BenchmarkRun,
    write_benchmark_results,
)
from tests.analysis.test_revised_statistics import SEEDS, TASKS, _attempt, _task

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"
BOOKS = sorted(NOTEBOOKS.glob("*.ipynb"))
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))


def _common() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "figures_common", NOTEBOOKS / "figures_common.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("figures_common", module)
    spec.loader.exec_module(module)
    return sys.modules["figures_common"]


common = _common()


def definitions(prefix: str) -> dict[str, Any]:
    """Execute the ``definitions`` cells of one notebook, as its kernel would."""
    book = nbformat.read(
        next(p for p in BOOKS if p.name.startswith(prefix)), as_version=4
    )
    # The kernel would give the cells a module in sys.modules; dataclasses resolve
    # string annotations through it, so the test module stands in for it.
    namespace: dict[str, Any] = {"__name__": __name__}
    exec("import matplotlib.pyplot as plt", namespace)
    for cell in book.cells:
        if cell.cell_type == "code" and "definitions" in cell.metadata.get("tags", []):
            exec(compile(cell.source, f"{prefix}:{cell.id}", "exec"), namespace)
    return namespace


PROTOCOL = common.ENVIRONMENTS["keydoor"].protocol
COUNTRECALL_PROTOCOL = common.ENVIRONMENTS["countrecall"].protocol
MAZERUNNER_PROTOCOL = common.ENVIRONMENTS["mazerunner"].protocol


def _contract(layout: Any, environment: str) -> Any:
    contract = layout.contract(environment)
    return replace(
        contract,
        evaluation=replace(
            contract.evaluation,
            splits={
                name: EvaluationSplit(split.source, len(TASKS), split.offset)
                for name, split in contract.evaluation.splits.items()
            },
        ),
    )


def _write_keydoor_panel(
    layout: Any, condition: str, *, split: str, history: str, doors: int
) -> None:
    contract = layout.contract("keydoor")
    for seed in SEEDS:
        run = layout.run_directory("keydoor", condition, seed)
        run.mkdir(parents=True, exist_ok=True)
        events = [
            replace(e, split=split, history=history, checkpoint="policy_epoch_999")  # type: ignore[arg-type]
            for task in contract.roster(split)
            for e in _task(condition, seed, task, doors)
        ]
        record = BenchmarkRun(
            PROTOCOL,
            "dark_key_to_door",
            condition,
            seed,
            "policy_epoch_999",
            split,
            history,
            "completed",
            checkpoint_rule="endpoint",
            retention="complete",
            metric="doors_completed",
            charged_calls=1000,
            physical_actions=900,
            reset_only_steps=100,
        )
        panel = run / "eval" / evaluation_directory(split, history, "endpoint")
        write_benchmark_results(
            panel / "benchmark_results.json", [contract], [record], events
        )


@pytest.fixture
def layout(tmp_path: Path) -> Any:
    built = common.default_layout(ROOT, outputs=tmp_path / "outputs")
    built = replace(
        built,
        contracts={env: _contract(built, env) for env in common.ALL_ENVIRONMENTS},
    )
    _write_keydoor_panel(
        built, "raw_summary", split="development", history="retained", doors=8
    )
    _write_keydoor_panel(
        built, "raw_summary", split="development", history="summary-cleared", doors=1
    )
    _write_keydoor_panel(
        built, "raw_segment", split="development", history="retained", doors=1
    )
    solved = {42: [1, 1], 100: [1, 1], 2026: [1, 0]}
    blind = {42: [1, 0], 100: [1, 0], 2026: [1, 0]}
    for history, outcome in (("retained", solved), ("summary-cleared", blind)):
        _write_tmaze_panel(
            built,
            "raw_summary",
            length=128,
            success=outcome,
            split="development",
            history=history,
        )
    _write_tmaze_panel(
        built,
        "raw_segment",
        length=128,
        success=blind,
        split="development",
        history="retained",
    )
    return built


@pytest.mark.parametrize("path", BOOKS, ids=lambda p: p.stem)
def test_notebook_is_output_free_and_reads_only(path: Path) -> None:
    """Committed without outputs; imports the shared plumbing and the study's
    pure estimators only, never the runtime, a subprocess or an evaluator."""
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    source = "\n".join(c.source for c in book.cells if c.cell_type == "code")
    for cell in book.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
            assert not cell.get("outputs") and cell.get("execution_count") is None
    modules = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "figures_common" in modules
    allowed = {"reasoned_icrl.analysis.statistics", "reasoned_icrl.analysis.tier"}
    for module in modules:
        if module.startswith("reasoned_icrl"):
            assert module in allowed or module.startswith(
                "reasoned_icrl.experiments."
            ), module
    for forbidden in (
        "reasoned_icrl.runtime",
        "subprocess",
        "runpy",
        "%run",
        "sha256",
    ):
        assert forbidden not in source, forbidden
    tags = [t for c in book.cells for t in c.metadata.get("tags", [])]
    assert tags.count("definitions") == 1 and tags.count("setup") == 1


def test_roster_selection_keeps_the_methods_together_on_one_split(layout: Any) -> None:
    contract = layout.contract("keydoor")
    selection = common.choose_split(
        layout,
        "keydoor",
        ["raw_summary", "raw_segment", "full_context"],
        contract=contract,
    )
    assert selection.split == "development" and selection.provisional
    assert selection.present == ("raw_summary", "raw_segment")
    assert selection.missing == ("full_context",)
    assert selection.tasks == len(TASKS)
    assert "provisional" in selection.title("Key-to-Door")
    # The title carries the environment and, only off the paper roster, the
    # roster; what is missing goes to the legend.
    assert selection.title("Key-to-Door").startswith("Key-to-Door")
    # One cell on the paper split does not move the column there while the
    # development split still holds more cells: the methods stay together.
    _write_keydoor_panel(
        layout, "raw_segment", split="confirmation", history="retained", doors=2
    )
    selection = common.choose_split(
        layout, "keydoor", ["raw_summary", "raw_segment"], contract=contract
    )
    assert selection.split == "development" and selection.present == (
        "raw_summary",
        "raw_segment",
    )
    # Once every cell has confirmation records the column moves to the paper
    # split; the development panels are never mixed in.
    _write_keydoor_panel(
        layout, "raw_summary", split="confirmation", history="retained", doors=5
    )
    selection = common.choose_split(
        layout, "keydoor", ["raw_summary", "raw_segment"], contract=contract
    )
    assert selection.split == "confirmation" and not selection.provisional
    assert selection.present == ("raw_summary", "raw_segment")
    # Equal coverage goes to the earlier preference.
    selection = common.choose_split(
        layout,
        "keydoor",
        ["raw_summary", "raw_segment", "full_context"],
        contract=contract,
    )
    assert selection.split == "confirmation" and selection.missing == ("full_context",)
    empty = common.choose_split(
        layout, "countrecall", ["raw_summary"], contract=layout.contract("countrecall")
    )
    assert empty.split is None and empty.title("CountRecall") == "CountRecall"
    notes = common.pending_handles(
        {"CountRecall": common.select_cells(["raw_summary", "full_context"])},
        common.select_cells(["raw_summary", "full_context"]),
    )
    assert [n.get_label() for n in notes] == ["CountRecall: every cell pending"]
    one = common.pending_handles(
        {"CountRecall": common.select_cells(["full_context"])}, common.DRAWN_CELLS
    )
    assert one[0].get_label() == "CountRecall: Full-history Transformer pending"
    assert all(n.get_color() == "none" for n in [*notes, *one])


def test_a_supplementary_cell_never_moves_the_paper_cells_off_their_roster(
    layout: Any,
) -> None:
    """The residual rewrite has development records only while the paper cells
    have confirmation records: the confirmation roster is chosen, not
    provisional, and the residual is listed as pending there."""
    for condition in ("raw_summary", "raw_segment"):
        _write_keydoor_panel(
            layout, condition, split="confirmation", history="retained", doors=4
        )
    _write_keydoor_panel(
        layout,
        common.RESIDUAL_CONDITION,
        split="development",
        history="retained",
        doors=4,
    )
    contract = layout.contract("keydoor")
    selection = common.choose_split(
        layout,
        "keydoor",
        ["raw_summary", "raw_segment", common.RESIDUAL_CONDITION],
        contract=contract,
    )
    assert selection.split == "confirmation" and not selection.provisional
    assert selection.present == ("raw_summary", "raw_segment")
    assert selection.missing == (common.RESIDUAL_CONDITION,)
    # With only the residual asked for, its own records decide the roster.
    alone = common.choose_split(
        layout, "keydoor", [common.RESIDUAL_CONDITION], contract=contract
    )
    assert alone.split == "development" and alone.provisional


def test_window_rates_refine_the_cumulative_curve(layout: Any) -> None:
    ns = definitions("figure2")
    selection = common.choose_split(
        layout, "keydoor", ["raw_summary"], contract=layout.contract("keydoor")
    )
    events = selection.panels["raw_summary"]
    windows = ns["keydoor_window_rates"](events, outer_length=500, width=50, samples=20)
    assert [(w["first"], w["last"]) for w in windows] == [
        (1 + 50 * i, 50 + 50 * i) for i in range(10)
    ]
    # Eight quick doors end at calls 10, 21, ..., 87: four in the first window,
    # four in the second, none later; doors per 100 calls doubles the count.
    assert [w["estimate"] for w in windows[:3]] == [8.0, 8.0, 0.0]
    total = sum(w["estimate"] * (w["last"] - w["first"] + 1) / 100 for w in windows)
    (final,) = [
        p for p in cumulative_curves(events, grid=(500,), samples=20) if p.step == 500
    ]
    assert total == pytest.approx(final.estimate) == pytest.approx(8.0)
    assert all(
        w["lower"] <= w["estimate"] <= w["upper"] and w["seeds"] == 3 for w in windows
    )


def test_attempt_view_stops_at_the_budget_bound(layout: Any) -> None:
    ns = definitions("figure2")
    contract = layout.contract("keydoor")
    selection = common.choose_split(
        layout, "keydoor", ["raw_summary", "raw_segment"], contract=contract
    )
    events = [e for rows in selection.panels.values() for e in rows or ()]
    rows, cutoff = ns["keydoor_attempt_view"](events, contract, samples=20)
    assert (
        cutoff == 9
    )  # a failure costs 50 steps + a reset; 9 * 51 - 1 <= 500 < 10 * 51 - 1
    assert all(r["shown"] == (r["attempt"] <= cutoff) for r in rows)
    quick = [r for r in rows if r["condition"] == "raw_summary" and r["attempt"] <= 8]
    assert all(r["estimate"] == 1.0 and r["risk_set"] == 6 for r in quick)


def _query(
    condition: str, seed: int, task: int, index: int, correct: bool
) -> BenchmarkEvent:
    return BenchmarkEvent(
        COUNTRECALL_PROTOCOL,
        "count_recall",
        condition,
        seed,
        "policy_epoch_999",
        "confirmation",
        "retained",
        task,
        task,
        0,
        "query",
        index,
        index,
        int(correct),
        1,
        (1.0 if correct else -1.0) / 103,
        answer=int(correct),
        true_count=1,
        start_step=index,
        end_step=index,
        checkpoint_rule="endpoint",
    )


def test_block_accuracy_scores_each_block_of_the_stream() -> None:
    ns = definitions("figure2")
    events = [
        _query("raw_segment", seed, task, index, correct=index <= 32 or index % 2 == 0)
        for seed in SEEDS
        for task in TASKS
        for index in range(1, 104)
    ]
    blocks = ns["countrecall_block_accuracy"](events, samples=20)
    assert [(b["first"], b["last"]) for b in blocks] == list(ns["COUNT_RECALL_BLOCKS"])
    assert blocks[0]["estimate"] == 1.0
    assert blocks[1]["estimate"] == pytest.approx(0.5)
    assert blocks[3]["estimate"] == pytest.approx(
        3 / 7
    )  # queries 98, 100, 102 of 97..103


def test_learning_helpers_map_labels_to_measured_calls_and_censor_crossings() -> None:
    ns = definitions("figure4")
    # AMAGO logs the collection counter after the epoch it belongs to, so the
    # telemetry row of Epoch e reads 8,000 (e + 2): label 999 sits at the 8M
    # endpoint and label 500 at 4.008M.
    rows = [
        {"panel": "train-rollout", "Epoch": epoch, "total_frames": 8000 * (epoch + 2)}
        for epoch in range(0, 999)
    ]
    calls = ns["label_calls"](rows)
    assert calls[999] == 8_000_000 and calls[500] == 4_008_000
    points = [(50, 400_000, 0.2), (100, 800_000, 0.95), (150, 1_200_000, 0.7)]
    assert ns["first_crossing"](points, 0.9) == 800_000
    assert ns["first_crossing"](points, 0.99) is None
    smoothed = ns["moving_average"]([0.0, 1.0, 0.0, 1.0, 0.0], 3)
    assert smoothed.shape == (5,) and smoothed[2] == pytest.approx(2 / 3)
    assert ns["moving_average"]([1.0, 2.0], 5).tolist() == [1.0, 2.0]


def test_figure2_draws_what_exists_and_writes_its_tables(
    layout: Any, tmp_path: Path
) -> None:
    ns = definitions("figure2")
    # The paper's RSM is the residual rewrite; the
    # fixture holds the overwrite cell's panels, so it is named as the method.
    assert ns["Figure2Options"]().method == common.METHOD_CONDITION
    options = ns["Figure2Options"](
        samples=20,
        method="raw_summary",
        cells=common.select_cells(["raw_summary", "raw_segment", "full_context"]),
    )
    built = ns["build_figure2"](layout, options)
    keydoor = built.notes["keydoor"]
    assert keydoor["split"] == "development" and keydoor["provisional"] is True
    assert keydoor["present"] == ["raw_summary", "raw_segment"] and keydoor[
        "missing"
    ] == ["full_context"]
    assert built.notes["countrecall"]["split"] is None
    tmaze = built.notes["tmaze"]
    assert tmaze["split"] == "development" and tmaze["provisional"] is True
    assert tmaze["present"] == ["raw_summary", "raw_segment"]
    assert tmaze["cue_blind"] == 0.5
    assert {
        "keydoor_cumulative",
        "keydoor_windows",
        "keydoor_attempts",
        "keydoor_summary_cleared",
        "tmaze_success",
        "tmaze_cumulative",
        "tmaze_interventions",
        "tmaze_cue_side",
    } <= set(built.tables)
    assert "countrecall_cumulative" not in built.tables
    # The intervention view holds the retained panel of both cells and the
    # summary-cleared panel of RSM only; success is read per (seed, task) unit.
    interventions = {
        (r["condition"], r["history"]): r for r in built.tables["tmaze_interventions"]
    }
    assert set(interventions) == {
        ("raw_summary", "retained"),
        ("raw_summary", "summary-cleared"),
        ("raw_segment", "retained"),
    }
    assert interventions[("raw_summary", "retained")]["estimate"] == pytest.approx(
        5 / 6
    )
    assert interventions[("raw_summary", "summary-cleared")]["estimate"] == 0.5
    assert len(built.tables["tmaze_cue_side"]) == 6
    titles = [ax.get_title(loc="left") for ax in built.figure.axes]
    assert any("provisional" in t for t in titles)
    assert not any("pending" in t or "no records" in t for t in titles)
    legend_title = built.figure.legends[0].get_title().get_text()
    assert "CountRecall: every cell pending" in legend_title
    assert "MazeRunner: every cell pending" in legend_title
    assert built.sources and all(
        s.endswith(("benchmark_results.json", "benchmark_secondary.json"))
        for s in built.sources
    )
    assert set(built.panels) == {
        "keydoor_cumulative_doors",
        "keydoor_success_by_attempt",
        "keydoor_success_by_attempt_difference",
        "keydoor_doors_per_100_calls",
        "keydoor_tasks_reaching_attempt",
        "keydoor_cumulative_doors_summary_cleared",
        "countrecall_cumulative_accuracy",
        "countrecall_accuracy_per_block",
        "mazerunner_goals_reached",
        "mazerunner_interventions",
        "tmaze_cumulative_goals",
        "tmaze_success_bars",
        "tmaze_success_by_cue",
        "tmaze_interventions",
    }
    # One row, one panel per main-text benchmark (door success by attempt,
    # cumulative exact accuracy, maps reaching each goal); the T-Maze and
    # every other view are standalone only.
    assert len(built.figure.axes) == 3
    assert not any(ax.patches for ax in built.figure.axes)
    assert built.figure.axes[2].get_title(loc="left") == "MazeRunner"
    assert built.notes["mazerunner"]["split"] is None
    # The T-Maze standalone panel is still the intervention view.
    tmaze_axis = built.panels["tmaze_interventions"].axes[0]
    assert tmaze_axis.get_title(loc="left").endswith("interventions")
    legend_labels = {t.get_text() for t in built.figure.legends[0].get_texts()}
    assert "RSM, summary cleared" not in legend_labels
    # The RSM family, the memoryless control and every comparator, Memo with
    # fixed segments among them.
    assert ns["FIGURE2_CELLS"] == common.DRAWN_CELLS
    assert [c.label for c in common.DRAWN_CELLS if c.condition == "raw_segment"] == [
        "w/o memory"
    ]
    cumulative = built.tables["tmaze_cumulative"]
    assert [r["episode"] for r in cumulative if r["condition"] == "raw_summary"] == [
        1,
        2,
    ]
    last = [r for r in cumulative if r["condition"] == "raw_summary"][-1]
    assert (last["seed_42"], last["seed_100"], last["seed_2026"]) == (2, 2, 1)
    assert built.notes["tmaze"]["cue_up_fraction"] == 0.5
    # The paired-by-attempt panel: the method against each other cell on the
    # tasks whose attempt finished under both, the memoryless control omitted.
    differences = built.tables["keydoor_attempt_differences"]
    assert {r["condition"] for r in differences} == {"raw_segment"}
    assert all(r["method"] == "raw_summary" for r in differences)
    assert all(r["units"] > 0 for r in differences)
    assert all(len(panel.axes) == 1 for panel in built.panels.values())
    written = {p.name for p in built.save(tmp_path / "figures", tmp_path / "data")}
    assert {
        "figure2_in_context_adaptation.pdf",
        "figure2_in_context_adaptation.png",
        "figure2_in_context_adaptation_panel_keydoor_success_by_attempt.png",
        "figure2_in_context_adaptation_panel_countrecall_accuracy_per_block.png",
        "figure2_in_context_adaptation_keydoor_attempts.csv",
        "figure2_in_context_adaptation_panel_tmaze_interventions.png",
        "figure2_in_context_adaptation_tmaze_cue_side.csv",
        "figure2_in_context_adaptation_notes.json",
    } <= written
    # The composite is saved as vector PDF beside the PNG; panels are PNG only.
    assert not any(name.endswith(".pdf") and "_panel_" in name for name in written)
    built.close()
    notes = json.loads(
        (tmp_path / "data" / "figure2_in_context_adaptation_notes.json").read_text()
    )
    assert notes["notes"]["keydoor"]["split"] == "development"
    assert notes["sources"] == sorted(set(built.sources))
    assert (tmp_path / "figures" / "figure2_in_context_adaptation.png").is_file()


def test_figure4_excludes_fits_without_a_scored_full_budget(layout: Any) -> None:
    ns = definitions("figure4")
    built = ns["build_figure4"](
        layout, ns["Figure4Options"](cells=common.select_cells(["raw_summary"]))
    )
    assert [(r["environment"], r["condition"]) for r in built.tables["excluded"]] == [
        ("keydoor", "raw_summary"),
        ("countrecall", "raw_summary"),
        ("mazerunner", "raw_summary"),
        ("tmaze", "raw_summary"),
    ]
    assert built.tables["first_crossing"] == []
    assert set(built.panels) == {
        "keydoor_collection_return",
        "keydoor_development_primary",
        "countrecall_collection_return",
        "countrecall_development_primary",
        "mazerunner_collection_return",
        "mazerunner_development_primary",
        "tmaze_collection_return",
        "tmaze_development_primary",
    }
    assert len(built.figure.axes) == 8
    assert built.notes["excluded_in_legend"]["MazeRunner"] == ["raw_summary"]
    built.close()
    # The T-Maze is an appendix build of its own.
    appendix = ns["build_figure4"](
        layout,
        ns["Figure4Options"](
            cells=common.select_cells(["raw_summary"]),
            environments=("tmaze",),
            name="figureA1_learning_tmaze",
        ),
    )
    assert appendix.name == "figureA1_learning_tmaze" and len(appendix.figure.axes) == 2
    appendix.close()


def test_telemetry_reader_recovers_rows_from_a_damaged_line(tmp_path: Path) -> None:
    """A resume after a full disk can leave a truncated row with the next row on
    the same line: the complete rows are kept, the fragment dropped and noted."""
    ns = definitions("figure4")
    good = json.dumps({"panel": "train-rollout", "Epoch": 1, "total_frames": 8000})
    fragment = '{"panel": "train-update", "Epoch": 2, "Actor Lo'
    following = json.dumps(
        {"panel": "train-rollout", "Epoch": 2, "total_frames": 16000}
    )
    (tmp_path / "training_metrics.jsonl").write_text(
        good + "\n" + fragment + following + "\n"
    )
    repairs: list[dict[str, object]] = []
    rows = ns["read_telemetry"](tmp_path, repairs)
    assert [r["Epoch"] for r in rows] == [1, 2]
    assert repairs == [
        {"file": str(tmp_path / "training_metrics.jsonl"), "line": 2, "recovered": 1}
    ]


def test_sustained_crossing_needs_the_threshold_held_to_the_endpoint() -> None:
    ns = definitions("figure4")
    points = [(50, 1.0, 1.0), (100, 2.0, 0.4), (150, 3.0, 1.0), (200, 4.0, 1.0)]
    assert ns["first_crossing"](points, 1.0) == 1.0
    assert ns["sustained_crossing"](points, 1.0) == 3.0
    assert ns["sustained_crossing"](points, 0.95) == 3.0
    collapsed = [(50, 1.0, 1.0), (100, 2.0, 1.0), (150, 3.0, 0.6)]
    assert ns["first_crossing"](collapsed, 1.0) == 1.0
    assert ns["sustained_crossing"](collapsed, 1.0) is None
    assert [t[2] for t in ns["THRESHOLDS"]["tmaze"]] == ["sustained", "sustained"]


def test_csv_and_preview_take_the_union_of_columns(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [{"a": 1.0}, {"a": 2.0, "b": "x"}]
    common.write_csv(tmp_path / "t.csv", rows)
    assert (tmp_path / "t.csv").read_text().splitlines()[0] == "a,b"
    preview = common.markdown_preview(rows, limit=1)
    assert preview.splitlines()[0] == "| a | b |" and "more rows" in preview
    assert common.markdown_preview([]) == "_empty_"
    assert np.isfinite(common.BAND_ALPHA)


# ---------------------------------------------------------------- Figure 3


def _horizon_attempts(
    condition: str, seed: int, task: int, *, budget: int, doors_per_window: list[int]
) -> list[BenchmarkEvent]:
    """Attempts tiling ``budget`` calls exactly: per 500-call window the given
    number of quick doors (9 steps + reset, a multiple of 5) then time-outs
    (49 steps + reset; the task's last attempt runs 50 steps to the budget)."""
    rows: list[BenchmarkEvent] = []
    step, index = 1, 1
    windows = budget // 500
    for w, doors in enumerate(doors_per_window):
        assert doors % 5 == 0
        fails = (500 - doors * 10) // 50
        for _ in range(doors):
            rows.append(
                _attempt(
                    condition, seed, task, index, start=step, steps=9, success=True
                )
            )
            step += 10
            index += 1
        for f in range(fails):
            final = w == windows - 1 and f == fails - 1
            steps = 50 if final else 49
            rows.append(
                _attempt(
                    condition, seed, task, index, start=step, steps=steps, success=False
                )
            )
            step += steps + (0 if final else 1)
            index += 1
    assert step - 1 == budget
    return [
        replace(
            r, split="confirmation", checkpoint="policy_epoch_999", outer_length=budget
        )
        for r in rows
    ]


def _write_horizon_panel(
    layout: Any,
    condition: str,
    *,
    budget: int,
    doors_per_window: list[int],
    check: str = "passed",
) -> None:
    contract = layout.contract("keydoor")
    extended = common.horizon_contract(contract, budget)
    for seed in SEEDS:
        run = layout.run_directory("keydoor", condition, seed)
        events = [
            e
            for task in contract.roster("confirmation")
            for e in _horizon_attempts(
                condition, seed, task, budget=budget, doors_per_window=doors_per_window
            )
        ]
        record = BenchmarkRun(
            PROTOCOL,
            "dark_key_to_door",
            condition,
            seed,
            "policy_epoch_999",
            "confirmation",
            "retained",
            "completed",
            checkpoint_rule="endpoint",
            retention="complete",
            metric="doors_completed",
            charged_calls=budget * len(TASKS),
            physical_actions=budget * len(TASKS) - 40,
            reset_only_steps=40,
            outer_length=budget,
        )
        panel = run / "eval" / f"confirmation-retained-endpoint-h{budget}"
        write_benchmark_results(
            panel / "benchmark_results.json", [extended], [record], events
        )
        (panel / "prefix_check.json").write_text(
            json.dumps(
                {"status": check, "base_panel": "confirmation-retained-endpoint-h500"}
            )
        )
        (run / "systems.json").write_text(
            json.dumps(
                {
                    "persistent_state_bytes": 249_868,
                    "state_tensor_bytes": {
                        "memory": 4096,
                        "lengths": 4,
                        "keys": 245_768,
                    },
                    "context_actions": 500,
                }
            )
        )


def _write_tmaze_panel(
    layout: Any,
    condition: str,
    *,
    length: int,
    success: dict[int, list[int]],
    split: str = "confirmation",
    history: str = "retained",
) -> None:
    """One trained-corridor panel per seed; ``success[seed]`` holds each task's
    outcome. The panel is the trained-length endpoint panel
    (``<split>-<history>-endpoint``, with the evaluator's secondary summary)."""
    contract = layout.contract("tmaze")
    for seed in SEEDS:
        run = layout.run_directory("tmaze", condition, seed)
        events = [
            BenchmarkEvent(
                common.ENVIRONMENTS["tmaze"].protocol,
                "tmaze",
                condition,
                seed,
                "policy_epoch_999",
                split,
                history,
                task,
                task,
                0,
                "episode",
                1,
                length + 1,
                int(won),
                1,
                1.0 if won else -1.0,
                start_step=1,
                end_step=length + 1,
                checkpoint_rule="endpoint",
                outer_length=length + 1,
            )
            for task, won in zip(contract.roster(split), success[seed], strict=True)
        ]
        record = BenchmarkRun(
            common.ENVIRONMENTS["tmaze"].protocol,
            "tmaze",
            condition,
            seed,
            "policy_epoch_999",
            split,
            history,
            "completed",
            checkpoint_rule="endpoint",
            retention="scored-band",
            metric="goal_success",
            charged_calls=(length + 1) * len(TASKS),
            physical_actions=(length + 1) * len(TASKS),
            reset_only_steps=0,
            outer_length=length + 1,
        )
        panel = run / "eval" / f"{split}-{history}-endpoint"
        write_benchmark_results(
            panel / "benchmark_results.json", [contract], [record], events
        )
        won = [int(w) for w in success[seed]]
        (panel / "benchmark_secondary.json").write_text(
            json.dumps(
                {
                    "tasks": float(len(won)),
                    "goal_success": float(sum(won) / len(won)),
                    "success_cue_up": float(won[0]),
                    "success_cue_down": float(won[-1]),
                    "tasks_cue_up": 1.0,
                    "tasks_cue_down": float(len(won) - 1),
                    "forward_moves_mean": float(length),
                }
            )
        )


def _write_countrecall_panel(
    layout: Any,
    condition: str,
    *,
    history: str = "retained",
    accuracy_by_stratum: dict[int, float],
    strata: bool = True,
) -> None:
    """A complete confirmation stream panel per seed (103 queries per stream),
    every queried suit first dealt in block 1, so the stratum (segments since
    the first evidence) is the query's C32 block: 0 for queries 1-32, 1 for
    33-64, 2 for 65-96, 3 for 97-103. The summary carriers record the stratum
    (``strata``), the full history and GRU do not. ``accuracy_by_stratum``
    gives the correct fraction of the roster's streams per stratum."""
    contract = layout.contract("countrecall")
    tasks = list(contract.roster("confirmation"))
    for seed in SEEDS:
        run = layout.run_directory("countrecall", condition, seed)
        run.mkdir(parents=True, exist_ok=True)
        events = []
        for t_index, task in enumerate(tasks):
            for index in range(1, 104):
                stratum = min((index - 1) // 32, 3)
                correct = (t_index + 1) / len(tasks) <= accuracy_by_stratum[stratum]
                event = replace(
                    _query(condition, seed, task, index, correct),
                    history=history,
                    writes_before_decision=stratum if strata else None,
                    evidence_age_writes=stratum if strata else None,
                    count_before_current_segment=int(stratum > 0) if strata else None,
                    count_in_current_segment=int(stratum == 0) if strata else None,
                )
                events.append(event)  # type: ignore[arg-type]
        record = BenchmarkRun(
            COUNTRECALL_PROTOCOL,
            "count_recall",
            condition,
            seed,
            "policy_epoch_999",
            "confirmation",
            history,
            "completed",
            checkpoint_rule="endpoint",
            retention="complete",
            metric="exact_accuracy",
            charged_calls=104 * len(tasks),
            physical_actions=104 * len(tasks),
            reset_only_steps=0,
        )
        panel = run / "eval" / evaluation_directory("confirmation", history, "endpoint")
        write_benchmark_results(
            panel / "benchmark_results.json", [contract], [record], events
        )
        (run / "systems.json").write_text(
            json.dumps(
                {
                    "persistent_state_bytes": 4096,
                    "state_tensor_bytes": {"memory": 4096},
                    "context_actions": 103,
                }
            )
        )


def _write_mazerunner_panel(
    layout: Any,
    condition: str,
    *,
    size: int = 15,
    goals_by_seed: dict[int, list[int]],
    split: str = "development",
    history: str = "retained",
    plain: bool = True,
) -> None:
    """One MazeRunner episode per roster map and seed: ``goals_by_seed`` gives
    the goals completed per map (of three); an episode ends at the third goal
    or at the budget. At the trained size the plain endpoint panel is written
    (``plain``), a larger size lands under the adapter's ``-h<S>`` panel."""
    contract = layout.contract("mazerunner")
    environment = contract.environment
    native = int(environment.size)
    budget = common.maze_budget(size, environment)
    tasks = list(contract.roster(split))
    for seed in SEEDS:
        run = layout.run_directory("mazerunner", condition, seed)
        run.mkdir(parents=True, exist_ok=True)
        events = []
        for task, goals in zip(tasks, goals_by_seed[seed], strict=True):
            steps = budget if goals < 3 else max(3, budget // 2)
            events.append(
                BenchmarkEvent(
                    MAZERUNNER_PROTOCOL,
                    "mazerunner",
                    condition,
                    seed,
                    "policy_epoch_999",
                    split,
                    history,
                    task,
                    task,
                    0,
                    "episode",
                    1,
                    steps,
                    goals,
                    3,
                    float(goals),
                    start_step=1,
                    end_step=steps,
                    checkpoint_rule="endpoint",
                    outer_length=None if (plain and size == native) else budget,
                )
            )
        record = BenchmarkRun(
            MAZERUNNER_PROTOCOL,
            "mazerunner",
            condition,
            seed,
            "policy_epoch_999",
            split,
            history,
            "completed",
            checkpoint_rule="endpoint",
            retention=contract.evaluation.retention,
            metric="goal_fraction",
            charged_calls=budget * len(tasks),
            physical_actions=budget * len(tasks),
            reset_only_steps=0,
            outer_length=None if (plain and size == native) else budget,
        )
        if plain and size == native:
            panel = run / "eval" / evaluation_directory(split, history, "endpoint")
            written = contract
        else:
            panel = (
                run
                / "eval"
                / evaluation_directory(split, history, "endpoint", horizon=size)
            )
            written = common.horizon_contract(contract, size)
        write_benchmark_results(
            panel / "benchmark_results.json", [written], [record], events
        )
        (run / "systems.json").write_text(
            json.dumps(
                {
                    "persistent_state_bytes": 4096,
                    "state_tensor_bytes": {"memory": 4096},
                    "context_actions": 500,
                }
            )
        )


def _write_streams_panel(
    layout: Any,
    condition: str,
    *,
    streams: int,
    accuracy_by_pair: list[float],
    history: str = "retained",
) -> None:
    """A CountRecall task continued over ``streams`` deck pairs per roster task
    and seed (103 scored decisions per pair): ``accuracy_by_pair`` gives the
    correct fraction of the roster's tasks in each pair."""
    contract = layout.contract("countrecall")
    tasks = list(contract.roster("confirmation"))
    continued = common.streams_contract(contract, streams)
    outer = continued.environment.outer_length
    for seed in SEEDS:
        run = layout.run_directory("countrecall", condition, seed)
        run.mkdir(parents=True, exist_ok=True)
        events = []
        for t_index, task in enumerate(tasks):
            for pair in range(streams):
                for index in range(1, 104):
                    correct = (t_index + 1) / len(tasks) <= accuracy_by_pair[pair]
                    decision = pair * 103 + index  # scored decisions, contiguous
                    events.append(
                        replace(
                            _query(condition, seed, task, index, correct),
                            history=history,
                            event_index=decision,
                            step=decision,
                            start_step=decision,
                            end_step=decision,
                            outer_length=outer,
                        )
                    )
        record = BenchmarkRun(
            COUNTRECALL_PROTOCOL,
            "count_recall",
            condition,
            seed,
            "policy_epoch_999",
            "confirmation",
            history,
            "completed",
            checkpoint_rule="endpoint",
            retention="complete",
            metric="exact_accuracy",
            charged_calls=outer * len(tasks),
            physical_actions=103 * streams * len(tasks),
            reset_only_steps=(streams - 1) * len(tasks),
            outer_length=outer,
        )
        panel = (
            run
            / "eval"
            / evaluation_directory("confirmation", history, "endpoint", streams=streams)
        )
        write_benchmark_results(
            panel / "benchmark_results.json", [continued], [record], events
        )


@pytest.fixture
def horizon_layout(tmp_path: Path) -> Any:
    built = common.default_layout(ROOT, outputs=tmp_path / "outputs")
    built = replace(
        built,
        contracts={env: _contract(built, env) for env in common.ALL_ENVIRONMENTS},
    )
    _write_horizon_panel(built, "raw_summary", budget=1000, doors_per_window=[10, 20])
    _write_horizon_panel(built, "raw_segment", budget=1000, doors_per_window=[5, 5])
    _write_countrecall_panel(
        built, "raw_summary", accuracy_by_stratum={0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5}
    )
    _write_countrecall_panel(
        built,
        "raw_summary",
        history="summary-cleared",
        accuracy_by_stratum={0: 1.0, 1: 0.5, 2: 0.5, 3: 0.5},
    )
    _write_countrecall_panel(
        built,
        "full_context",
        accuracy_by_stratum={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        strata=False,
    )
    _write_streams_panel(
        built, "raw_summary", streams=4, accuracy_by_pair=[1.0, 1.0, 1.0, 0.5]
    )
    _write_streams_panel(
        built, "full_context", streams=4, accuracy_by_pair=[1.0, 1.0, 0.5, 0.0]
    )
    _write_streams_panel(
        built,
        "raw_summary",
        streams=4,
        accuracy_by_pair=[1.0, 1.0, 1.0, 1.0],
        history="attempt-cleared",
    )
    for size, goals in ((15, [3, 3]), (17, [3, 2]), (21, [1, 0])):
        _write_mazerunner_panel(
            built,
            "raw_summary",
            size=size,
            goals_by_seed={42: goals, 100: goals, 2026: goals},
            split="confirmation",
        )
    _write_mazerunner_panel(
        built,
        "full_context",
        size=15,
        goals_by_seed={42: [3, 3], 100: [3, 3], 2026: [3, 3]},
        split="confirmation",
    )
    return built


def test_countrecall_strata_are_borrowed_from_the_method(horizon_layout: Any) -> None:
    """Every cell is scored on the same queries per stratum: the full history,
    which records no writes, takes the strata of the method's panel, or of the
    overwrite cell's while the method's is absent (the strata depend on the
    task and the fixed C32 schedule); the summary-cleared trace is the method's
    own and is left out while its panel is absent; the opportunity counts are
    the reference's."""
    ns = definitions("figure3")
    fallback = ns["countrecall_retention"](
        horizon_layout,
        horizon_layout.contract("countrecall"),
        ns["Figure3Options"](samples=20, method="raw_summary_residual"),
        [],
    )
    # The ablation (RSM-R) as the method has no panel here: the other write
    # rule stands in for the strata and no cleared trace is invented.
    assert fallback["reference"] == "raw_summary" and fallback["cleared"] == []
    options = ns["Figure3Options"](samples=20, method="raw_summary")
    sources: list[str] = []
    data = ns["countrecall_retention"](
        horizon_layout, horizon_layout.contract("countrecall"), options, sources
    )
    assert data["reference"] == "raw_summary"
    assert [c.condition for c in data["present"]] == ["raw_summary", "full_context"]
    # Every memory carrier is requested now, so a
    # cell without a panel on the roster is reported as missing, Memo fixed
    # included; the memoryless control is not a carrier and is never asked for.
    assert {"full_gru", "memo", "memo_fixed"} <= set(data["missing"])
    assert "raw_segment" not in data["missing"]
    rows = {(r["condition"], r["stratum"]): r for r in data["rows"]}
    assert rows[("full_context", 3)]["estimate"] == 1.0
    assert rows[("raw_summary", 3)]["estimate"] == 0.5
    assert rows[("raw_summary", 1)]["estimate"] == 1.0
    assert [rows[("full_context", s)]["queries_per_seed"] for s in range(4)] == [
        32 * len(TASKS),
        32 * len(TASKS),
        32 * len(TASKS),
        7 * len(TASKS),
    ]
    cleared = {r["stratum"]: r["estimate"] for r in data["cleared"]}
    assert cleared == {0: 1.0, 1: 0.5, 2: 0.5, 3: 0.5}
    endpoints = {r["condition"]: r["estimate"] for r in data["endpoints"]}
    assert endpoints["full_context"] == 1.0
    assert endpoints["raw_summary"] == pytest.approx((1.0 + 96 / 103) / 2)
    assert {r["condition"]: r["bytes"] for r in data["states"]} == {
        "raw_summary": 4096,
        "full_context": 4096,
    }
    assert any("summary-cleared" in s for s in sources)


def test_horizon_contract_extends_only_the_budget(layout: Any) -> None:
    keydoor = layout.contract("keydoor")
    extended = common.horizon_contract(keydoor, 4000)
    assert extended.environment.outer_length == 4000
    assert list(extended.roster("confirmation")) == list(keydoor.roster("confirmation"))
    with pytest.raises(ContractError, match="no extended-horizon adapter"):
        common.horizon_contract(layout.contract("tmaze"), 512)


def test_window_rates_and_contrasts_are_paired_per_window(horizon_layout: Any) -> None:
    ns = definitions("figure3")
    # The figure's default cells are the memory carriers; no carry is asked for
    # here to check the paired carry contrast at the horizon.
    assert "raw_segment" not in {c.condition for c in ns["HORIZON_CELLS"]}
    options = ns["Figure3Options"](samples=20, cells=common.PAPER_CELLS)
    sources: list[str] = []
    horizon, cubes = ns["choose_horizon"](
        horizon_layout, horizon_layout.contract("keydoor"), options, sources
    )
    assert horizon == 1000  # the only horizon with panels; four cells pending there
    assert cubes["raw_summary"].shape == (3, len(TASKS), 2) and cubes["memo"] is None
    windows = ns["window_rates"](cubes, samples=20, seed=0, seeds=SEEDS)
    rsm = [w for w in windows if w["condition"] == "raw_summary"]
    assert [w["estimate"] for w in rsm] == [10.0, 20.0]
    assert [w["cumulative_estimate"] for w in rsm] == [10.0, 30.0]
    assert rsm[1]["first_call"] == 501 and rsm[1]["last_call"] == 1000
    contrasts = {
        r["contrast"]: r
        for r in ns["paired_contrasts"](
            cubes, samples=20, seed=0, seeds=SEEDS, method="raw_summary"
        )
    }
    # The method is RSM-O here: no write-rule row.
    assert "RSM-O - RSM-O" not in contrasts
    rise = contrasts["H4a RSM-O last window - first window"]
    assert rise["estimate"] == 10.0 and rise["consistent_gain"]
    assert not rise["within_delta_every_seed"] and rise["no_fall_every_seed"]
    carry = contrasts["RSM-O - no carry"]
    assert carry["estimate"] == 15.0 and carry["positive_seeds"] == 3
    assert "H4b RSM-O - Memo" not in contrasts  # Memo has no panel: never invented
    assert len(sources) == 6 and all("-h1000" in s for s in sources)


def test_state_bytes_follow_each_cells_growth_rule() -> None:
    """Two accountings of one growth rule per cell: the peak an actor holds
    within the task (the tables) and the mean over its records (Figure 3's state row);
    the allocation is a third column."""
    ns = definitions("figure3")
    mean = ns["state_bytes_at"]  # Figure 3's Key-to-Door name for the mean

    def peak(systems, condition, horizon):
        return common.live_state_bytes(
            systems, condition, horizon + 1, segment_length=32, summary_tokens=4
        )

    fixed = {"persistent_state_bytes": 3072, "state_tensor_bytes": {"hidden": 3072}}
    assert peak(fixed, "full_gru", 4000)[0] == 3072
    assert mean(fixed, "full_gru", 4000, segment_length=32, summary_tokens=4)[0] == 3072
    history = {
        "persistent_state_bytes": 3_078_148,
        "state_tensor_bytes": {
            "key_cache": 1_539_072,
            "val_cache": 1_539_072,
            "seq_lens": 4,
        },
        "context_actions": 500,
    }
    # One cache slot (6,144 bytes) per record over 4 fixed bytes: the peak is
    # the last record, the mean (records + 1) / 2 slots.
    assert peak(history, "full_context", 500)[0] == 3_078_148
    assert peak(history, "full_context", 4000)[0] == 24_582_148
    assert (
        mean(history, "full_context", 500, segment_length=32, summary_tokens=4)[0]
        == 4 + 6144 * 251
    )
    assert (
        mean(history, "full_context", 4000, segment_length=32, summary_tokens=4)[0]
        == 4 + 6144 * 2001
    )
    memo = {
        "state_growth": {
            "bytes_per_slot": 6144,
            "counter_bytes": 12,
            "live_slots_by_prefix": {"1": 1, "32": 32, "33": 5, "64": 36, "501": 81},
        }
    }
    # Memo's schedule peaks on the last record of a full segment (every
    # earlier summary plus the 32 open records): a 4,000-call task peaks at
    # record 4,000 with 124 summaries written, a 103-query stream at record 96;
    # the mean averages the schedule over every record of the task.
    total, method = peak(memo, "memo", 4000)
    assert total == (4 * 124 + 32) * 6144 + 12 and "peak" in method
    assert peak(memo, "memo", 103)[0] == (4 * 2 + 32) * 6144 + 12
    slots = [common.memo_live_slots(p, 32, 4) for p in range(1, 4002)]
    total, method = mean(memo, "memo", 4000, segment_length=32, summary_tokens=4)
    assert total == round(sum(slots) / len(slots) * 6144 + 12) and "mean" in method
    assert common.memo_peak_slots(20, 32, 4) == 20
    allocated = {"state_growth": {**memo["state_growth"], "capacity_slots": 96}}
    assert common.allocated_state_bytes(allocated) == (
        96 * 6144 + 12,
        "capacity allocated for the trained horizon",
    )
    assert common.allocated_state_bytes(fixed) == (3072, "measured persistent state")
    broken = {
        "state_growth": {**memo["state_growth"], "live_slots_by_prefix": {"33": 6}}
    }
    with pytest.raises(ValueError):
        mean(broken, "memo", 4000, segment_length=32, summary_tokens=4)
    with pytest.raises(ValueError):
        peak(broken, "memo", 4000)


def test_figure3_draws_what_exists_and_writes_its_tables(
    horizon_layout: Any, tmp_path: Path
) -> None:
    ns = definitions("figure3")
    # The paper's long-run figure: Dark Key-to-Door
    # alone — the rate per window, the total, the last window against live
    # state — with only its standalone panels; every environment's tables are
    # still computed from the one read.
    # The fixture holds the overwrite cell's panels, so it is named as the method.
    prepared = ns["prepare_figure3"](
        horizon_layout,
        ns["Figure3Options"](
            samples=20, cells=common.PAPER_CELLS, method="raw_summary"
        ),
    )
    main = ns["build_figure3"](
        horizon_layout,
        ns["Figure3Options"](
            samples=20, cells=common.PAPER_CELLS, method="raw_summary"
        ),
        prepared=prepared,
    )
    assert len(main.figure.axes) == 3
    titles = [ax.get_title(loc="left") for ax in main.figure.axes]
    assert [t.split(", ")[-1] for t in titles] == ["rate", "total", "retention"]
    assert all(t.startswith("Dark Key-to-Door") for t in titles)
    assert set(main.panels) == {
        "keydoor_doors_per_window",
        "keydoor_cumulative_doors",
        "keydoor_retention_against_state",
    }
    assert {"countrecall_retention", "mazerunner_sizes"} <= set(main.tables)
    assert not any(key.startswith("tmaze") for key in main.tables)
    # The continued pairs: the longest count every cell has on every seed, the
    # method's cleared companion, the paired contrasts per pair.
    streams = main.notes["countrecall"]
    assert streams["streams"] == 4 and streams["streams_present"] == [
        "raw_summary",
        "full_context",
    ]
    rows = {
        (r["condition"], r["history"], r["pair"]): r["estimate"]
        for r in main.tables["countrecall_streams"]
    }
    assert rows[("raw_summary", "retained", 4)] == pytest.approx(0.5)
    assert rows[("full_context", "retained", 3)] == pytest.approx(0.5)
    assert rows[("full_context", "retained", 4)] == pytest.approx(0.0)
    assert rows[("raw_summary", "attempt-cleared", 4)] == pytest.approx(1.0)
    contrasts = {
        (r["contrast"], r["pair"]): r["estimate"]
        for r in main.tables["countrecall_stream_contrasts"]
    }
    assert contrasts[("RSM-O - full history", 4)] == pytest.approx(0.5)
    assert contrasts[("RSM-O - full history", 1)] == pytest.approx(0.0)
    # The maze sizes: the seed rows per size, the state panel on the largest
    # size every present cell finished (15, the full history has no larger one).
    maze = main.notes["mazerunner"]
    assert maze["sizes"] == [15, 17, 21] and maze["state_size"] == 15
    assert maze["budgets"] == {15: 500, 17: 642, 19: 802, 21: 980, 25: 1389}
    sizes = {
        (r["condition"], r["size"], r["seed"]): r["goal_fraction"]
        for r in main.tables["mazerunner_sizes"]
    }
    assert sizes[("raw_summary", 15, 42)] == pytest.approx(1.0)
    assert sizes[("raw_summary", 17, 42)] == pytest.approx(5 / 6)
    assert sizes[("raw_summary", 21, 42)] == pytest.approx(1 / 6)
    assert sizes[("full_context", 15, 100)] == pytest.approx(1.0)
    assert ("full_context", 17, 42) not in sizes
    main.close()
    # A second figure from the same payload reads nothing again.
    reused = ns["build_figure3"](
        horizon_layout,
        ns["Figure3Options"](
            samples=20,
            cells=common.PAPER_CELLS,
            method="raw_summary",
            environments=common.PAPER_ENVIRONMENTS,
            name="figureA5_beyond_the_training_horizon_all_benchmarks",
        ),
        prepared=prepared,
    )
    assert len(reused.figure.axes) == 6
    assert reused.sources == prepared["sources"]
    reused.close()
    # A build restricted to one environment reads nothing else.
    only = ns["build_figure3"](
        horizon_layout,
        ns["Figure3Options"](
            samples=20,
            cells=common.PAPER_CELLS,
            method="raw_summary",
            environments=("countrecall",),
            compute=("countrecall",),
            name="figureA5_countrecall_one_stream",
        ),
    )
    assert only.notes["keydoor"]["horizon"] is None
    assert only.notes["countrecall"]["streams"] == 4
    assert only.tables["keydoor_windows"] == []
    assert not any(
        s.startswith("outputs/summary-memory") and "keydoor" in s for s in only.sources
    )
    assert "countrecall_retention_by_segment" in only.panels
    only.close()
    # The appendix composite draws the three benchmarks with the
    # retention-against-state row; the T-Maze has no beyond-horizon axis.
    built = ns["build_figure3"](
        horizon_layout,
        ns["Figure3Options"](
            samples=20, cells=common.PAPER_CELLS, environments=common.PAPER_ENVIRONMENTS
        ),
    )
    keydoor = built.notes["keydoor"]
    assert keydoor["horizon"] == 1000 and keydoor["present"] == [
        "raw_summary",
        "raw_segment",
    ]
    assert keydoor["units"] == len(TASKS) and keydoor["prefix_checks"] == ["passed"]
    assert "tmaze" not in built.notes and "tmaze_state" not in built.notes
    assert built.notes["state"] == {
        "raw_summary": "measured (fixed state)",
        "raw_segment": "measured (fixed state)",
    }
    countrecall = built.notes["countrecall"]
    assert countrecall["reference"] == "raw_summary" and countrecall["strata"] == [
        0,
        1,
        2,
        3,
    ]
    assert countrecall["present"] == ["raw_summary", "full_context"]
    assert set(built.panels) == {
        "keydoor_doors_per_window",
        "keydoor_cumulative_doors",
        "keydoor_retention_against_state",
        "countrecall_accuracy_by_pair",
        "countrecall_retention_by_segment",
        "countrecall_accuracy_against_state",
        "mazerunner_goal_fraction_by_size",
        "mazerunner_goal_fraction_against_state",
    }
    # Two rows, one column per benchmark; cumulative doors is standalone only.
    assert len(built.figure.axes) == 6
    titles = [ax.get_title(loc="left") for ax in built.figure.axes]
    assert any("1,000-call task" in t for t in titles)
    assert not any("Passive T-Maze" in t for t in titles)
    # Nothing about a missing cell is written on a panel: the legend says it.
    assert not any("pending" in t for t in titles)
    legend_title = built.figure.legends[0].get_title().get_text()
    assert "pending (no complete panels)" in legend_title
    assert sum("CountRecall" in t for t in titles) == 2
    assert sum("MazeRunner" in t for t in titles) == 2
    written = {p.name for p in built.save(tmp_path / "figures", tmp_path / "data")}
    assert {
        "figure3_beyond_the_training_horizon.pdf",
        "figure3_beyond_the_training_horizon.png",
        "figure3_beyond_the_training_horizon_panel_countrecall_retention_by_segment.png",
        "figure3_beyond_the_training_horizon_keydoor_contrasts.csv",
        "figure3_beyond_the_training_horizon_countrecall_retention.csv",
        "figure3_beyond_the_training_horizon_notes.json",
    } <= written
    built.close()


def test_tables_skip_an_empty_saved_figure_table(tmp_path: Path) -> None:
    """A figure table written empty (a lone ``status`` column, as
    ``write_csv`` writes it) reads back as no rows, so the main table and
    the claims matrix say pending instead of failing on a missing column."""
    ns = definitions("tables")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    common.write_csv(data_dir / ns["FIGURE_TABLES"]["countrecall_retention"], [])
    assert ns["_read_saved"](data_dir, "countrecall_retention") == []
    options = ns["TableOptions"](cells=common.select_cells(["raw_summary"]))
    rows = ns["table_main"]([], options, data_dir)
    assert rows[0][ns["MAIN_TABLE_HEADERS"]["last_pair"]] == "pending"
