"""The shared plotting helpers and the history-contrast estimator.

The condition style map is one colour-blind-safe mapping across every figure:
hue and marker name the attention family, line style names the memory regime.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from reasoned_icrl.analysis import HistoryContrast, condition_style, history_contrasts
from reasoned_icrl.experiments.contracts import ResultValidationError
from reasoned_icrl.experiments.summary_memory.configs import (
    load_retired_summary_memory_study,
)
from tests.analysis.test_summary_memory import SEEDS, fixture


def test_history_contrasts_pair_one_condition_across_its_histories() -> None:
    _, retained = fixture(conditions=("raw", "raw_summary"))
    cleared = [
        replace(e, history="attempt-cleared", numerator=0, native_return=0.0)
        for e in retained
        if e.condition == "raw"
    ]
    rows = history_contrasts([*retained, *cleared], "retained", "attempt-cleared")
    assert len(rows) == 1 and isinstance(rows[0], HistoryContrast)
    assert rows[0].condition == "raw" and rows[0].estimate == pytest.approx(6 / 8)
    assert rows[0].per_seed == {seed: pytest.approx(6 / 8) for seed in SEEDS}
    assert (rows[0].left_history, rows[0].right_history) == (
        "retained",
        "attempt-cleared",
    )
    assert history_contrasts(retained, "retained", "attempt-cleared") == ()
    # A whole task missing from one history breaks the pairing; one attempt
    # fewer would still leave every seed/task cell in place.
    task = cleared[-1].task_id
    partial = [e for e in cleared if not (e.training_seed == 2 and e.task_id == task)]
    with pytest.raises(ResultValidationError, match="one roster"):
        history_contrasts([*retained, *partial], "retained", "attempt-cleared")


def test_condition_styles_encode_attention_by_hue_and_regime_by_line() -> None:
    styles = {
        c: condition_style(c) for c in load_retired_summary_memory_study().conditions
    }
    assert styles["raw"].family == "ordinary" and styles["raw"].regime == "full"
    assert styles["raw_segment"].color == styles["raw"].color
    assert styles["raw_segment"].linestyle != styles["raw"].linestyle
    assert styles["raw_dat_summary"].color == styles["raw_dat"].color
    assert styles["raw_dat_summary"].linestyle == styles["raw_summary"].linestyle
    assert styles["raw_gru"].family == "gru"
    assert styles["raw_dat_summary_relational_write_off"].family == "write_off"
    assert styles["raw_dual_content_summary"].family == "dual_content"
    assert styles["raw_window"].regime == "window"
    assert len({s.color for s in styles.values()}) == 5
    assert len({(s.color, s.marker) for s in styles.values()}) == 5
    assert condition_style("transition").family == "ordinary"
