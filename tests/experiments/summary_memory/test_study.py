"""The summary-memory study roster and its refusals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from reasoned_icrl.experiments.benchmarks import experiment_config
from reasoned_icrl.experiments.contracts import (
    DAT_ARCHITECTURE_ID,
    GRU_HISTORY_ARCHITECTURE_ID,
    HISTORY_ARCHITECTURE_ID,
    ContractError,
)
from reasoned_icrl.experiments.summary_memory.configs import (
    RETIRED_STUDY,
    RETIRED_STUDY_NAME,
    load_retired_summary_memory_study,
    load_summary_memory_study,
)
from reasoned_icrl.experiments.summary_memory.environments import (
    ENVIRONMENTS,
    environment,
    interventions,
)

ROOT = Path(__file__).resolve().parents[3]


def _study_copy(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    raw = yaml.safe_load((ROOT / RETIRED_STUDY).read_text())
    raw["contracts"] = [str(ROOT / "configs" / entry) for entry in raw["contracts"]]
    return raw, tmp_path / "study.yaml"


def test_the_declared_roster_is_the_key_to_door_matrix() -> None:
    study = load_retired_summary_memory_study()
    assert study.name == RETIRED_STUDY_NAME
    assert study.conditions == (
        "raw",
        "raw_dat",
        "raw_dual_content",
        "raw_gru",
        "raw_segment",
        "raw_summary",
        "raw_dat_segment",
        "raw_dat_summary",
        "raw_dat_summary_relational_write_off",
        "raw_dual_content_summary",
        "raw_window",
    )
    assert study.model["window"] == {"segment_length": 40, "cache_dtype": "float32"}
    assert study.model["summary"] == {
        "segment_length": 32,
        "memory_tokens": 4,
        "detach": "none",
        "position": "segment-local",
        "cache_dtype": "float32",
    }
    assert study.training_seeds == (0, 1, 2)
    assert study.pilot_seeds == () and study.control_conditions == ()
    assert study.control_protocol is None
    assert [c.protocol for c in study.contracts] == [
        "native-keydoor-fixed500-first8",
        "concentration-easy",
        "xland-r1-9x9-small1m-goal-visible-5attempts",
        "count-recall-hard",
        "mazerunner-15-randomized-actions",
    ]
    assert study.output_root == "outputs/summary-memory"
    assert set(ENVIRONMENTS) >= {c.name for c in study.contracts}


@pytest.mark.parametrize(
    "benchmark",
    (
        "dark_key_to_door",
        "concentration",
        "xland_minigrid",
        "count_recall",
        "mazerunner",
    ),
)
def test_every_cell_resolves_to_the_shared_recipe_and_its_own_carrier(
    benchmark: str,
) -> None:
    study = load_retired_summary_memory_study()
    contract = study.contract(benchmark)
    configs = {
        condition: experiment_config(
            contract, study, condition=condition, seed=0, repository=ROOT
        )
        for condition in study.conditions
    }
    reference = configs["raw"]
    assert reference.model.architecture_id == HISTORY_ARCHITECTURE_ID
    assert configs["raw_dat"].model.architecture_id == DAT_ARCHITECTURE_ID
    assert configs["raw_gru"].model.architecture_id == GRU_HISTORY_ARCHITECTURE_ID
    for condition, config in configs.items():
        assert config.environment == reference.environment, condition
        assert config.training == reference.training, condition
        assert config.run_directory == (
            ROOT / "outputs/summary-memory" / contract.protocol / condition / "seed-0"
        )
        runtime = config.as_runtime_mapping()["model"]
        assert runtime["evidence"] == "raw" and runtime["bypass"] is False
    assert configs["raw_dat"].model.dat is not None
    assert configs["raw_dat"].model.dat.max_relative_distance == 500
    assert configs["raw_gru"].model.dat is None and reference.model.dat is None
    for condition, config in configs.items():
        bounded = condition.endswith(("segment", "summary", "relational_write_off"))
        assert (config.model.summary is not None) == bounded, condition
        assert (config.model.window is not None) == (condition == "raw_window")
        capacity = 24 if benchmark == "concentration" else 40
        if bounded and config.model.dat is not None:
            assert config.model.dat.max_relative_distance == capacity
        if config.model.summary is not None:
            assert config.model.summary.capacity == capacity


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda d: d.update(name="dat_benchmarks"), "is not a summary-memory study"),
        (lambda d: d.update(pilot_seeds=[101]), "no pilot seed"),
        (lambda d: d.update(control_conditions=["raw_dual_content"]), "no control arm"),
        (
            lambda d: d.update(control_protocol="mazerunner-randomized-actions"),
            "no control arm",
        ),
        (lambda d: d.update(conditions=["raw_dat", "raw_gru"]), "reference"),
        (lambda d: d.update(conditions=["raw", "transition"]), "raw packet"),
        (lambda d: d.update(conditions=["raw", "raw_bypass"]), "raw packet"),
        (lambda d: d["model"].update(attention_backend="flash"), "vanilla"),
    ],
)
def test_rosters_outside_the_matched_design_are_refused(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    raw, path = _study_copy(tmp_path)
    mutation(raw)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ContractError, match=message):
        load_summary_memory_study(path)


def test_study_environments_and_interventions() -> None:
    study = load_retired_summary_memory_study()
    env = environment(study, "dark_key_to_door", split="development")
    try:
        assert env.split == "development"
    finally:
        env.close()
    hard = environment(study, "count_recall", split="development")
    try:
        # The study's CountRecall contract is the Hard protocol (M5.1).
        assert hard.protocol == "count-recall-hard" and hard.action_space.n == 17
    finally:
        hard.close()
    board = environment(study, "concentration", split="development")
    try:
        # The study's Concentration contract is the colour deck (decision 19);
        # both rank decks failed their gate and left the study.
        assert board.protocol == "concentration-easy" and board.action_space.n == 52
    finally:
        board.close()
    maze = environment(study, "mazerunner", split="development")
    try:
        # The study's MazeRunner contract is the 15x15 randomized protocol (M6.1).
        assert maze.protocol == "mazerunner-15-randomized-actions"
        assert maze.size == 15 and maze.randomized_actions and maze.horizon == 500
    finally:
        maze.close()
    assert interventions("mazerunner") == (
        "retained",
        "goal-cleared",
        "summary-cleared",
    )
    assert interventions("dark_key_to_door") == (
        "retained",
        "attempt-cleared",
        "summary-cleared",
    )
    assert interventions("count_recall") == (
        "retained",
        "current-token",
        "summary-cleared",
    )
    assert interventions("concentration") == (
        "retained",
        "current-token",
        "summary-cleared",
    )
    xland = environment(study, "xland_minigrid", split="development")
    try:
        # The extended benchmark (decision 15): five episodes, the last two scored.
        assert xland.protocol == "xland-r1-9x9-small1m-goal-visible-5attempts"
        assert xland.attempts == 5 and xland.scored_from == 4 and xland.horizon == 243
        assert xland.action_space.n == 6
    finally:
        xland.close()
    assert interventions("xland_minigrid") == (
        "retained",
        "attempt-cleared",
        "summary-cleared",
    )
    with pytest.raises(ContractError, match="Unknown environment"):
        study.contract("darkroom")
