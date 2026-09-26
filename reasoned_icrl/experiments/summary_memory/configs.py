"""The summary-memory study roster: memory regime x attention on the raw token.

Every cell reads AMAGO's native timestep token (``raw``) without the state
bypass; the cells differ in the carrier, the attention of the selected block
and how much history a decision may read. The ordinary full-prefix cell is the
sufficient-history reference and qualifies the environment from its own
training seeds, so the study declares no pilot seed and no control arm.

The active roster is the 8M revision; the 4M roster is retired and reachable
only by explicit path. See :data:`DEFAULT_STUDY` and :data:`RETIRED_STUDY`.
"""

from __future__ import annotations

from pathlib import Path

from reasoned_icrl.experiments.benchmarks import Study, load_study
from reasoned_icrl.experiments.contracts import ALL_CONDITIONS, ContractError
from reasoned_icrl.utils import repository_root

STUDY_NAME = "summary_memory_8m"
REFERENCE_CONDITION = "full_context"
DEFAULT_STUDY = Path("configs/summary_memory_8m.yaml")
"""The active study. Loading without a path loads this roster."""

RETIRED_STUDY_NAME = "summary_memory"
RETIRED_REFERENCE_CONDITION = "raw"
RETIRED_STUDY = Path("configs/summary_memory.yaml")
"""The 4M roster, retired.

It is reachable only by passing this path explicitly, so nothing in the active
code base selects it. It stays loadable because the 4M records, checkpoints and
the evidence ledger are interpreted against it; it is not a recipe for new work,
and its environment contracts under ``configs/environments/`` are bound to no
active study."""

MEMO_STUDY_NAME = "memo_key_to_door_8m"
MEMO_STUDY = Path("configs/memo_key_to_door_8m.yaml")
MEMO_COUNT_RECALL_STUDY_NAME = "memo_count_recall_8m"
MEMO_COUNT_RECALL_STUDY = Path("configs/memo_count_recall_8m.yaml")
"""The Memo comparator studies (ME0-ME5): ``memo`` and
``memo_fixed`` fitted on the tier-0 Key-to-Door contract and on the tier-2
CountRecall contract at the locked 8M recipes under their own roots, compared
against the 8M study's saved endpoints, which each tier lists as frozen
reference cells read from the 8M root. Reachable by explicit path only."""

TMAZE_V3_STUDY_NAME = "tmaze_v3_8m"
TMAZE_V3_STUDY = Path("configs/tmaze_v3_8m.yaml")
"""The bounded-summary paper's third benchmark: the six paper cells fitted on
the passive T-Maze contract at the 8M recipe under their own root, no frozen
reference cells. Reachable by explicit path only."""

XLAND_STUDY_NAME = "xland_one_rule_8m"
XLAND_STUDY = Path("configs/xland_one_rule_8m.yaml")
"""The conditional XLand one-rule application: the six required cells on the
one-rule contract (manifest roster, fixed lifetime layouts, curriculum) at the
8M recipe under its own root. Reachable by explicit path only; qualification
of the ordinary reference precedes every other cell."""

MEMO_STUDY_REFERENCES = {
    MEMO_STUDY_NAME: REFERENCE_CONDITION,
    MEMO_COUNT_RECALL_STUDY_NAME: "full_dual_relational",
    # The capacity ablation reads
    # full_context and full_gru frozen from the 8M root exactly as the Memo
    # comparators do; it is a comparator study in this sense.
    "keydoor_capacity_8m": REFERENCE_CONDITION,
    # The ablation's other arms,
    # Key-to-Door at M = 1 and the T-Maze v3 at M = 1 and M = 16, read the
    # frozen full_context and full_gru endpoints of their benchmark's root.
    "keydoor_capacity_m1_8m": REFERENCE_CONDITION,
    # The summary-length ablation
    # moved to RSM-O and gained Key-to-Door at M = 8, read against the same
    # frozen full_context and full_gru endpoints of the 8M root.
    "keydoor_capacity_m8_8m": REFERENCE_CONDITION,
}
"""The comparator studies (every roster that reads frozen reference cells from
the 8M root) and the frozen reference each inherits its environment's
qualification through (tier 0's ``full_context``, tier 2's
``full_dual_relational``)."""

STUDY_REFERENCES = {
    "match_pattern_8m": REFERENCE_CONDITION,
    STUDY_NAME: REFERENCE_CONDITION,
    RETIRED_STUDY_NAME: RETIRED_REFERENCE_CONDITION,
    **MEMO_STUDY_REFERENCES,
    XLAND_STUDY_NAME: REFERENCE_CONDITION,
    # v3, the training corridor
    # drawn per task; tiered like v2, the reference resolves per tier.
    TMAZE_V3_STUDY_NAME: REFERENCE_CONDITION,
    # MazeRunner 15x15 with
    # randomised actions at the paper's 8M recipe, the fourth benchmark; tiered,
    # the reference resolves per tier.
    "mazerunner_8m": REFERENCE_CONDITION,
}
"""Each roster's ordinary full-prefix cell, which qualifies its environment.

The 8M study's reference is ``full_context`` and the retired 4M study's is
``raw``. They name the same operator and different study identities, so a gate
always resolves the reference from the roster it was given rather than from a
module constant. A Memo comparator inherits its environment's qualification
through the frozen reference of the 8M root.
"""


def reference_condition(study: Study, environment: str | None = None) -> str:
    """Return the full-history reference cell that qualifies one environment.

    A tiered study (the 8M revision since R4) resolves it from the
    environment's tier plan: ``full_context`` on Key-to-Door, Concentration and
    XLand, ``full_dual_relational`` on CountRecallMedium. The retired 4M roster
    has one study-wide reference, ``raw``, and ignores ``environment``.
    """
    if study.tiered:
        if environment is None:
            raise ContractError(
                f"{study.name} resolves its qualification reference per "
                "environment; name the environment or protocol."
            )
        return study.tier(environment).qualification_reference
    try:
        return STUDY_REFERENCES[study.name]
    except KeyError:
        raise ContractError(f"{study.name!r} declares no reference cell.") from None


def load_summary_memory_study(path: str | Path | None = None) -> Study:
    """Load a summary-memory roster and check every cell stays matched.

    With no path this is the active 8M study. The retired 4M roster still loads
    from :data:`RETIRED_STUDY`, for reading historical runs rather than
    producing new ones; both share the packet, the no-bypass rule and the
    vanilla FP32 attention backend."""
    source = repository_root() / DEFAULT_STUDY if path is None else Path(path)
    study = load_study(source)
    if study.name not in STUDY_REFERENCES:
        raise ContractError(f"{source} is not a summary-memory study.")
    if study.control_conditions or study.control_protocol is not None:
        raise ContractError("The summary-memory study has no control arm.")
    if study.pilot_seeds:
        raise ContractError(
            "The summary-memory study qualifies from its training seeds; "
            "it declares no pilot seed."
        )
    if study.tiered:
        for plan in study.tiers.values():
            if ALL_CONDITIONS[plan.qualification_reference].memory != "full":
                raise ContractError(
                    f"Tier {plan.environment}: the qualification reference must "
                    "be a full-history cell."
                )
            if (plan.reference_cells != ()) != (study.name in MEMO_STUDY_REFERENCES):
                raise ContractError(
                    "Frozen reference cells belong to the Memo comparator studies "
                    "alone; the 8M study fits every cell it compares."
                )
            if study.name in MEMO_STUDY_REFERENCES and (
                plan.qualification_reference != MEMO_STUDY_REFERENCES[study.name]
                or plan.qualification_reference not in plan.reference_cells
            ):
                raise ContractError(
                    f"{study.name} inherits its qualification through the frozen "
                    f"{MEMO_STUDY_REFERENCES[study.name]!r} reference."
                )
    else:
        reference = STUDY_REFERENCES[study.name]
        if reference not in study.conditions:
            raise ContractError(
                f"The {study.name} roster needs its {reference!r} reference."
            )
    for name in study.conditions:
        spec = ALL_CONDITIONS[name]
        if spec.evidence != "raw" or spec.bypass:
            raise ContractError(
                "Summary-memory cells share the raw packet without the state bypass."
            )
    if study.model.get("attention_backend") != "vanilla":
        raise ContractError(
            "The summary-memory comparison uses vanilla FP32 attention."
        )
    return study


def load_retired_summary_memory_study() -> Study:
    """Load the retired 4M roster.

    Use this only to interpret historical runs, checkpoints and records. It is
    never the recipe for new work, and its environment contracts are bound to no
    active study. Calling it is the one way to reach the retired roster, which
    keeps the active code base free of implicit 4M defaults."""
    return load_summary_memory_study(repository_root() / RETIRED_STUDY)


def load_memo_study(environment: str = "dark_key_to_door") -> Study:
    """Load a Memo comparator roster by its explicit path: the Key-to-Door
    study (default) or the CountRecall study."""
    if environment == "dark_key_to_door":
        return load_summary_memory_study(repository_root() / MEMO_STUDY)
    if environment == "count_recall":
        return load_summary_memory_study(repository_root() / MEMO_COUNT_RECALL_STUDY)
    raise ContractError(f"No Memo comparator study on {environment!r}.")


def load_xland_one_rule_study() -> Study:
    """Load the XLand one-rule application roster by its explicit path."""
    return load_summary_memory_study(repository_root() / XLAND_STUDY)


def load_tmaze_v3_study() -> Study:
    """Load the T-Maze v3 study (randomised training corridors) by its explicit path."""
    return load_summary_memory_study(repository_root() / TMAZE_V3_STUDY)


__all__ = [
    "DEFAULT_STUDY",
    "MEMO_COUNT_RECALL_STUDY",
    "MEMO_COUNT_RECALL_STUDY_NAME",
    "MEMO_STUDY",
    "MEMO_STUDY_NAME",
    "MEMO_STUDY_REFERENCES",
    "REFERENCE_CONDITION",
    "RETIRED_REFERENCE_CONDITION",
    "RETIRED_STUDY",
    "RETIRED_STUDY_NAME",
    "STUDY_NAME",
    "STUDY_REFERENCES",
    "TMAZE_V3_STUDY",
    "TMAZE_V3_STUDY_NAME",
    "XLAND_STUDY",
    "XLAND_STUDY_NAME",
    "load_memo_study",
    "load_retired_summary_memory_study",
    "load_summary_memory_study",
    "load_tmaze_v3_study",
    "load_xland_one_rule_study",
    "reference_condition",
]
