"""Property test: constraint simplification changes the extracted rows only in the direction it claims.

Simplification leaves every window (and so every label) alone and changes only criteria. Each simplified
criterion is exact, weaker, or stronger than the original; ACES must then extract a superset of the original
rows when none got stronger, a subset when none got weaker, and the same rows when all are exact, with the
same label at every prediction time both extract (for a single trigger).
"""

import copy

import polars as pl
from aces.config import TaskExtractorConfig, WindowConfig
from hypothesis import event, given
from hypothesis import strategies as st

from aces_relaxations import ConstraintTarget, extract, simplify_constraints

from .strategies import CRITERIA_PREDICATES, SETTINGS, arbitrary_task_configs, config, datasets, meds_data

Constraint = tuple[int | None, int | None]
KEYS = ["subject_id", "prediction_time"]

# Criteria the default rules keep exactly, plus "at most 1", which a positive ``never_cutoff`` strengthens.
STRENGTHENABLE = {(None, 0), (1, None), (None, 1)}


def _direction(before: Constraint | None, after: Constraint | None) -> str:
    """Whether ``after`` accepts the same counts as ``before``, more of them, fewer, or neither."""
    lo, hi = before or (None, None)
    lo2, hi2 = after or (None, None)
    lo, lo2 = lo or 0, lo2 or 0
    hi, hi2 = (float("inf") if hi is None else hi), (float("inf") if hi2 is None else hi2)
    if (lo, hi) == (lo2, hi2):
        return "exact"
    if lo2 <= lo and hi2 >= hi:
        return "weaker"
    if lo2 >= lo and hi2 <= hi:
        return "stronger"
    return "mixed"


@st.composite
def configs_and_rules(draw: st.DrawFn) -> tuple[TaskExtractorConfig, dict]:
    """A configuration and simplification rules: random ones, or ones that can only strengthen criteria.

    Random rules on random criteria almost always weaken something, so half the time the criteria are
    restricted to ``STRENGTHENABLE`` shapes and the rules to a ``never_cutoff``.
    """
    task_cfg = draw(arbitrary_task_configs())
    if draw(st.booleans()):
        overrides = st.dictionaries(
            st.sampled_from(CRITERIA_PREDICATES), st.sampled_from(list(ConstraintTarget)), max_size=1
        )
        rules = {
            "ever_cutoff": draw(st.none() | st.integers(1, 3)),
            "never_cutoff": draw(st.integers(0, 2)),
            "predicate_overrides": draw(overrides),
        }
        return task_cfg, rules
    windows = {
        name: WindowConfig(
            w.start,
            w.end,
            w.start_inclusive,
            w.end_inclusive,
            has={p: f"({lo}, {hi})" for p, (lo, hi) in w.has.items() if (lo, hi) in STRENGTHENABLE},
            label=w.label,
            index_timestamp=w.index_timestamp,
        )
        for name, w in task_cfg.windows.items()
    }
    return config(windows), {"never_cutoff": draw(st.integers(1, 2))}


def _labels_at_shared_times(original: pl.DataFrame, after: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Both results at the prediction times each extracts exactly once.

    Several triggers can share a prediction time (e.g. the end of the record) with different labels, and a
    criterion can keep some of them and not others, so only a time with a single row identifies one trigger.
    """
    single = (
        original.group_by(KEYS)
        .len()
        .filter(pl.col("len") == 1)
        .join(after.group_by(KEYS).len().filter(pl.col("len") == 1), on=KEYS)
    )
    return original.join(single, on=KEYS, how="semi"), after.join(single, on=KEYS, how="semi")


@SETTINGS
@given(configs_and_rules(), meds_data())
def test_simplification_moves_rows_in_the_claimed_direction(
    cfg_and_rules: tuple[TaskExtractorConfig, dict], MEDS_df: pl.DataFrame
):
    """Weakening only adds rows, strengthening only removes them, and labels never change."""
    task_cfg, rules = cfg_and_rules
    simplified = simplify_constraints(copy.deepcopy(task_cfg), **rules)
    directions = {
        _direction(window.has.get(p), simplified.windows[name].has.get(p))
        for name, window in task_cfg.windows.items()
        for p in {*window.has, *simplified.windows[name].has}
    } - {"exact"}
    event(f"directions: {sorted(directions)}")

    for df in datasets(MEDS_df):
        try:
            original = extract(task_cfg, df)
        except Exception:
            event("ACES failed on the original configuration")
            return
        after = extract(simplified, df)
        before_labels, after_labels = _labels_at_shared_times(original, after)
        assert before_labels.equals(after_labels), f"labels changed:\n{before_labels}\n{after_labels}"
        if not directions - {"weaker"}:
            assert len(original.join(after, on=KEYS, how="anti")) == 0, f"weakening dropped rows:\n{original}"
        if not directions - {"stronger"}:
            assert len(after.join(original, on=KEYS, how="anti")) == 0, f"strengthening added rows:\n{after}"
    event("population rows extracted" if len(original) else "no population rows extracted")
