"""Property tests for the redundancy pass of the normalized form.

Constructed configurations exercise each rule (same-region windows merge, subsumed criteria go, "bounding
event occurs" criteria implied by an asserted landing go, unconstrained realizability branches implied by
criteria go) and must simplify as intended while extracting exactly the rows ACES extracts from the original
configuration. Configurations that *look* subsumed but are not, because ACES counts a derived predicate at
most once per timestamp, must keep both criteria. The general label-preservation tests in
``test_normalize_property`` cover the pass on arbitrary configurations.
"""

import polars as pl
from aces.config import TaskExtractorConfig, WindowConfig
from hypothesis import given
from hypothesis import strategies as st

from aces_relaxations import normalize

from .strategies import (
    CODES,
    CRITERIA_PREDICATES,
    PREDICATES,
    SETTINGS,
    assert_same_rows,
    config,
    identity_hop_configs,
    implied_by,
    input_window,
    meds_data,
)


def _target(draw: st.DrawFn, ref: str, days: int) -> WindowConfig:
    return WindowConfig(ref, f"start + {days}d", False, True, label=draw(st.sampled_from(PREDICATES)))


def _criteria_names(task_cfg: TaskExtractorConfig) -> list[str]:
    return [b.terminal.name for b in normalize(task_cfg).criteria_branches]


@st.composite
def same_region_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """Two constrained windows with the same bound from the same position, plus a label window."""
    days = draw(st.integers(1, 7))
    s, e = draw(st.booleans()), draw(st.booleans())
    p1, p2 = draw(st.sampled_from(CRITERIA_PREDICATES)), draw(st.sampled_from(CRITERIA_PREDICATES))
    windows = {
        "input": input_window(draw),
        "w1": WindowConfig("input.end", f"start + {days}d", s, e, has={p1: "(None, 0)"}),
        "w2": WindowConfig("input.end", f"start + {days}d", s, e, has={p2: "(1, None)"}),
        "target": _target(draw, "input.end", days + 3),
    }
    return config(windows)


@SETTINGS
@given(same_region_configs(), meds_data())
def test_same_region_windows_merge(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """The two windows become one carrying both criteria."""
    names = _criteria_names(task_cfg)
    assert len(names) == 1 and "w1" in names[0] and "w2" in names[0], names
    assert_same_rows(task_cfg, normalize(task_cfg), MEDS_df)


def _nested(draw: st.DrawFn, outer_has: dict[str, str], inner_has: dict[str, str]) -> dict[str, WindowConfig]:
    """An outer and an inner window provably inside it, both from the prediction time, and a label window."""
    outer_days = draw(st.integers(2, 10))
    inner_days = draw(st.integers(1, outer_days - 1))
    outer_s = draw(st.booleans())
    inner_s = outer_s and draw(st.booleans())
    return {
        "input": input_window(draw),
        "outer": WindowConfig("input.end", f"start + {outer_days}d", outer_s, True, has=outer_has),
        "inner": WindowConfig("input.end", f"start + {inner_days}d", inner_s, True, has=inner_has),
        "target": _target(draw, "input.end", 12),
    }


@st.composite
def subsumed_configs(draw: st.DrawFn) -> tuple[TaskExtractorConfig, str]:
    """A criterion, and one it implies on a window it contains (at most) or that contains it (at least).

    Returns the configuration and the name of the window carrying the stronger criterion.
    """
    p = draw(st.sampled_from(CODES))
    q = implied_by(draw, p)
    if draw(st.booleans()):
        # At most k q over the outer window implies at most k q over the inner one; "never" also transfers to
        # p, which q implies.
        k = draw(st.integers(0, 2))
        windows = _nested(draw, {q: f"(None, {k})"}, {(p if k == 0 else q): f"(None, {k})"})
        return config(windows), "outer"
    # At least k p over the inner window implies at least k p over the outer one; a single occurrence also
    # transfers to q.
    k = draw(st.integers(1, 3))
    windows = _nested(draw, {(p if k > 1 else q): f"({k}, None)"}, {p: f"({k}, None)"})
    return config(windows), "inner"


@SETTINGS
@given(subsumed_configs(), meds_data())
def test_subsumed_criteria_are_dropped(
    cfg_and_strong: tuple[TaskExtractorConfig, str], MEDS_df: pl.DataFrame
):
    """Only the stronger criterion survives."""
    task_cfg, strong = cfg_and_strong
    assert _criteria_names(task_cfg) == [strong], _criteria_names(task_cfg)
    assert_same_rows(task_cfg, normalize(task_cfg), MEDS_df)


@st.composite
def per_timestamp_count_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """Count criteria that only *look* subsumed: a plain predicate counts codes, a derived one timestamps."""
    p = draw(st.sampled_from(["A", "B"]))
    k = draw(st.integers(1, 2))
    if draw(st.booleans()):
        return config(_nested(draw, {"A_or_B": f"(None, {k})"}, {p: f"(None, {k})"}))
    return config(_nested(draw, {"A_or_B": f"({k + 1}, None)"}, {p: f"({k + 1}, None)"}))


@SETTINGS
@given(per_timestamp_count_configs(), meds_data())
def test_counts_do_not_transfer_to_derived_predicates(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """Neither criterion is implied by the other, so both stay, and labels are preserved."""
    assert sorted(_criteria_names(task_cfg)) == ["inner", "outer"], _criteria_names(task_cfg)
    assert_same_rows(task_cfg, normalize(task_cfg), MEDS_df)


@SETTINGS
@given(identity_hop_configs(), meds_data())
def test_identity_hop_realizability_is_dropped(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """Only the first hop's realizability criterion remains."""
    names = _criteria_names(task_cfg)
    assert not any("w2" in n for n in names) and any("w1" in n for n in names), names
    assert_same_rows(task_cfg, normalize(task_cfg), MEDS_df)


@st.composite
def bare_leaf_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """An unconstrained event-bounded window off the label path (a bare realizability branch)."""
    windows = {
        "input": input_window(draw),
        "leaf": WindowConfig(
            "input.end", f"start -> {draw(st.sampled_from(PREDICATES))}", draw(st.booleans()), True
        ),
        "target": _target(draw, "input.end", draw(st.integers(1, 5))),
    }
    return config(windows)


@SETTINGS
@given(bare_leaf_configs(), meds_data())
def test_bare_realizability_branches_are_dropped(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """The explicit "bounding event occurs" criterion stays; the unconstrained branch behind it goes."""
    branches = normalize(task_cfg).criteria_branches
    assert all(b.terminal.is_constrained for b in branches), [str(b) for b in branches]
    assert any("leaf" in b.terminal.name for b in branches), [str(b) for b in branches]
    assert_same_rows(task_cfg, normalize(task_cfg), MEDS_df)
