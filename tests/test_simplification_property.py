"""Property tests for the redundancy pass of the normalized form.

Constructed configurations exercise each rule (same-region windows merge, subsumed criteria go, "bounding
event occurs" criteria implied by an asserted landing go, unconstrained realizability branches implied by
criteria go) and must simplify as intended while extracting exactly the rows ACES extracts from the original
configuration. (A criterion window provably inside a hop's gap is not expressible in ACES except trivially, so
that rule only matters for the hop collapse.) The general label-preservation tests in
``test_normalize_property`` cover the pass on arbitrary configurations.
"""

import os

import polars as pl
from aces.config import (
    DerivedPredicateConfig,
    EventConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    WindowConfig,
)
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from aces_relaxations.extraction import extract, extract_at
from aces_relaxations.normalization import normalize

from .test_normalize_property import CODES, PREDICATES, meds_data

SETTINGS = settings(
    max_examples=int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "60")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


def _predicates():
    predicates = {c: PlainPredicateConfig(c) for c in CODES}
    predicates["A_or_B"] = DerivedPredicateConfig("or(A, B)")
    return predicates


def _config(windows: dict[str, WindowConfig]) -> TaskExtractorConfig:
    try:
        return TaskExtractorConfig(predicates=_predicates(), trigger=EventConfig("A"), windows=windows)
    except (ValueError, KeyError, TypeError):
        assume(False)


def _input(draw: st.DrawFn) -> WindowConfig:
    return WindowConfig("trigger", f"start + {draw(st.integers(1, 3))}d", False, True, index_timestamp="end")


def _check_rows(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame) -> None:
    original = extract(task_cfg, MEDS_df)
    assume(len(original) > 0)
    assert extract_at(normalize(task_cfg), MEDS_df, original).equals(original)


@st.composite
def same_region_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """Two constrained windows with the same bound from the same position, plus a label window."""
    days = draw(st.integers(1, 7))
    s, e = draw(st.booleans()), draw(st.booleans())
    p1, p2 = draw(st.sampled_from(PREDICATES)), draw(st.sampled_from(PREDICATES))
    windows = {
        "input": _input(draw),
        "w1": WindowConfig("input.end", f"start + {days}d", s, e, has={p1: "(None, 0)"}),
        "w2": WindowConfig("input.end", f"start + {days}d", s, e, has={p2: "(1, None)"}),
        "target": WindowConfig(
            "input.end", f"start + {days + 3}d", False, True, label=draw(st.sampled_from(PREDICATES))
        ),
    }
    return _config(windows)


@SETTINGS
@given(same_region_configs(), meds_data())
def test_same_region_windows_merge(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """The two windows become one carrying both criteria."""
    normalized = normalize(task_cfg)
    terminals = [b.terminal for b in normalized.criteria_branches]
    assert len(terminals) == 1, [str(b) for b in normalized.criteria_branches]
    assert "w1" in terminals[0].name and "w2" in terminals[0].name
    _check_rows(task_cfg, MEDS_df)


@st.composite
def subsumed_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """A criterion on a window and a weaker one on a window it contains (never) or that contains it (ever)."""
    outer_days = draw(st.integers(2, 10))
    inner_days = draw(st.integers(1, outer_days - 1))
    outer_s = draw(st.booleans())
    inner_s = outer_s and draw(st.booleans())
    p = draw(st.sampled_from(CODES))
    implied_by = "A_or_B" if p in ("A", "B") and draw(st.booleans()) else p
    if draw(st.booleans()):
        # never implied_by over the outer window implies never p over the inner window
        strong = WindowConfig(
            "input.end", f"start + {outer_days}d", outer_s, True, has={implied_by: "(None, 0)"}
        )
        weak = WindowConfig("input.end", f"start + {inner_days}d", inner_s, True, has={p: "(None, 0)"})
    else:
        # at least 2 p over the inner window implies at least 1 implied_by over the outer window
        strong = WindowConfig("input.end", f"start + {inner_days}d", inner_s, True, has={p: "(2, None)"})
        weak = WindowConfig(
            "input.end", f"start + {outer_days}d", outer_s, True, has={implied_by: "(1, None)"}
        )
    windows = {
        "input": _input(draw),
        "strong": strong,
        "weak": weak,
        "target": WindowConfig(
            "input.end", f"start + {outer_days + 2}d", False, True, label=draw(st.sampled_from(PREDICATES))
        ),
    }
    return _config(windows)


@SETTINGS
@given(subsumed_configs(), meds_data())
def test_subsumed_criteria_are_dropped(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """Only the stronger criterion survives."""
    normalized = normalize(task_cfg)
    names = [b.terminal.name for b in normalized.criteria_branches]
    assert names == ["strong"], [str(b) for b in normalized.criteria_branches]
    _check_rows(task_cfg, MEDS_df)


@st.composite
def identity_hop_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """A hop to P then an inclusive hop to P' (implied by P), whose realizability criterion is redundant."""
    p = draw(st.sampled_from(CODES))
    p_prime = "A_or_B" if p in ("A", "B") and draw(st.booleans()) else p
    windows = {
        "input": _input(draw),
        "w1": WindowConfig("input.end", f"start -> {p}", draw(st.booleans()), True),
        "w2": WindowConfig("w1.end", f"start -> {p_prime}", draw(st.booleans()), True),
        "target": WindowConfig(
            "w2.end",
            f"start + {draw(st.integers(1, 5))}d",
            False,
            True,
            label=draw(st.sampled_from(PREDICATES)),
        ),
    }
    return _config(windows)


@SETTINGS
@given(identity_hop_configs(), meds_data())
def test_identity_hop_realizability_is_dropped(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """Only the first hop's realizability criterion remains."""
    normalized = normalize(task_cfg)
    names = [b.terminal.name for b in normalized.criteria_branches]
    assert not any("w2" in n for n in names), [str(b) for b in normalized.criteria_branches]
    assert any("w1" in n for n in names), [str(b) for b in normalized.criteria_branches]
    _check_rows(task_cfg, MEDS_df)


@st.composite
def bare_leaf_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """An unconstrained event-bounded window off the label path (a bare realizability branch)."""
    windows = {
        "input": _input(draw),
        "leaf": WindowConfig(
            "input.end", f"start -> {draw(st.sampled_from(PREDICATES))}", draw(st.booleans()), True
        ),
        "target": WindowConfig(
            "input.end",
            f"start + {draw(st.integers(1, 5))}d",
            False,
            True,
            label=draw(st.sampled_from(PREDICATES)),
        ),
    }
    return _config(windows)


@SETTINGS
@given(bare_leaf_configs(), meds_data())
def test_bare_realizability_branches_are_dropped(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """The explicit "bounding event occurs" criterion stays; the unconstrained branch behind it goes."""
    normalized = normalize(task_cfg)
    branches = list(normalized.criteria_branches)
    assert all(b.terminal.is_constrained for b in branches), [str(b) for b in branches]
    assert any("leaf" in b.terminal.name for b in branches), [str(b) for b in branches]
    _check_rows(task_cfg, MEDS_df)
