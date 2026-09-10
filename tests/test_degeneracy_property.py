"""Property tests for the structural degeneracy detectors.

Two directions:

- **Constructed** configurations that are degenerate by construction (a criterion guaranteeing the label
  predicate inside the label window, a criterion forbidding it over an enclosing window, a hop whose event
  is forbidden after the trigger, nested counts that contradict, ...) must be detected, and ACES's behavior
  on random data must agree: every extracted label ``True`` / ``False``, or no row extracted at all.
- **Arbitrary** configurations (no structural restrictions): whenever a detector fires, ACES's behavior on
  random data must agree. This is the soundness check; the detectors are allowed to miss degeneracies.
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
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st

from aces_relaxations.degeneracy import degeneracies
from aces_relaxations.extraction import extract

from .test_normalize_property import CODES, PREDICATES, arbitrary_task_configs, meds_data, task_configs

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


def _prefix(draw: st.DrawFn) -> tuple[dict[str, WindowConfig], str]:
    """A random chain of windows from the trigger, and the reference for its last endpoint."""
    windows: dict[str, WindowConfig] = {
        "input": WindowConfig(
            "trigger", f"start + {draw(st.integers(1, 3))}d", False, True, index_timestamp="end"
        )
    }
    ref = "input.end"
    for i in range(draw(st.integers(0, 2))):
        name = f"pre{i}"
        if draw(st.booleans()):
            end = f"start + {draw(st.integers(1, 5))}d"
        else:
            end = f"start -> {draw(st.sampled_from(PREDICATES))}"
        windows[name] = WindowConfig(ref, end, draw(st.booleans()), True)
        ref = f"{name}.end"
    return windows, ref


@st.composite
def always_positive_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """Configurations whose label is forced to ``True`` by construction."""
    windows, ref = _prefix(draw)
    label = draw(st.sampled_from(PREDICATES))
    variant = draw(st.sampled_from(["nested", "bounding_event", "trigger"]))
    if variant == "nested":
        # The label predicate is required within a window inside the label window.
        outer_days = draw(st.integers(2, 10))
        inner_days = draw(st.integers(1, outer_days - 1))
        outer_start_inclusive = draw(st.booleans())
        inner_start_inclusive = outer_start_inclusive and draw(st.booleans())
        windows["target"] = WindowConfig(
            ref, f"start + {outer_days}d", outer_start_inclusive, draw(st.booleans()), label=label
        )
        windows["forced"] = WindowConfig(
            ref,
            f"start + {inner_days}d",
            inner_start_inclusive,
            draw(st.booleans()),
            has={label: "(1, None)"},
        )
    elif variant == "bounding_event":
        # The label window ends at the next occurrence of the label predicate and includes both ends.
        windows["target"] = WindowConfig(ref, f"start -> {label}", True, True, label=label)
    else:
        # The label is the trigger predicate and the label window starts (inclusively) at the trigger.
        label = "A" if draw(st.booleans()) else "A_or_B"
        windows = {"input": windows["input"]}
        windows["target"] = WindowConfig(
            "trigger", f"start + {draw(st.integers(1, 10))}d", True, draw(st.booleans()), label=label
        )
    event(f"always_positive:{variant}")
    return _config(windows)


@st.composite
def always_negative_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """Configurations whose label is forced to ``False`` by construction."""
    windows, ref = _prefix(draw)
    label = draw(st.sampled_from(PREDICATES))
    variant = draw(st.sampled_from(["nested", "before_next"]))
    if variant == "nested":
        # The label predicate is forbidden over a window containing the label window.
        outer_days = draw(st.integers(2, 10))
        inner_days = draw(st.integers(1, outer_days - 1))
        outer_start_inclusive = draw(st.booleans())
        inner_start_inclusive = outer_start_inclusive and draw(st.booleans())
        windows["forbidden"] = WindowConfig(
            ref,
            f"start + {outer_days}d",
            outer_start_inclusive,
            draw(st.booleans()),
            has={label: "(None, 0)"},
        )
        windows["target"] = WindowConfig(
            ref, f"start + {inner_days}d", inner_start_inclusive, draw(st.booleans()), label=label
        )
    else:
        # The label window runs up to (excluding) the next occurrence of the label predicate.
        # The found endpoint is exclusive, so an occurrence at the start is skipped: the start must be too.
        windows["target"] = WindowConfig(ref, f"start -> {label}", False, False, label=label)
    event(f"always_negative:{variant}")
    return _config(windows)


@st.composite
def unsatisfiable_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """Configurations whose criteria can never all hold, by construction."""
    windows, ref = _prefix(draw)
    label = draw(st.sampled_from(PREDICATES))
    variant = draw(st.sampled_from(["forbidden_hop", "nested_counts", "before_next"]))
    if variant == "forbidden_hop":
        # Some window hops to an event that is forbidden everywhere after the trigger.
        hop = draw(st.sampled_from(PREDICATES))
        windows["rest"] = WindowConfig("trigger", None, draw(st.booleans()), True, has={hop: "(None, 0)"})
        windows["hop"] = WindowConfig(ref, f"start -> {hop}", draw(st.booleans()), True)
        windows["target"] = WindowConfig(
            "hop.end", f"start + {draw(st.integers(1, 5))}d", False, True, label=label
        )
    elif variant == "nested_counts":
        # More occurrences are required within an inner window than allowed over an outer one.
        outer_days = draw(st.integers(2, 10))
        inner_days = draw(st.integers(1, outer_days - 1))
        predicate = draw(st.sampled_from(PREDICATES))
        at_most = draw(st.integers(0, 2))
        at_least = at_most + draw(st.integers(1, 2))
        outer_start_inclusive = draw(st.booleans())
        inner_start_inclusive = outer_start_inclusive and draw(st.booleans())
        windows["outer"] = WindowConfig(
            ref, f"start + {outer_days}d", outer_start_inclusive, True, has={predicate: f"(None, {at_most})"}
        )
        windows["inner"] = WindowConfig(
            ref, f"start + {inner_days}d", inner_start_inclusive, True, has={predicate: f"({at_least}, None)"}
        )
        windows["target"] = WindowConfig(
            "outer.end", f"start + {draw(st.integers(1, 5))}d", False, True, label=label
        )
    else:
        # An occurrence is required strictly before the next occurrence.
        predicate = draw(st.sampled_from(PREDICATES))
        windows["gap"] = WindowConfig(
            ref, f"start -> {predicate}", False, False, has={predicate: "(1, None)"}
        )
        windows["target"] = WindowConfig(
            "gap.end", f"start + {draw(st.integers(1, 5))}d", False, True, label=label
        )
    event(f"unsatisfiable:{variant}")
    return _config(windows)


def _kinds(task_cfg: TaskExtractorConfig) -> set[str]:
    return {f.kind for f in degeneracies(task_cfg)}


def _check_empirically(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame, kinds: set[str]) -> None:
    """ACES's behavior on the data must agree with every detected kind."""
    labeled = extract(task_cfg, MEDS_df)
    event("rows extracted" if len(labeled) else "no rows extracted")
    if "unsatisfiable" in kinds:
        assert len(labeled) == 0, f"unsatisfiable, yet ACES extracted rows:\n{labeled}"
    if "always_positive" in kinds:
        assert labeled["boolean_value"].all(), (
            f"always positive, yet ACES extracted a False label:\n{labeled}"
        )
    if "always_negative" in kinds:
        assert not labeled["boolean_value"].any(), (
            f"always negative, yet ACES extracted a True label:\n{labeled}"
        )


@SETTINGS
@given(always_positive_configs(), meds_data())
def test_constructed_always_positive(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """Configurations built to force a positive label are detected, and ACES agrees."""
    kinds = _kinds(task_cfg)
    assert "always_positive" in kinds, f"not detected:\n{task_cfg.windows}"
    _check_empirically(task_cfg, MEDS_df, kinds)


@SETTINGS
@given(always_negative_configs(), meds_data())
def test_constructed_always_negative(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """Configurations built to force a negative label are detected, and ACES agrees."""
    kinds = _kinds(task_cfg)
    assert "always_negative" in kinds, f"not detected:\n{task_cfg.windows}"
    _check_empirically(task_cfg, MEDS_df, kinds)


@SETTINGS
@given(unsatisfiable_configs(), meds_data())
def test_constructed_unsatisfiable(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """Configurations built to be unsatisfiable are detected, and ACES extracts nothing."""
    kinds = _kinds(task_cfg)
    assert "unsatisfiable" in kinds, f"not detected:\n{task_cfg.windows}"
    _check_empirically(task_cfg, MEDS_df, kinds)


@SETTINGS
@given(arbitrary_task_configs(), meds_data())
def test_detectors_are_sound_on_arbitrary_configs(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """Whatever the detectors report on an unrestricted configuration, ACES's behavior agrees."""
    kinds = _kinds(task_cfg)
    for kind in sorted(kinds):
        event(f"detected {kind}")
    if not kinds:
        event("nothing detected")
    _check_empirically(task_cfg, MEDS_df, kinds)


@SETTINGS
@given(task_configs(end_rooted=True), meds_data())
def test_detectors_are_sound_on_future_only_configs(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """The same, on the future-only configurations (which carry criteria on random windows)."""
    kinds = _kinds(task_cfg)
    for kind in sorted(kinds):
        event(f"detected {kind}")
    if not kinds:
        event("nothing detected")
    _check_empirically(task_cfg, MEDS_df, kinds)
