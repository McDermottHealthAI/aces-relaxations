"""Property tests: normalization never changes the label ACES assigns to any prediction-time row.

For random future-only task configurations and random MEDS data, the rows ACES extracts with the original
configuration must be exactly the rows it extracts with the normalized configuration anchored at those
prediction times, with identical labels. Normalization discards cohort selection before the prediction time,
so anchoring at the original rows is the right comparison: on those rows the two must agree, and the
normalized configuration must drop none of them.
"""

import polars as pl
import pytest
from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
from hypothesis import event, given
from hypothesis import strategies as st

from aces_relaxations import extract, is_future_only, normalize, require_bounding_events

from .strategies import (
    CODES,
    PREDICATES,
    SETTINGS,
    arbitrary_task_configs,
    assert_same_rows,
    config,
    datasets,
    identity_hop_configs,
    implied_by,
    input_window,
    meds_data,
    task_configs,
)


@SETTINGS
@given(task_configs(), meds_data())
def test_normalization_preserves_labels(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """The strict normalized form, anchored at the original rows, reproduces them exactly."""
    assert_same_rows(task_cfg, normalize(task_cfg), MEDS_df)


@SETTINGS
@given(task_configs(end_rooted=True), meds_data())
def test_universal_normalization_preserves_labels(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """The universal form (end-rooted windows allowed) also reproduces the original rows exactly."""
    assert_same_rows(task_cfg, normalize(task_cfg, allow_end_rooted=True), MEDS_df)


@SETTINGS
@given(task_configs(end_rooted=True), st.booleans())
def test_normalization_is_structurally_normal(task_cfg: TaskExtractorConfig, allow_end_rooted: bool):
    """The window tree is one stick per branch, ending in the label or in criteria, and nothing else.

    Every branch chains from the prediction time, only its last window carries criteria or the label, and the
    label branch carries no criteria. The strict form never looks back; the universal form only in end-rooted
    windows.
    """
    try:
        normalized = normalize(task_cfg, allow_end_rooted=allow_end_rooted)
    except ValueError:
        assert not allow_end_rooted, "the universal form rejected a configuration built to be expressible"
        event("strict form rejected an end-rooted look-back")
        return

    assert normalized.label_branch.terminal.label is not None
    assert sum(b.is_label_branch for b in normalized.branches) == 1
    assert not normalized.label_branch.terminal.is_constrained
    for branch in normalized.branches:
        assert all(not e.is_constrained and e.label is None for e in branch.edges[:-1]), str(branch)
        assert (
            branch.terminal.is_constrained
            or branch.terminal.label is not None
            or not branch.terminal.is_temporal
        )
    sticks = normalized.window_tree.children
    assert len(sticks) == len(normalized.branches), [n.node_name for n in sticks]
    for stick in sticks:
        assert all(len(node.children) <= 1 for node in stick.descendants) and len(stick.children) <= 1
        *inner, _leaf = (stick, *stick.descendants)
        for node in inner:
            window = normalized.windows[node.node_name.rsplit(".", 1)[0]]
            assert not window.has and window.label is None, node.node_name
        if not allow_end_rooted:
            assert not any(e.is_end_rooted for e in branch.edges), str(branch)
    assert normalized.trigger.predicate == "index_event"
    assert normalized.windows[normalized.index_timestamp_window].start == "trigger"
    for name, window in normalized.windows.items():
        anchor = window.end if window.root_node == "end" else window.start
        assert anchor == "trigger" or anchor.endswith((".end", ".start")), (name, window.start, window.end)
        if not allow_end_rooted:
            assert window.root_node == "start", name


def test_lookback_across_an_event_bound_needs_the_universal_form():
    """A day before the next ``B`` is provably after the prediction time, but not a start-rooted chain."""
    cfg = TaskExtractorConfig(
        predicates={c: PlainPredicateConfig(c) for c in CODES},
        trigger=EventConfig("A"),
        windows={
            "input": WindowConfig("trigger", "start + 1d", False, True, index_timestamp="end"),
            "wait": WindowConfig("input.end", "start + 2d", False, True),
            "stay": WindowConfig("wait.end", "start -> B", False, True),
            "last_day": WindowConfig("end - 1d", "stay.end", False, True, has={"C": "(None, 0)"}),
            "target": WindowConfig("stay.end", "start + 7d", False, True, label="D"),
        },
    )
    with pytest.raises(ValueError, match="across the event-bounded window 'stay'"):
        normalize(cfg)
    assert normalize(cfg, allow_end_rooted=True).windows["last_day"].start == "end - 1 day"


@SETTINGS
@given(arbitrary_task_configs(), meds_data(), st.booleans())
def test_normalization_rejects_or_preserves(
    task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame, allow_end_rooted: bool
):
    """On arbitrary ACES configurations, either form of normalization raises `ValueError` or matches ACES."""
    try:
        normalized = normalize(task_cfg, allow_end_rooted=allow_end_rooted)
    except ValueError:
        event("rejected")
        return

    try:
        extract(task_cfg, MEDS_df)
    except Exception:
        event("ACES failed on the original configuration")
        return

    event("normalized and compared")
    assert_same_rows(task_cfg, normalized, MEDS_df)


@SETTINGS
@given(arbitrary_task_configs())
def test_forms_are_consistent(task_cfg: TaskExtractorConfig):
    """The strict form is a special case of the universal one, and both imply the future-only property."""
    try:
        strict = normalize(task_cfg)
    except ValueError:
        strict = None
    try:
        universal = normalize(task_cfg, allow_end_rooted=True)
    except ValueError:
        universal = None

    event(f"strict: {strict is not None}, universal: {universal is not None}")
    if strict is not None:
        assert universal is not None
        assert universal.branches == strict.branches
    if universal is not None:
        assert is_future_only(task_cfg)


@SETTINGS
@given(arbitrary_task_configs(), meds_data())
def test_require_bounding_events_is_a_no_op_for_aces(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """Making bounding events explicit is idempotent and changes no extracted row or label."""
    explicit = require_bounding_events(task_cfg)
    assert require_bounding_events(explicit).windows == explicit.windows

    for df in datasets(MEDS_df):
        try:
            original = extract(task_cfg, df)
        except Exception:
            event("ACES failed on the original configuration")
            return
        assert extract(explicit, df).equals(original)
    event("rows extracted" if len(original) else "no rows extracted")


@st.composite
def hop_chain_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """A chain ``pt -> w1 (forbids Q) -> w2: start -> Q -> target``, with random inclusivity and shapes.

    ``w1`` is temporal or event-bounded and forbids ``Q`` (or a predicate ``Q`` implies); ``w2`` hops to the
    next ``Q``; the label window follows. The normalized label branch should merge ``w1`` and ``w2`` into one
    hop exactly when the flags make that provably exact.
    """
    q = draw(st.sampled_from(CODES))
    if draw(st.booleans()):
        w1_end = f"start + {draw(st.integers(1, 5))}d"
    else:
        w1_end = f"start -> {draw(st.sampled_from(PREDICATES))}"
    s1, e1 = draw(st.booleans()), draw(st.booleans())
    windows = {
        "input": input_window(draw),
        "w1": WindowConfig("input.end", w1_end, s1, e1, has={implied_by(draw, q): "(None, 0)"}),
        "w2": WindowConfig("w1.end", f"start -> {q}", draw(st.booleans()), draw(st.booleans())),
        "target": WindowConfig(
            "w2.end",
            f"start + {draw(st.integers(1, 7))}d",
            draw(st.booleans()),
            True,
            label=draw(st.sampled_from(PREDICATES)),
        ),
    }
    return config(windows)


@SETTINGS
@given(hop_chain_configs(), meds_data())
def test_hop_collapse_preserves_labels_and_fires_when_exact(
    task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame
):
    """The merged hop preserves labels, and is applied at least when the window's own flags make it exact.

    The merge can also fire for other reasons (an identity hop, a hop gap of the first window covering the
    second hop's event), so the flag condition is sufficient, not necessary.
    """
    w1, w2 = task_cfg.windows["w1"], task_cfg.windows["w2"]
    expected = (w2.end_inclusive or w1.end_inclusive) and (
        w1.start_inclusive or w1.end.startswith("start +") or not w1.end_inclusive
    )
    normalized = normalize(task_cfg)
    names = [e.name for e in normalized.label_branch.edges]
    # Either the two windows merged into one hop, or the second hop was an identity hop (a search for the
    # event the first hop landed on) and vanished from the chain.
    merged = "w1__w2" in names or "w2" not in names
    event("merged" if merged else "kept")
    assert merged or not expected, f"not merged although exact by flags:\n{normalized.label_branch}"
    assert_same_rows(task_cfg, normalized, MEDS_df)


@SETTINGS
@given(identity_hop_configs(), meds_data())
def test_identity_hops_leave_the_chain(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """An inclusive hop to an event the position already is drops out of the chain, preserving labels."""
    normalized = normalize(task_cfg)
    names = [e.name for e in normalized.label_branch.edges]
    assert not any("w2" in n for n in names), str(normalized.label_branch)
    assert_same_rows(task_cfg, normalized, MEDS_df)
