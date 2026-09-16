"""Property test: normalization never changes the label ACES assigns to any prediction-time row.

For random future-only task configurations and random MEDS data, the rows ACES extracts with the original
configuration must be exactly the rows it extracts with the normalized configuration anchored at those
prediction times, with identical labels. Normalization discards cohort selection before the prediction time,
so anchoring at the original rows is the right comparison: on those rows the two must agree, and the
normalized configuration must drop none of them.
"""

import os
from datetime import datetime, timedelta

import polars as pl
import pytest
from aces.config import (
    DerivedPredicateConfig,
    EventConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    WindowConfig,
)
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st

from aces_relaxations.bounding_events import require_bounding_events
from aces_relaxations.extraction import extract, extract_at
from aces_relaxations.future_only import is_future_only
from aces_relaxations.normalization import normalize

CODES = ["A", "B", "C", "D"]
PREDICATES = [*CODES, "A_or_B"]
CONSTRAINTS = ["(None, 0)", "(1, None)", "(None, 1)", "(1, 2)"]
EPOCH = datetime(2020, 1, 1)


@st.composite
def task_configs(draw: st.DrawFn, end_rooted: bool = False) -> TaskExtractorConfig:
    """A random future-only task: a temporal input window, then a random tree of future windows.

    Args:
        draw: Hypothesis's draw function.
        end_rooted: Whether to also emit look-backs that only the universal normalized form can express: a
            fixed look-back across an event-bounded window (within the *cumulative* guaranteed time) and a
            look-back to a prior event that the path guarantees (by having hopped to it).
    """
    predicates = {c: PlainPredicateConfig(c) for c in CODES}
    predicates["A_or_B"] = DerivedPredicateConfig("or(A, B)")

    windows: dict[str, WindowConfig] = {}
    input_has = {}
    if draw(st.booleans()):
        input_has[draw(st.sampled_from(PREDICATES))] = "(None, 0)"
    windows["input"] = WindowConfig(
        "trigger", f"start + {draw(st.integers(1, 3))}d", False, True, index_timestamp="end", has=input_has
    )

    n_future = draw(st.integers(1, 4))
    # For every referenceable endpoint: the days of temporal windows since the last event-bounded endpoint
    # (how far a look-back can go and still be a start-rooted chain), the cumulative guaranteed days since
    # the prediction time, and the events the path to it guarantees to have occurred after the prediction
    # time (those it hopped to).
    slack: dict[str, int] = {"input.end": 0}
    total: dict[str, int] = {"input.end": 0}
    hopped: dict[str, list[str]] = {"input.end": []}
    for i in range(n_future):
        name = f"w{i}"
        parent = draw(st.sampled_from(sorted(slack)))
        has = {p: draw(st.sampled_from(CONSTRAINTS)) for p in PREDICATES if draw(st.integers(0, 3)) == 0}
        start_inclusive, end_inclusive = draw(st.booleans()), draw(st.booleans())

        limit = total[parent] if end_rooted else slack[parent]
        if limit > 0 and draw(st.integers(0, 3)) == 0:
            # A window defined backward from its end, kept within the guaranteed elapsed time.
            back = draw(st.integers(1, limit))
            windows[name] = WindowConfig(f"end - {back}d", parent, start_inclusive, end_inclusive, has=has)
            slack[f"{name}.start"] = max(slack[parent] - back, 0)
            total[f"{name}.start"] = total[parent] - back
            hopped[f"{name}.start"] = []
            continue
        if end_rooted and hopped[parent] and draw(st.integers(0, 3)) == 0:
            # A look-back to a prior event the path is guaranteed to have passed.
            # ACES includes the end event itself in a `<-` search exactly when `start_inclusive` is set,
            # which is what makes the hopped-to event the guaranteed target.
            p = draw(st.sampled_from(hopped[parent]))
            windows[name] = WindowConfig(f"end <- {p}", parent, True, end_inclusive, has=has)
            slack[f"{name}.start"] = 0
            total[f"{name}.start"] = 0
            hopped[f"{name}.start"] = []
            continue

        offset = draw(st.integers(1, 2)) if draw(st.integers(0, 4)) == 0 else 0
        start = f"{parent} + {offset}d" if offset else parent
        if draw(st.booleans()):
            size = draw(st.integers(1, 7))
            end = f"start + {size}d"
            slack[f"{name}.end"] = slack[parent] + offset + size
            total[f"{name}.end"] = total[parent] + offset + size
            hopped[f"{name}.end"] = hopped[parent]
        elif draw(st.integers(0, 9)) == 0:
            end = None
        else:
            p = draw(st.sampled_from(PREDICATES))
            end = f"start -> {p}"
            slack[f"{name}.end"] = 0
            total[f"{name}.end"] = total[parent] + offset
            hopped[f"{name}.end"] = [*hopped[parent], p]
        windows[name] = WindowConfig(start, end, start_inclusive, end_inclusive, has=has)

    label_window = draw(st.sampled_from([n for n in windows if n != "input"]))
    windows[label_window].label = draw(st.sampled_from(PREDICATES))

    return TaskExtractorConfig(predicates=predicates, trigger=EventConfig("A"), windows=windows)


@st.composite
def meds_data(draw: st.DrawFn) -> pl.DataFrame:
    """A random MEDS dataframe over the test codes with hourly timestamps within a month."""
    rows = []
    for subject_id in range(1, draw(st.integers(1, 4)) + 1):
        hours = draw(st.lists(st.integers(0, 24 * 30), min_size=1, max_size=12))
        for h in hours:
            rows.append((subject_id, EPOCH + timedelta(hours=h), draw(st.sampled_from(CODES))))
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "time": [r[1] for r in rows],
            "code": [r[2] for r in rows],
            "numeric_value": [None] * len(rows),
        },
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.String,
            "numeric_value": pl.Float64,
        },
    ).sort("subject_id", "time")


SETTINGS = settings(
    max_examples=int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "60")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@SETTINGS
@given(task_configs(), meds_data())
def test_normalization_preserves_labels(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """The normalized config, anchored at the original rows, reproduces them exactly."""
    original = extract(task_cfg, MEDS_df)
    normalized = extract_at(normalize(task_cfg), MEDS_df, original.select("subject_id", "prediction_time"))
    assert normalized.equals(original), f"\noriginal:\n{original}\nnormalized:\n{normalized}"


@SETTINGS
@given(task_configs(end_rooted=True), meds_data())
def test_universal_normalization_preserves_labels(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """The universal form (end-rooted windows allowed) also reproduces the original rows exactly."""
    original = extract(task_cfg, MEDS_df)
    normalized = normalize(task_cfg, allow_end_rooted=True)
    anchored = extract_at(normalized, MEDS_df, original.select("subject_id", "prediction_time"))
    assert anchored.equals(original), f"\noriginal:\n{original}\nnormalized:\n{anchored}"


@SETTINGS
@given(task_configs())
def test_normalization_is_structurally_normal(task_cfg: TaskExtractorConfig):
    """Every branch chains start to end, has criteria only on its last window, and never looks back."""
    normalized = normalize(task_cfg)
    assert normalized.label_branch.terminal.label is not None
    for branch in normalized.branches:
        assert all(not e.is_constrained for e in branch.edges[:-1])
        assert all(e.days is None or e.days > 0 for e in branch.edges)
        if branch.is_label_branch:
            assert not branch.terminal.is_constrained
    for name, window in normalized.windows.items():
        assert window.root_node == "start", name
        assert window.start == "trigger" or window.start.endswith(".end"), (name, window.start)


def test_lookback_past_prediction_time_is_rejected():
    """A window that could reach before the prediction time is not guaranteeably future-only."""
    cfg = TaskExtractorConfig(
        predicates={c: PlainPredicateConfig(c) for c in CODES},
        trigger=EventConfig("A"),
        windows={
            "input": WindowConfig("trigger", "start + 1d", False, True, index_timestamp="end"),
            "stay": WindowConfig("input.end", "start -> B", False, True),
            "last_day": WindowConfig("end - 1d", "stay.end", False, True, has={"C": "(None, 0)"}),
            "target": WindowConfig("stay.end", "start + 7d", False, True, label="D"),
        },
    )
    with pytest.raises(ValueError, match="across the event-bounded window 'stay'"):
        normalize(cfg)


# ----------------------------------------------------------------------------------------------------------
# Unconstrained configurations: anything ACES accepts. Normalization must either reject the configuration
# (it is not future-only, or not expressible as start-rooted chains) or reproduce ACES's labels exactly.
# ----------------------------------------------------------------------------------------------------------

RELATIVE_FORMS = ["start + {k}d", "start -> {p}", "end - {k}d", "end <- {p}"]


@st.composite
def arbitrary_task_configs(draw: st.DrawFn, criteria: bool = True) -> TaskExtractorConfig:
    """A random ACES configuration with no structural restrictions beyond what ACES itself accepts.

    Windows reference any earlier endpoint (or the trigger) with any offset form, forward or backward,
    temporal or event-bounded, or a record boundary; the index timestamp and label land on any window.
    Configurations ACES rejects are discarded.

    Args:
        draw: Hypothesis's draw function.
        criteria: Whether windows may carry inclusion / exclusion criteria.
    """
    predicates = {c: PlainPredicateConfig(c) for c in CODES}
    predicates["A_or_B"] = DerivedPredicateConfig("or(A, B)")

    n = draw(st.integers(1, 5))
    refs = ["trigger"]
    windows: dict[str, WindowConfig] = {}
    for i in range(n):
        name = f"w{i}"
        k, p = draw(st.integers(1, 7)), draw(st.sampled_from(PREDICATES))
        ref = draw(st.sampled_from(refs))
        external = draw(
            st.sampled_from([ref, f"{ref} + {k}d", f"{ref} - {k}d", f"{ref} -> {p}", f"{ref} <- {p}"])
        )
        k2, p2 = draw(st.integers(1, 7)), draw(st.sampled_from(PREDICATES))
        if draw(st.booleans()):
            start = external
            end = draw(st.sampled_from([None, f"start + {k2}d", f"start -> {p2}"]))
        else:
            start = draw(st.sampled_from([None, f"end - {k2}d", f"end <- {p2}"]))
            end = external
        has = {}
        if criteria:
            has = {q: draw(st.sampled_from(CONSTRAINTS)) for q in PREDICATES if draw(st.integers(0, 3)) == 0}
        try:
            windows[name] = WindowConfig(start, end, draw(st.booleans()), draw(st.booleans()), has=has)
        except ValueError:
            assume(False)
        refs.extend([f"{name}.start", f"{name}.end"])

    names = list(windows)
    windows[draw(st.sampled_from(names))].index_timestamp = draw(st.sampled_from(["start", "end"]))
    windows[draw(st.sampled_from(names))].label = draw(st.sampled_from(PREDICATES))

    try:
        return TaskExtractorConfig(predicates=predicates, trigger=EventConfig("A"), windows=windows)
    except (ValueError, KeyError, TypeError):
        assume(False)


@SETTINGS
@given(arbitrary_task_configs(), meds_data(), st.booleans())
def test_normalization_rejects_or_preserves(
    task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame, allow_end_rooted: bool
):
    """On arbitrary ACES configurations, either form of normalization raises `ValueError` or matches ACES."""
    try:
        normalized = normalize(task_cfg, allow_end_rooted=allow_end_rooted)
    except ValueError:
        event("rejected as not normalizable")
        return

    try:
        original = extract(task_cfg, MEDS_df)
    except Exception:
        event("ACES failed on the original configuration")
        return

    event("normalized and compared")
    anchored = extract_at(normalized, MEDS_df, original.select("subject_id", "prediction_time"))
    assert anchored.equals(original), f"\noriginal:\n{original}\nnormalized:\n{anchored}"


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

    try:
        original = extract(task_cfg, MEDS_df)
    except Exception:
        event("ACES failed on the original configuration")
        return
    assert extract(explicit, MEDS_df).equals(original)


@st.composite
def hop_chain_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """A chain ``pt -> w1 (forbids Q) -> w2: start -> Q -> target``, with random inclusivity and shapes.

    ``w1`` is temporal or event-bounded and forbids ``Q`` (or a predicate ``Q`` implies); ``w2`` hops to the
    next ``Q``; the label window follows. The normalized label branch should merge ``w1`` and ``w2`` into one
    hop exactly when the flags make that provably exact.
    """
    predicates = {c: PlainPredicateConfig(c) for c in CODES}
    predicates["A_or_B"] = DerivedPredicateConfig("or(A, B)")
    q = draw(st.sampled_from(CODES))
    forbidden = "A_or_B" if q in ("A", "B") and draw(st.booleans()) else q
    s1, e1, e2 = draw(st.booleans()), draw(st.booleans()), draw(st.booleans())
    if draw(st.booleans()):
        w1_end = f"start + {draw(st.integers(1, 5))}d"
    else:
        w1_end = f"start -> {draw(st.sampled_from(PREDICATES))}"
    windows = {
        "input": WindowConfig(
            "trigger", f"start + {draw(st.integers(1, 3))}d", False, True, index_timestamp="end"
        ),
        "w1": WindowConfig("input.end", w1_end, s1, e1, has={forbidden: "(None, 0)"}),
        "w2": WindowConfig("w1.end", f"start -> {q}", draw(st.booleans()), e2),
        "target": WindowConfig(
            "w2.end",
            f"start + {draw(st.integers(1, 7))}d",
            draw(st.booleans()),
            True,
            label=draw(st.sampled_from(PREDICATES)),
        ),
    }
    return TaskExtractorConfig(predicates=predicates, trigger=EventConfig("A"), windows=windows)


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

    original = extract(task_cfg, MEDS_df)
    assume(len(original) > 0)
    assert extract_at(normalized, MEDS_df, original).equals(original)


@st.composite
def identity_hop_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """A hop to ``P``, then an inclusive hop to ``P'`` implied by ``P``, then the target window."""
    predicates = {c: PlainPredicateConfig(c) for c in CODES}
    predicates["A_or_B"] = DerivedPredicateConfig("or(A, B)")
    p = draw(st.sampled_from(CODES))
    p_prime = "A_or_B" if p in ("A", "B") and draw(st.booleans()) else p
    windows = {
        "input": WindowConfig(
            "trigger", f"start + {draw(st.integers(1, 3))}d", False, True, index_timestamp="end"
        ),
        "w1": WindowConfig("input.end", f"start -> {p}", draw(st.booleans()), True),
        "w2": WindowConfig("w1.end", f"start -> {p_prime}", draw(st.booleans()), True),
        "target": WindowConfig(
            "w2.end",
            f"start + {draw(st.integers(1, 7))}d",
            draw(st.booleans()),
            True,
            label=draw(st.sampled_from(PREDICATES)),
        ),
    }
    return TaskExtractorConfig(predicates=predicates, trigger=EventConfig("A"), windows=windows)


@SETTINGS
@given(identity_hop_configs(), meds_data())
def test_identity_hops_leave_the_chain(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """An inclusive hop to an event the position already is drops out of the chain, preserving labels."""
    normalized = normalize(task_cfg)
    names = [e.name for e in normalized.label_branch.edges]
    assert "w2" not in names and not any("w2" in n for n in names), str(normalized.label_branch)
    original = extract(task_cfg, MEDS_df)
    assume(len(original) > 0)
    assert extract_at(normalized, MEDS_df, original).equals(original)
