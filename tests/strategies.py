"""Hypothesis strategies and shared checks for the property tests.

Everything here is built so that the checks are *informative*, not merely passing:

- The predicates include a disjunction, a conjunction, and a conjunction over a disjunction, so implication
  between predicates matters; criteria may also count ``_ANY_EVENT``.
- The data put several codes (including repeats of the same code) at one timestamp often, because ACES counts
  plain predicates per code but derived predicates and ``_ANY_EVENT`` once per timestamp, and because ties
  between an event and a window endpoint are where inclusivity matters.
- Every label comparison runs on the data hypothesis draws (which shrinks to a small counterexample) *and* on
  a fixed population large enough that most configurations extract rows, so that a comparison of two empty
  results is the exception rather than the rule.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from aces.config import (
    DerivedPredicateConfig,
    EventConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    WindowConfig,
)
from hypothesis import HealthCheck, assume, event, settings
from hypothesis import strategies as st

from aces_relaxations import extract, extract_at
from aces_relaxations.tree import endpoint_node, prediction_time_node

CODES = ["A", "B", "C", "D"]
DERIVED = {"A_or_B": "or(A, B)", "A_and_C": "and(A, C)", "AB_and_C": "and(A_or_B, C)"}
PREDICATES = [*CODES, *DERIVED]
CRITERIA_PREDICATES = [*PREDICATES, "_ANY_EVENT"]
CONSTRAINTS = ["(None, 0)", "(1, None)", "(2, None)", "(None, 1)", "(1, 2)"]
EPOCH = datetime(2020, 1, 1)  # noqa: DTZ001 - MEDS timestamps are naive

SETTINGS = settings(
    max_examples=int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "60")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

MEDS_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float64,
}


def predicates() -> dict[str, PlainPredicateConfig | DerivedPredicateConfig]:
    """The predicates every generated configuration uses."""
    out: dict[str, PlainPredicateConfig | DerivedPredicateConfig] = {
        c: PlainPredicateConfig(c) for c in CODES
    }
    out.update({name: DerivedPredicateConfig(expr) for name, expr in DERIVED.items()})
    return out


def config(windows: dict[str, WindowConfig]) -> TaskExtractorConfig:
    """A configuration over the shared predicates, triggered by ``A``; rejected ones are discarded."""
    try:
        return TaskExtractorConfig(predicates=predicates(), trigger=EventConfig("A"), windows=windows)
    except (ValueError, KeyError, TypeError):
        assume(False)


def _meds_frame(rows: list[tuple[int, datetime, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "time": [r[1] for r in rows],
            "code": [r[2] for r in rows],
            "numeric_value": [None] * len(rows),
        },
        schema=MEDS_SCHEMA,
    ).sort("subject_id", "time")


@st.composite
def meds_data(draw: st.DrawFn) -> pl.DataFrame:
    """A small MEDS dataframe on a six-hour grid over three weeks, with frequent same-time events."""
    rows = []
    for subject_id in range(1, draw(st.integers(1, 4)) + 1):
        for slot in draw(st.lists(st.integers(0, 4 * 21), min_size=1, max_size=12)):
            time = EPOCH + timedelta(hours=6 * slot)
            for code in draw(st.lists(st.sampled_from(CODES), min_size=1, max_size=3)):
                rows.append((subject_id, time, code))
    return _meds_frame(rows)


def population(seed: int = 0, n_subjects: int = 150) -> pl.DataFrame:
    """A fixed population on a six-hour grid over three weeks, dense enough that most tasks extract rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject_id in range(1, n_subjects + 1):
        for slot in rng.integers(0, 4 * 21, size=rng.integers(8, 40)):
            time = EPOCH + timedelta(hours=6 * int(slot))
            for code in rng.choice(CODES, size=rng.choice([1, 1, 1, 2, 3])):
                rows.append((subject_id, time, str(code)))
    return _meds_frame(rows)


POPULATION = population()


def datasets(drawn: pl.DataFrame) -> list[pl.DataFrame]:
    """The drawn data, and the fixed population."""
    return [drawn, POPULATION]


def assert_same_rows(
    original_cfg: TaskExtractorConfig, rewritten_cfg: TaskExtractorConfig, drawn: pl.DataFrame
):
    """``rewritten_cfg``, anchored at the rows ``original_cfg`` extracts, reproduces them exactly."""
    for df in datasets(drawn):
        original = extract(original_cfg, df)
        anchored = extract_at(rewritten_cfg, df, original.select("subject_id", "prediction_time"))
        assert anchored.equals(original), f"\noriginal:\n{original}\nrewritten:\n{anchored}"
    event("population rows extracted" if len(original) else "no population rows extracted")


@st.composite
def task_configs(draw: st.DrawFn, end_rooted: bool = False) -> TaskExtractorConfig:
    """A random future-only task: a temporal input window, then a random tree of future windows.

    Args:
        draw: Hypothesis's draw function.
        end_rooted: Whether to also emit look-backs that only the universal normalized form can express: a
            fixed look-back across an event-bounded window (within the *cumulative* guaranteed time) and a
            look-back to a prior event that the path guarantees (by having hopped to it).
    """
    windows: dict[str, WindowConfig] = {}
    input_has = {}
    if draw(st.booleans()):
        input_has[draw(st.sampled_from(PREDICATES))] = "(None, 0)"
    windows["input"] = WindowConfig(
        "trigger", f"start + {draw(st.integers(1, 3))}d", False, True, index_timestamp="end", has=input_has
    )

    # For every referenceable endpoint: the days of temporal windows since the last event-bounded endpoint
    # (how far a look-back can go and still be a start-rooted chain), the cumulative guaranteed days since
    # the prediction time, and the events the path to it guarantees to have occurred after the prediction
    # time (those it hopped to).
    slack: dict[str, int] = {"input.end": 0}
    total: dict[str, int] = {"input.end": 0}
    hopped: dict[str, list[str]] = {"input.end": []}
    for i in range(draw(st.integers(1, 4))):
        name = f"w{i}"
        parent = draw(st.sampled_from(sorted(slack)))
        has = {
            p: draw(st.sampled_from(CONSTRAINTS)) for p in CRITERIA_PREDICATES if draw(st.integers(0, 4)) == 0
        }
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
            # A look-back to a prior event the path is guaranteed to have passed. ACES includes the end event
            # itself in a `<-` search exactly when `start_inclusive` is set, which is what makes the
            # hopped-to event the guaranteed target.
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
    return config(windows)


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
    refs = ["trigger"]
    windows: dict[str, WindowConfig] = {}
    for i in range(draw(st.integers(1, 5))):
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
            has = {
                q: draw(st.sampled_from(CONSTRAINTS))
                for q in CRITERIA_PREDICATES
                if draw(st.integers(0, 4)) == 0
            }
        try:
            windows[name] = WindowConfig(start, end, draw(st.booleans()), draw(st.booleans()), has=has)
        except ValueError:
            assume(False)
        refs.extend([f"{name}.start", f"{name}.end"])

    names = list(windows)
    windows[draw(st.sampled_from(names))].index_timestamp = draw(st.sampled_from(["start", "end"]))

    # Half the time, put the label on a window reachable from the prediction time; most random placements are
    # not, and those are rejected before anything more interesting is checked.
    unlabeled = config(windows)
    pt = prediction_time_node(unlabeled)
    reachable = [n for n in names if pt in endpoint_node(unlabeled, n, "end").node_path[:-1]]
    candidates = reachable if reachable and draw(st.booleans()) else names
    windows[draw(st.sampled_from(candidates))].label = draw(st.sampled_from(PREDICATES))
    return config(windows)


def input_window(draw: st.DrawFn) -> WindowConfig:
    """A temporal input window from the trigger whose end is the prediction time."""
    return WindowConfig("trigger", f"start + {draw(st.integers(1, 3))}d", False, True, index_timestamp="end")


def implied_by(draw: st.DrawFn, code: str) -> str:
    """``code`` itself, or a derived predicate every occurrence of ``code`` is an occurrence of."""
    return "A_or_B" if code in ("A", "B") and draw(st.booleans()) else code


@st.composite
def identity_hop_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """A hop to ``P``, then an inclusive hop to ``P'`` implied by ``P``, then the label window."""
    p = draw(st.sampled_from(CODES))
    windows = {
        "input": input_window(draw),
        "w1": WindowConfig("input.end", f"start -> {p}", draw(st.booleans()), True),
        "w2": WindowConfig("w1.end", f"start -> {implied_by(draw, p)}", draw(st.booleans()), True),
        "target": WindowConfig(
            "w2.end",
            f"start + {draw(st.integers(1, 7))}d",
            draw(st.booleans()),
            True,
            label=draw(st.sampled_from(PREDICATES)),
        ),
    }
    return config(windows)
