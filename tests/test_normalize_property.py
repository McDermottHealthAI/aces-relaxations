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
from aces.config import (
    DerivedPredicateConfig,
    EventConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    WindowConfig,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aces_relaxations.extraction import extract, extract_at
from aces_relaxations.normalization import normalize

CODES = ["A", "B", "C", "D"]
PREDICATES = [*CODES, "A_or_B"]
CONSTRAINTS = ["(None, 0)", "(1, None)", "(None, 1)", "(1, 2)"]
EPOCH = datetime(2020, 1, 1)


@st.composite
def task_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """A random future-only task: a temporal input window, then a random tree of future windows."""
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
    ends = ["input.end"]
    for i in range(n_future):
        name = f"w{i}"
        parent = draw(st.sampled_from(ends))
        start = f"{parent} + {draw(st.integers(1, 2))}d" if draw(st.integers(0, 4)) == 0 else parent
        if draw(st.booleans()):
            end = f"start + {draw(st.integers(1, 7))}d"
        elif draw(st.integers(0, 9)) == 0:
            end = None
        else:
            end = f"start -> {draw(st.sampled_from(PREDICATES))}"
        has = {p: draw(st.sampled_from(CONSTRAINTS)) for p in PREDICATES if draw(st.integers(0, 3)) == 0}
        windows[name] = WindowConfig(start, end, draw(st.booleans()), draw(st.booleans()), has=has)
        if end is not None:
            ends.append(f"{name}.end")

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


@settings(
    max_examples=int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "60")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(task_configs(), meds_data())
def test_normalization_preserves_labels(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame):
    """The normalized config, anchored at the original rows, reproduces them exactly."""
    original = extract(task_cfg, MEDS_df)
    normalized = extract_at(normalize(task_cfg), MEDS_df, original.select("subject_id", "prediction_time"))
    assert normalized.equals(original), f"\noriginal:\n{original}\nnormalized:\n{normalized}"
