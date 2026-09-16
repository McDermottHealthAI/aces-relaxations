"""Property test: a target window noted ``"exact"`` extracts the original label from the label window's start.

``last_link_target`` claims that, re-rooted at the realized start of the original label window, its single
window counts the label predicate over exactly the original label window. Anchoring a one-window configuration
built from it at those starts must therefore reproduce every original label.
"""

from datetime import timedelta

import polars as pl
from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
from hypothesis import event, given

from aces_relaxations import TargetWindow, extract, extract_at, last_link_target, normalize, predicate_expr
from aces_relaxations.branches import render_timedelta
from aces_relaxations.extraction import extract_label_window_bounds

from .strategies import SETTINGS, datasets, meds_data, predicates, task_configs


def _name(task_cfg: TaskExtractorConfig, expr) -> str:
    return next(n for n in task_cfg.predicates if predicate_expr(task_cfg, n) == expr)


def _as_config(task_cfg: TaskExtractorConfig, target: TargetWindow) -> TaskExtractorConfig:
    """A one-window configuration triggered at the re-rooted start."""
    if target.duration_days is not None:
        end = f"start + {render_timedelta(timedelta(days=target.duration_days))}"
    elif target.to_record_end:
        end = None
    else:
        end = f"start -> {_name(task_cfg, target.bounding_event)}"
    window = WindowConfig(
        "trigger",
        end,
        target.start_inclusive,
        target.end_inclusive,
        label=_name(task_cfg, target.label),
        index_timestamp="start",
    )
    all_predicates = {**predicates(), "index_event": PlainPredicateConfig("_INDEX_EVENT_")}
    return TaskExtractorConfig(
        predicates=all_predicates, trigger=EventConfig("index_event"), windows={"t": window}
    )


@SETTINGS
@given(task_configs(end_rooted=True), meds_data())
def test_exact_targets_reproduce_labels_from_the_label_window_start(
    task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame
):
    """Every original row's label is the re-rooted window's label at that row's label window start."""
    try:
        target = last_link_target(normalize(task_cfg, allow_end_rooted=True))
    except ValueError:
        event("no forward re-rooting")
        return
    event(f"note: {'exact' if target.note == 'exact' else 'approximate'}")
    if target.note != "exact":
        return
    rerooted = _as_config(task_cfg, target)

    for df in datasets(MEDS_df):
        keys = ["subject_id", "prediction_time"]
        starts = extract_label_window_bounds(task_cfg, df).join(extract(task_cfg, df), on=keys)
        expected = (
            starts.select(
                "subject_id", pl.col("label_window_start").alias("prediction_time"), "boolean_value"
            )
            .unique()
            .sort("subject_id", "prediction_time", "boolean_value")
        )
        got = extract_at(rerooted, df, expected.select(keys))
        assert got.equals(expected), f"\nexpected:\n{expected}\ngot:\n{got}\ntarget: {target.describe()}"
    event("population rows extracted" if len(expected) else "no population rows extracted")
