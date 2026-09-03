"""Run ACES extraction in-process, including on normalized configurations anchored at given prediction times.

These helpers exist mainly to *test* relaxations against ACES itself: a relaxation that claims to preserve
labels can be checked by extracting labels with the original configuration and with the relaxed one on the
same data. Configurations are deep-copied before querying, because ACES mutates the ``offset`` of the window
tree's bounds in place on every query.

A normalized configuration (see ``aces_relaxations.normalization``) is triggered by a synthetic index-event
predicate. To anchor it at the prediction times of interest, that predicate is switched on in the *predicate
table* rather than by adding events to the data, so that an anchor at a time with no real event does not count
as an event (``_ANY_EVENT``) in the windows that follow.
"""

import copy
import logging
import tempfile
from contextlib import contextmanager
from datetime import datetime

import polars as pl
from aces.config import TaskExtractorConfig
from aces.predicates import get_predicates_df
from aces.query import query
from omegaconf import DictConfig

from .future_only import backward_edge_reason
from .normalization import INDEX_PREDICATE
from .tree import WindowNode, prediction_time_node, resolve_node

SUBJECT_ID = "subject_id"
PREDICTION_TIME = "prediction_time"
BOOLEAN_VALUE = "boolean_value"
ANY_EVENT = "_ANY_EVENT"


@contextmanager
def _quiet():
    """Silences ACES's chatty logging."""
    logger = logging.getLogger("aces")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(level)


def get_predicates(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame) -> pl.DataFrame:
    """Computes the ACES predicate table for a MEDS dataframe.

    Args:
        task_cfg: The task configuration whose predicates to realize.
        MEDS_df: A MEDS-format dataframe (``subject_id``, ``time``, ``code``, ``numeric_value``).

    Returns:
        One row per ``(subject_id, timestamp)`` with an integer count column per predicate.

    Examples:
        >>> get_predicates(sample_ACES_cfg, sample_MEDS_df).head(3)
        shape: (3, 6)
        ┌────────────┬─────────────────────┬───────────────┬───────────────┬───────┬────────────────────┐
        │ subject_id ┆ timestamp           ┆ icu_admission ┆ icu_discharge ┆ death ┆ discharge_or_death │
        │ ---        ┆ ---                 ┆ ---           ┆ ---           ┆ ---   ┆ ---                │
        │ i64        ┆ datetime[μs]        ┆ i64           ┆ i64           ┆ i64   ┆ i64                │
        ╞════════════╪═════════════════════╪═══════════════╪═══════════════╪═══════╪════════════════════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ 1             ┆ 0             ┆ 0     ┆ 0                  │
        │ 1          ┆ 2020-01-01 12:00:00 ┆ 0             ┆ 0             ┆ 0     ┆ 0                  │
        │ 1          ┆ 2020-01-05 00:00:00 ┆ 0             ┆ 1             ┆ 0     ┆ 1                  │
        └────────────┴─────────────────────┴───────────────┴───────────────┴───────┴────────────────────┘
    """
    with tempfile.NamedTemporaryFile(suffix=".parquet") as fp, _quiet():
        MEDS_df.write_parquet(fp.name, use_pyarrow=True)
        return get_predicates_df(copy.deepcopy(task_cfg), DictConfig({"path": fp.name, "standard": "meds"}))


def _node_time(task_cfg: TaskExtractorConfig, ref: WindowNode) -> pl.Expr:
    """The realized timestamp of a (resolved) window-endpoint node in an ACES result.

    A node's summary holds the timestamps of both ends of the edge into it, named by traversal direction
    rather than by time, so the node itself is the later one for a forward edge and the earlier one for a
    backward edge. (ACES's own ``index_timestamp`` column uses the names literally, which for a window defined
    backward from its end reports the wrong endpoint; the helpers here therefore take the prediction time from
    the tree instead.)
    """
    if ref.root is None:
        return pl.col("trigger")
    summary = pl.col(f"{ref.node_name}_summary").struct
    ends = [summary.field("timestamp_at_start"), summary.field("timestamp_at_end")]
    forward = backward_edge_reason(task_cfg.window_nodes[ref.node_name]) is None
    return pl.max_horizontal(ends) if forward else pl.min_horizontal(ends)


def _prediction_time(task_cfg: TaskExtractorConfig) -> pl.Expr:
    return _node_time(task_cfg, prediction_time_node(task_cfg)).cast(pl.Datetime("us")).alias(PREDICTION_TIME)


def _tidy(task_cfg: TaskExtractorConfig, result: pl.DataFrame) -> pl.DataFrame:
    """Reduces an ACES result to ``(subject_id, prediction_time, boolean_value)``, sorted and deduplicated.

    Rows whose prediction time or label is null are dropped: ACES emits such rows when a window's end falls
    before its start (e.g. a window to the end of the record anchored after the last event), and they cannot
    be prediction-task rows.
    """
    schema = {SUBJECT_ID: pl.Int64, PREDICTION_TIME: pl.Datetime("us"), BOOLEAN_VALUE: pl.Boolean}
    if len(result) == 0 or "label" not in result.columns:
        return pl.DataFrame(schema=schema)
    tidy = (
        result.select(
            pl.col(SUBJECT_ID).cast(pl.Int64),
            _prediction_time(task_cfg),
            (pl.col("label") > 0).alias(BOOLEAN_VALUE),
        )
        .drop_nulls()
        .unique()
        .sort(SUBJECT_ID, PREDICTION_TIME)
    )
    return tidy if len(tidy) else pl.DataFrame(schema=schema)


def extract(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame) -> pl.DataFrame:
    """Extracts ``(subject_id, prediction_time, boolean_value)`` rows with ACES.

    Args:
        task_cfg: A task configuration with a label and an index timestamp.
        MEDS_df: A MEDS-format dataframe.

    Returns:
        The extracted rows, sorted.

    Examples:
        >>> extract(sample_ACES_cfg, sample_MEDS_df)
        shape: (2, 3)
        ┌────────────┬─────────────────────┬───────────────┐
        │ subject_id ┆ prediction_time     ┆ boolean_value │
        │ ---        ┆ ---                 ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ bool          │
        ╞════════════╪═════════════════════╪═══════════════╡
        │ 1          ┆ 2020-01-02 00:00:00 ┆ false         │
        │ 2          ┆ 2020-01-02 00:00:00 ┆ true          │
        └────────────┴─────────────────────┴───────────────┘
    """
    with _quiet():
        return _tidy(task_cfg, query(copy.deepcopy(task_cfg), get_predicates(task_cfg, MEDS_df)))


def with_index_events(
    predicates_df: pl.DataFrame, index_df: pl.DataFrame, index_predicate: str = INDEX_PREDICATE
) -> pl.DataFrame:
    """Switches the index-event predicate on at the given prediction times.

    Rows are added for prediction times at which the subject has no event, with every other predicate (and
    ``_ANY_EVENT``, if present) zero, so that the anchor itself never counts as an event.

    Args:
        predicates_df: A predicate table containing an ``index_predicate`` column (all zero).
        index_df: The ``(subject_id, prediction_time)`` rows to anchor at.
        index_predicate: The index-event predicate's column name.

    Returns:
        The predicate table with the index predicate set.

    Examples:
        >>> predicates_df = pl.DataFrame({
        ...     "subject_id": [1, 1],
        ...     "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 3)],
        ...     "death": [0, 1],
        ...     "index_event": [0, 0],
        ...     "_ANY_EVENT": [1, 1],
        ... })
        >>> index_df = pl.DataFrame({
        ...     "subject_id": [1, 1], "prediction_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        ... })
        >>> with_index_events(predicates_df, index_df)
        shape: (3, 5)
        ┌────────────┬─────────────────────┬───────┬─────────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ death ┆ index_event ┆ _ANY_EVENT │
        │ ---        ┆ ---                 ┆ ---   ┆ ---         ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64   ┆ i64         ┆ i64        │
        ╞════════════╪═════════════════════╪═══════╪═════════════╪════════════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ 0     ┆ 1           ┆ 1          │
        │ 1          ┆ 2020-01-02 00:00:00 ┆ 0     ┆ 1           ┆ 0          │
        │ 1          ┆ 2020-01-03 00:00:00 ┆ 1     ┆ 0           ┆ 1          │
        └────────────┴─────────────────────┴───────┴─────────────┴────────────┘
    """
    anchors = index_df.select(
        pl.col(SUBJECT_ID).cast(predicates_df.schema[SUBJECT_ID]),
        pl.col(PREDICTION_TIME).cast(predicates_df.schema["timestamp"]).alias("timestamp"),
        pl.lit(1).alias("__anchor"),
    ).unique()

    joined = predicates_df.join(anchors, on=[SUBJECT_ID, "timestamp"], how="full", coalesce=True)
    value_cols = [c for c in predicates_df.columns if c not in (SUBJECT_ID, "timestamp")]
    is_new = pl.col(index_predicate).is_null()

    filled = {c: pl.col(c).fill_null(0) for c in value_cols}
    filled[index_predicate] = (
        pl.when(pl.col("__anchor") == 1).then(1).otherwise(0).cast(predicates_df.schema[index_predicate])
    )
    if ANY_EVENT in value_cols:
        filled[ANY_EVENT] = (
            pl.when(is_new).then(0).otherwise(pl.col(ANY_EVENT)).cast(predicates_df.schema[ANY_EVENT])
        )

    return (
        joined.with_columns(*(expr.alias(c) for c, expr in filled.items()))
        .drop("__anchor")
        .select(predicates_df.columns)
        .sort(SUBJECT_ID, "timestamp")
    )


def extract_at(
    task_cfg: TaskExtractorConfig,
    MEDS_df: pl.DataFrame,
    index_df: pl.DataFrame,
    index_predicate: str = INDEX_PREDICATE,
) -> pl.DataFrame:
    """Extracts labels with a normalized configuration anchored at the given prediction times.

    Args:
        task_cfg: A normalized task configuration (triggered by ``index_predicate``).
        MEDS_df: A MEDS-format dataframe.
        index_df: The ``(subject_id, prediction_time)`` rows to anchor at.
        index_predicate: The index-event predicate's name.

    Returns:
        The extracted rows, sorted. Anchors whose windows cannot be realized or whose criteria fail are
        absent.

    Examples:
        >>> from aces_relaxations.normalization import normalize
        >>> original = extract(sample_ACES_cfg, sample_MEDS_df)
        >>> extract_at(normalize(sample_ACES_cfg), sample_MEDS_df, original).equals(original)
        True
    """
    with _quiet():
        predicates_df = with_index_events(get_predicates(task_cfg, MEDS_df), index_df, index_predicate)
        return _tidy(task_cfg, query(copy.deepcopy(task_cfg), predicates_df))


def prediction_times(result: pl.DataFrame) -> list[tuple[int, datetime]]:
    """The ``(subject_id, prediction_time)`` pairs of an extraction result, for quick comparisons.

    Examples:
        >>> prediction_times(extract(sample_ACES_cfg, sample_MEDS_df))
        [(1, datetime.datetime(2020, 1, 2, 0, 0)), (2, datetime.datetime(2020, 1, 2, 0, 0))]
    """
    return list(result.select(SUBJECT_ID, PREDICTION_TIME).iter_rows())


def extract_label_window_bounds(task_cfg: TaskExtractorConfig, MEDS_df: pl.DataFrame) -> pl.DataFrame:
    """Extracts, per row, the realized start and end timestamps of the label window.

    This exposes what ACES actually resolved the label window to on real data, which is how the structural
    future-only property can be checked empirically: a guaranteeably future-only task never yields a label
    window starting before its prediction time.

    Args:
        task_cfg: A task configuration with a label and an index timestamp.
        MEDS_df: A MEDS-format dataframe.

    Returns:
        ``subject_id``, ``prediction_time``, ``label_window_start``, ``label_window_end``, sorted.

    Examples:
        >>> extract_label_window_bounds(sample_ACES_cfg, sample_MEDS_df)
        shape: (2, 4)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
        │ subject_id ┆ prediction_time     ┆ label_window_start  ┆ label_window_end    │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╡
        │ 1          ┆ 2020-01-02 00:00:00 ┆ 2020-01-03 00:00:00 ┆ 2020-01-05 00:00:00 │
        │ 2          ┆ 2020-01-02 00:00:00 ┆ 2020-01-03 00:00:00 ┆ 2020-01-04 00:00:00 │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┘

    A label window that looks back from the next ``B`` event can start before the prediction time:

        >>> from aces.config import EventConfig, PlainPredicateConfig, WindowConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", False, True, index_timestamp="end"),
        ...         "target": WindowConfig("end - 3d", "input.end -> B", False, True, label="A"),
        ...     },
        ... )
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1],
        ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 3), datetime(2020, 1, 5)],
        ...     "code": ["A", "B", "A"],
        ...     "numeric_value": [None, None, None],
        ... })
        >>> extract_label_window_bounds(cfg, df)
        shape: (1, 4)
        ┌────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
        │ subject_id ┆ prediction_time     ┆ label_window_start  ┆ label_window_end    │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---                 │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        │
        ╞════════════╪═════════════════════╪═════════════════════╪═════════════════════╡
        │ 1          ┆ 2020-01-02 00:00:00 ┆ 2019-12-31 00:00:00 ┆ 2020-01-03 00:00:00 │
        └────────────┴─────────────────────┴─────────────────────┴─────────────────────┘
    """
    with _quiet():
        result = query(copy.deepcopy(task_cfg), get_predicates(task_cfg, MEDS_df))

    schema = {
        SUBJECT_ID: pl.Int64,
        PREDICTION_TIME: pl.Datetime("us"),
        "label_window_start": pl.Datetime("us"),
        "label_window_end": pl.Datetime("us"),
    }
    if len(result) == 0 or "label" not in result.columns:
        return pl.DataFrame(schema=schema)

    def realized(side: str) -> pl.Expr:
        ref = resolve_node(task_cfg, root_node=WindowNode(task_cfg.label_window, side))
        return _node_time(task_cfg, ref).cast(pl.Datetime("us")).alias(f"label_window_{side}")

    return (
        result.select(
            pl.col(SUBJECT_ID).cast(pl.Int64), _prediction_time(task_cfg), realized("start"), realized("end")
        )
        .drop_nulls(subset=[PREDICTION_TIME])
        .unique()
        .sort(SUBJECT_ID, PREDICTION_TIME)
    )
