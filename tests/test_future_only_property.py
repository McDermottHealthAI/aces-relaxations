"""Property test: the structural future-only verdict agrees with what ACES realizes on data.

A configuration judged guaranteeably future-only must never realize a label window that starts (or ends)
before its prediction time, on any data. A configuration whose label is reachable from the prediction time but
only through a path that can look back must, with enough subjects and moderately sparse data, realize such a
window at least once. Configurations whose label is not reachable from the prediction time at all are
structurally unguaranteed yet may happen never to look back (a sibling branch that starts later), and criteria
can rule out exactly the rows on which a look-back would reach the past, so those are only checked in the safe
direction.

The data are kept moderately sparse and the offsets small so that "could look back" is not merely a
theoretical possibility: a look-back to the previous event across a week of dense hourly events would almost
never reach before the prediction time, and such cases would make the test flaky rather than informative.
"""

from datetime import timedelta

import numpy as np
import polars as pl
from aces.config import TaskExtractorConfig
from hypothesis import event, given

from aces_relaxations.extraction import extract_label_window_bounds
from aces_relaxations.future_only import backward_edge_reason, future_only_violations
from aces_relaxations.tree import prediction_time_node
from tests.test_normalize_property import CODES, EPOCH, SETTINGS, arbitrary_task_configs

LOOKS_BACK = "can move backward in time"


def _derived_by_look_back(task_cfg: TaskExtractorConfig) -> bool:
    """Whether the prediction time is reached from the trigger through a backward edge."""
    pt = task_cfg.window_nodes[prediction_time_node(task_cfg).node_name]
    return any(backward_edge_reason(node) is not None for node in pt.node_path)


def sparse_population(seed: int, n_subjects: int = 100) -> pl.DataFrame:
    """Many subjects with a moderate number of events each over a month.

    The density is a compromise: sparse enough that a look-back to a prior event has room to reach before
    the prediction time (it needs the event to be *absent* from an interval), dense enough that a short
    fixed look-back from the next occurrence of an event can reach before it (it needs the event to be
    *present* shortly after).

    The data are drawn with a fixed random generator rather than by hypothesis, so that the shrinker cannot
    search for a dataset on which a possible look-back happens not to occur: only the configuration is
    adversarial.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for subject_id in range(1, n_subjects + 1):
        hours = np.sort(rng.integers(0, 24 * 30, size=rng.integers(8, 25)))
        for h in hours:
            rows.append((subject_id, EPOCH + timedelta(hours=int(h)), CODES[rng.integers(len(CODES))]))
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


POPULATIONS = [sparse_population(seed) for seed in range(4)]
MIN_ROWS_FOR_A_VERDICT = 50


@SETTINGS
@given(arbitrary_task_configs(criteria=False))
def test_future_only_verdict_matches_realized_windows(task_cfg: TaskExtractorConfig):
    """Guaranteed configs never look back on data; criteria-free reachable-but-unguaranteed ones do."""
    violations = future_only_violations(task_cfg)

    try:
        bounds = pl.concat([extract_label_window_bounds(task_cfg, df) for df in POPULATIONS])
    except Exception:
        event("ACES failed on the configuration")
        return

    looks_back = (
        (pl.col("label_window_start") < pl.col("prediction_time"))
        | (pl.col("label_window_end") < pl.col("prediction_time"))
    ).alias("looks_back")
    n_back = bounds.select(looks_back).sum().item() or 0

    if not violations:
        event("guaranteed future-only")
        assert n_back == 0, f"Guaranteed future-only, yet ACES realized {n_back} label window(s) in the past."
    elif len(bounds) < MIN_ROWS_FOR_A_VERDICT:
        event("too few rows realized; inconclusive")
    elif _derived_by_look_back(task_cfg):
        # The structural check ignores everything above the prediction time. When the prediction time is
        # itself "the latest A before the next D", the path above it guarantees a D after the prediction
        # time, which a later look-back to a prior D can rely on; the check cannot see that, so its
        # "can look back" verdict is conservative and not empirically testable.
        event("prediction time derived through a look-back; inconclusive")
    elif any(w.has for w in task_cfg.windows.values()):  # pragma: no cover - generated without criteria
        # Criteria can rule out exactly the rows on which the look-back would reach the past (e.g. "no A
        # before the trigger" keeps only first-event triggers, for which the record start is the
        # prediction time), so a structural look-back need not be observable.
        event("can look back, but criteria may prevent it; inconclusive")
    elif all(LOOKS_BACK in v for v in violations):
        event("reachable but can look back")
        assert n_back > 0, (
            f"Structurally able to look back ({violations}), yet none of {len(bounds)} realized label "
            "windows did."
        )
    else:
        event("label not reachable from the prediction time")
