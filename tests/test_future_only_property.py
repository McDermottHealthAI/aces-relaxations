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
from aces.config import TaskExtractorConfig, WindowConfig
from hypothesis import event, given
from hypothesis import strategies as st

from aces_relaxations import future_only_violations
from aces_relaxations.extraction import extract_label_window_bounds
from aces_relaxations.tree import backward_edge_reason, prediction_time_node

from .strategies import CODES, EPOCH, MEDS_SCHEMA, PREDICATES, SETTINGS, arbitrary_task_configs, config

LOOKS_BACK = "can move backward in time"


def _derived_by_look_back(task_cfg: TaskExtractorConfig) -> bool:
    """Whether the prediction time is reached from the trigger through a backward edge."""
    return any(backward_edge_reason(node) is not None for node in prediction_time_node(task_cfg).node_path)


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
            # A quarter of the events share their time with a second one, so conjunctions occur too.
            for _ in range(1 + (rng.random() < 0.25)):
                rows.append((subject_id, EPOCH + timedelta(hours=int(h)), CODES[rng.integers(len(CODES))]))
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "time": [r[1] for r in rows],
            "code": [r[2] for r in rows],
            "numeric_value": [None] * len(rows),
        },
        schema=MEDS_SCHEMA,
    ).sort("subject_id", "time")


POPULATIONS = [sparse_population(seed) for seed in range(4)]
MIN_ROWS_FOR_A_VERDICT = 50

LOOKS_BACK_EXPR = (
    (pl.col("label_window_start") < pl.col("prediction_time"))
    | (pl.col("label_window_end") < pl.col("prediction_time"))
).alias("looks_back")


def _realized_look_backs(task_cfg: TaskExtractorConfig) -> tuple[int, int] | None:
    """How many label windows the populations realize, and how many reach before the prediction time."""
    try:
        bounds = pl.concat([extract_label_window_bounds(task_cfg, df) for df in POPULATIONS])
    except Exception:
        event("ACES failed on the configuration")
        return None
    return len(bounds), bounds.select(LOOKS_BACK_EXPR).sum().item() or 0


@SETTINGS
@given(arbitrary_task_configs(criteria=False))
def test_future_only_verdict_matches_realized_windows(task_cfg: TaskExtractorConfig):
    """Guaranteed configs never look back on data; criteria-free reachable-but-unguaranteed ones do."""
    violations = future_only_violations(task_cfg)
    if (realized := _realized_look_backs(task_cfg)) is None:
        return
    n_rows, n_back = realized

    if not violations:
        event("guaranteed future-only")
        assert n_back == 0, f"Guaranteed future-only, yet ACES realized {n_back} label window(s) in the past."
    elif n_rows < MIN_ROWS_FOR_A_VERDICT:
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
            f"Structurally able to look back ({violations}), yet none of {n_rows} realized label windows did."
        )
    else:
        event("label not reachable from the prediction time")


@st.composite
def unsafe_look_back_configs(draw: st.DrawFn) -> TaskExtractorConfig:
    """A forward chain from the prediction time, then a label window looking back further than guaranteed.

    The look-back is either a fixed offset longer than the temporal time on the chain (event hops guarantee
    none), or a search for a prior event the chain never hopped to.
    """
    windows = {
        "input": WindowConfig(
            "trigger", f"start + {draw(st.integers(1, 3))}d", False, True, index_timestamp="end"
        )
    }
    ref, guaranteed, hopped = "input.end", 0, set()
    for i in range(draw(st.integers(0, 2))):
        if draw(st.booleans()):
            days = draw(st.integers(1, 3))
            end, guaranteed = f"start + {days}d", guaranteed + days
        else:
            p = draw(st.sampled_from(CODES))
            end = f"start -> {p}"
            hopped.add(p)
        windows[f"w{i}"] = WindowConfig(ref, end, draw(st.booleans()), draw(st.booleans()))
        ref = f"w{i}.end"
    unhopped = [p for p in CODES if p not in hopped]
    if unhopped and draw(st.booleans()):
        start = f"end <- {draw(st.sampled_from(unhopped))}"
    else:
        start = f"end - {guaranteed + draw(st.integers(1, 2))}d"
    windows["target"] = WindowConfig(
        start, ref, draw(st.booleans()), draw(st.booleans()), label=draw(st.sampled_from(PREDICATES))
    )
    return config(windows)


@SETTINGS
@given(unsafe_look_back_configs())
def test_unguaranteed_look_backs_are_flagged_and_happen(task_cfg: TaskExtractorConfig):
    """A look-back past what the chain guarantees is a violation, and does reach before the prediction time.

    This is the direction the arbitrary-configuration test rarely reaches: most of its unguaranteed
    configurations are inconclusive there.
    """
    violations = future_only_violations(task_cfg)
    assert violations and all(LOOKS_BACK in v for v in violations), violations
    if (realized := _realized_look_backs(task_cfg)) is None:
        return
    n_rows, n_back = realized
    if n_rows < MIN_ROWS_FOR_A_VERDICT:
        event("too few rows realized; inconclusive")
        return
    event("checked")
    assert n_back > 0, f"None of {n_rows} realized label windows looked back ({violations})."
