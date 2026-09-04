# aces-relaxations

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/aces-relaxations)](https://pypi.org/project/aces-relaxations/)
[![Tests](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/tests.yaml/badge.svg)](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/McDermottHealthAI/aces-relaxations.svg)](https://github.com/McDermottHealthAI/aces-relaxations/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/McDermottHealthAI/aces-relaxations/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

Relaxations and structural properties for [ACES](https://eventstreamaces.readthedocs.io/) task
configurations, so that a future-only ACES task can be posed to systems with a more limited query language
than ACES itself: generative trajectory models (labeling sampled futures), "ever / never within a region"
query-answering models, time-to-event models, and so on.

This package holds the *relaxations themselves*. Model-specific converters that use them live in sibling
repositories (e.g. `aces-to-eq`, `aces-to-motor`, `aces-to-ethos`, `aces-to-meds-eic-ar`), and inference
over generated trajectories stays in
[`MEDS_trajectory_evaluation`](https://github.com/mmcdermott/MEDS_trajectory_evaluation), from which the
relaxations here were extracted.

> [!WARNING]
> This package is a work in progress and is not yet stable. The API may change in future releases.

## Quick start

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. To get set up:

```bash
git clone https://github.com/McDermottHealthAI/aces-relaxations.git
cd aces-relaxations
uv sync
uv run pre-commit install
```

Run the tests:

```bash
uv run pytest -v
```

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for the full development guide (build system, testing
conventions, code style, PR workflow).

The main entry point is `relax`, which validates a configuration is future-only, prunes it to the windows
after the prediction time, and applies named relaxations:

```python
from aces.config import TaskExtractorConfig
from aces_relaxations import relax, print_ACES

cfg = TaskExtractorConfig.load("task.yaml", "predicates.yaml")
relaxations = {"remove_all_criteria": True, "collapse_temporal_gap_windows": True}
relaxed = relax(cfg, relaxations)
print_ACES(relaxed)
```

## Terminology

| Term           | Description                                                                                                                                                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ACES           | [ACES](https://eventstreamaces.readthedocs.io/en/latest/) is a domain specific language for describing task cohorts and a tool to automatically extract them from EHR datasets. It is the "source of truth" for task definitions in this work.      |
| Task Config    | The (original / raw) ACES configuration file that describes the task cohort.                                                                                                                                                                        |
| Input Window   | The window in the ACES config defining the "prediction time". This is indicated via the `index_timestamp` marker in the ACES config.                                                                                                                |
| Target Window  | The window in the ACES config over which the label is extracted. This is indicated via the `label` marker in the ACES config.                                                                                                                       |
| Future-only    | A config is *guaranteeably future-only* when both endpoints of its target window are reachable from the prediction time through edges that only move forward in time, so the label can only depend on data at or after the prediction time.         |
| Relaxed Config | A config re-rooted at the prediction time and pruned to the windows between it and the target (plus any windows after the target). All relaxations operate on relaxed configs.                                                                      |
| Relaxation     | A modification to the task config that removes constraints or simplifies the relationships between window endpoints. These are used to simplify or broaden the set of identified empirical labels when the task is posed to a reduced query system. |

## The future-only property

Relaxations only make sense for tasks whose label lies in the future of the prediction time. Whether that
holds is a purely structural property of the ACES window tree, checked by `is_future_only` /
`validate_future_only`:

```pycon
>>> from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
>>> from aces_relaxations import is_future_only, validate_future_only, print_ACES
>>> predicates = {"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")}
>>> future_cfg = TaskExtractorConfig(
...     predicates=predicates,
...     trigger=EventConfig("A"),
...     windows={
...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
...         "target": WindowConfig("input.end", "start + 7d", False, True, label="B"),
...     },
... )
>>> is_future_only(future_cfg)
True

```

A config whose target window reaches back before the prediction time fails the check, and
`validate_future_only` explains why:

```pycon
>>> past_cfg = TaskExtractorConfig(
...     predicates=predicates,
...     trigger=EventConfig("A"),
...     windows={
...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
...         "target": WindowConfig("input.end - 2d", "start + 7d", False, True, label="B"),
...     },
... )
>>> is_future_only(past_cfg)
False
>>> validate_future_only(past_cfg)
Traceback (most recent call last):
    ...
ValueError: The task configuration is not guaranteeably future-only:
  - The edge into target.start on the path from the prediction time to the label window can move
    backward in time: temporal bound -2 days, 0:00:00 is negative.
Got tree:
trigger
└── input.end
    └── target.start
        └── target.end
<BLANKLINE>

```

The check is conservative: it never accepts a config that could look into the past, but it can reject
configs that are future-only only by arithmetic (e.g. `end - 1d` where the end is one day after the
prediction time).

## Normalized form

Relaxations change what a task means. Before changing anything, it helps to have a canonical way of writing
a future-only task that *does not* change its meaning but separates the two things every downstream
converter needs separately: the **target** (the chain of windows leading to the label window) and the
**conditions** (inclusion / exclusion criteria, and event-bounded windows that must be realizable). That is
the *normalized form*, produced by `normalize`:

1. Every event-bounded window is given an explicit criterion that its bounding event occurs
    (`require_bounding_events`, below), so nothing downstream has to know that ACES drops rows whose
    event-bounded windows cannot be realized.
2. The tree is re-rooted at the prediction time. Everything before it (the trigger, the input window's
    criteria, look-back windows) selects the cohort but cannot change the label of a row in it, so it is
    discarded. Branches that start by moving backward from the prediction time are discarded with a warning.
3. Every window carrying criteria is pulled out into its own **branch**: unconstrained copies of the windows
    leading to it, then the constrained window. The label window's branch carries no criteria at all, and
    unconstrained event-bounded leaf windows stay as branches because ACES drops rows whose event-bounded
    windows cannot be realized.
4. Along each branch, runs of unconstrained, unlabeled, temporal windows collapse into one window.
5. Branches sharing an unconstrained prefix share those windows, so the result is again a tree in which
    every root-to-leaf path chains `start` to the previous `end`, and criteria sit only on the last window.

The result is a genuine ACES configuration, triggered by a synthetic `index_event` predicate that marks the
prediction time.

```pycon
>>> from aces_relaxations import normalize
>>> normalized = normalize(post_discharge_mortality_cfg)
>>> print_ACES(normalized)
trigger; **Prediction Time**
├── (next discharge) hospitalization.end
│   ├── (+1 day, 0:00:00) gap.end
│   │   └── (+29 days, 0:00:00) target.end; **Label: Presence of death**
│   ├── (+1 day, 0:00:00) gap_criteria.end (no admission, death)
│   └── (next discharge) hospitalization_bounding_event.end (at least 1 discharge)
├── (+1 day, 0:00:00) post_input.end (no admission, discharge_or_death)
└── (next discharge) hospitalization_criteria.end (no death)
>>> print(normalized.label_branch)
hospitalization: start -> discharge -> gap: start + 1 day -> target: start + 29 days; label=death
>>> for branch in normalized.criteria_branches:
...     print(branch)
post_input: start + 1 day (admission: (None, 0), discharge_or_death: (None, 0))
hospitalization: start -> discharge (death: (None, 0))
hospitalization: start -> discharge -> gap: start + 1 day (admission: (None, 0), death: (None, 0))
hospitalization: start -> discharge -> hospitalization_bounding_event: start -> discharge (discharge: (1,
None))

```

Compare the original: the `hospitalization` and `gap` windows each carried criteria *and* led to the label.

```pycon
>>> print_ACES(post_discharge_mortality_cfg)
trigger
├── (start of record) sufficient_history.start (at least 5 event(s))
└── (+1 day, 0:00:00) input.end (no admission, discharge_or_death); **Prediction Time**
    ├── (+1 day, 0:00:00) post_input.end (no admission, discharge_or_death)
    └── (next discharge) hospitalization.end (no death)
        └── (+1 day, 0:00:00) gap.end (no admission, death)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

### Why it preserves labels

With every edge after the prediction time moving forward in time, the timestamp of every window endpoint is
a deterministic function of the data and the prediction time, computed along its path from the root. Copying
a path realizes exactly the same endpoints, so a criterion evaluated on the copy has the same truth value,
and the conjunction of criteria across branches equals the original conjunction. Merging two adjacent
unconstrained temporal windows `(a, b]` and `(b, c]` into `(a, c]` changes no endpoint anything refers to,
and the merged window is itself unconstrained.

### Making bounding events explicit

`start -> discharge` only exists if a discharge occurs after the start. ACES enforces that implicitly by
dropping rows whose event-bounded windows cannot be realized; another interpreter could just as reasonably
read the window as extending to infinity when no discharge occurs. `require_bounding_events` removes the
ambiguity by stating the requirement as an ordinary criterion, and works on any ACES configuration:

- if both endpoints are inclusive, the bounding event is always counted by the window, which gets
    `P: (1, None)` itself;
- otherwise it may not be: an exclusive bounding endpoint never counts it, and even an inclusive one does
    not when the window is zero-length with an exclusive other endpoint (`[t, t)` is empty), which happens
    whenever the bounding event coincides with the other endpoint. Then a zero-length auxiliary window
    anchored at the bounding event (`start: x.end`, `end: start -> P`, both endpoints inclusive) carries
    the criterion. ACES resolves it to the bounding event itself, so it always holds when `x` is
    realizable; under infinity semantics it fails exactly when `P` never occurs.

It never adds a criterion that is already implied, so it is idempotent, and it is a no-op for ACES, which
a property test checks on arbitrary configurations. Both normalized forms apply it first.

```pycon
>>> from aces_relaxations import require_bounding_events
>>> print_ACES(require_bounding_events(in_hosp_mortality_cfg))
trigger
├── (start of record) sufficient_history.start (at least 5 event(s))
└── (+1 day, 0:00:00) input.end (no admission, discharge_or_death); **Prediction Time**
    └── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
        └── (next discharge_or_death) target.end; **Label: Presence of death**
            └── (next discharge_or_death) target_bounding_event.end (at least 1 discharge_or_death)

```

### Look-back windows, and the strict versus universal forms

Windows defined backward from a later endpoint (`start: end - 1d`, `end: week.end`) are kept whenever they
are *guaranteeably* still in the future. When the temporal windows since the last event-bounded endpoint
cover the look-back, the window is rewritten as a forward chain from the rewound position, and its
descendants continue from there. That keeps every window start-rooted, which is the **strict** form
(`normalize(cfg)`), but it is not universal: a look-back from an event-bounded endpoint (the day before the
next discharge) is provably in the future when enough temporal time precedes the event, yet no forward chain
from the prediction time can reach "one day before the discharge".

The **universal** form (`normalize(cfg, allow_end_rooted=True)`) keeps such a window end-rooted, with its
`end` referencing the chain and its `start` defined backward from it, but only when a forward rewrite is
impossible and the look-back is guaranteed not to reach before the prediction time:

- `start: end - d` needs the end to be guaranteed at least `d` after the prediction time, where guarantees
    come from the temporal windows on the path (an event-bounded hop guarantees nothing, since the event may
    occur immediately);
- `start: end <- P` needs `P` to be guaranteed to occur between the prediction time and the end, either
    because the path hopped to the next `P` or because a criterion on the path requires at least one `P`.

Guarantees propagate through such windows (a window hanging off `last_day.start` inherits the buffer), and
are tracked symbolically since positions after an event-bounded hop are data-dependent
(`aces_relaxations.guarantees`). Every other property is unchanged: criteria sit only on a branch's last
window, the label branch is unconstrained, temporal runs collapse, prefixes share. A look-back that could
reach before the prediction time is an error in both forms; the future-only check (`is_future_only`) uses
the same guarantees, so the universal form succeeds on every future-only configuration whose windows after
the prediction time are themselves guaranteeably future-only.

```pycon
>>> stay_cfg = TaskExtractorConfig(
...     predicates={
...         "admission": PlainPredicateConfig("ADMISSION"),
...         "discharge": PlainPredicateConfig("DISCHARGE"),
...         "lab": PlainPredicateConfig("LAB"),
...     },
...     trigger=EventConfig("admission"),
...     windows={
...         "input": WindowConfig("trigger", "start + 1d", False, True, index_timestamp="end"),
...         "wait": WindowConfig("input.end", "start + 5d", False, True),
...         "stay": WindowConfig("wait.end", "start -> discharge", False, True),
...         "last_day": WindowConfig("end - 1d", "stay.end", False, True, has={"lab": "(1, None)"}),
...         "target": WindowConfig("stay.end", "start + 30d", False, True, label="admission"),
...     },
... )
>>> normalize(stay_cfg)
Traceback (most recent call last):
    ...
ValueError: Window 'last_day' looks back 1 day across the event-bounded window 'stay', so it cannot be
written as a start-rooted chain.
>>> print_ACES(normalize(stay_cfg, allow_end_rooted=True))
trigger; **Prediction Time**
└── (+5 days, 0:00:00) wait.end
    └── (next discharge) stay.end
        ├── (+30 days, 0:00:00) target.end; **Label: Presence of admission**
        ├── (-1 day, 0:00:00) last_day.start (at least 1 lab)
        └── (next discharge) stay_bounding_event.end (at least 1 discharge)

```

Two things are *not* preserved, by design. Cohort selection before the prediction time is gone, so the
normalized configuration can only be compared on rows the original produces. And a branch that *starts* by
moving backward from the prediction time (a look-back window anchored at the prediction time itself) is
discarded with the rest of the pre-prediction-time structure, with a warning.

This is checked against ACES itself: `tests/test_normalize_property.py` generates random future-only
configurations and random MEDS data with [hypothesis](https://hypothesis.readthedocs.io/), extracts labels
with the original configuration, then extracts with the normalized configuration anchored at those rows
(`aces_relaxations.extraction.extract_at`), and requires the two results to be identical.

```pycon
>>> from aces_relaxations import extract, extract_at
>>> original = extract(sample_ACES_cfg, sample_MEDS_df)
>>> original
shape: (2, 3)
┌────────────┬─────────────────────┬───────────────┐
│ subject_id ┆ prediction_time     ┆ boolean_value │
│ ---        ┆ ---                 ┆ ---           │
│ i64        ┆ datetime[μs]        ┆ bool          │
╞════════════╪═════════════════════╪═══════════════╡
│ 1          ┆ 2020-01-02 00:00:00 ┆ false         │
│ 2          ┆ 2020-01-02 00:00:00 ┆ true          │
└────────────┴─────────────────────┴───────────────┘
>>> extract_at(normalize(sample_ACES_cfg), sample_MEDS_df, original).equals(original)
True

```

Anchoring is done in the predicate table rather than by adding events to the data, so that an anchor at a
time with no real event does not count towards `_ANY_EVENT` criteria. A second property test draws
*arbitrary* ACES configurations (any reference, direction, bound type, record boundary, or index / label
placement) and requires either form of `normalize` to raise or match ACES exactly; a third checks that the
strict form is a special case of the universal one and that both imply the future-only property.

The future-only property itself is checked empirically too (`tests/test_future_only_property.py`): on
fixed random populations, a configuration judged guaranteeably future-only never realizes a label window
starting before its prediction time, and a criteria-free configuration whose label is reachable only
through a path that can look back realizes one at least once. The data are deliberately moderately sparse
so that "can look back" is a real possibility rather than a theoretical one.

> [!NOTE]
> The extraction helpers take the prediction time from the resolved window tree rather than from ACES's
> `index_timestamp` column. ACES names the two timestamps in a node's summary by traversal direction rather
> than by time, so for a window defined backward from its end (`start: end - 1d`, `end: trigger`) with
> `index_timestamp: end`, its column reports the window's *start*. ACES also accumulates into each bound's
> `offset` on every query, so configurations are deep-copied before querying.

## Supported Config Relaxations

We support a few different relaxations that can help make reduced-language label extraction simpler and
more accommodating. These relaxations are not always appropriate for all tasks, but they can be useful in
some cases. To understand them deeply, we'll use several examples, which we'll set up first.

### Example Configurations

To explore these relaxations, we'll use a few simple example task configs. To construct them, we first need
to import the relevant ACES config classes:

```pycon
>>> from aces.config import (
...     PlainPredicateConfig,
...     EventConfig,
...     TaskExtractorConfig,
...     WindowConfig,
...     DerivedPredicateConfig,
... )

```

We'll also use the `print_ACES` helper function (imported above) to visualize the task configs.

#### Example 1: In-hospital mortality prediction

```pycon
>>> in_hosp_mortality_cfg = TaskExtractorConfig(
...     predicates={
...         "admission": PlainPredicateConfig("ADMISSION"),
...         "discharge": PlainPredicateConfig("DISCHARGE"),
...         "death": PlainPredicateConfig("MEDS_DEATH"),
...         "discharge_or_death": DerivedPredicateConfig("or(discharge, death)"),
...     },
...     trigger=EventConfig("admission"),
...     windows={
...         "sufficient_history": WindowConfig(None, "trigger", True, False, has={"_ANY_EVENT": "(5, None)"}),
...         "input": WindowConfig(
...             "trigger",
...             "start + 24h",
...             False,
...             True,
...             index_timestamp="end",
...             has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
...         ),
...         "gap": WindowConfig(
...             "input.end",
...             "start + 24h",
...             False,
...             True,
...             has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
...         ),
...         "target": WindowConfig("gap.end", "start -> discharge_or_death", False, True, label="death"),
...     },
... )
>>> print_ACES(in_hosp_mortality_cfg)
trigger
├── (start of record) sufficient_history.start (at least 5 event(s))
└── (+1 day, 0:00:00) input.end (no admission, discharge_or_death); **Prediction Time**
    └── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
        └── (next discharge_or_death) target.end; **Label: Presence of death**

```

#### Example 2: 30-day post discharge mortality prediction

Given a hospital admission, we'll use the first 24 hours of data to predict whether or not the patient will
die within 30 days of discharge (with a 1-day gap window post discharge to avoid future leakage). We'll also
impose another gap window after the admission to ensure that the hospitalization itself lasts at least 48
hours.

```pycon
>>> post_discharge_mortality_cfg = TaskExtractorConfig(
...     predicates={
...         "admission": PlainPredicateConfig("ADMISSION"),
...         "discharge": PlainPredicateConfig("DISCHARGE"),
...         "death": PlainPredicateConfig("MEDS_DEATH"),
...         "discharge_or_death": DerivedPredicateConfig("or(discharge, death)"),
...     },
...     trigger=EventConfig("admission"),
...     windows={
...         "sufficient_history": WindowConfig(None, "trigger", True, False, has={"_ANY_EVENT": "(5, None)"}),
...         "input": WindowConfig(
...             "trigger",
...             "start + 24h",
...             False,
...             True,
...             index_timestamp="end",
...             has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
...         ),
...         "post_input": WindowConfig(
...             "input.end",
...             "start + 1d",
...             False,
...             True,
...             has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
...         ),
...         "hospitalization": WindowConfig(
...             "input.end",
...             "start -> discharge",
...             False,
...             True,
...             has={"death": "(None, 0)"},
...         ),
...         "gap": WindowConfig(
...             "hospitalization.end",
...             "start + 1d",
...             False,
...             True,
...             has={"admission": "(None, 0)", "death": "(None, 0)"},
...         ),
...         "target": WindowConfig("gap.end", "start + 29d", False, True, label="death"),
...     },
... )
>>> print_ACES(post_discharge_mortality_cfg)
trigger
├── (start of record) sufficient_history.start (at least 5 event(s))
└── (+1 day, 0:00:00) input.end (no admission, discharge_or_death); **Prediction Time**
    ├── (+1 day, 0:00:00) post_input.end (no admission, discharge_or_death)
    └── (next discharge) hospitalization.end (no death)
        └── (+1 day, 0:00:00) gap.end (no admission, death)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

#### Example 3: 30-day readmission prediction with censoring

This example features a 30-day readmission risk prediction task, but with a post-target censoring protection
window.

```pycon
>>> readmission_cfg = TaskExtractorConfig(
...     predicates={
...         "admission": PlainPredicateConfig("ADMISSION"),
...         "discharge": PlainPredicateConfig("DISCHARGE"),
...         "death": PlainPredicateConfig("MEDS_DEATH"),
...         "discharge_or_death": DerivedPredicateConfig("or(discharge, death)"),
...     },
...     trigger=EventConfig("discharge"),
...     windows={
...         "sufficient_history": WindowConfig(
...             None, "hospitalization.start", True, False, has={"_ANY_EVENT": "(5, None)"}
...         ),
...         "hospitalization": WindowConfig(
...             "end <- admission", "trigger", True, True, has={"_ANY_EVENT": "(10, None)"}, index_timestamp="end"
...         ),
...         "gap": WindowConfig(
...             "hospitalization.end",
...             "start + 1d",
...             False,
...             True,
...             has={"admission": "(None, 0)", "death": "(None, 0)"},
...         ),
...         "target": WindowConfig("gap.end", "start + 29d", False, True, label="admission"),
...         "censoring_protection": WindowConfig(
...             "target.end",
...             None,
...             True,
...             True,
...             has={"_ANY_EVENT": "(1, None)"},
...         ),
...     },
... )
>>> print_ACES(readmission_cfg)
trigger; **Prediction Time**
├── (prior admission) hospitalization.start (at least 10 event(s))
│   └── (start of record) sufficient_history.start (at least 5 event(s))
└── (+1 day, 0:00:00) gap.end (no admission, death)
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
        └── (end of record) censoring_protection.end (at least 1 event(s))

```

#### Example 4: Two-stage Infusion

In this hypothetical example, we are examining a cohort of patients who are given an infusion, then given a
drug, then (within 10 minutes) have their infusion stopped temporarily, then resumed. We are interested in
predicting, at the time of the drug being given, about an adverse event within their second infusion stage.
The reason to have such a task is to explore when relaxations are or aren't appropriate in more complex
set-ups.

```pycon
>>> two_stage_cfg = TaskExtractorConfig(
...     predicates={
...         "infusion_start": PlainPredicateConfig("INFUSION//START"),
...         "infusion_end": PlainPredicateConfig("INFUSION//END"),
...         "drug_given": PlainPredicateConfig("special_drug"),
...         "adverse_event": PlainPredicateConfig("special_adverse_event"),
...     },
...     trigger=EventConfig("drug_given"),
...     windows={
...         "1st_infusion": WindowConfig(
...             "trigger",
...             "start -> infusion_end",
...             True,
...             True,
...             has={"adverse_event": "(None, 0)"},
...             index_timestamp="start",
...         ),
...         "2nd_infusion": WindowConfig(
...             "1st_infusion.end -> infusion_start", "start -> infusion_end", True, True, label="adverse_event"
...         ),
...     },
... )
>>> print_ACES(two_stage_cfg)
trigger; **Prediction Time**
└── (next infusion_end) 1st_infusion.end (no adverse_event)
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

#### Example 5: Counted criteria

This example carries count-based criteria that most reduced query languages cannot express: at least three
labs and no more than one ICU transfer in the week after admission.

```pycon
>>> counted_cfg = TaskExtractorConfig(
...     predicates={
...         "admission": PlainPredicateConfig("ADMISSION"),
...         "lab": PlainPredicateConfig(code={"regex": "LAB//.*"}),
...         "icu_transfer": PlainPredicateConfig("ICU_TRANSFER"),
...         "death": PlainPredicateConfig("MEDS_DEATH"),
...     },
...     trigger=EventConfig("admission"),
...     windows={
...         "input": WindowConfig("trigger", "start + 24h", True, True, index_timestamp="end"),
...         "monitoring": WindowConfig(
...             "input.end",
...             "start + 6d",
...             False,
...             True,
...             has={"lab": "(3, None)", "icu_transfer": "(None, 1)", "death": "(None, 0)"},
...         ),
...         "target": WindowConfig("monitoring.end", "start + 23d", False, True, label="death"),
...     },
... )
>>> print_ACES(counted_cfg)
trigger
└── (+1 day, 0:00:00) input.end; **Prediction Time**
    └── (+6 days, 0:00:00) monitoring.end (at least 3 lab; no more than 1 icu_transfer; no death)
        └── (+23 days, 0:00:00) target.end; **Label: Presence of death**

```

#### Other examples we can't reflect:

1. What if we only want to count something as a readmission only if the next admission has a discharge
    associated with a particular diagnosis code? We can't reflect this in ACES currently, but it would pose
    additional challenges.

### Relaxations

We can perform any of the relaxations with the `relax` function and a mapping from relaxation name to either
a boolean (apply with defaults) or a mapping of parameters. Let's import that now for use with our examples:

```pycon
>>> from aces_relaxations import relax

```

Even without any relaxations, `relax` naturally prunes the tree to include only those nodes between the
prediction time window and the label window or after the label window.

```pycon
>>> print_ACES(relax(in_hosp_mortality_cfg))
input.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
    └── (next discharge_or_death) target.end; **Label: Presence of death**

```

> [!WARNING]
> This can remove some criteria that you may still want to leverage. See, for example, how the post discharge
> config has lost the window asserting the hospitalization is at least 48 hours. This could be corrected by
> having the hospitalization window depend directly on the post input window, rather than the input window.

```pycon
>>> print_ACES(relax(post_discharge_mortality_cfg))
input.end; **Prediction Time**
└── (next discharge) hospitalization.end (no death)
    └── (+1 day, 0:00:00) gap.end (no admission, death)
        └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

We still retain the prediction time, label, and relevant criteria in this view.

```pycon
>>> print_ACES(relax(readmission_cfg))
hospitalization.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no admission, death)
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
        └── (end of record) censoring_protection.end (at least 1 event(s))
>>> print_ACES(relax(two_stage_cfg))
1st_infusion.start; **Prediction Time**
└── (next infusion_end) 1st_infusion.end (no adverse_event)
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

#### 1. `remove_all_criteria`: Remove inclusion/exclusion criteria

This relaxation removes all inclusion/exclusion criteria from the task config, but does not change the window
boundaries that are used to compile the task cohort.

> [!NOTE]
> Using this relaxation does _not_ mean that predictions are made over task samples that failed to meet the
> task criteria (with respect to their real data). Rather, it just means that generated trajectories will not
> be discarded on the basis of failing to meet post-input window inclusion/exclusion criteria.

##### On Example 1: In Hospital Mortality

```pycon
>>> print_ACES(relax(in_hosp_mortality_cfg, {"remove_all_criteria": True}))
input.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end
    └── (next discharge_or_death) target.end; **Label: Presence of death**

```

Here, this may be a mistake, as it will classify trajectories as true if they die after discharge, provided
discharge is within 1 day. However, using this in conjunction with absorbing gap windows is likely suitable.

##### On Example 2: Post-discharge Mortality

```pycon
>>> print_ACES(relax(post_discharge_mortality_cfg, {"remove_all_criteria": True}))
input.end; **Prediction Time**
└── (next discharge) hospitalization.end
    └── (+1 day, 0:00:00) gap.end
        └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

Here, this may be a mistake, as it will classify as negative trajectories who die within 1 day after
discharge (whereas previously such trajectories would be excluded). However, in concert with gap window
absorption, this may be suitable.

##### On Example 3: Readmission

```pycon
>>> print_ACES(relax(readmission_cfg, {"remove_all_criteria": True}))
hospitalization.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
        └── (end of record) censoring_protection.end

```

In this example, there are both good and bad aspects of these changes. First, this will now label trajectories
as negative if they are admitted within 1 day (previously, they would have been excluded), which is likely
problematic. But it also renders the censoring window moot, which may improve the efficiency.

##### On Example 4: 2nd infusion stage adverse event

```pycon
>>> print_ACES(relax(two_stage_cfg, {"remove_all_criteria": True}))
1st_infusion.start; **Prediction Time**
└── (next infusion_end) 1st_infusion.end
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

This may be suitable here; it still tracks the right target (adverse events within the 2nd infusion period),
but now will include labels for patients who have adverse events in both, which may improve the predictive
quality or efficiency of the trajectory-driven predictor.

#### 2. `simplify_constraints`: Reduce every criterion to "ever", "never", or nothing

Many reduced query languages can only ask whether a predicate *ever* occurs in a region, or assert that it
*never* does. This relaxation maps every `(min, max)` count constraint onto one of those two shapes, or
drops it entirely (`null`), according to a small rule set:

| Original constraint    | Meaning             | Default target | Soundness of the default       |
| ---------------------- | ------------------- | -------------- | ------------------------------ |
| `(None, 0)` / `(0, 0)` | never               | `never`        | exact                          |
| `(1, None)`            | at least once       | `ever`         | exact                          |
| `(k, None)`, `k > 1`   | at least `k` times  | `ever`         | weakening (accepts a superset) |
| `(k, m)`, `k >= 1`     | between `k` and `m` | `ever`         | weakening (accepts a superset) |
| `(None, m)`, `m > 0`   | at most `m` times   | `null`         | weakening (constraint dropped) |
| `(None, None)`         | unconstrained       | `null`         | exact                          |

Two optional cutoffs adjust the defaults, and per-predicate or per-window overrides can force any target:

- `ever_cutoff`: "at least `k`" constraints are only kept (as `ever`) when `k <= ever_cutoff`; larger `k`
    is dropped. With no cutoff, they are always kept.
- `never_cutoff`: "at most `m`" constraints with `0 < m <= never_cutoff` are *strengthened* to `never`;
    larger `m` is dropped. The default of `0` never strengthens.
- `predicate_overrides`: `{predicate: target}`, applied in every window.
- `window_overrides`: `{window: {predicate: target}}`, taking precedence over `predicate_overrides`.

##### On Example 5: Counted criteria

With the default rules, "at least 3 lab" weakens to "at least 1 lab" and "no more than 1 icu_transfer" is
dropped:

```pycon
>>> print_ACES(relax(counted_cfg, {"simplify_constraints": True}))
input.end; **Prediction Time**
└── (+6 days, 0:00:00) monitoring.end (at least 1 lab; no death)
    └── (+23 days, 0:00:00) target.end; **Label: Presence of death**

```

With an `ever_cutoff` of 2, the lab constraint is dropped instead; with a `never_cutoff` of 1, the ICU
transfer constraint is strengthened to "never":

```pycon
>>> print_ACES(relax(counted_cfg, {"simplify_constraints": {"ever_cutoff": 2, "never_cutoff": 1}}))
input.end; **Prediction Time**
└── (+6 days, 0:00:00) monitoring.end (no icu_transfer, death)
    └── (+23 days, 0:00:00) target.end; **Label: Presence of death**

```

Overrides force specific targets regardless of the constraint's shape:

```pycon
>>> print_ACES(
...     relax(
...         counted_cfg,
...         {
...             "simplify_constraints": {
...                 "predicate_overrides": {"death": "null"},
...                 "window_overrides": {
...                     "monitoring": {"icu_transfer": "never"},
...                 },
...             }
...         },
...     )
... )
input.end; **Prediction Time**
└── (+6 days, 0:00:00) monitoring.end (at least 1 lab; no icu_transfer)
    └── (+23 days, 0:00:00) target.end; **Label: Presence of death**

```

##### On Example 3: Readmission

The gap window's "no admission, death" criteria are already in "never" form and are unchanged, as is the
"at least 1 event" censoring criterion:

```pycon
>>> print_ACES(relax(readmission_cfg, {"simplify_constraints": True}))
hospitalization.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no admission, death)
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
        └── (end of record) censoring_protection.end (at least 1 event(s))

```

#### 3. `collapse_temporal_gap_windows`: Absorb temporal gap windows into target

This relaxation absorbs any chain of temporal windows between the input and target window terminating at the
target window into the target window. This can only be used if the constraints of these windows are all
removed (or if the remove all criteria relaxation is applied as well). This relaxation allows you to make
predictions with fewer generated tokens and simpler early stopping criteria.

> [!NOTE]
> This does not remove event bounded windows, though it does remove temporal windows directly before event
> bound windows or absorb adjacent temporal windows together.

```pycon
>>> relaxations = {"remove_all_criteria": True, "collapse_temporal_gap_windows": True}

```

##### On Example 1: In Hospital Mortality

```pycon
>>> print_ACES(relax(in_hosp_mortality_cfg, relaxations))
input.end; **Prediction Time**
└── (next discharge_or_death) target.end; **Label: Presence of death**

```

This is likely appropriate, as we will now simply classify if there is any death observed before the next
discharge.

##### On Example 2: Post-discharge Mortality

```pycon
>>> print_ACES(relax(post_discharge_mortality_cfg, relaxations))
input.end; **Prediction Time**
└── (next discharge) hospitalization.end
    └── (+30 days, 0:00:00) target.end; **Label: Presence of death**

```

This is likely suitable; we have simply streamlined the prediction target to be anytime within the 30 days post
discharge, giving the trajectory labeler a more flexible target.

##### On Example 3: Readmission

```pycon
>>> print_ACES(relax(readmission_cfg, relaxations))
hospitalization.end; **Prediction Time**
└── (+30 days, 0:00:00) target.end; **Label: Presence of admission**
    └── (end of record) censoring_protection.end

```

This is likely an improvement over the basic config, because it is more accommodating to the target, but it
still has a censoring prediction window we may want to remove.

##### On Example 4: 2nd infusion stage adverse event

```pycon
>>> print_ACES(relax(two_stage_cfg, relaxations))
1st_infusion.start; **Prediction Time**
└── (next infusion_end) 1st_infusion.end
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

This makes no difference as there are no temporal gap windows in this example.

##### On Example 5: Counted criteria

Because constraint-modifying relaxations run before structural ones, dropping the monitoring window's
criteria (here via overrides) lets the window collapse into the target:

```pycon
>>> print_ACES(
...     relax(
...         counted_cfg,
...         {
...             "simplify_constraints": {
...                 "predicate_overrides": {"lab": "null", "icu_transfer": "null", "death": "null"}
...             },
...             "collapse_temporal_gap_windows": True,
...         },
...     )
... )
input.end; **Prediction Time**
└── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

#### 4. `remove_post_label_windows`: Removes all post-label windows from the task config

This relaxation removes all windows that are after the label window. This is useful for removing censoring
protection windows which expand the generation scope necessary to resolve a window.

##### On Example 1: In Hospital Mortality

```pycon
>>> print_ACES(relax(in_hosp_mortality_cfg, {"remove_post_label_windows": True}))
input.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
    └── (next discharge_or_death) target.end; **Label: Presence of death**

```

This makes no difference as there are no post-label windows in this example.

##### On Example 2: Post-discharge Mortality

```pycon
>>> print_ACES(relax(post_discharge_mortality_cfg, {"remove_post_label_windows": True}))
input.end; **Prediction Time**
└── (next discharge) hospitalization.end (no death)
    └── (+1 day, 0:00:00) gap.end (no admission, death)
        └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

This makes no difference as there are no post-label windows in this example.

##### On Example 3: Readmission

```pycon
>>> print_ACES(relax(readmission_cfg, {"remove_post_label_windows": True}))
hospitalization.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no admission, death)
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**

```

This is likely an improvement, as the censoring protection may complicate generation and reduce the
efficiency.

##### On Example 4: 2nd infusion stage adverse event

```pycon
>>> print_ACES(relax(two_stage_cfg, {"remove_post_label_windows": True}))
1st_infusion.start; **Prediction Time**
└── (next infusion_end) 1st_infusion.end (no adverse_event)
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

This makes no difference as there are no post-label windows in this example.

#### 5. `assume_independent_codes`: Drop everything that cannot matter if codes are independent

Time-to-event models such as MOTOR predict, for each code, the distribution of the time until that code next
occurs, independently of every other code. Under that assumption, a criterion over codes unrelated to the
label cannot change the label's distribution, and a window after the label is irrelevant. This relaxation
therefore:

1. drops every criterion whose predicate shares no plain predicate with the label predicate (including
    `_ANY_EVENT` criteria),
2. narrows "never" criteria on disjunctions to the label-related plain predicates (`no discharge_or_death`
    becomes `no death` when the label is `death`) and keeps other label-related criteria unchanged, and
3. removes all post-label windows.

What remains are exactly the criteria such a model *can* honor, because they condition the very
time-to-event distribution being queried.

##### On Example 1: In Hospital Mortality

```pycon
>>> print_ACES(relax(in_hosp_mortality_cfg, {"assume_independent_codes": True}))
input.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no death)
    └── (next discharge_or_death) target.end; **Label: Presence of death**

```

##### On Example 2: Post-discharge Mortality

```pycon
>>> print_ACES(relax(post_discharge_mortality_cfg, {"assume_independent_codes": True}))
input.end; **Prediction Time**
└── (next discharge) hospitalization.end (no death)
    └── (+1 day, 0:00:00) gap.end (no death)
        └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

##### On Example 3: Readmission

The censoring-protection window and the `no death` criterion go; `no admission` stays because `admission` is
the label:

```pycon
>>> print_ACES(relax(readmission_cfg, {"assume_independent_codes": True}))
hospitalization.end; **Prediction Time**
└── (+1 day, 0:00:00) gap.end (no admission)
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**

```

##### On Example 4: 2nd infusion stage adverse event

The `no adverse_event` criterion is on the label predicate itself, so it stays:

```pycon
>>> print_ACES(relax(two_stage_cfg, {"assume_independent_codes": True}))
1st_infusion.start; **Prediction Time**
└── (next infusion_end) 1st_infusion.end (no adverse_event)
    └── (next infusion_start) 2nd_infusion.start
        └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

```

##### On Example 5: Counted criteria

Both counted criteria are unrelated to the label and vanish. The `no death` criterion stays, so the monitoring
window is still a constrained gap and does not collapse into the target:

```pycon
>>> print_ACES(relax(counted_cfg, {"assume_independent_codes": True, "collapse_temporal_gap_windows": True}))
input.end; **Prediction Time**
└── (+6 days, 0:00:00) monitoring.end (no death)
    └── (+23 days, 0:00:00) target.end; **Label: Presence of death**

```

#### 6. `restrict_to_vocabulary`: Rewrite predicates over a model's code vocabulary

Tokenized and time-to-event models can only reason about a fixed set of codes. This relaxation rewrites each
plain predicate into the vocabulary codes it matches (an exact code, an unanchored regex, or an `any` list),
drops value constraints such models cannot express (a weakening), and replaces predicates that match nothing
according to the `unmatched` policy: the default `"marginal"` substitutes the sentinel code `_MARGINAL_`,
which a converter can interpret as "behaves like the average expressible code"; `"keep"` leaves them alone;
`"error"` raises.

```pycon
>>> vocabulary = ["ADMISSION", "ICU_TRANSFER", "LAB//A1C", "LAB//GLUCOSE", "MEDS_DEATH"]
>>> cfg = relax(counted_cfg, {"restrict_to_vocabulary": {"vocabulary": vocabulary}})
>>> for name, pred in cfg.plain_predicates.items():
...     print(f"{name}: {pred.code}")
admission: ADMISSION
lab: {'any': ['LAB//A1C', 'LAB//GLUCOSE']}
icu_transfer: ICU_TRANSFER
death: MEDS_DEATH
>>> cfg = relax(counted_cfg, {"restrict_to_vocabulary": {"vocabulary": ["ADMISSION"]}})
>>> for name, pred in cfg.plain_predicates.items():
...     print(f"{name}: {pred.code}")
admission: ADMISSION
lab: _MARGINAL_
icu_transfer: _MARGINAL_
death: _MARGINAL_
>>> relax(counted_cfg, {"restrict_to_vocabulary": {"vocabulary": ["ADMISSION"], "unmatched": "error"}})
Traceback (most recent call last):
    ...
ValueError: Predicates match no vocabulary code: ['lab', 'icu_transfer', 'death'].

```

The helper `vocabulary_matches` reports the matches without modifying anything:

```pycon
>>> from aces_relaxations import vocabulary_matches
>>> vocabulary_matches(counted_cfg, vocabulary)
{'admission': ['ADMISSION'], 'lab': ['LAB//A1C', 'LAB//GLUCOSE'], 'icu_transfer': ['ICU_TRANSFER'],
 'death': ['MEDS_DEATH']}

```

### Order of application and extension

`relax` applies relaxations in the fixed order of the `RELAXATIONS` registry, regardless of the order in the
mapping you pass: predicate-level relaxations (`restrict_to_vocabulary`) run first, then criteria-level ones
(`remove_all_criteria`, `assume_independent_codes`, `simplify_constraints`), then structural ones
(`collapse_temporal_gap_windows`, `remove_post_label_windows`).

```pycon
>>> from aces_relaxations import RELAXATIONS
>>> for name in RELAXATIONS:
...     print(name)
restrict_to_vocabulary
remove_all_criteria
assume_independent_codes
simplify_constraints
collapse_temporal_gap_windows
remove_post_label_windows

```

Each relaxation is also a plain function operating in place on a `RelaxedTaskConfig` (the output of
`strip_to_relevant_windows`), so they can be composed by hand or registered with new names by downstream
packages:

```pycon
>>> from aces_relaxations import strip_to_relevant_windows, remove_all_criteria, collapse_temporal_gap_windows
>>> cfg = strip_to_relevant_windows(in_hosp_mortality_cfg)
>>> print_ACES(collapse_temporal_gap_windows(remove_all_criteria(cfg)))
input.end; **Prediction Time**
└── (next discharge_or_death) target.end; **Label: Presence of death**

```

## Model-agnostic reductions

Beyond named relaxations, `aces_relaxations.reductions` holds the two structural moves every
reduced-query-language converter shares, operating on the normalized form:

- **`label_only(normalized)`** keeps the label branch alone, dropping every condition branch. This is
  the *conditional independence* assumption: criteria (and event-bounded realizability requirements)
  are taken not to change the label's distribution. It is provably violated when a dropped criterion
  shares plain-predicate codes with the label (e.g. a gap window forbidding the label code, or an
  inclusion criterion requiring it) — converters should surface the dropped branches so such tasks can
  be recognized.
- **`last_link_target(normalized)`** reduces the label branch to its last window, re-rooted at that
  window's own start. This is the *TTE stationarity* assumption: the model's prediction at the
  prediction time stands in for the prediction it would make at the label window's true start. The
  label window keeps its own extent; the discarded intermediate windows' time is gone. The result is a
  `TargetWindow`: a fixed-duration, event-bounded, or to-record-end window over the label predicate,
  with a `note` recording any approximation made in re-rooting a look-back window.

```pycon
>>> from aces_relaxations import label_only, last_link_target
>>> print(label_only(normalize(post_discharge_mortality_cfg)))
hospitalization: start -> discharge -> gap: start + 1 day -> target: start + 29 days; label=death
>>> last_link_target(normalize(post_discharge_mortality_cfg)).describe()
'death within 29.0 days of the window start (start exclusive, end inclusive)'
>>> last_link_target(normalize(in_hosp_mortality_cfg)).describe()
'death before the next or(discharge, death) after the window start (start exclusive, end inclusive)'

```

Model-specific interpretation of the `TargetWindow` (e.g. MOTOR's hazard compositions, marginal
sentinels for out-of-vocabulary predicates, and the treatment of never-occurring bounding events)
lives in the converter packages (`aces-to-motor` and siblings), typically in their decode step.

## Migrating from `MEDS_trajectory_evaluation`

| `MEDS_trajectory_evaluation`                              | `aces_relaxations`                            |
| --------------------------------------------------------- | --------------------------------------------- |
| `aces_utils.ZeroShotTaskConfig`                           | `RelaxedTaskConfig`                           |
| `aces_utils.WindowNode`, `_resolve_node`, ...             | `WindowNode`, `resolve_node`, ...             |
| `aces_utils.print_ACES`, `get_constraint_str`             | `print_ACES`, `get_constraint_str`            |
| `task_config.validate_task_cfg`                           | `validate_future_only` (and `is_future_only`) |
| `task_config._strip_to_rel_windows`                       | `strip_to_relevant_windows`                   |
| `task_config.convert_to_zero_shot(cfg, labeler_cfg)`      | `relax(cfg, relaxations)`                     |
| `task_config.ALLOWED_RELAXATIONS`                         | `RELAXATIONS`                                 |
| `labeler.remove_all_criteria=True`, ... (Hydra overrides) | `{"remove_all_criteria": True, ...}`          |

Predicate evaluation over MEDS data (`get_MEDS_predicates`, `get_MEDS_plain_predicates`) and trajectory
labeling remain in `MEDS_trajectory_evaluation`.

## Roadmap

Relaxations planned for reduced query languages used by other foundation models:

- Mapping predicates over codes absent from a model vocabulary onto *nearby* in-vocabulary codes (e.g. via
    a code hierarchy) rather than the marginal sentinel.
- Mapping value-based predicates onto code-set predicates given quantized values joined with codes.
- Simplifying complex multi-window queries into single-window approximations.

## Repository management

This repository lives within the
[McDermottHealthAI](https://github.com/McDermottHealthAI) GitHub organization, which sets default
issue labels that propagate to new repos. All changes to `main` go through pull requests; see the
"Pull Request Workflow" section of [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for the expected flow.
Versioning follows semantic versioning, managed through `git` tags (e.g., `git tag 0.0.1`) —
`setuptools-scm` reads the tag and stamps the package version automatically.
