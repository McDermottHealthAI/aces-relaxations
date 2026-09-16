# aces-relaxations

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/aces-relaxations)](https://pypi.org/project/aces-relaxations/)
[![Tests](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/tests.yaml/badge.svg)](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/code-quality-main.yaml)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

[ACES](https://eventstreamaces.readthedocs.io/) describes a prediction task over
[MEDS](https://medical-event-data-standard.github.io/) data as a tree of windows anchored at a trigger event:
a prediction time, a label window, and inclusion / exclusion criteria on the windows around them. ACES extracts
such a task from data. **`aces-relaxations` reasons about the task definition itself**, before any data is
seen, so that the task can be posed to systems that cannot run ACES: models that generate future trajectories,
models that answer "does this ever / never happen in this region" queries, time-to-event models, and so on.

It answers three questions about an ACES task, and produces the rewrites those systems need:

- **Is the task well posed?** Does the label depend only on data at or after the prediction time
    (`is_future_only`)? Does the configuration force its own label, or exclude every row (`degeneracies`)?
- **What does it ask, exactly?** `normalize` rewrites a future-only task, without changing any label, into a
    *normal form*: re-rooted at the prediction time, with the chain of windows leading to the label separated
    from the conditions on it.
- **How can it be approximated?** `simplify_constraints` maps count criteria onto "ever", "never", or nothing;
    `last_link_target` reduces the label to a single window from its own start. Each approximation states the
    direction in which it changes the task.

Every rewrite is checked against ACES itself: property tests generate random task configurations and random
MEDS data and require the rewritten task to extract exactly the rows and labels the original does (or, for an
approximation, to change them only in the stated direction).

> [!WARNING]
> This package is pre-1.0; its API may still change.

## Installation

```bash
pip install aces-relaxations
```

or, with [`uv`](https://docs.astral.sh/uv/), `uv add aces-relaxations`.

## A worked example

Take 30-day post-discharge mortality, predicted 24 hours into an admission for patients with at least three
labs by then, among those who survive the hospitalization and are not readmitted within a day of discharge.
In practice you would load your own task file with `TaskExtractorConfig.load("task.yaml")`; here `yaml_disk`
writes one to a temporary directory so that the example runs as a test.

```python
>>> from aces.config import TaskExtractorConfig
>>> from aces_relaxations import print_ACES
>>> with yaml_disk("""
... task.yaml:
...   predicates:
...     admission: {code: ADMISSION}
...     discharge: {code: DISCHARGE}
...     death: {code: MEDS_DEATH}
...     lab: {code: {regex: "^LAB//.*"}}
...   trigger: admission
...   windows:
...     input:
...       start: null
...       end: trigger + 24h
...       start_inclusive: true
...       end_inclusive: true
...       has:
...         lab: (3, None)
...       index_timestamp: end
...     hospitalization:
...       start: input.end
...       end: start -> discharge
...       start_inclusive: false
...       end_inclusive: true
...       has:
...         death: (None, 0)
...     gap:
...       start: hospitalization.end
...       end: start + 1d
...       start_inclusive: false
...       end_inclusive: true
...       has:
...         admission: (None, 0)
...         death: (None, 0)
...     target:
...       start: gap.end
...       end: start + 29d
...       start_inclusive: false
...       end_inclusive: true
...       label: death
... """) as root:
...     task = TaskExtractorConfig.load(root / "task.yaml")
>>> print_ACES(task)
trigger
└── (+1 day, 0:00:00) input.end; **Prediction Time**
    ├── (start of record) input.start (at least 3 lab)
    └── (next discharge) hospitalization.end (no death)
        └── (+1 day, 0:00:00) gap.end (no admission, death)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

`print_ACES` shows the window tree: how each endpoint is reached from its parent, the criteria of the window
it closes, and where the prediction time and label are.

### Is the task well posed?

The label window is reached from the prediction time only by moving forward, so the label cannot peek at the
past; and nothing in the configuration fixes the label or rules out every row:

```python
>>> from aces_relaxations import degeneracies, is_future_only
>>> is_future_only(task)
True
>>> degeneracies(task)
[]

```

### The normal form

`normalize` discards what happens before the prediction time (it selects the cohort, but cannot change the
label of a row in it) and pulls every criterion onto its own *branch* from the prediction time: a chain of
windows ending in either the label or one window of criteria. What remains is a *label branch* with no
criteria, and *criteria branches* that condition it. It also makes explicit that the
hospitalization must end: ACES silently drops rows whose `start -> discharge` window never closes.

```python
>>> from aces_relaxations import normalize
>>> normalized = normalize(task)
>>> print(normalized.label_branch)
hospitalization: start -> discharge -> gap: start + 1 day -> target: start + 29 days; label=death
>>> for branch in normalized.criteria_branches:
...     print(branch)
hospitalization: start -> discharge (death: (None, 0))
hospitalization: start -> discharge -> gap: start + 1 day (admission: (None, 0), death: (None, 0))
hospitalization_bounding_event: start -> discharge (discharge: (1, None))

```

The result is itself an ACES configuration, triggered at the prediction time, whose window tree is exactly
those branches: each hangs off the prediction time as a separate chain (copying any windows it has in common
with another branch), so no window has more than one child.

```python
>>> print_ACES(normalized)
trigger; **Prediction Time**
├── (next discharge) hospitalization.end
│   └── (+1 day, 0:00:00) gap.end
│       └── (+29 days, 0:00:00) target.end; **Label: Presence of death**
├── (next discharge) hospitalization_criteria.end (no death)
├── (next discharge) gap_criteria__hospitalization.end
│   └── (+1 day, 0:00:00) gap_criteria.end (no admission, death)
└── (next discharge) hospitalization_bounding_event.end (at least 1 discharge)

```

### Checking against ACES

`extract` runs ACES in memory on a MEDS dataframe. `extract_at` runs a normalized task anchored at given
prediction times, which is how the property tests compare the two: on the rows the original task extracts, the
normalized one must give the same labels.

```python
>>> import polars as pl
>>> from datetime import datetime
>>> from aces_relaxations import extract, extract_at
>>> def day(d, h=0):
...     return datetime(2020, 1, d, h)
>>> data = pl.DataFrame({
...     "subject_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
...     "time": [
...         day(1), day(1, 1), day(1, 2), day(1, 3), day(4), day(20),
...         day(1), day(1, 1), day(1, 2), day(1, 3), day(3),
...         day(1), day(1, 1), day(5),
...     ],
...     "code": [
...         "ADMISSION", "LAB//K", "LAB//NA", "LAB//CR", "DISCHARGE", "MEDS_DEATH",
...         "ADMISSION", "LAB//K", "LAB//NA", "LAB//CR", "DISCHARGE",
...         "ADMISSION", "LAB//K", "DISCHARGE",
...     ],
...     "numeric_value": [None] * 14,
... })
>>> rows = extract(task, data)
>>> rows
shape: (2, 3)
┌────────────┬─────────────────────┬───────────────┐
│ subject_id ┆ prediction_time     ┆ boolean_value │
│ ---        ┆ ---                 ┆ ---           │
│ i64        ┆ datetime[μs]        ┆ bool          │
╞════════════╪═════════════════════╪═══════════════╡
│ 1          ┆ 2020-01-02 00:00:00 ┆ true          │
│ 2          ┆ 2020-01-02 00:00:00 ┆ false         │
└────────────┴─────────────────────┴───────────────┘
>>> extract_at(normalized, data, rows).equals(rows)
True

```

Subject 3 has only one lab in the first 24 hours, so the original task excludes them; the normalized task,
which no longer looks before the prediction time, is only ever compared on the rows the original extracts.

### Approximations

A query language that can only ask whether something *ever* or *never* happens cannot express "at least 3
labs". `simplify_constraints` maps every count criterion onto one of those shapes, or drops it:

| Original constraint    | Meaning             | Default target | Effect of the default          |
| ---------------------- | ------------------- | -------------- | ------------------------------ |
| `(None, 0)` / `(0, 0)` | never               | `never`        | exact                          |
| `(1, None)`            | at least once       | `ever`         | exact                          |
| `(k, None)`, `k > 1`   | at least `k` times  | `ever`         | weakening (accepts a superset) |
| `(k, m)`, `k >= 1`     | between `k` and `m` | `ever`         | weakening (accepts a superset) |
| `(None, m)`, `m > 0`   | at most `m` times   | `null`         | weakening (constraint dropped) |
| `(None, None)`         | unconstrained       | `null`         | exact                          |

`ever_cutoff` drops "at least `k`" criteria above a cutoff instead, `never_cutoff` strengthens "at most `m`"
criteria up to a cutoff to "never", and `predicate_overrides` / `window_overrides` force a target. The
configuration is modified in place:

```python
>>> import copy
>>> from aces_relaxations import simplify_constraints
>>> print_ACES(simplify_constraints(copy.deepcopy(task)))
trigger
└── (+1 day, 0:00:00) input.end; **Prediction Time**
    ├── (start of record) input.start (at least 1 lab)
    └── (next discharge) hospitalization.end (no death)
        └── (+1 day, 0:00:00) gap.end (no admission, death)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of death**

```

A model that answers a question about a single window from "now" cannot follow the chain from the prediction
time to the label window. `last_link_target` keeps the label window alone, re-rooted at its own start, and
notes whether that is exactly the original window:

```python
>>> from aces_relaxations import last_link_target
>>> target = last_link_target(normalized)
>>> target.describe()
'death within 29.0 days of the window start (start exclusive, end inclusive)'
>>> target.note
'exact'

```

Whether it is acceptable to ask that question at the prediction time instead of at the label window's start,
or to drop the criteria branches, depends on the model; those decisions are left to the caller.

## More

- **Look-back windows.** A window defined backward from a later endpoint (`start: end - 1d`) is kept when it is
    guaranteed not to reach before the prediction time. The default, *strict* normal form rewrites it as a
    forward chain; `normalize(task, allow_end_rooted=True)` gives the *universal* form, which also keeps
    look-backs that have no forward rewrite (the day before the next discharge) as end-rooted windows.
- **Explicit bounding events.** `require_bounding_events` adds, to any ACES configuration, a criterion that each
    event-bounded window's bounding event occurs; `normalize` applies it first.
- **Degeneracies.** `degeneracies` reports labels forced to `True` or `False` and criteria no row can satisfy,
    with the facts responsible. It is sound but incomplete: everything it reports is forced, but it can miss
    degeneracies that depend on orderings it cannot prove from the window tree.
- **Predicates.** `predicate_expr` expands a derived predicate into a `PredicateExpr` tree over plain
    predicates.

Each function's docstring has further examples.

## Contributing

Development uses [`uv`](https://docs.astral.sh/uv/): `uv sync`, then `uv run pytest`. See
[`CONTRIBUTORS.md`](CONTRIBUTORS.md) for conventions and the pull request workflow.
