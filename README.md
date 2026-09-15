# aces-relaxations

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/aces-relaxations)](https://pypi.org/project/aces-relaxations/)
[![Tests](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/tests.yaml/badge.svg)](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/McDermottHealthAI/aces-relaxations.svg)](https://github.com/McDermottHealthAI/aces-relaxations/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/McDermottHealthAI/aces-relaxations/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

The normalized form, structural properties, and count-constraint simplification of
[ACES](https://eventstreamaces.readthedocs.io/) task configurations, so that a future-only ACES task can be
posed to systems with a more limited query language than ACES itself: generative trajectory models (labeling
sampled futures), "ever / never within a region" query-answering models, time-to-event models, and so on.

This package holds the model-agnostic pieces every such converter needs:

- the **future-only** check, a structural property of the window tree;
- **degeneracy** detection, for configurations that fix their own label or that no row can satisfy;
- the label-preserving **normalized form**, which separates the chain of windows leading to the label from
    the criteria and realizability requirements that condition it, and the two reductions of it that the
    converters share;
- **constraint simplification**, the one approximation that is the same for every reduced query language:
    mapping count criteria onto "ever", "never", or nothing;
- `print_ACES`, a tree pretty-printer for configurations.

Model-specific converters that consume the normalized form live in sibling repositories (e.g. `aces-to-eq`,
`aces-to-motor`, `aces-to-ethos`, `aces-to-meds-eic-ar`), and inference over generated trajectories stays in
[`MEDS_trajectory_evaluation`](https://github.com/mmcdermott/MEDS_trajectory_evaluation), from which this
package was extracted. The broader suite of named relaxations this package once carried (criteria removal,
temporal-gap collapsing, post-label pruning, code independence, vocabulary restriction, and the `relax`
registry that composed them) is preserved on the
[`extended-relaxations`](https://github.com/McDermottHealthAI/aces-relaxations/tree/extended-relaxations)
branch; those approximations are model-specific decisions, and each converter now makes them itself on top
of the normalized form.

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

The main entry point is `normalize`, which checks that a configuration is future-only, re-roots it at the
prediction time, and separates the target chain from its conditions without changing any label. The other
analyses work on the original configuration:

```python
from aces.config import TaskExtractorConfig
from aces_relaxations import degeneracies, normalize, print_ACES, simplify_constraints

cfg = TaskExtractorConfig.load("task.yaml", "predicates.yaml")

# Labels the configuration forces, and criteria no row can satisfy.
for finding in degeneracies(cfg):
    print(finding)

# Optional, in place: count criteria become "ever", "never", or nothing.
simplify_constraints(cfg)

# The universal normalized form: the target chain, separated from its conditions.
normalized = normalize(cfg, allow_end_rooted=True)
print_ACES(normalized)
print(normalized.label_branch)
for branch in normalized.criteria_branches:
    print(branch)
```

## Terminology

| Term            | Description                                                                                                                                                                                                                                                                                |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ACES            | [ACES](https://eventstreamaces.readthedocs.io/en/latest/) is a domain specific language for describing task cohorts and a tool to automatically extract them from EHR datasets. It is the "source of truth" for task definitions in this work.                                             |
| Task Config     | The (original / raw) ACES configuration file that describes the task cohort.                                                                                                                                                                                                               |
| Input Window    | The window in the ACES config defining the "prediction time". This is indicated via the `index_timestamp` marker in the ACES config.                                                                                                                                                       |
| Target Window   | The window in the ACES config over which the label is extracted. This is indicated via the `label` marker in the ACES config.                                                                                                                                                              |
| Future-only     | A config is *guaranteeably future-only* when both endpoints of its target window are reachable from the prediction time through edges that only move forward in time (or look back by a provably guaranteed amount), so the label can only depend on data at or after the prediction time. |
| Normalized form | A label-preserving rewrite of a future-only config, re-rooted at the prediction time, in which the chain of windows leading to the label (the *target*) is separated from the criteria and realizability requirements that condition it (the *conditions*), each on its own branch.        |
| Simplification  | A modification of the criteria that weakens (or, on request, strengthens) count constraints so that a reduced query language can express them. Unlike normalization, it changes what the task means, in a documented direction.                                                            |

## Example configurations

The sections below share a few example task configurations. To construct them, we first need
to import the relevant ACES config classes:

```pycon
>>> from aces.config import (
...     PlainPredicateConfig,
...     EventConfig,
...     TaskExtractorConfig,
...     WindowConfig,
...     DerivedPredicateConfig,
... )
>>> from aces_relaxations import print_ACES

```

We'll also use the `print_ACES` helper function to visualize the task configs.

### Example 1: In-hospital mortality prediction

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

### Example 2: 30-day post discharge mortality prediction

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

### Example 3: 30-day readmission prediction with censoring

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

### Example 4: Two-stage Infusion

In this hypothetical example, we are examining a cohort of patients who are given an infusion, then given a
drug, then (within 10 minutes) have their infusion stopped temporarily, then resumed. We are interested in
predicting, at the time of the drug being given, about an adverse event within their second infusion stage.
The reason to have such a task is to explore how the normalized form and its reductions behave in more
complex set-ups.

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

### Example 5: Counted criteria

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

### Other examples we can't reflect:

1. What if we only want to count something as a readmission only if the next admission has a discharge
    associated with a particular diagnosis code? We can't reflect this in ACES currently, but it would pose
    additional challenges.

## The future-only property

Normalization only makes sense for tasks whose label lies in the future of the prediction time. Whether that
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

## Degenerate configurations

A configuration can fix its own outcome before any data is seen: a criterion may guarantee an occurrence of
the label predicate inside the label window (every label is `True`), forbid one over a window enclosing it
(every label is `False`), or contradict another criterion (no row ever satisfies them all). `degeneracies`
lists such findings, and `is_always_positive` / `is_always_negative` / `is_unsatisfiable` summarize them:

```pycon
>>> from aces.config import EventConfig, PlainPredicateConfig, WindowConfig
>>> from aces_relaxations import degeneracies
>>> predicates = {"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")}
>>> cfg = TaskExtractorConfig(
...     predicates=predicates,
...     trigger=EventConfig("A"),
...     windows={
...         "week": WindowConfig("trigger", "start + 7d", False, True, has={"B": "(1, None)"}),
...         "target": WindowConfig("trigger", "start + 10d", False, True, label="B"),
...     },
... )
>>> for finding in degeneracies(cfg):
...     print(finding)
always_positive: 'week: B (1, None)' guarantees B within the label window (pos(trigger), pos(target.end)]
>>> cfg = TaskExtractorConfig(
...     predicates=predicates,
...     trigger=EventConfig("A"),
...     windows={
...         "rest": WindowConfig("trigger", None, False, True, has={"B": "(None, 0)"}),
...         "target": WindowConfig("trigger -> B", "start + 1d", False, True, label="A"),
...     },
... )
>>> for finding in degeneracies(cfg):
...     print(finding)
unsatisfiable: 'rest: B (None, 0)' allows at most 0 but 'target.start is an occurrence of B' guarantees 1

```

The analysis works on the window tree symbolically. Every hop to the next (or prior) occurrence of a
predicate guarantees one at its landing point and forbids any strictly between the search's start and that
point; the trigger predicate occurs at the trigger; `(k, None)` criteria are lower bounds and `(None, k)`
criteria upper bounds over their windows. Lower bounds propagate to enclosing intervals and upper bounds to
enclosed ones, wherever the enclosure can be proven from the tree's structure (temporal offsets compare
exactly; an event endpoint is only known to follow the position it was searched from). It is sound but
incomplete: everything it reports is truly forced, checked by property tests against ACES on random data,
but it can miss degeneracies that depend on orderings it cannot prove. Inclusivity matters: a label window
`(trigger, next B]` is *not* always positive, because a `B` at the trigger is the next `B` and makes the
window empty.

## Normalized form

Every converter approximates a task in its own way. Before changing anything, it helps to have a canonical
way of writing a future-only task that *does not* change its meaning but separates the two things every
downstream converter needs separately: the **target** (the chain of windows leading to the label window) and the
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
4. Along each branch, runs of unconstrained, unlabeled windows whose composition is a single window
    collapse into it: two temporal windows into one of the summed size, and a window followed by a hop to
    the next `Q` into one hop to the next `Q` from the first window's start whenever `Q` is provably absent
    wherever the longer search would additionally look (by a criterion on any window covering it, or by an
    earlier hop's gap, e.g. `-> A_or_B` then `-> A`). A hop that lands where it starts (an inclusive search
    for an event the position already is) drops out of the chain. Exactness at the endpoints is checked:
    with ACES's usual `(start, next P]` window a `Q` exactly at the start is not forbidden, so that shape
    only merges when the first window includes its start or excludes its end, or when a `Q` there would
    have stopped the previous hop anyway.
5. Redundant criteria and branches are removed (`aces_relaxations.simplification`): constrained windows
    with the same prefix and bound merge into one; a criterion implied by another (a tighter bound on an
    implying / implied predicate over a containing / contained window), by a hop's gap, or by an occurrence
    guaranteed inside its window goes; a branch that asserts nothing (an unconstrained temporal window, or an
    unconstrained hop whose landing a criterion guarantees) goes. A hop's landing counts as a guarantee only
    when some *other* criterion explicitly asserts that the hop lands, so the "bounding event occurs"
    criteria of step 3 stay: they are what makes realizability explicit for downstream converters.
6. Branches sharing an unconstrained prefix share those windows, so the result is again a tree in which
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
│   └── (+1 day, 0:00:00) gap_criteria.end (no admission, death)
├── (+1 day, 0:00:00) post_input.end (no admission, discharge_or_death)
├── (next discharge) hospitalization_criteria.end (no death)
└── (next discharge) hospitalization_bounding_event.end (at least 1 discharge)
>>> print(normalized.label_branch)
hospitalization: start -> discharge -> gap: start + 1 day -> target: start + 29 days; label=death
>>> for branch in normalized.criteria_branches:
...     print(branch)
post_input: start + 1 day (admission: (None, 0), discharge_or_death: (None, 0))
hospitalization: start -> discharge (death: (None, 0))
hospitalization: start -> discharge -> gap: start + 1 day (admission: (None, 0), death: (None, 0))
hospitalization_bounding_event: start -> discharge (discharge: (1, None))

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
    whenever the bounding event coincides with the other endpoint. Then an auxiliary window carries the
    criterion, in whichever of two forms is exact. A forward bound with an inclusive end gets a *sibling*
    repeating the same search from the same anchor, both endpoints inclusive: it resolves to the same
    bounding event and counts it, and under infinity semantics it reads directly as an infinite window
    containing no `P`, failing the criterion. Any other bound (a strict forward search, which skips a `P`
    at its anchor, or a look-back, which ACES resolves strictly before an external anchor) gets a
    zero-length *singleton* anchored at the bounding event itself, which ACES resolves to `[t, t]` there;
    no sibling search can be exact for those.

It never adds a criterion that is already implied, so it is idempotent, and it is a no-op for ACES, which
a property test checks on arbitrary configurations. Both normalized forms apply it first.

```pycon
>>> from aces_relaxations import require_bounding_events
>>> print_ACES(require_bounding_events(in_hosp_mortality_cfg))
trigger
├── (start of record) sufficient_history.start (at least 5 event(s))
└── (+1 day, 0:00:00) input.end (no admission, discharge_or_death); **Prediction Time**
    └── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
        ├── (next discharge_or_death) target.end; **Label: Presence of death**
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
    ├── (next discharge) stay.end
    │   ├── (+30 days, 0:00:00) target.end; **Label: Presence of admission**
    │   └── (-1 day, 0:00:00) last_day.start (at least 1 lab)
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

## Simplifying count constraints

Many reduced query languages can only ask whether a predicate *ever* occurs in a region, or assert that it
*never* does. `simplify_constraints` maps every `(min, max)` count constraint onto one of those two shapes, or
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

Unlike normalization, this changes what the task means, in the direction the table states. It modifies the
configuration in place and returns it, so copy first when the original is still needed. It composes with
`normalize`: simplify, then normalize.

### On Example 5: Counted criteria

With the default rules, "at least 3 lab" weakens to "at least 1 lab" and "no more than 1 icu_transfer" is
dropped:

```pycon
>>> import copy
>>> from aces_relaxations import simplify_constraints
>>> print_ACES(simplify_constraints(copy.deepcopy(counted_cfg)))
trigger
└── (+1 day, 0:00:00) input.end; **Prediction Time**
    └── (+6 days, 0:00:00) monitoring.end (at least 1 lab; no death)
        └── (+23 days, 0:00:00) target.end; **Label: Presence of death**

```

With an `ever_cutoff` of 2, the lab constraint is dropped instead; with a `never_cutoff` of 1, the ICU
transfer constraint is strengthened to "never":

```pycon
>>> print_ACES(simplify_constraints(copy.deepcopy(counted_cfg), ever_cutoff=2, never_cutoff=1))
trigger
└── (+1 day, 0:00:00) input.end; **Prediction Time**
    └── (+6 days, 0:00:00) monitoring.end (no icu_transfer, death)
        └── (+23 days, 0:00:00) target.end; **Label: Presence of death**

```

Overrides force specific targets regardless of the constraint's shape:

```pycon
>>> print_ACES(
...     simplify_constraints(
...         copy.deepcopy(counted_cfg),
...         predicate_overrides={"death": "null"},
...         window_overrides={"monitoring": {"icu_transfer": "never"}},
...     )
... )
trigger
└── (+1 day, 0:00:00) input.end; **Prediction Time**
    └── (+6 days, 0:00:00) monitoring.end (at least 1 lab; no icu_transfer)
        └── (+23 days, 0:00:00) target.end; **Label: Presence of death**

```

The simplified configuration normalizes like any other:

```pycon
>>> print_ACES(normalize(simplify_constraints(copy.deepcopy(counted_cfg))))
trigger; **Prediction Time**
├── (+6 days, 0:00:00) monitoring.end
│   └── (+23 days, 0:00:00) target.end; **Label: Presence of death**
└── (+6 days, 0:00:00) monitoring_criteria.end (no death; at least 1 lab)

```

### On Example 3: Readmission

The gap window's "no admission, death" criteria are already in "never" form and are unchanged, as is the
"at least 1 event" censoring criterion. The look-back "at least 10 events" and "at least 5 events" criteria
weaken to "at least 1 event" (they select the cohort, so normalization discards them anyway):

```pycon
>>> print_ACES(simplify_constraints(copy.deepcopy(readmission_cfg)))
trigger; **Prediction Time**
├── (prior admission) hospitalization.start (at least 1 event(s))
│   └── (start of record) sufficient_history.start (at least 1 event(s))
└── (+1 day, 0:00:00) gap.end (no admission, death)
    └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
        └── (end of record) censoring_protection.end (at least 1 event(s))

```

## Reductions of the normalized form

`aces_relaxations.reductions` holds the two structural moves every
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

| `MEDS_trajectory_evaluation`                  | `aces_relaxations`                            |
| --------------------------------------------- | --------------------------------------------- |
| `aces_utils.WindowNode`, `_resolve_node`, ... | `WindowNode`, `resolve_node`, ...             |
| `aces_utils.print_ACES`, `get_constraint_str` | `print_ACES`, `get_constraint_str`            |
| `task_config.validate_task_cfg`               | `validate_future_only` (and `is_future_only`) |
| `task_config.convert_to_zero_shot(cfg, ...)`  | `normalize(cfg)`, then the converter's decode |

`convert_to_zero_shot` and its Hydra-selected relaxations (`remove_all_criteria`, ...) have no direct
counterpart here: the equivalent `relax` registry, `strip_to_relevant_windows`, and the named relaxations
live on the `extended-relaxations` branch, and the converters make those approximations themselves.
Predicate evaluation over MEDS data (`get_MEDS_predicates`, `get_MEDS_plain_predicates`) and trajectory
labeling remain in `MEDS_trajectory_evaluation`.

## Roadmap

- Ordering-dependent degeneracies and simplifications that the symbolic analysis cannot prove (it only
    sees orderings that follow from the window tree's structure).
- Value-based predicates and `and` (co-occurrence) predicates, which the implication machinery treats
    conservatively.

## Repository management

This repository lives within the
[McDermottHealthAI](https://github.com/McDermottHealthAI) GitHub organization, which sets default
issue labels that propagate to new repos. All changes to `main` go through pull requests; see the
"Pull Request Workflow" section of [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for the expected flow.
Versioning follows semantic versioning, managed through `git` tags (e.g., `git tag 0.0.1`) —
`setuptools-scm` reads the tag and stamps the package version automatically.
