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

### Order of application and extension

`relax` applies relaxations in the fixed order of the `RELAXATIONS` registry, regardless of the order in the
mapping you pass: constraint-modifying relaxations (`remove_all_criteria`, `simplify_constraints`) run before
structural ones (`collapse_temporal_gap_windows`, `remove_post_label_windows`).

```pycon
>>> from aces_relaxations import RELAXATIONS
>>> list(RELAXATIONS)
['remove_all_criteria', 'simplify_constraints', 'collapse_temporal_gap_windows', 'remove_post_label_windows']

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

- Mapping predicates over codes absent from a tokenized model vocabulary onto approximate tokenized forms.
- Mapping value-based predicates onto code-set predicates given quantized values joined with codes.
- Simplifying complex multi-window queries into single-window approximations.

## Repository management

This repository lives within the
[McDermottHealthAI](https://github.com/McDermottHealthAI) GitHub organization, which sets default
issue labels that propagate to new repos. All changes to `main` go through pull requests; see the
"Pull Request Workflow" section of [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for the expected flow.
Versioning follows semantic versioning, managed through `git` tags (e.g., `git tag 0.0.1`) —
`setuptools-scm` reads the tag and stamps the package version automatically.
