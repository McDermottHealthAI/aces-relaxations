"""Relaxations and structural properties for ACES task configurations.

This package lets you take a future-only [ACES](https://eventstreamaces.readthedocs.io/) task configuration
and (a) check that it is *guaranteeably future-only*, (b) prune it to the windows that matter after the
prediction time, and (c) apply named *relaxations* that simplify its constraints and window structure so it
can be expressed in reduced query languages (generative trajectory labeling, "ever / never within a region"
queries, and so on).

Key entry points:
  - [`relax`][aces_relaxations.relaxations.relax] -- validate, prune, and apply relaxations by name.
  - [`is_future_only`][aces_relaxations.future_only.is_future_only] /
    [`validate_future_only`][aces_relaxations.future_only.validate_future_only] -- the structural property.
  - [`print_ACES`][aces_relaxations.printing.print_ACES] -- a tree pretty-printer for configurations.
"""

from .constraint_simplification import (
    ConstraintSimplificationRules,
    ConstraintTarget,
    classify_constraint,
    simplify_constraints,
)
from .extraction import extract, extract_at, get_predicates, with_index_events
from .future_only import future_only_violations, is_future_only, validate_future_only
from .printing import get_constraint_str, print_ACES
from .relaxations import (
    RELAXATIONS,
    collapse_temporal_gap_windows,
    relax,
    remove_all_criteria,
    remove_post_label_windows,
    strip_to_relevant_windows,
)
from .tree import RelaxedTaskConfig, WindowNode, label_node, prediction_time_node, resolve_node

__all__ = [
    "RELAXATIONS",
    "ConstraintSimplificationRules",
    "ConstraintTarget",
    "RelaxedTaskConfig",
    "WindowNode",
    "classify_constraint",
    "collapse_temporal_gap_windows",
    "extract",
    "extract_at",
    "future_only_violations",
    "get_constraint_str",
    "get_predicates",
    "is_future_only",
    "label_node",
    "prediction_time_node",
    "print_ACES",
    "relax",
    "remove_all_criteria",
    "remove_post_label_windows",
    "resolve_node",
    "simplify_constraints",
    "strip_to_relevant_windows",
    "validate_future_only",
]
