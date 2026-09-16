"""Normalized form, structural properties, and count-constraint simplification for ACES task configurations.

This package takes a future-only [ACES](https://eventstreamaces.readthedocs.io/) task configuration and
(a) checks that it is *guaranteeably future-only*, (b) detects *degeneracies* (labels the configuration
forces before any data is seen, and criteria no row can satisfy), (c) rewrites it into a label-preserving
*normalized form* that separates the target chain from its conditions, and (d) optionally simplifies count
criteria to the "ever" / "never" shapes that reduced query languages can express. Model-specific converters
that consume the normalized form live in sibling repositories.

Key entry points:
  - [`normalize`][aces_relaxations.normalization.normalize] -- the normalized form: the target chain separated
    from condition branches, label-preserving (checked against ACES in the tests).
  - [`is_future_only`][aces_relaxations.future_only.is_future_only] /
    [`validate_future_only`][aces_relaxations.future_only.validate_future_only] -- the structural property.
  - [`degeneracies`][aces_relaxations.degeneracy.degeneracies] -- structurally forced labels (always 1 / 0)
    and unsatisfiable criteria.
  - [`simplify_constraints`][aces_relaxations.constraint_simplification.simplify_constraints] -- count
    criteria to "ever", "never", or nothing.
  - [`label_only`][aces_relaxations.reductions.label_only] /
    [`last_link_target`][aces_relaxations.reductions.last_link_target] -- reductions of the normalized form
    shared by the converters.
  - [`print_ACES`][aces_relaxations.printing.print_ACES] -- a tree pretty-printer for configurations.
"""

from .bounding_events import require_bounding_events
from .constraint_simplification import (
    ConstraintSimplificationRules,
    ConstraintTarget,
    classify_constraint,
    simplify_constraints,
)
from .degeneracy import Finding, degeneracies, is_always_negative, is_always_positive, is_unsatisfiable
from .extraction import (
    extract,
    extract_at,
    extract_label_window_bounds,
    get_predicates,
    with_index_events,
)
from .future_only import future_only_violations, is_future_only, validate_future_only
from .normalization import Branch, Edge, NormalizedTaskConfig, branches, normalize
from .predicates import PredicateExpr, label_predicate_expr, leaf_predicates, predicate_expr
from .printing import get_constraint_str, print_ACES
from .reductions import TargetWindow, label_only, last_link_target
from .simplification import simplify_branches
from .tree import WindowNode, label_node, prediction_time_node, resolve_node

__all__ = [
    "Branch",
    "ConstraintSimplificationRules",
    "ConstraintTarget",
    "Edge",
    "Finding",
    "NormalizedTaskConfig",
    "PredicateExpr",
    "TargetWindow",
    "WindowNode",
    "branches",
    "classify_constraint",
    "degeneracies",
    "extract",
    "extract_at",
    "extract_label_window_bounds",
    "future_only_violations",
    "get_constraint_str",
    "get_predicates",
    "is_always_negative",
    "is_always_positive",
    "is_future_only",
    "is_unsatisfiable",
    "label_node",
    "label_only",
    "label_predicate_expr",
    "last_link_target",
    "leaf_predicates",
    "normalize",
    "predicate_expr",
    "prediction_time_node",
    "print_ACES",
    "require_bounding_events",
    "resolve_node",
    "simplify_branches",
    "simplify_constraints",
    "validate_future_only",
    "with_index_events",
]
