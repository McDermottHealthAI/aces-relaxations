"""Relaxations and structural properties for ACES task configurations.

This package lets you take a future-only [ACES](https://eventstreamaces.readthedocs.io/) task configuration
and (a) check that it is *guaranteeably future-only*, (b) prune it to the windows that matter after the
prediction time, and (c) apply named *relaxations* that simplify its constraints and window structure so it
can be expressed in reduced query languages (generative trajectory labeling, "ever / never within a region"
queries, and so on).

Key entry points:
  - [`normalize`][aces_relaxations.normalization.normalize] -- the normalized form: the target chain separated
    from condition branches, label-preserving (checked against ACES in the tests).
  - [`relax`][aces_relaxations.registry.relax] -- validate, prune, and apply relaxations by name.
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
from .extraction import (
    extract,
    extract_at,
    extract_label_window_bounds,
    get_predicates,
    with_index_events,
)
from .future_only import future_only_violations, is_future_only, validate_future_only
from .independence import assume_independent_codes
from .normalization import Branch, Edge, NormalizedTaskConfig, branches, normalize
from .predicates import PredicateExpr, label_predicate_expr, leaf_predicates, predicate_expr
from .printing import get_constraint_str, print_ACES
from .registry import RELAXATIONS, relax
from .relaxations import (
    collapse_temporal_gap_windows,
    remove_all_criteria,
    remove_post_label_windows,
    strip_to_relevant_windows,
)
from .tree import RelaxedTaskConfig, WindowNode, label_node, prediction_time_node, resolve_node
from .vocabulary import MARGINAL_CODE, matching_codes, restrict_to_vocabulary, vocabulary_matches

__all__ = [
    "MARGINAL_CODE",
    "RELAXATIONS",
    "Branch",
    "ConstraintSimplificationRules",
    "ConstraintTarget",
    "Edge",
    "NormalizedTaskConfig",
    "PredicateExpr",
    "RelaxedTaskConfig",
    "WindowNode",
    "assume_independent_codes",
    "branches",
    "classify_constraint",
    "collapse_temporal_gap_windows",
    "extract",
    "extract_at",
    "extract_label_window_bounds",
    "future_only_violations",
    "get_constraint_str",
    "get_predicates",
    "is_future_only",
    "label_node",
    "label_predicate_expr",
    "leaf_predicates",
    "matching_codes",
    "normalize",
    "predicate_expr",
    "prediction_time_node",
    "print_ACES",
    "relax",
    "remove_all_criteria",
    "remove_post_label_windows",
    "resolve_node",
    "restrict_to_vocabulary",
    "simplify_constraints",
    "strip_to_relevant_windows",
    "validate_future_only",
    "vocabulary_matches",
    "with_index_events",
]
