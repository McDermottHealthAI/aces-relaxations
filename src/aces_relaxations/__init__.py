"""Structural analysis and a label-preserving normal form for ACES task configurations.

[ACES](https://eventstreamaces.readthedocs.io/) describes a prediction task over MEDS data as a tree of
windows anchored at a trigger event. This package reasons about such a configuration *before* any data is
seen, and rewrites it into shapes that simpler query languages (and the models behind them) can answer:

- **Properties.** [`is_future_only`][aces_relaxations.future_only.is_future_only] checks that the label only
  depends on data at or after the prediction time;
  [`degeneracies`][aces_relaxations.degeneracy.degeneracies] finds labels the configuration forces and
  criteria no row can satisfy.
- **Label-preserving rewrites.**
  [`require_bounding_events`][aces_relaxations.bounding_events.require_bounding_events] makes the implicit
  "this event-bounded window can be realized" requirement explicit;
  [`normalize`][aces_relaxations.normalization.normalize] re-roots a future-only configuration at the
  prediction time and separates the chain of windows leading to the label from the conditions on it.
- **Approximations.**
  [`simplify_constraints`][aces_relaxations.constraint_simplification.simplify_constraints] maps count
  criteria onto "ever", "never", or nothing;
  [`last_link_target`][aces_relaxations.reductions.last_link_target] reduces the label to a single window
  from its own start.
- **Utilities.** [`print_ACES`][aces_relaxations.printing.print_ACES] pretty-prints a window tree;
  [`extract`][aces_relaxations.extraction.extract] and [`extract_at`][aces_relaxations.extraction.extract_at]
  run ACES in memory, the latter anchored at given prediction times, which is how every rewrite is checked
  against ACES itself.
"""

from .bounding_events import require_bounding_events
from .branches import Branch, Edge
from .constraint_simplification import ConstraintSimplificationRules, ConstraintTarget, simplify_constraints
from .degeneracy import Finding, degeneracies, is_always_negative, is_always_positive, is_unsatisfiable
from .extraction import extract, extract_at
from .future_only import future_only_violations, is_future_only, validate_future_only
from .normalization import NormalizedTaskConfig, normalize
from .predicates import PredicateExpr, predicate_expr
from .printing import print_ACES
from .reductions import TargetWindow, last_link_target

__all__ = [
    "Branch",
    "ConstraintSimplificationRules",
    "ConstraintTarget",
    "Edge",
    "Finding",
    "NormalizedTaskConfig",
    "PredicateExpr",
    "TargetWindow",
    "degeneracies",
    "extract",
    "extract_at",
    "future_only_violations",
    "is_always_negative",
    "is_always_positive",
    "is_future_only",
    "is_unsatisfiable",
    "last_link_target",
    "normalize",
    "predicate_expr",
    "print_ACES",
    "require_bounding_events",
    "simplify_constraints",
    "validate_future_only",
]
