"""Simplify window inclusion / exclusion constraints into "ever", "never", or no constraint at all.

ACES window constraints (``has``) are inclusive count ranges ``(min, max)`` on a predicate within a window.
Many downstream query languages cannot express arbitrary count ranges; they can only ask whether a predicate
*ever* occurs in a region, or assert that it *never* does. This module maps every constraint onto one of three
targets, controlled by a small set of rules and cutoffs:

| Original constraint        | Meaning                    | Default target | Soundness of the default       |
| -------------------------- | -------------------------- | -------------- | ------------------------------ |
| ``(None, 0)`` / ``(0, 0)`` | never                      | ``never``      | exact                          |
| ``(1, None)``              | at least once              | ``ever``       | exact                          |
| ``(k, None)``, ``k > 1``   | at least ``k`` times       | ``ever``       | weakening (accepts a superset) |
| ``(k, m)``, ``k >= 1``     | between ``k`` and ``m``    | ``ever``       | weakening (accepts a superset) |
| ``(None, m)``, ``m > 0``   | at most ``m`` times        | ``null``       | weakening (constraint dropped) |
| ``(None, None)``           | unconstrained              | ``null``       | exact                          |

Two cutoffs adjust the defaults:

- ``ever_cutoff``: "at least ``k``" and "between" constraints are only kept (as ``ever``) when ``k`` is at
    most this cutoff; larger ``k`` is dropped (``null``). With no cutoff, they are always kept.
- ``never_cutoff``: "at most ``m``" constraints (``m > 0``) are *strengthened* to ``never`` when ``m`` is at
    most this cutoff; larger ``m`` is dropped. The default of ``0`` never strengthens.

Per-predicate and per-window overrides can force any target regardless of the constraint shape.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aces.config import TaskExtractorConfig

from .tree import constraint_node_name

Constraint = tuple[int | None, int | None]


class ConstraintTarget(StrEnum):
    """The three shapes a constraint can be simplified into."""

    EVER = "ever"
    NEVER = "never"
    NULL = "null"


EVER: Constraint = (1, None)
NEVER: Constraint = (None, 0)


def to_constraint(target: ConstraintTarget) -> Constraint | None:
    """Returns the ACES ``(min, max)`` constraint for a simplification target, or None for ``null``.

    Examples:
        >>> to_constraint(ConstraintTarget.EVER)
        (1, None)
        >>> to_constraint(ConstraintTarget.NEVER)
        (None, 0)
        >>> print(to_constraint(ConstraintTarget.NULL))
        None
    """
    match ConstraintTarget(target):
        case ConstraintTarget.EVER:
            return EVER
        case ConstraintTarget.NEVER:
            return NEVER
        case ConstraintTarget.NULL:
            return None


def classify_constraint(
    constraint: Constraint, ever_cutoff: int | None = None, never_cutoff: int = 0
) -> ConstraintTarget:
    """Maps a single ``(min, max)`` constraint onto a simplification target given the cutoffs.

    Args:
        constraint: The inclusive ``(min, max)`` count range, where either side may be ``None``.
        ever_cutoff: "At least ``k``" constraints with ``k`` above this cutoff become ``null``. ``None``
            means no cutoff.
        never_cutoff: "At most ``m``" constraints with ``0 < m <= never_cutoff`` become ``never``; larger
            ``m`` become ``null``.

    Returns:
        The target shape.

    Raises:
        ValueError: If the constraint's minimum exceeds its maximum.

    Examples:
        >>> classify_constraint((None, 0))
        <ConstraintTarget.NEVER: 'never'>
        >>> classify_constraint((0, 0))
        <ConstraintTarget.NEVER: 'never'>
        >>> classify_constraint((1, None))
        <ConstraintTarget.EVER: 'ever'>
        >>> classify_constraint((5, None))
        <ConstraintTarget.EVER: 'ever'>
        >>> classify_constraint((5, None), ever_cutoff=3)
        <ConstraintTarget.NULL: 'null'>
        >>> classify_constraint((2, 4), ever_cutoff=3)
        <ConstraintTarget.EVER: 'ever'>
        >>> classify_constraint((None, 2))
        <ConstraintTarget.NULL: 'null'>
        >>> classify_constraint((None, 2), never_cutoff=2)
        <ConstraintTarget.NEVER: 'never'>
        >>> classify_constraint((0, 3), never_cutoff=2)
        <ConstraintTarget.NULL: 'null'>
        >>> classify_constraint((None, None))
        <ConstraintTarget.NULL: 'null'>
        >>> classify_constraint((3, 1))
        Traceback (most recent call last):
            ...
        ValueError: Constraint minimum 3 exceeds maximum 1.
    """
    lo, hi = constraint
    lo = 0 if lo is None else lo

    if hi is not None and hi < lo:
        raise ValueError(f"Constraint minimum {lo} exceeds maximum {hi}.")

    if hi == 0:
        return ConstraintTarget.NEVER

    if lo == 0:
        if hi is None:
            return ConstraintTarget.NULL
        return ConstraintTarget.NEVER if hi <= never_cutoff else ConstraintTarget.NULL

    if ever_cutoff is None or lo <= ever_cutoff:
        return ConstraintTarget.EVER
    return ConstraintTarget.NULL


@dataclass(frozen=True)
class ConstraintSimplificationRules:
    """The rules governing how constraints are simplified.

    Attributes:
        ever_cutoff: See
            [`classify_constraint`][aces_relaxations.constraint_simplification.classify_constraint].
        never_cutoff: See
            [`classify_constraint`][aces_relaxations.constraint_simplification.classify_constraint].
        predicate_overrides: Forced targets by predicate name, applied in every window.
        window_overrides: Forced targets by window name and then predicate name. These take precedence over
            ``predicate_overrides``.

    Examples:
        >>> rules = ConstraintSimplificationRules()
        >>> rules.target_for((None, 3), predicate="death", window="gap")
        <ConstraintTarget.NULL: 'null'>
        >>> rules = ConstraintSimplificationRules(never_cutoff=3)
        >>> rules.target_for((None, 3), predicate="death", window="gap")
        <ConstraintTarget.NEVER: 'never'>

    Overrides may be given as strings and are normalized to targets:

        >>> rules = ConstraintSimplificationRules(
        ...     predicate_overrides={"death": "never"}, window_overrides={"gap": {"death": "ever"}}
        ... )
        >>> rules.predicate_overrides
        {'death': <ConstraintTarget.NEVER: 'never'>}
        >>> rules.target_for((None, 3), predicate="death", window="target")
        <ConstraintTarget.NEVER: 'never'>
        >>> rules.target_for((None, 3), predicate="death", window="gap")
        <ConstraintTarget.EVER: 'ever'>
        >>> rules.target_for((None, 3), predicate="death")
        <ConstraintTarget.NEVER: 'never'>

    Invalid rules are rejected:

        >>> ConstraintSimplificationRules(ever_cutoff=0)
        Traceback (most recent call last):
            ...
        ValueError: ever_cutoff must be at least 1; got 0.
        >>> ConstraintSimplificationRules(never_cutoff=-1)
        Traceback (most recent call last):
            ...
        ValueError: never_cutoff must be non-negative; got -1.
        >>> ConstraintSimplificationRules(predicate_overrides={"death": "sometimes"})
        Traceback (most recent call last):
            ...
        ValueError: 'sometimes' is not a valid ConstraintTarget
    """

    ever_cutoff: int | None = None
    never_cutoff: int = 0
    predicate_overrides: Mapping[str, ConstraintTarget] = field(default_factory=dict)
    window_overrides: Mapping[str, Mapping[str, ConstraintTarget]] = field(default_factory=dict)

    def __post_init__(self):
        """Validates the cutoffs and normalizes override targets to ``ConstraintTarget`` values."""
        if self.ever_cutoff is not None and self.ever_cutoff < 1:
            raise ValueError(f"ever_cutoff must be at least 1; got {self.ever_cutoff}.")
        if self.never_cutoff < 0:
            raise ValueError(f"never_cutoff must be non-negative; got {self.never_cutoff}.")

        object.__setattr__(
            self,
            "predicate_overrides",
            {p: ConstraintTarget(t) for p, t in (self.predicate_overrides or {}).items()},
        )
        object.__setattr__(
            self,
            "window_overrides",
            {
                w: {p: ConstraintTarget(t) for p, t in (preds or {}).items()}
                for w, preds in (self.window_overrides or {}).items()
            },
        )

    def target_for(
        self, constraint: Constraint, *, predicate: str, window: str | None = None
    ) -> ConstraintTarget:
        """Returns the simplification target for a constraint on ``predicate`` in ``window``."""
        if window is not None and predicate in self.window_overrides.get(window, {}):
            return self.window_overrides[window][predicate]
        if predicate in self.predicate_overrides:
            return self.predicate_overrides[predicate]
        return classify_constraint(constraint, self.ever_cutoff, self.never_cutoff)


def simplify_constraints(
    task_cfg: TaskExtractorConfig,
    rules: ConstraintSimplificationRules | Mapping[str, Any] | None = None,
    **rule_kwargs: Any,
) -> TaskExtractorConfig:
    """Simplifies every window constraint in the configuration into an "ever", "never", or no constraint.

    The configuration is modified in place (both the ``has`` dictionaries of the windows and the
    ``constraints`` of the corresponding tree nodes) and returned for convenience.

    Args:
        task_cfg: The task configuration to simplify.
        rules: The rules to apply, as a
            [`ConstraintSimplificationRules`][aces_relaxations.constraint_simplification.ConstraintSimplificationRules]
            object or a mapping of its fields. Defaults to the default rules.
        **rule_kwargs: Alternatively, the rule fields may be given as keyword arguments. Cannot be combined
            with ``rules``.

    Returns:
        The modified configuration.

    Raises:
        ValueError: If both ``rules`` and ``rule_kwargs`` are given.

    Examples:
        >>> from aces.config import EventConfig, PlainPredicateConfig
        >>> predicates = {"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")}
        >>> def make_cfg():
        ...     return TaskExtractorConfig(
        ...         predicates=predicates, trigger=EventConfig("A"),
        ...         windows={
        ...             "input": WindowConfig(
        ...                 "trigger", "start + 1d", True, True, index_timestamp="end",
        ...                 has={"A": "(None, 0)", "B": "(2, None)", "_ANY_EVENT": "(None, 3)"},
        ...             ),
        ...             "target": WindowConfig(
        ...                 "input.end", "start + 3d", False, True, label="B", has={"A": "(1, 2)"}
        ...             ),
        ...         },
        ...     )
        >>> print_ACES(make_cfg())
        trigger
        └── (+1 day, 0:00:00) input.end (no A; at least 2 B; no more than 3 event(s)); **Prediction Time**
            └── (+3 days, 0:00:00) target.end (between 1-2 A); **Label: Presence of B**

    With the default rules, "at least 2 B" weakens to "at least 1 B", the "no more than 3 events" constraint
    is dropped, and "between 1-2 A" weakens to "at least 1 A":

        >>> print_ACES(simplify_constraints(make_cfg()))
        trigger
        └── (+1 day, 0:00:00) input.end (no A; at least 1 B); **Prediction Time**
            └── (+3 days, 0:00:00) target.end (at least 1 A); **Label: Presence of B**

    Cutoffs and overrides change this. Here, "at least 2 B" is dropped because 2 exceeds the ever cutoff,
    "no more than 3 events" is strengthened to "no events", and the target window's ``A`` constraint is
    forced to "never":

        >>> cfg = simplify_constraints(
        ...     make_cfg(), ever_cutoff=1, never_cutoff=3, window_overrides={"target": {"A": "never"}}
        ... )
        >>> print_ACES(cfg)
        trigger
        └── (+1 day, 0:00:00) input.end (no A, event(s)); **Prediction Time**
            └── (+3 days, 0:00:00) target.end (no A); **Label: Presence of B**

    The tree nodes are kept in sync with the window definitions:

        >>> cfg.window_nodes["input.end"].constraints
        {'A': (None, 0), '_ANY_EVENT': (None, 0)}
        >>> cfg.window_nodes["target.end"].constraints
        {'A': (None, 0)}

    Rules may also be passed as an object or a mapping:

        >>> rules = ConstraintSimplificationRules(predicate_overrides={"A": "null"})
        >>> print_ACES(simplify_constraints(make_cfg(), rules))
        trigger
        └── (+1 day, 0:00:00) input.end (at least 1 B); **Prediction Time**
            └── (+3 days, 0:00:00) target.end; **Label: Presence of B**
        >>> print_ACES(simplify_constraints(make_cfg(), {"predicate_overrides": {"A": "null"}}))
        trigger
        └── (+1 day, 0:00:00) input.end (at least 1 B); **Prediction Time**
            └── (+3 days, 0:00:00) target.end; **Label: Presence of B**
        >>> simplify_constraints(make_cfg(), rules, ever_cutoff=2)
        Traceback (most recent call last):
            ...
        ValueError: Pass either rules or keyword arguments, not both.
    """
    if rules is not None and rule_kwargs:
        raise ValueError("Pass either rules or keyword arguments, not both.")

    if rules is None:
        rules = ConstraintSimplificationRules(**rule_kwargs)
    elif not isinstance(rules, ConstraintSimplificationRules):
        rules = ConstraintSimplificationRules(**dict(rules))

    for name, window in task_cfg.windows.items():
        if not window.has:
            continue

        new_has = {}
        for predicate, constraint in window.has.items():
            target = rules.target_for(constraint, predicate=predicate, window=name)
            simplified = to_constraint(target)
            if simplified is not None:
                new_has[predicate] = simplified

        window.has.clear()
        window.has.update(new_has)

        node = task_cfg.window_nodes.get(constraint_node_name(name, window))
        if node is not None:
            node.constraints = window.has

    return task_cfg
