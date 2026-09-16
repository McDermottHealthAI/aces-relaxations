"""Predicate expression trees over the plain predicates of an ACES task configuration, and implication.

ACES predicates are either *plain* (a code, optionally with value bounds) or *derived* (``and(...)`` /
``or(...)`` over other predicates, possibly nested). Several analyses need to know which plain predicates a
criterion or label ultimately depends on, and whether an occurrence of one predicate is necessarily an
occurrence of another; this module expands derived predicates into explicit expression trees and decides that
implication conservatively.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from aces.config import DerivedPredicateConfig, TaskExtractorConfig
from aces.types import ANY_EVENT_COLUMN

from .tree import RECORD_BOUNDARIES


@dataclass(frozen=True)
class PredicateExpr:
    """An expression tree over plain predicate names.

    Attributes:
        op: ``"leaf"`` for a plain predicate, or ``"and"`` / ``"or"`` for a derived predicate.
        name: The plain predicate name (leaves only).
        children: The sub-expressions (``and`` / ``or`` only).

    Examples:
        >>> expr = PredicateExpr.or_(PredicateExpr.leaf("a"), PredicateExpr.and_(
        ...     PredicateExpr.leaf("b"), PredicateExpr.leaf("c")
        ... ))
        >>> print(expr)
        or(a, and(b, c))
        >>> sorted(expr.leaves)
        ['a', 'b', 'c']
        >>> expr.is_or_only
        False
        >>> PredicateExpr.or_(PredicateExpr.leaf("a"), PredicateExpr.leaf("b")).is_or_only
        True
        >>> PredicateExpr.from_dict(expr.to_dict()) == expr
        True
    """

    op: Literal["leaf", "and", "or"]
    name: str | None = None
    children: tuple["PredicateExpr", ...] = ()

    @classmethod
    def leaf(cls, name: str) -> "PredicateExpr":
        """A plain predicate."""
        return cls("leaf", name=name)

    @classmethod
    def and_(cls, *children: "PredicateExpr") -> "PredicateExpr":
        """A conjunction."""
        return cls("and", children=tuple(children))

    @classmethod
    def or_(cls, *children: "PredicateExpr") -> "PredicateExpr":
        """A disjunction."""
        return cls("or", children=tuple(children))

    @property
    def leaves(self) -> set[str]:
        """The plain predicate names this expression depends on."""
        if self.op == "leaf":
            return {self.name}
        return set().union(*(child.leaves for child in self.children))

    @property
    def is_or_only(self) -> bool:
        """Whether the expression is a leaf or a (nested) disjunction of leaves."""
        return self.op == "leaf" or (self.op == "or" and all(c.is_or_only for c in self.children))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable representation."""
        if self.op == "leaf":
            return {"op": "leaf", "name": self.name}
        return {"op": self.op, "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PredicateExpr":
        """Inverse of ``to_dict``."""
        if d["op"] == "leaf":
            return cls.leaf(d["name"])
        return cls(d["op"], children=tuple(cls.from_dict(c) for c in d["children"]))

    def __str__(self) -> str:
        """Renders the expression in ACES syntax, e.g. ``or(a, and(b, c))``."""
        if self.op == "leaf":
            return self.name
        return f"{self.op}({', '.join(str(c) for c in self.children)})"


def predicate_expr(task_cfg: TaskExtractorConfig, name: str) -> PredicateExpr:
    """Expands a (possibly derived) predicate into an expression tree over plain predicates.

    Args:
        task_cfg: The task configuration defining the predicates.
        name: The predicate name, or ``_ANY_EVENT``.

    Returns:
        The expression tree.

    Raises:
        KeyError: If the predicate is not defined.

    Examples:
        >>> print(predicate_expr(sample_ACES_cfg, "discharge_or_death"))
        or(icu_discharge, death)
        >>> print(predicate_expr(sample_ACES_cfg, "death"))
        death
        >>> print(predicate_expr(sample_ACES_cfg, "_ANY_EVENT"))
        _ANY_EVENT
        >>> predicate_expr(sample_ACES_cfg, "nope")
        Traceback (most recent call last):
            ...
        KeyError: "Predicate 'nope' is not defined."

    Nested derived predicates are expanded recursively:

        >>> from aces.config import DerivedPredicateConfig, EventConfig, PlainPredicateConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={
        ...         "a": PlainPredicateConfig("A"),
        ...         "b": PlainPredicateConfig("B"),
        ...         "c": PlainPredicateConfig("C"),
        ...         "ab": DerivedPredicateConfig("and(a, b)"), "abc": DerivedPredicateConfig("or(ab, c)"),
        ...     },
        ...     trigger=EventConfig("a"),
        ...     windows={"w": WindowConfig("trigger", "start + 1d", True, True)},
        ... )
        >>> print(predicate_expr(cfg, "abc"))
        or(and(a, b), c)
    """
    if name == ANY_EVENT_COLUMN:
        return PredicateExpr.leaf(ANY_EVENT_COLUMN)
    if name not in task_cfg.predicates:
        raise KeyError(f"Predicate '{name}' is not defined.")

    pred = task_cfg.predicates[name]
    if not isinstance(pred, DerivedPredicateConfig):
        return PredicateExpr.leaf(name)

    children = tuple(predicate_expr(task_cfg, child) for child in pred.input_predicates)
    return PredicateExpr.and_(*children) if pred.is_and else PredicateExpr.or_(*children)


def implication(task_cfg: TaskExtractorConfig) -> Callable[[str, str], bool]:
    """Returns ``implies(g, p)``: whether an occurrence of predicate ``g`` is necessarily one of ``p``.

    The check is conservative (it may answer ``False`` for an implication that holds, never ``True`` for one
    that does not): ``p`` must be a (nested) disjunction of plain predicates, and every disjunct of ``g`` (or
    some conjunct of it) must be among them. Every occurrence is an event (``_ANY_EVENT``). The record
    boundaries are positions rather than predicates, and imply and are implied by nothing.

    This is implication of *occurrences*, not of counts: ACES counts every matching code of a plain predicate
    at a timestamp, but a derived predicate (and ``_ANY_EVENT``) at most once, so two ``a`` codes at one time
    are two occurrences of ``a`` and one of ``or(a, b)``. A count bound above 1 therefore only transfers
    between identical predicates.

    Examples:
        >>> from aces.config import DerivedPredicateConfig, EventConfig, PlainPredicateConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={
        ...         "a": PlainPredicateConfig("A"),
        ...         "b": PlainPredicateConfig("B"),
        ...         "ab": DerivedPredicateConfig("and(a, b)"),
        ...         "a_or_b": DerivedPredicateConfig("or(a, b)"),
        ...     },
        ...     trigger=EventConfig("a"),
        ...     windows={"w": WindowConfig("trigger", "start + 1d", True, True)},
        ... )
        >>> implies = implication(cfg)
        >>> implies("a", "a"), implies("a", "a_or_b"), implies("a_or_b", "a"), implies("a", "_ANY_EVENT")
        (True, True, False, True)

    A co-occurrence implies each of its parts, but nothing implies a co-occurrence (except itself):

        >>> implies("ab", "a"), implies("a", "ab"), implies("ab", "ab")
        (True, False, True)
        >>> implies("_RECORD_END", "_RECORD_END"), implies("a", "_RECORD_START")
        (False, False)

    A conjunction implies ``p`` only through a conjunct that does, so ``and(or(a, b), c)`` does not imply
    ``a``:

        >>> cfg.predicates["c"] = PlainPredicateConfig("C")
        >>> cfg.predicates["a_or_b_and_c"] = DerivedPredicateConfig("and(a_or_b, c)")
        >>> implies("a_or_b_and_c", "a"), implies("a_or_b_and_c", "a_or_b")
        (False, True)
    """

    def occurs_within(expr: PredicateExpr, allowed: set[str]) -> bool:
        if expr.op == "leaf":
            return expr.name in allowed
        if expr.op == "and":
            return any(occurs_within(child, allowed) for child in expr.children)
        return all(occurs_within(child, allowed) for child in expr.children)

    def implies(g: str, p: str) -> bool:
        if g in RECORD_BOUNDARIES or p in RECORD_BOUNDARIES:
            return False
        if g == p or p == ANY_EVENT_COLUMN:
            return True
        p_expr = predicate_expr(task_cfg, p)
        return p_expr.is_or_only and occurs_within(predicate_expr(task_cfg, g), p_expr.leaves)

    return implies
