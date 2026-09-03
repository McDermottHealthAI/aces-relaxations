"""Predicate expression trees over the plain predicates of an ACES task configuration.

ACES predicates are either *plain* (a code, optionally with value bounds) or *derived* (``and(...)`` /
``or(...)`` over other predicates, possibly nested). Several relaxations need to reason about which plain
predicates a window criterion or label ultimately depends on; this module expands derived predicates into an
explicit expression tree whose leaves are plain predicate names.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from aces.config import DerivedPredicateConfig, TaskExtractorConfig

ANY_EVENT = "_ANY_EVENT"


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
        return set(self.iter_leaves())

    def iter_leaves(self) -> Iterator[str]:
        """Yields the plain predicate names this expression depends on, in order, with repeats."""
        if self.op == "leaf":
            yield self.name
        else:
            for child in self.children:
                yield from child.iter_leaves()

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

        >>> from aces.config import DerivedPredicateConfig, EventConfig, PlainPredicateConfig, WindowConfig
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
    if name == ANY_EVENT:
        return PredicateExpr.leaf(ANY_EVENT)
    if name not in task_cfg.predicates:
        raise KeyError(f"Predicate '{name}' is not defined.")

    pred = task_cfg.predicates[name]
    if not isinstance(pred, DerivedPredicateConfig):
        return PredicateExpr.leaf(name)

    children = tuple(predicate_expr(task_cfg, child) for child in pred.input_predicates)
    return PredicateExpr.and_(*children) if pred.is_and else PredicateExpr.or_(*children)


def leaf_predicates(task_cfg: TaskExtractorConfig, name: str) -> set[str]:
    """The plain predicates a (possibly derived) predicate depends on.

    Examples:
        >>> sorted(leaf_predicates(sample_ACES_cfg, "discharge_or_death"))
        ['death', 'icu_discharge']
    """
    return predicate_expr(task_cfg, name).leaves


def label_predicate_expr(task_cfg: TaskExtractorConfig) -> PredicateExpr:
    """The expression tree of the label predicate.

    Examples:
        >>> print(label_predicate_expr(sample_ACES_cfg))
        death
        >>> cfg = RelaxedTaskConfig(
        ...     predicates=sample_ACES_cfg.predicates, trigger=sample_ACES_cfg.trigger,
        ...     windows={"w": WindowConfig("trigger", "start + 1d", True, True)},
        ... )
        >>> label_predicate_expr(cfg)
        Traceback (most recent call last):
            ...
        ValueError: The task configuration has no label window.
    """
    if task_cfg.label_window is None:
        raise ValueError("The task configuration has no label window.")
    return predicate_expr(task_cfg, task_cfg.windows[task_cfg.label_window].label)
