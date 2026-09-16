"""Detect structurally degenerate task configurations: forced labels and unsatisfiable criteria.

A configuration is *degenerate* when its windows and criteria alone, before any data is seen, fix the
outcome: every extracted label is ``True`` (the criteria guarantee an occurrence of the label predicate
inside the label window), every extracted label is ``False`` (the criteria forbid one), or no row can ever
satisfy the criteria (a guaranteed occurrence falls inside a window that forbids it, or lower-bound counts
inside a window exceed its upper bound).

The analysis is *sound but incomplete*: whatever it reports is truly forced by the configuration's
structure, but it only sees orderings it can prove symbolically (see ``aces_relaxations.ordering``). Facts are
*count bounds* on a predicate over an interval:

- every hop to the next (or prior) occurrence of ``P`` guarantees a ``P`` exactly at the landing point, and
  forbids ``P`` strictly between the search's start and the landing point;
- the trigger predicate occurs at the trigger, and every event endpoint is an occurrence of ``_ANY_EVENT``;
- a window criterion ``(k, m)`` bounds the count of its predicate over the window.

Lower bounds propagate to enclosing intervals and upper bounds to enclosed ones, so a lower bound of ``k``
on an interval provably inside an interval with an upper bound of ``j < k`` (for an implied predicate) is
a contradiction, as are lower bounds on provably disjoint sub-intervals whose sum exceeds the upper bound.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

from aces.config import TaskExtractorConfig
from aces.types import END_OF_RECORD_KEY, START_OF_RECORD_KEY, TemporalWindowBounds
from bigtree import Node

from .ordering import (
    RECORD_END_ANCHOR,
    RECORD_START_ANCHOR,
    ROOT,
    Anchor,
    CountBound,
    Interval,
    Ordering,
    Position,
    transferable_count,
)
from .predicates import implication
from .tree import endpoint_node, search_includes_anchor

Kind = Literal["always_positive", "always_negative", "unsatisfiable"]


@dataclass(frozen=True)
class Finding:
    """One way a configuration is degenerate.

    Attributes:
        kind: ``"always_positive"``, ``"always_negative"``, or ``"unsatisfiable"``.
        reason: A human-readable explanation, naming the criteria or window-tree facts responsible.
    """

    kind: Kind
    reason: str

    def __str__(self) -> str:
        """Renders the finding."""
        return f"{self.kind}: {self.reason}"


@dataclass
class Analysis:
    """The positions and count bounds of a configuration's window tree, and how they compare.

    Examples:
        >>> analysis = Analysis.of(sample_ACES_cfg)
        >>> print(analysis.describe(analysis.window("gap")))
        (pos(input.end), pos(gap.end)]
        >>> for b in analysis.bounds: print(b.source)
        the trigger is an occurrence of icu_admission
        the trigger is an event
        target.end is an occurrence of discharge_or_death
        target.end is an event
        target.end is the next discharge_or_death after gap.end
        input: icu_admission (None, 0)
        input: discharge_or_death (None, 0)
        gap: icu_admission (None, 0)
        gap: discharge_or_death (None, 0)

    Positions along a temporal chain compare exactly; an event endpoint is only known to follow the
    position it was searched from:

        >>> pt, gap_end, target_end = (analysis.positions[n] for n in ("input.end", "gap.end", "target.end"))
        >>> order = analysis.ordering
        >>> order.leq(pt, gap_end), order.leq(gap_end, pt), order.leq(gap_end, target_end)
        (True, False, True)
        >>> order.leq(pt, target_end, strict=True), order.leq(target_end, pt)
        (True, False)

    Positions render by the node they are an offset from, or as a record boundary:

        >>> later, earlier = pt.shifted(timedelta(days=2)), pt.shifted(-timedelta(hours=6))
        >>> for p in (Position(RECORD_START_ANCHOR), Position(RECORD_END_ANCHOR), later, earlier):
        ...     print(analysis.describe(p))
        pos(record_start)
        pos(record_end)
        pos(trigger) + 3 days, 0:00:00
        pos(trigger) + 18:00:00
    """

    task_cfg: TaskExtractorConfig
    implies: Callable[[str, str], bool]
    ordering: Ordering
    positions: dict[str, Position] = field(default_factory=dict)
    bounds: list[CountBound] = field(default_factory=list)

    @classmethod
    def of(cls, task_cfg: TaskExtractorConfig) -> "Analysis":
        """Analyzes a configuration's window tree."""
        self = cls(task_cfg, implication(task_cfg), Ordering([Anchor(task_cfg.trigger.predicate)]))
        root = task_cfg.window_tree
        self.positions[root.node_name] = Position(ROOT)
        self._event_at(Position(ROOT), task_cfg.trigger.predicate, "the trigger")
        for node in root.descendants:
            self._place(node)
        for name, window in task_cfg.windows.items():
            interval = self.window(name)
            for predicate, (lo, hi) in window.has.items():
                self.bounds.append(
                    CountBound(predicate, interval, lo or 0, hi, f"{name}: {predicate} ({lo}, {hi})")
                )
        return self

    def describe(self, pos_or_interval: Position | Interval) -> str:
        """Renders a position (by the tree node it is an offset from) or an interval."""
        if isinstance(pos_or_interval, Interval):
            i = pos_or_interval
            left, right = "[" if i.lo_inclusive else "(", "]" if i.hi_inclusive else ")"
            return f"{left}{self.describe(i.lo)}, {self.describe(i.hi)}{right}"
        pos = pos_or_interval
        named = next((n for n, p in self.positions.items() if p == pos), None)
        if named is not None:
            return f"pos({named})"
        base = {RECORD_START_ANCHOR: "record_start", RECORD_END_ANCHOR: "record_end"}.get(pos.anchor)
        if base is None:
            base = next(n for n, p in self.positions.items() if p == Position(pos.anchor))
        if not pos.offset:
            return f"pos({base})"
        sign = "+" if pos.offset > timedelta(0) else "-"
        return f"pos({base}) {sign} {abs(pos.offset)}"

    def _event_at(self, pos: Position, predicate: str, what: str) -> None:
        self.bounds.append(
            CountBound(predicate, Interval.point(pos), 1, None, f"{what} is an occurrence of {predicate}")
        )
        self.bounds.append(CountBound("_ANY_EVENT", Interval.point(pos), 1, None, f"{what} is an event"))

    def _place(self, node: Node) -> None:
        parent = self.positions[node.parent.node_name]
        bound = node.endpoint_expr
        if isinstance(bound, TemporalWindowBounds):
            self.positions[node.node_name] = parent.shifted(bound.window_size)
            return
        event = bound.end_event.lstrip("-")
        if event in (START_OF_RECORD_KEY, END_OF_RECORD_KEY):
            anchor = RECORD_START_ANCHOR if event == START_OF_RECORD_KEY else RECORD_END_ANCHOR
            self.positions[node.node_name] = Position(anchor)
            return
        window_name, side = node.node_name.rsplit(".", 1)
        backward = bound.end_event.startswith("-")
        strict = not search_includes_anchor(self.task_cfg.windows[window_name], side)
        anchor = (
            Anchor(event, upper=parent, strict=strict)
            if backward
            else Anchor(event, lower=parent, strict=strict)
        )
        n_anchors = len(self.ordering.anchors)
        pos = self.positions[node.node_name] = self.ordering.add(anchor)
        if len(self.ordering.anchors) == n_anchors:
            return  # The same search from the same position lands on the same, already described, time.
        self._event_at(pos, event, node.node_name)
        # The search found the *next* / *prior* occurrence, so none lies strictly in between (nor at the
        # search start, when an occurrence there would have been found).
        gap = (
            Interval(pos, parent, False, not strict) if backward else Interval(parent, pos, not strict, False)
        )
        direction, relation = ("prior", "before") if backward else ("next", "after")
        source = f"{node.node_name} is the {direction} {event} {relation} {node.parent.node_name}"
        self.bounds.append(CountBound(event, gap, 0, 0, source))

    def window(self, name: str) -> Interval:
        """A window's interval."""
        window = self.task_cfg.windows[name]
        start = self.positions[endpoint_node(self.task_cfg, name, "start").node_name]
        end = self.positions[endpoint_node(self.task_cfg, name, "end").node_name]
        return Interval(start, end, window.start_inclusive, window.end_inclusive)

    def guaranteed_within(self, predicate: str, interval: Interval) -> CountBound | None:
        """A bound guaranteeing an occurrence of (something implying) ``predicate`` inside ``interval``."""
        return next(
            (
                b
                for b in self.bounds
                if b.at_least >= 1
                and self.implies(b.predicate, predicate)
                and self.ordering.contains(interval, b.interval)
            ),
            None,
        )

    def forbidden_over(self, predicate: str, interval: Interval) -> CountBound | None:
        """A bound forbidding (something implied by) ``predicate`` over all of ``interval``."""
        return next(
            (
                b
                for b in self.bounds
                if b.at_most == 0
                and self.implies(predicate, b.predicate)
                and self.ordering.contains(b.interval, interval)
            ),
            None,
        )

    def contradictions(self) -> list[Finding]:
        """Upper bounds exceeded by lower bounds provably inside them, and lower bounds on empty intervals."""
        out = []
        for b in self.bounds:
            if b.at_least >= 1 and self.ordering.is_empty(b.interval):
                where = self.describe(b.interval)
                out.append(
                    Finding(
                        "unsatisfiable", f"'{b.source}' requires an occurrence in the empty window {where}"
                    )
                )
        for hi in self.bounds:
            if hi.at_most is None:
                continue
            inside = [
                (lo, transferable_count(lo.at_least, lo.predicate, hi.predicate))
                for lo in self.bounds
                if lo.at_least >= 1
                and self.implies(lo.predicate, hi.predicate)
                and self.ordering.contains(hi.interval, lo.interval)
            ]
            chosen: list[tuple[CountBound, int]] = []
            for lo, count in sorted(inside, key=lambda x: -x[1]):
                if all(self.ordering.disjoint(lo.interval, c.interval) for c, _ in chosen):
                    chosen.append((lo, count))
            total = sum(count for _, count in chosen)
            if total > hi.at_most:
                sources = "; ".join(f"'{c.source}'" for c, _ in chosen)
                out.append(
                    Finding(
                        "unsatisfiable",
                        f"'{hi.source}' allows at most {hi.at_most} but {sources} "
                        f"guarantee{'s' if len(chosen) == 1 else ''} {total}",
                    )
                )
        return out


def degeneracies(task_cfg: TaskExtractorConfig) -> list[Finding]:
    """Lists the ways a configuration is structurally degenerate.

    Args:
        task_cfg: The task configuration.

    Returns:
        Findings of kind ``always_positive`` (every extracted label is ``True``), ``always_negative``
        (every extracted label is ``False``), and ``unsatisfiable`` (no row satisfies the criteria), each
        with a reason. An empty list means nothing structural was found (which does not prove the task is
        non-degenerate).

    Examples:
        >>> degeneracies(sample_ACES_cfg)
        []

        A criterion requiring the label predicate inside the label window forces the label:

        >>> from aces.config import EventConfig, PlainPredicateConfig
        >>> predicates = {"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")}
        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "week": WindowConfig("trigger", "start + 7d", False, True, has={"B": "(1, None)"}),
        ...         "target": WindowConfig("trigger", "start + 10d", False, True, label="B"),
        ...     },
        ... )
        >>> for f in degeneracies(cfg): print(f)
        always_positive: 'week: B (1, None)' guarantees B within the label window
        (pos(trigger), pos(target.end)]

        So does a label that is the label window's own bounding event, or the trigger:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={"target": WindowConfig("trigger", "start -> B", True, True, label="B")},
        ... )
        >>> for f in degeneracies(cfg): print(f)
        always_positive: 'target.end is an occurrence of B' guarantees B within the label window
        [pos(trigger), pos(target.end)]

        With an exclusive start the same window is *not* forced: a ``B`` at the trigger itself is the next
        ``B`` (the found endpoint is inclusive), which makes the window empty and the label ``False``:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={"target": WindowConfig("trigger", "start -> B", False, True, label="B")},
        ... )
        >>> degeneracies(cfg)
        []
        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={"target": WindowConfig("trigger", "start + 1d", True, True, label="A")},
        ... )
        >>> for f in degeneracies(cfg): print(f)
        always_positive: 'the trigger is an occurrence of A' guarantees A within the label window
        [pos(trigger), pos(target.end)]

        Forbidding the label predicate over a window containing the label window forces it the other way:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "month": WindowConfig("trigger", "start + 30d", False, True, has={"B": "(None, 0)"}),
        ...         "target": WindowConfig("trigger", "start + 10d", False, True, label="B"),
        ...     },
        ... )
        >>> for f in degeneracies(cfg): print(f)
        always_negative: 'month: B (None, 0)' forbids B over the label window (pos(trigger), pos(target.end)]

        A hop to an event forbidden everywhere after the trigger can never land, so no row survives:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "rest": WindowConfig("trigger", None, False, True, has={"B": "(None, 0)"}),
        ...         "target": WindowConfig("trigger -> B", "start + 1d", False, True, label="A"),
        ...     },
        ... )
        >>> for f in degeneracies(cfg): print(f)
        unsatisfiable: 'rest: B (None, 0)' allows at most 0 but 'target.start is an occurrence of B'
        guarantees 1

        A lower bound on a window that is provably empty (a zero-length window with an exclusive end) is
        unsatisfiable, and a label window that is empty forces the label to ``False``:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "nothing": WindowConfig("trigger", "start + 0d", True, False, has={"B": "(1, None)"}),
        ...         "target": WindowConfig("trigger", "start + 10d", False, True, label="B"),
        ...     },
        ... )
        >>> for f in degeneracies(cfg): print(f)
        unsatisfiable: 'nothing: B (1, None)' requires an occurrence in the empty window [pos(trigger),
        pos(trigger))
        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={"target": WindowConfig("trigger", "start + 0d", True, False, label="B")},
        ... )
        >>> for f in degeneracies(cfg): print(f)
        always_negative: the label window [pos(trigger), pos(trigger)) is empty

        Counts contradict across nested windows too:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "week": WindowConfig("trigger", "start + 7d", False, True, has={"B": "(3, None)"}),
        ...         "month": WindowConfig("trigger", "start + 30d", False, True, has={"B": "(None, 2)"}),
        ...         "target": WindowConfig("month.end", "start + 10d", False, True, label="B"),
        ...     },
        ... )
        >>> for f in degeneracies(cfg): print(f)
        unsatisfiable: 'month: B (None, 2)' allows at most 2 but 'week: B (3, None)' guarantees 3

        But only for the same predicate: ACES counts a derived predicate at most once per timestamp, so three
        ``B`` codes at one time satisfy both "at least 3 ``B``" and "at most 2 ``or(A, B)``":

        >>> from aces.config import DerivedPredicateConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={**predicates, "A_or_B": DerivedPredicateConfig("or(A, B)")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "week": WindowConfig("trigger", "start + 7d", False, True, has={"B": "(3, None)"}),
        ...         "month": WindowConfig("trigger", "start + 30d", False, True, has={"A_or_B": "(None, 2)"}),
        ...         "target": WindowConfig("month.end", "start + 10d", False, True, label="B"),
        ...     },
        ... )
        >>> degeneracies(cfg)
        []
    """
    analysis = Analysis.of(task_cfg)
    findings = analysis.contradictions()
    if task_cfg.label_window is None:
        return findings
    label = task_cfg.windows[task_cfg.label_window].label
    interval = analysis.window(task_cfg.label_window)
    described = analysis.describe(interval)
    if analysis.ordering.is_empty(interval):
        findings.append(Finding("always_negative", f"the label window {described} is empty"))
    if (guarantee := analysis.guaranteed_within(label, interval)) is not None:
        reason = f"'{guarantee.source}' guarantees {label} within the label window {described}"
        findings.append(Finding("always_positive", reason))
    if (forbidden := analysis.forbidden_over(label, interval)) is not None:
        reason = f"'{forbidden.source}' forbids {label} over the label window {described}"
        findings.append(Finding("always_negative", reason))
    return findings


def is_always_positive(task_cfg: TaskExtractorConfig) -> bool:
    """Whether every label the configuration extracts is structurally ``True``.

    Examples:
        >>> is_always_positive(sample_ACES_cfg)
        False
    """
    return any(f.kind == "always_positive" for f in degeneracies(task_cfg))


def is_always_negative(task_cfg: TaskExtractorConfig) -> bool:
    """Whether every label the configuration extracts is structurally ``False``.

    Examples:
        >>> is_always_negative(sample_ACES_cfg)
        False
    """
    return any(f.kind == "always_negative" for f in degeneracies(task_cfg))


def is_unsatisfiable(task_cfg: TaskExtractorConfig) -> bool:
    """Whether the configuration's criteria can structurally never all hold.

    Examples:
        >>> is_unsatisfiable(sample_ACES_cfg)
        False
    """
    return any(f.kind == "unsatisfiable" for f in degeneracies(task_cfg))
