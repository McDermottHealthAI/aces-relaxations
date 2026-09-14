"""Detect structurally degenerate task configurations: forced labels and unsatisfiable criteria.

A configuration is *degenerate* when its windows and criteria alone, before any data is seen, fix the
outcome: every extracted label is ``True`` (the criteria guarantee an occurrence of the label predicate
inside the label window), every extracted label is ``False`` (the criteria forbid one), or no row can ever
satisfy the criteria (a guaranteed occurrence falls inside a window that forbids it, or lower-bound counts
inside a window exceed its upper bound).

The analysis is *sound but incomplete*: whatever it reports is truly forced by the configuration's
structure, but it only sees orderings it can prove symbolically. Positions in the window tree are expressed
as offsets from *anchors* (the trigger, each event-bounded endpoint, and the record's start and end), each
event anchor knowing only the position it was searched from. Two positions compare when a chain of such
bounds links them. Facts are *count bounds* on a predicate over an interval:

- every hop to the next (or prior) occurrence of ``P`` guarantees a ``P`` exactly at the landing point, and
  forbids ``P`` strictly between the search's start and the landing point;
- the trigger predicate occurs at the trigger, and every event endpoint is an occurrence of ``_ANY_EVENT``;
- an ``(k, None)`` criterion guarantees at least ``k`` occurrences over its window, a ``(None, 0)`` one
  forbids any, and other bounds are kept as they are.

Lower bounds propagate to enclosing intervals and upper bounds to enclosed ones, so a lower bound of ``k``
on an interval provably inside an interval with an upper bound of ``j < k`` (for an implied predicate) is
a contradiction, as are lower bounds on provably disjoint sub-intervals whose sum exceeds the upper bound.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

from aces.config import TaskExtractorConfig
from aces.types import TemporalWindowBounds
from bigtree import Node

from .future_only import _implies
from .guarantees import anchor_counts
from .predicates import ANY_EVENT
from .tree import WindowNode, resolve_node

RECORD_START = "_RECORD_START"
RECORD_END = "_RECORD_END"
TRIGGER, BOTTOM, TOP = 0, -1, -2

Kind = Literal["always_positive", "always_negative", "unsatisfiable"]


@dataclass(frozen=True)
class Position:
    """A time expressed as a fixed offset from an anchor.

    Anchor ``0`` is the trigger, ``-1`` the record's start, ``-2`` its end; others are event endpoints.
    """

    anchor: int
    offset: timedelta = timedelta(0)

    def shifted(self, delta: timedelta) -> "Position":
        """The position ``delta`` later (earlier if negative)."""
        return Position(self.anchor, self.offset + delta)


@dataclass(frozen=True)
class Anchor:
    """An event endpoint: the next (or prior) occurrence of a predicate searched from a known position.

    Attributes:
        predicate: The predicate found.
        lower: A position the anchor is guaranteed not to precede (the search start of a forward search).
        upper: A position the anchor is guaranteed not to follow (the search start of a backward search).
        strict: Whether the search excluded an occurrence exactly at its start, so that the bound is strict.
    """

    predicate: str | None
    lower: Position | None = None
    upper: Position | None = None
    strict: bool = False


@dataclass(frozen=True)
class Interval:
    """A time interval between two positions, with inclusivity."""

    lo: Position
    hi: Position
    lo_inclusive: bool = True
    hi_inclusive: bool = True

    @classmethod
    def point(cls, pos: Position) -> "Interval":
        """The interval containing exactly one time."""
        return cls(pos, pos, True, True)


@dataclass(frozen=True)
class CountBound:
    """``at_least <= count of predicate over interval <= at_most`` (``None``: unbounded), and why."""

    predicate: str
    interval: Interval
    at_least: int = 0
    at_most: int | None = None
    source: str = ""


@dataclass(frozen=True)
class Finding:
    """One way a configuration is degenerate."""

    kind: Kind
    reason: str

    def __str__(self) -> str:
        """Renders the finding."""
        return f"{self.kind}: {self.reason}"


@dataclass
class Analysis:
    """The positions, anchors, and count bounds of a configuration, and how to compare them.

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
        >>> analysis.leq(pt, gap_end), analysis.leq(gap_end, pt), analysis.leq(gap_end, target_end)
        (True, False, True)
        >>> analysis.leq(pt, target_end, strict=True), analysis.leq(target_end, pt)
        (True, False)

    Positions render by the node they are an offset from, or as a record boundary:

        >>> later, earlier = pt.shifted(timedelta(days=2)), pt.shifted(-timedelta(hours=6))
        >>> for p in (Position(BOTTOM), Position(TOP), later, earlier):
        ...     print(analysis.describe(p))
        pos(record_start)
        pos(record_end)
        pos(trigger) + 3 days, 0:00:00
        pos(trigger) + 18:00:00
    """

    task_cfg: TaskExtractorConfig
    implies: Callable[[str, str], bool]
    anchors: list[Anchor] = field(default_factory=list)
    positions: dict[str, Position] = field(default_factory=dict)
    bounds: list[CountBound] = field(default_factory=list)

    def describe(self, pos_or_interval: "Position | Interval") -> str:
        """Renders a position (by the tree node it is an offset from) or an interval."""
        if isinstance(pos_or_interval, Interval):
            i = pos_or_interval
            left, right = "[" if i.lo_inclusive else "(", "]" if i.hi_inclusive else ")"
            return f"{left}{self.describe(i.lo)}, {self.describe(i.hi)}{right}"
        pos = pos_or_interval
        named = next((n for n, p in self.positions.items() if p == pos), None)
        if named is not None:
            return f"pos({named})"
        base = {BOTTOM: "record_start", TOP: "record_end"}.get(pos.anchor)
        if base is None:
            base = next(n for n, p in self.positions.items() if p == Position(pos.anchor))
        if not pos.offset:
            return f"pos({base})"
        sign = "+" if pos.offset > timedelta(0) else "-"
        return f"pos({base}) {sign} {abs(pos.offset)}"

    @classmethod
    def of(cls, task_cfg: TaskExtractorConfig) -> "Analysis":
        """Analyzes a configuration's window tree."""
        implies = _implies(task_cfg)
        self = cls(task_cfg, lambda g, p: p == ANY_EVENT or implies(g, p))
        self.anchors = [Anchor(task_cfg.trigger.predicate)]
        root = task_cfg.window_tree
        self.positions[root.node_name] = Position(TRIGGER)
        self._event_at(Position(TRIGGER), task_cfg.trigger.predicate, "the trigger")
        for node in root.descendants:
            self._place(node)
        for name, window in task_cfg.windows.items():
            interval = self.window(name)
            for predicate, (lo, hi) in window.has.items():
                self.bounds.append(
                    CountBound(predicate, interval, lo or 0, hi, f"{name}: {predicate} ({lo}, {hi})")
                )
        return self

    # -- Construction ------------------------------------------------------------------------------------

    def _event_at(self, pos: Position, predicate: str, what: str) -> None:
        self.bounds.append(
            CountBound(predicate, Interval.point(pos), 1, None, f"{what} is an occurrence of {predicate}")
        )
        self.bounds.append(CountBound(ANY_EVENT, Interval.point(pos), 1, None, f"{what} is an event"))

    def _place(self, node: Node) -> None:
        parent = self.positions[node.parent.node_name]
        bound = node.endpoint_expr
        if isinstance(bound, TemporalWindowBounds):
            self.positions[node.node_name] = parent.shifted(bound.window_size)
            return
        window_name, side = node.node_name.rsplit(".", 1)
        window = self.task_cfg.windows[window_name]
        backward = bound.end_event.startswith("-")
        event = bound.end_event.lstrip("-")
        if event in (RECORD_START, RECORD_END):
            self.positions[node.node_name] = Position(BOTTOM if event == RECORD_START else TOP)
            return
        strict = not anchor_counts(window.start_inclusive, window.end_inclusive, side)
        anchor = (
            Anchor(event, upper=parent, strict=strict)
            if backward
            else Anchor(event, lower=parent, strict=strict)
        )
        if anchor in self.anchors:
            # The same search from the same position lands on the same time.
            self.positions[node.node_name] = Position(self.anchors.index(anchor))
            return
        self.anchors.append(anchor)
        pos = Position(len(self.anchors) - 1)
        self.positions[node.node_name] = pos
        self._event_at(pos, event, node.node_name)
        # The search found the *next* / *prior* occurrence, so none lies strictly in between (nor at the
        # search start, when an occurrence there would have been found).
        direction = "prior" if backward else "next"
        gap = (
            Interval(pos, parent, False, not strict) if backward else Interval(parent, pos, not strict, False)
        )
        parent_name = node.parent.node_name
        self.bounds.append(
            CountBound(
                event,
                gap,
                0,
                0,
                f"{node.node_name} is the {direction} {event} {'before' if backward else 'after'} "
                f"{parent_name}",
            )
        )

    def window(self, name: str) -> Interval:
        """A window's interval."""
        window = self.task_cfg.windows[name]
        start = self.positions[resolve_node(self.task_cfg, root_node=WindowNode(name, "start")).node_name]
        end = self.positions[resolve_node(self.task_cfg, root_node=WindowNode(name, "end")).node_name]
        return Interval(start, end, window.start_inclusive, window.end_inclusive)

    # -- Ordering ----------------------------------------------------------------------------------------

    def _is_event(self, pos: Position) -> bool:
        return pos.anchor != BOTTOM and pos.anchor != TOP and pos.offset == timedelta(0)

    def _closure(self, pos: Position, upward: bool) -> list[tuple[Position, bool]]:
        """Positions provably at or after (``upward``) / at or before ``pos``, with whether strictly so."""
        out: list[tuple[Position, bool]] = [(pos, False)]
        seen = {pos}
        frontier = [(pos, False)]
        while frontier:
            current, strict = frontier.pop()
            if current.anchor in (BOTTOM, TOP):
                continue
            anchor = self.anchors[current.anchor]
            bound = anchor.upper if upward else anchor.lower
            if bound is not None:
                nxt = bound.shifted(current.offset)
                if nxt not in seen:
                    seen.add(nxt)
                    out.append((nxt, strict or anchor.strict))
                    frontier.append((nxt, strict or anchor.strict))
            if self._is_event(current):
                extreme = Position(TOP if upward else BOTTOM)
                if extreme not in seen:
                    seen.add(extreme)
                    out.append((extreme, strict))
        return out

    def leq(self, a: Position, b: Position, strict: bool = False) -> bool:
        """Whether ``a`` is provably at or before ``b`` (strictly before, if ``strict``)."""
        for upper, s1 in self._closure(a, upward=True):
            for lower, s2 in self._closure(b, upward=False):
                if upper.anchor != lower.anchor:
                    continue
                if upper.offset < lower.offset or (upper.offset == lower.offset and (s1 or s2 or not strict)):
                    return True
        return False

    def contains(self, outer: Interval, inner: Interval) -> bool:
        """Whether ``inner`` provably lies within ``outer``."""
        lo_ok = self.leq(outer.lo, inner.lo, strict=inner.lo_inclusive and not outer.lo_inclusive)
        hi_ok = self.leq(inner.hi, outer.hi, strict=inner.hi_inclusive and not outer.hi_inclusive)
        return lo_ok and hi_ok

    def is_empty(self, interval: Interval) -> bool:
        """Whether an interval provably contains no time at all."""
        both = interval.lo_inclusive and interval.hi_inclusive
        return self.leq(interval.hi, interval.lo, strict=both)

    def disjoint(self, a: Interval, b: Interval) -> bool:
        """Whether two intervals provably share no time."""
        return self.leq(a.hi, b.lo, strict=a.hi_inclusive and b.lo_inclusive) or self.leq(
            b.hi, a.lo, strict=b.hi_inclusive and a.lo_inclusive
        )

    # -- Findings ----------------------------------------------------------------------------------------

    def guaranteed_within(self, predicate: str, interval: Interval) -> CountBound | None:
        """A bound guaranteeing an occurrence of (something implying) ``predicate`` inside ``interval``."""
        for b in self.bounds:
            if (
                b.at_least >= 1
                and self.implies(b.predicate, predicate)
                and self.contains(interval, b.interval)
            ):
                return b
        return None

    def forbidden_over(self, predicate: str, interval: Interval) -> CountBound | None:
        """A bound forbidding (something implied by) ``predicate`` over all of ``interval``."""
        for b in self.bounds:
            if (
                b.at_most == 0
                and self.implies(predicate, b.predicate)
                and self.contains(b.interval, interval)
            ):
                return b
        return None

    def contradictions(self) -> list[Finding]:
        """Upper bounds exceeded by lower bounds provably inside them, and lower bounds on empty intervals."""
        out = []
        for b in self.bounds:
            if b.at_least >= 1 and self.is_empty(b.interval):
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
                lo
                for lo in self.bounds
                if lo.at_least >= 1
                and self.implies(lo.predicate, hi.predicate)
                and self.contains(hi.interval, lo.interval)
            ]
            chosen: list[CountBound] = []
            for lo in sorted(inside, key=lambda x: -x.at_least):
                if all(self.disjoint(lo.interval, c.interval) for c in chosen):
                    chosen.append(lo)
            total = sum(c.at_least for c in chosen)
            if total > hi.at_most:
                sources = "; ".join(f"'{c.source}'" for c in chosen)
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

        >>> from aces.config import EventConfig, PlainPredicateConfig, WindowConfig
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
    """
    analysis = Analysis.of(task_cfg)
    findings = analysis.contradictions()
    if task_cfg.label_window is not None:
        label = task_cfg.windows[task_cfg.label_window].label
        interval = analysis.window(task_cfg.label_window)
        described = analysis.describe(interval)
        if analysis.is_empty(interval):
            findings.append(Finding("always_negative", f"the label window {described} is empty"))
        if (guarantee := analysis.guaranteed_within(label, interval)) is not None:
            findings.append(
                Finding(
                    "always_positive",
                    f"'{guarantee.source}' guarantees {label} within the label window {described}",
                )
            )
        if (forbidden := analysis.forbidden_over(label, interval)) is not None:
            findings.append(
                Finding(
                    "always_negative",
                    f"'{forbidden.source}' forbids {label} over the label window {described}",
                )
            )
    return findings


def _has(task_cfg: TaskExtractorConfig, kind: Kind) -> bool:
    return any(f.kind == kind for f in degeneracies(task_cfg))


def is_always_positive(task_cfg: TaskExtractorConfig) -> bool:
    """Whether every label the configuration extracts is structurally ``True``.

    Examples:
        >>> is_always_positive(sample_ACES_cfg)
        False
    """
    return _has(task_cfg, "always_positive")


def is_always_negative(task_cfg: TaskExtractorConfig) -> bool:
    """Whether every label the configuration extracts is structurally ``False``.

    Examples:
        >>> is_always_negative(sample_ACES_cfg)
        False
    """
    return _has(task_cfg, "always_negative")


def is_unsatisfiable(task_cfg: TaskExtractorConfig) -> bool:
    """Whether the configuration's criteria can structurally never all hold.

    Examples:
        >>> is_unsatisfiable(sample_ACES_cfg)
        False
    """
    return _has(task_cfg, "unsatisfiable")
