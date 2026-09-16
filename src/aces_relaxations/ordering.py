"""Symbolic positions in time, and what can be proven about their order.

After an event-bounded hop, the time a window endpoint lands on depends on the data, so the analyses in this
package cannot compare endpoints with plain arithmetic. Instead, every position is a fixed offset from an
*anchor*: the root of the analysis (the trigger or the prediction time), one of the record's boundaries, or
the time an event search resolves to. An event anchor knows only what its search guarantees: a forward search
lands at or after (strictly after, if it skips an event at its start) the position it searched from, and a
backward search lands at or before it. Two positions are provably ordered when a chain of such facts links
them.

On top of positions sit *intervals* and *count bounds* (``at_least <= count(P over interval) <= at_most``),
which is the vocabulary of the future-only check, the degeneracy detector, and normalization's redundancy
pass.
"""

from dataclasses import dataclass, field
from datetime import timedelta

ROOT = 0
RECORD_START_ANCHOR = -1
RECORD_END_ANCHOR = -2


@dataclass(frozen=True)
class Position:
    """A time expressed as a fixed offset from an anchor.

    Attributes:
        anchor: The anchor's index: ``ROOT`` (0), an event anchor (positive), or ``RECORD_START_ANCHOR`` /
            ``RECORD_END_ANCHOR``.
        offset: The signed offset from the anchor's time.
    """

    anchor: int
    offset: timedelta = timedelta(0)

    def shifted(self, delta: timedelta) -> "Position":
        """The position ``delta`` later (earlier, if negative)."""
        return Position(self.anchor, self.offset + delta)


@dataclass(frozen=True)
class Anchor:
    """A data-dependent time, and the positions it is guaranteed not to precede or follow.

    Attributes:
        predicate: The predicate whose occurrence the anchor is (``None`` for the root).
        lower: A position the anchor is guaranteed not to precede (e.g. where a forward search started).
        upper: A position the anchor is guaranteed not to follow (e.g. where a backward search started).
        strict: Whether those guarantees are strict (the search skipped an occurrence at its start).
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


@dataclass
class Ordering:
    """A set of anchors, and the order relations between positions on them that they prove.

    Examples:
        >>> ordering = Ordering()
        >>> root = Position(ROOT)
        >>> week = root.shifted(timedelta(days=7))
        >>> discharge = ordering.add(Anchor("discharge", lower=week, strict=True))
        >>> ordering.leq(root, week), ordering.leq(week, root)
        (True, False)

    An event found by a forward search follows its search start, strictly so if the search skipped an event
    there, but nothing bounds it from above:

        >>> ordering.leq(week, discharge, strict=True), ordering.leq(discharge, week)
        (True, False)
        >>> ordering.leq(discharge.shifted(-timedelta(days=7)), root)
        False
        >>> ordering.leq(root, discharge.shifted(-timedelta(days=7)))
        True

    Every event lies within the record:

        >>> ordering.leq(Position(RECORD_START_ANCHOR), discharge), ordering.leq(discharge, root)
        (True, False)

    The same search from the same position is the same time:

        >>> ordering.add(Anchor("discharge", lower=week, strict=True)) == discharge
        True
    """

    anchors: list[Anchor] = field(default_factory=lambda: [Anchor(None)])

    def add(self, anchor: Anchor) -> Position:
        """The position of an anchor, adding it unless an identical one exists."""
        if anchor not in self.anchors:
            self.anchors.append(anchor)
        return Position(self.anchors.index(anchor))

    def _is_event(self, pos: Position) -> bool:
        return pos.anchor >= 0 and pos.offset == timedelta(0)

    def _closure(self, pos: Position, upward: bool) -> list[tuple[Position, bool]]:
        """Positions provably at or after (``upward``) / at or before ``pos``, with whether strictly so."""
        out: list[tuple[Position, bool]] = [(pos, False)]
        seen = {pos}
        frontier = [(pos, False)]
        while frontier:
            current, strict = frontier.pop()
            if current.anchor < 0:
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
                extreme = Position(RECORD_END_ANCHOR if upward else RECORD_START_ANCHOR)
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


def transferable_count(count: int, from_predicate: str, to_predicate: str) -> int:
    """How many occurrences of ``to_predicate`` ``count`` occurrences of an implying ``from_predicate`` prove.

    ACES counts every matching code of a plain predicate at a timestamp, but a derived predicate (and
    ``_ANY_EVENT``) at most once, so beyond a single occurrence counts only carry over to the same predicate.

    Examples:
        >>> transferable_count(3, "A", "A"), transferable_count(3, "A", "A_or_B")
        (3, 1)
    """
    return count if from_predicate == to_predicate else min(count, 1)
