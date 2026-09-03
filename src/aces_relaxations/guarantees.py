"""Symbolic bookkeeping of what is guaranteed along a path of windows after the prediction time.

After an event-bounded hop the position on a path is data-dependent, so "how far after the prediction time
are we?" and "has predicate ``P`` certainly occurred by now?" cannot be answered with plain arithmetic. This
module tracks positions relative to *anchors* (the prediction time, and the time each event-bounded hop
resolves to), each with a lower bound expressed as a position on an earlier anchor, and tracks *guarantees*:
windows within which a predicate certainly occurs at least once, because the path hopped to its next
occurrence or because a criterion on the path requires it.

That is enough to decide the two questions a look-back window raises:

- ``start: end - d``: is the end guaranteed to be at least ``d`` after the prediction time?
- ``start: end <- P``: is ``P`` guaranteed to occur between the prediction time and the end, so that the
  most recent ``P`` before the end is not before the prediction time?
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta

Constraint = tuple[int | None, int | None]


@dataclass(frozen=True)
class Position:
    """A time expressed as an offset from an anchor.

    Attributes:
        anchor: The anchor index (0 is the prediction time).
        offset: The signed offset from the anchor's time.
    """

    anchor: int
    offset: timedelta = timedelta(0)

    def shifted(self, delta: timedelta) -> "Position":
        """The position ``delta`` later (or earlier, if negative)."""
        return Position(self.anchor, self.offset + delta)


@dataclass(frozen=True)
class Anchor:
    """A data-dependent reference time.

    Attributes:
        kind: ``"prediction_time"``, ``"next"`` (the next occurrence of an event, at or after ``lower``), or
            ``"prior"`` (the most recent occurrence of an event, guaranteed to be at or after ``lower``).
        lower: A position on an earlier anchor that this anchor's time is guaranteed not to precede.
        strict: Whether the anchor's time is guaranteed to be strictly after ``lower`` (a hop that excludes
            the event at its starting point).
    """

    kind: str
    lower: Position | None = None
    strict: bool = False


@dataclass(frozen=True)
class Guarantee:
    """A window within which a predicate certainly occurs at least once."""

    predicate: str
    start: Position
    end: Position


@dataclass
class PathState:
    """The position along a path after the prediction time, and everything guaranteed so far.

    Attributes:
        implies: ``implies(g, p)`` says whether an occurrence of predicate ``g`` is also an occurrence of
            predicate ``p`` (e.g. ``death`` implies ``or(discharge, death)``).
        anchors: The anchors created so far.
        guarantees: The predicate-occurrence guarantees collected so far.
        pos: The current position.

    Examples:
        >>> state = PathState(lambda g, p: g == p)
        >>> state.lower_bound(state.pos)
        datetime.timedelta(0)

    Moving forward a week, then hopping to the next discharge, leaves a lower bound of a week and a
    guarantee that a discharge occurred at the hop:

        >>> state.advance_temporal(timedelta(days=7))
        >>> state.advance_event("discharge")
        >>> state.lower_bound(state.pos)
        datetime.timedelta(days=7)
        >>> state.can_look_back(timedelta(days=7)), state.can_look_back(timedelta(days=8))
        (True, False)
        >>> print(state.guarantee_for("discharge", anchor_counts=True))
        Guarantee(predicate='discharge', start=Position(anchor=1, offset=datetime.timedelta(0)),
                  end=Position(anchor=1, offset=datetime.timedelta(0)))

    Looking back to that discharge from a day later is guaranteed, but looking *past* it (an exclusive
    look-back from the discharge itself) is not, and neither is looking back for an event never guaranteed:

        >>> state.advance_temporal(timedelta(days=1))
        >>> state.guarantee_for("discharge", anchor_counts=False) is not None
        True
        >>> state.look_back_temporal(timedelta(days=1))
        >>> state.guarantee_for("discharge", anchor_counts=False) is None
        True
        >>> state.guarantee_for("admission", anchor_counts=True) is None
        True
    """

    implies: Callable[[str, str], bool]
    anchors: list[Anchor] = field(default_factory=lambda: [Anchor("prediction_time")])
    guarantees: list[Guarantee] = field(default_factory=list)
    pos: Position = field(default_factory=lambda: Position(0))

    def copy(self) -> "PathState":
        """An independent copy (anchors and guarantees are shared prefixes, so copying lists suffices)."""
        return replace(self, anchors=list(self.anchors), guarantees=list(self.guarantees))

    # ------------------------------------------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------------------------------------------

    def _lower(self, pos: Position) -> Position | None:
        """A position on an earlier anchor that ``pos`` is guaranteed not to precede, if any."""
        lower = self.anchors[pos.anchor].lower
        return None if lower is None else lower.shifted(pos.offset)

    def lower_bound(self, pos: Position) -> timedelta:
        """A lower bound on how long after the prediction time ``pos`` is."""
        while pos.anchor != 0:
            pos = self._lower(pos)
        return pos.offset

    def not_after(self, earlier: Position, later: Position, strict: bool = False) -> bool:
        """Whether ``earlier`` is guaranteed not to be after ``later`` (strictly before, if ``strict``)."""
        while later.anchor != earlier.anchor:
            if later.anchor < earlier.anchor:
                return False
            if self.anchors[later.anchor].strict:
                strict = False
            later = self._lower(later)
        return earlier.offset < later.offset if strict else earlier.offset <= later.offset

    # ------------------------------------------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------------------------------------------

    def advance_temporal(self, size: timedelta) -> None:
        """Moves forward by a fixed offset."""
        self.pos = self.pos.shifted(size)

    def advance_event(self, predicate: str, strict: bool = False) -> None:
        """Hops to the next occurrence of ``predicate``, which is thereby guaranteed to occur there.

        Args:
            predicate: The predicate hopped to.
            strict: Whether the hop excludes an occurrence exactly at the current position (in ACES, a
                ``start -> P`` search includes the start event exactly when the found endpoint's
                inclusivity flag is set), so that the new anchor is strictly later.
        """
        self.anchors.append(Anchor("next", self.pos, strict=strict))
        self.pos = Position(len(self.anchors) - 1)
        self.guarantees.append(Guarantee(predicate, self.pos, self.pos))

    def add_criteria(self, has: Mapping[str, Constraint], start: Position, end: Position) -> None:
        """Records the guarantees implied by a window's criteria: any ``(k, ...)`` with ``k >= 1``."""
        for predicate, (lo, _) in has.items():
            if lo is not None and lo >= 1:
                self.guarantees.append(Guarantee(predicate, start, end))

    def can_look_back(self, days: timedelta) -> bool:
        """Whether the position ``days`` earlier is guaranteed not to precede the prediction time."""
        return self.lower_bound(self.pos.shifted(-days)) >= timedelta(0)

    def look_back_temporal(self, days: timedelta) -> None:
        """Moves back by a fixed offset (the caller has checked ``can_look_back``)."""
        self.pos = self.pos.shifted(-days)

    def guarantee_for(self, predicate: str, anchor_counts: bool) -> Guarantee | None:
        """A guarantee that ``predicate`` occurs between the prediction time and the current position.

        Args:
            predicate: The predicate a look-back searches for.
            anchor_counts: Whether an occurrence exactly at the current position counts. In ACES a
                ``start: end <- P`` search includes the end event itself exactly when the window's
                ``start_inclusive`` flag is set; otherwise the guaranteeing window must end strictly before.
        """
        if predicate == "_RECORD_START":
            # The record start is a fixed point, so inclusivity is irrelevant: it is guaranteed only when the
            # prediction time is itself the record start.
            return next((g for g in self.guarantees if g.predicate == predicate), None)
        if predicate == "_RECORD_END":
            return None
        for g in self.guarantees:
            if g.predicate.startswith("_RECORD_"):
                continue
            if self.implies(g.predicate, predicate) and self.not_after(
                g.end, self.pos, strict=not anchor_counts
            ):
                return g
        return None

    def guarantee_at_prediction_time(self, predicate: str) -> None:
        """Records that ``predicate`` occurs exactly at the prediction time (e.g. the trigger event)."""
        self.guarantees.append(Guarantee(predicate, Position(0), Position(0)))

    def look_back_event(self, guarantee: Guarantee) -> None:
        """Moves to the most recent occurrence of a guaranteed predicate before the current position."""
        self.anchors.append(Anchor("prior", guarantee.start))
        self.pos = Position(len(self.anchors) - 1)


def anchor_counts(window_start_inclusive: bool, window_end_inclusive: bool, side: str) -> bool:
    """Whether an event search that finds a window's ``side`` endpoint counts an event at its starting point.

    In ACES, ``start -> P`` includes the start event itself exactly when ``end_inclusive`` is set (the found
    endpoint is the end), and ``end <- P`` includes the end event exactly when ``start_inclusive`` is set (the
    found endpoint is the start).

    Examples:
        >>> anchor_counts(True, False, "start"), anchor_counts(True, False, "end")
        (True, False)
    """
    return window_start_inclusive if side == "start" else window_end_inclusive
