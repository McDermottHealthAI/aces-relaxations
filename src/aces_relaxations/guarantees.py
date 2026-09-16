"""What is guaranteed along a path of windows after the prediction time.

Walking a path of the window tree from the prediction time, a look-back window raises one of two questions:

- ``start: end - d``: is the end guaranteed to be at least ``d`` after the prediction time?
- ``start: end <- P``: is ``P`` guaranteed to occur between the prediction time and the end, so that the most
  recent ``P`` before the end is not before the prediction time?

``PathState`` answers them by tracking the current position symbolically (see ``aces_relaxations.ordering``)
and collecting *guarantees*: intervals within which a predicate certainly occurs, because the path hopped to
its next occurrence or because a criterion on the path requires it.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta

from aces.config import TaskExtractorConfig
from aces.types import END_OF_RECORD_KEY, START_OF_RECORD_KEY, ToEventWindowBounds
from bigtree import Node

from .ordering import ROOT, Anchor, Interval, Ordering, Position
from .predicates import implication

Constraint = tuple[int | None, int | None]


@dataclass(frozen=True)
class Guarantee:
    """An interval within which a predicate certainly occurs at least once."""

    predicate: str
    interval: Interval


@dataclass
class PathState:
    """The position along a path after the prediction time, and everything guaranteed so far.

    Attributes:
        implies: ``implies(g, p)`` says whether an occurrence of predicate ``g`` is also an occurrence of
            predicate ``p`` (see [`implication`][aces_relaxations.predicates.implication]).
        ordering: The anchors created so far (anchor 0 is the prediction time).
        guarantees: The predicate-occurrence guarantees collected so far.
        pos: The current position.

    Examples:
        Moving forward a week, then hopping to the next discharge, leaves the position at least a week after
        the prediction time and guarantees a discharge at the hop:

        >>> state = PathState(lambda g, p: g == p)
        >>> state.advance_temporal(timedelta(days=7))
        >>> state.advance_event("discharge")
        >>> state.can_look_back(timedelta(days=7)), state.can_look_back(timedelta(days=8))
        (True, False)
        >>> state.guarantee_for("discharge", anchor_counts=True).interval.lo == state.pos
        True

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
    ordering: Ordering = field(default_factory=Ordering)
    guarantees: list[Guarantee] = field(default_factory=list)
    pos: Position = field(default_factory=lambda: Position(ROOT))

    @classmethod
    def at(cls, task_cfg: TaskExtractorConfig, node: Node) -> "PathState":
        """A path state at a node of the window tree, taken as the prediction time.

        If the node is the trigger, or is itself the occurrence of an event (the endpoint of a ``-> P`` or
        ``<- P`` bound), that event is known to occur exactly there, which lets an inclusive look-back to it
        resolve to the prediction time itself. Likewise, if the node is the start of the record, windows
        bounded by the record start begin there.

        Examples:
            >>> [g.predicate for g in PathState.at(sample_ACES_cfg, sample_ACES_cfg.window_tree).guarantees]
            ['icu_admission']
            >>> PathState.at(sample_ACES_cfg, sample_ACES_cfg.window_nodes["input.end"]).guarantees
            []
        """
        state = cls(implication(task_cfg))
        bound = getattr(node, "endpoint_expr", None)
        if node.node_name == "trigger":
            state.guarantee_at_prediction_time(task_cfg.trigger.predicate)
        elif isinstance(bound, ToEventWindowBounds) and bound.end_event.lstrip("-") != END_OF_RECORD_KEY:
            state.guarantee_at_prediction_time(bound.end_event.lstrip("-"))
        return state

    def advance_temporal(self, size: timedelta) -> None:
        """Moves forward by a fixed offset."""
        self.pos = self.pos.shifted(size)

    def advance_event(self, predicate: str, strict: bool = False) -> None:
        """Hops to the next occurrence of ``predicate``, which is thereby guaranteed to occur there.

        Args:
            predicate: The predicate hopped to.
            strict: Whether the hop excludes an occurrence exactly at the current position, so that the new
                anchor is strictly later.
        """
        self.pos = self._new_anchor(Anchor(predicate, lower=self.pos, strict=strict))
        self.guarantees.append(Guarantee(predicate, Interval.point(self.pos)))

    def add_criteria(self, has: Mapping[str, Constraint], start: Position, end: Position) -> None:
        """Records the guarantees implied by a window's criteria: any ``(k, ...)`` with ``k >= 1``."""
        for predicate, (lo, _) in has.items():
            if lo is not None and lo >= 1:
                self.guarantees.append(Guarantee(predicate, Interval(start, end)))

    def can_look_back(self, delta: timedelta) -> bool:
        """Whether the position ``delta`` earlier is guaranteed not to precede the prediction time."""
        return self.ordering.leq(Position(ROOT), self.pos.shifted(-delta))

    def look_back_temporal(self, delta: timedelta) -> None:
        """Moves back by a fixed offset (the caller has checked ``can_look_back``)."""
        self.pos = self.pos.shifted(-delta)

    def guarantee_for(self, predicate: str, anchor_counts: bool) -> Guarantee | None:
        """A guarantee that ``predicate`` occurs between the prediction time and the current position.

        Args:
            predicate: The predicate a look-back searches for.
            anchor_counts: Whether an occurrence exactly at the current position counts (see
                [`search_includes_anchor`][aces_relaxations.tree.search_includes_anchor]); otherwise the
                guaranteeing interval must end strictly before it.

        Examples:
            The record boundaries are positions rather than events: the record's end is never guaranteed,
            and guarantees about the record's start do not stand in for events:

            >>> state = PathState(lambda g, p: g == p and not g.startswith("_RECORD_"))
            >>> state.guarantee_at_prediction_time("_RECORD_START")
            >>> state.guarantee_for("_RECORD_START", True) is not None
            True
            >>> print(state.guarantee_for("_RECORD_END", True), state.guarantee_for("admission", True))
            None None
        """
        if predicate == START_OF_RECORD_KEY:
            # The record start is a fixed point, so inclusivity is irrelevant: it is guaranteed only when the
            # prediction time is itself the record start.
            return next((g for g in self.guarantees if g.predicate == predicate), None)
        for g in self.guarantees:
            if self.implies(g.predicate, predicate) and self.ordering.leq(
                g.interval.hi, self.pos, strict=not anchor_counts
            ):
                return g
        return None

    def guarantee_at_prediction_time(self, predicate: str) -> None:
        """Records that ``predicate`` occurs exactly at the prediction time (e.g. the trigger event)."""
        self.guarantees.append(Guarantee(predicate, Interval.point(Position(ROOT))))

    def look_back_event(self, guarantee: Guarantee) -> None:
        """Moves to the most recent occurrence of a guaranteed predicate before the current position."""
        self.pos = self._new_anchor(Anchor(guarantee.predicate, lower=guarantee.interval.lo))

    def _new_anchor(self, anchor: Anchor) -> Position:
        # Along a single path, anchors are never shared: a look-back and a forward search from the same
        # position are different times even when their recorded bounds coincide.
        self.ordering.anchors.append(anchor)
        return Position(len(self.ordering.anchors) - 1)
