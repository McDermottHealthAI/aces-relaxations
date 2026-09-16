"""The building blocks of the normalized form: edges (single windows) and branches (chains of them).

A normalized task configuration (see [`normalize`][aces_relaxations.normalization.normalize]) is a set of
*branches* hanging off the prediction time. Each branch is a chain of *edges*, one per window, in which every
window starts where the previous one exits, and in which criteria, if any, sit only on the last window.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta

from aces.types import END_OF_RECORD_KEY, TemporalWindowBounds, ToEventWindowBounds

Constraint = tuple[int | None, int | None]
Bound = TemporalWindowBounds | ToEventWindowBounds


def bound_key(bound: Bound) -> tuple:
    """A hashable summary of a bound, sufficient to decide whether two windows coincide.

    The ``offset`` field of ACES bounds is deliberately ignored: ACES uses it as scratch state during
    extraction (and accumulates into it on every query), so it carries no meaning about the window.

    Examples:
        >>> bound_key(TemporalWindowBounds(False, timedelta(days=1), True, timedelta(hours=3)))
        ('time', False, True, datetime.timedelta(days=1))
        >>> bound_key(ToEventWindowBounds(True, "death", True))
        ('event', True, True, 'death')
    """
    if isinstance(bound, TemporalWindowBounds):
        return ("time", bound.left_inclusive, bound.right_inclusive, bound.window_size)
    return ("event", bound.left_inclusive, bound.right_inclusive, bound.end_event)


def render_timedelta(delta: timedelta) -> str:
    """Renders a positive timedelta in a form ACES's boundary parser accepts.

    Examples:
        >>> render_timedelta(timedelta(days=7))
        '7 days'
        >>> render_timedelta(timedelta(hours=36))
        '36 hours'
        >>> render_timedelta(timedelta(minutes=90))
        '90 minutes'
        >>> render_timedelta(timedelta(seconds=61))
        '61 seconds'
        >>> render_timedelta(timedelta(hours=1))
        '1 hour'
    """
    seconds = round(delta.total_seconds())
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)):
        if seconds % size == 0:
            n = seconds // size
            return f"{n} {unit}{'' if n == 1 else 's'}"
    raise ValueError(f"Cannot render {delta}.")  # pragma: no cover


def _constraints_str(constraints: Mapping[str, Constraint]) -> str:
    return ", ".join(f"{p}: ({lo}, {hi})" for p, (lo, hi) in constraints.items())


@dataclass(frozen=True, eq=False)
class Edge:
    """One window of a normalized branch: how its end is reached from its start, and what it asserts.

    Attributes:
        name: The window name.
        bound: How the window's far endpoint is reached from the endpoint the chain arrives at (a temporal
            offset or an event search, forward or backward), including the inclusivity of both endpoints.
        constraints: The window's criteria, as sorted ``(predicate, (min, max))`` pairs.
        label: The label predicate, if this is the label window.

    Examples:
        >>> edge = Edge("gap", TemporalWindowBounds(False, timedelta(days=1), True), (("death", (None, 0)),))
        >>> edge.is_temporal, edge.is_constrained, edge.days
        (True, True, 1.0)
        >>> print(edge)
        gap: start + 1 day (death: (None, 0))
        >>> print(edge.unconstrained())
        gap: start + 1 day
        >>> print(Edge("target", ToEventWindowBounds(False, "discharge", True), label="death"))
        target: start -> discharge; label=death
        >>> print(Edge("tail", ToEventWindowBounds(True, "_RECORD_END", True)))
        tail: start -> end of record
        >>> back = Edge("last_day", TemporalWindowBounds(False, timedelta(days=-1), True), label="death")
        >>> back.is_end_rooted, back.exit_side, back.days
        (True, 'start', -1.0)
        >>> print(back)
        last_day: end - 1 day; label=death
        >>> print(Edge("since", ToEventWindowBounds(False, "-admission", True)))
        since: end <- admission

    Edges are equal when their names, bounds (ignoring ACES's scratch ``offset``), criteria, and labels are:

        >>> edge == Edge("gap", TemporalWindowBounds(False, timedelta(days=1), True, timedelta(hours=2)),
        ...              (("death", (None, 0)),))
        True
    """

    name: str
    bound: Bound
    constraints: tuple[tuple[str, Constraint], ...] = ()
    label: str | None = None

    @property
    def is_temporal(self) -> bool:
        """Whether the window is a fixed offset (forward or backward)."""
        return isinstance(self.bound, TemporalWindowBounds)

    @property
    def is_end_rooted(self) -> bool:
        """Whether the window is defined from its end backward to its start (a look-back)."""
        if self.is_temporal:
            return self.bound.window_size < timedelta(0)
        return self.bound.end_event.startswith("-")

    @property
    def exit_side(self) -> str:
        """The endpoint the next window in the chain attaches to."""
        return "start" if self.is_end_rooted else "end"

    @property
    def is_constrained(self) -> bool:
        """Whether the window carries criteria."""
        return bool(self.constraints)

    @property
    def days(self) -> float | None:
        """The window's duration in days, for temporal windows (negative for look-backs)."""
        if not self.is_temporal:
            return None
        return self.bound.window_size.total_seconds() / 86400

    @property
    def event(self) -> str | None:
        """The predicate bounding the window, for event-bounded windows (without the look-back sign)."""
        return None if self.is_temporal else self.bound.end_event.lstrip("-")

    @property
    def shape(self) -> tuple:
        """What the window is, regardless of its name: its bound, criteria, and label."""
        return (bound_key(self.bound), self.constraints, self.label)

    def unconstrained(self) -> "Edge":
        """A copy without criteria."""
        return replace(self, constraints=())

    def __eq__(self, other: object) -> bool:
        """Edges are equal when their names and shapes are."""
        return isinstance(other, Edge) and (self.name, self.shape) == (other.name, other.shape)

    def __hash__(self) -> int:
        """Hashes the name and shape."""
        return hash((self.name, self.shape))

    def end_str(self) -> str | None:
        """The window's ``end`` in ACES syntax for a forward window (``None`` for "end of record")."""
        if self.is_temporal:
            return f"start + {render_timedelta(self.bound.window_size)}"
        if self.event == END_OF_RECORD_KEY:
            return None
        return f"start -> {self.event}"

    def start_str(self) -> str:
        """The window's ``start`` in ACES syntax for an end-rooted window."""
        if self.is_temporal:
            return f"end - {render_timedelta(-self.bound.window_size)}"
        return f"end <- {self.event}"

    def __str__(self) -> str:
        """Renders the edge in ACES-like syntax."""
        end = self.start_str() if self.is_end_rooted else (self.end_str() or "start -> end of record")
        parts = [f"{self.name}: {end}"]
        if self.constraints:
            parts.append(f" ({_constraints_str(dict(self.constraints))})")
        if self.label:
            parts.append(f"; label={self.label}")
        return "".join(parts)


@dataclass(frozen=True)
class Branch:
    """A chain of windows from the prediction time, whose criteria (if any) sit only on its last window.

    Examples:
        >>> gap = Edge("gap", TemporalWindowBounds(False, timedelta(days=1), True))
        >>> target = Edge("target", ToEventWindowBounds(False, "discharge", True), label="death")
        >>> branch = Branch((gap, target))
        >>> print(branch)
        gap: start + 1 day -> target: start -> discharge; label=death
        >>> branch.is_label_branch, branch.terminal.name
        (True, 'target')
    """

    edges: tuple[Edge, ...]

    @property
    def terminal(self) -> Edge:
        """The last window of the branch."""
        return self.edges[-1]

    @property
    def is_label_branch(self) -> bool:
        """Whether the branch ends in the label window."""
        return self.terminal.label is not None

    def __str__(self) -> str:
        """Renders the chain."""
        return " -> ".join(str(e) for e in self.edges)
