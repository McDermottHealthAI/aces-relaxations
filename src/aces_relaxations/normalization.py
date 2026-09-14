"""Normalized form for future-only ACES task configurations.

A future-only task configuration describes, from the prediction time onward, a *target* (the label window)
and a set of *conditions* (inclusion / exclusion criteria on windows, and the requirement that event-bounded
windows be realizable). ACES lets these be tangled together: a single window can both carry criteria and be
the parent of the label window, and criteria can sit anywhere along a chain. The **normalized form** untangles
them without changing the label of any prediction-time row:

0. Every event-bounded window is given an explicit criterion that its bounding event occurs
   (``aces_relaxations.bounding_events.require_bounding_events``), so that nothing downstream has to know
   that ACES drops rows whose event-bounded windows cannot be realized.
1. The tree is re-rooted at the prediction time; everything before it (the trigger, input-window criteria) is
   discarded, since it selects the cohort but cannot change the label of a row that is in it.
2. Every window that carries criteria is pulled out into its own *branch*: a chain of unconstrained copies of
   the windows leading to it, ending in the constrained window. The label window's branch carries no
   criteria (any criteria it had are pulled out likewise), and unconstrained event-bounded leaf windows are
   kept as branches of their own, because ACES drops rows whose event-bounded windows cannot be realized.
3. Along each branch, runs of unconstrained, unlabeled, temporal windows are collapsed into one window.
4. Branches sharing an identical unconstrained prefix share those windows, so the result is again a tree, in
   which every root-to-leaf path is a chain of windows whose ``start`` is the previous window's ``end`` and
   whose criteria (if any) sit only on its last window.

The result is a genuine ACES configuration whose trigger is a synthetic *index event* predicate marking the
prediction time, so it can be run through ACES itself (see ``aces_relaxations.extraction``), and whose
branches are exposed for downstream converters.

Windows defined backward from a later endpoint (``start: end - 1d``) are allowed when they cannot reach before
the prediction time, i.e. when the guaranteed elapsed time along the path (the sum of the temporal windows
on it; event-bounded windows guarantee nothing) covers the look-back; they are rewritten as forward chains.

Why this is label-preserving: with every edge in the prediction-time subtree moving forward in time, the
timestamp of every window endpoint is a deterministic function of the data and the prediction time, computed
along its path from the root. Copying a path therefore realizes exactly the same endpoints, so a criterion
evaluated on a copy has exactly the same truth value, and the conjunction of criteria across branches equals
the original conjunction. Merging two adjacent unconstrained temporal windows ``(a, b]`` and ``(b, c]``
into ``(a, c]`` changes no endpoint that anything downstream refers to (only ``a`` and ``c`` are referenced),
and the merged window is itself unconstrained. See the README for the property test that checks this against
ACES on random configurations and data.
"""

import copy
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any

from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
from aces.types import TemporalWindowBounds, ToEventWindowBounds
from bigtree import Node

from .bounding_events import require_bounding_events
from .degeneracy import Analysis, Interval
from .future_only import _implies, _initial_state, backward_edge_reason, validate_future_only
from .guarantees import PathState, anchor_counts
from .tree import prediction_time_node

logger = logging.getLogger(__name__)

INDEX_PREDICATE = "index_event"
INDEX_CODE = "_INDEX_EVENT_"
RECORD_END = "_RECORD_END"

Constraint = tuple[int | None, int | None]
Bound = TemporalWindowBounds | ToEventWindowBounds


def _bound_key(bound: Bound) -> tuple:
    """A hashable summary of a bound, sufficient to decide whether two windows coincide.

    The ``offset`` field of ACES bounds is deliberately ignored: ACES uses it as scratch state during
    extraction (and accumulates into it on every query), so it carries no meaning about the window.
    """
    if isinstance(bound, TemporalWindowBounds):
        return ("time", bound.left_inclusive, bound.right_inclusive, bound.window_size)
    return ("event", bound.left_inclusive, bound.right_inclusive, bound.end_event)


@dataclass(frozen=True, eq=False)
class Edge:
    """One window of a normalized branch: how its end is reached from its start, and what it asserts.

    Attributes:
        name: The window name.
        bound: How the end is reached from the start (a forward temporal offset or the next matching event),
            including the inclusivity of both endpoints.
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

    def unconstrained(self) -> "Edge":
        """A copy without criteria."""
        return replace(self, constraints=())

    @property
    def key(self) -> tuple:
        """A hashable identity: name, bound, criteria, and label."""
        return (self.name, _bound_key(self.bound), self.constraints, self.label)

    def __eq__(self, other: object) -> bool:
        """Edges are equal when their identities are."""
        return isinstance(other, Edge) and self.key == other.key

    def __hash__(self) -> int:
        """Hashes the identity."""
        return hash(self.key)

    def end_str(self) -> str | None:
        """The window's ``end`` in ACES syntax for a forward window (``None`` for "end of record")."""
        if self.is_temporal:
            return f"start + {_render_timedelta(self.bound.window_size)}"
        if self.event == RECORD_END:
            return None
        return f"start -> {self.event}"

    def start_str(self) -> str:
        """The window's ``start`` in ACES syntax for an end-rooted window."""
        if self.is_temporal:
            return f"end - {_render_timedelta(-self.bound.window_size)}"
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
        >>> branch.is_label_branch, branch.is_criteria_branch, branch.terminal.name
        (True, False, 'target')
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

    @property
    def is_criteria_branch(self) -> bool:
        """Whether the branch ends in a constrained window."""
        return self.terminal.is_constrained

    def __str__(self) -> str:
        """Renders the chain."""
        return " -> ".join(str(e) for e in self.edges)


class NormalizedTaskConfig(TaskExtractorConfig):
    """A task configuration in normalized form.

    This is a genuine ACES configuration (its trigger is the synthetic index-event predicate), with the
    normalized branches exposed on ``branches``.
    """

    branches: tuple[Branch, ...] = ()

    @property
    def label_branch(self) -> Branch:
        """The branch ending in the label window."""
        return next(b for b in self.branches if b.is_label_branch)

    @property
    def criteria_branches(self) -> tuple[Branch, ...]:
        """The branches ending in a constrained window, plus unconstrained realizability branches."""
        return tuple(b for b in self.branches if not b.is_label_branch)


def _render_timedelta(delta: timedelta) -> str:
    """Renders a positive timedelta in a form ACES's boundary parser accepts.

    Examples:
        >>> _render_timedelta(timedelta(days=7))
        '7 days'
        >>> _render_timedelta(timedelta(hours=36))
        '36 hours'
        >>> _render_timedelta(timedelta(minutes=90))
        '90 minutes'
        >>> _render_timedelta(timedelta(seconds=61))
        '61 seconds'
        >>> _render_timedelta(timedelta(hours=1))
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


def _own_node_name(window_name: str, window: WindowConfig) -> str:
    """The name of the node that carries a window's criteria and label (the far end of its span)."""
    return f"{window_name}.{'start' if window.root_node == 'end' else 'end'}"


def _rewind(stack: list[Edge], days: timedelta, window_name: str) -> list[Edge]:
    """Moves the current position back by ``days`` along a chain of unconstrained windows.

    Only temporal windows can be rewound through, because only they contribute *guaranteed* elapsed time:
    an event-bounded window may be arbitrarily short, so looking back across one could reach before the
    prediction time.

    Examples:
        >>> a = Edge("a", TemporalWindowBounds(False, timedelta(days=7), True))
        >>> for e in _rewind([a], timedelta(days=1), "b"): print(e)
        a__pre_b: start + 6 days
        >>> _rewind([a], timedelta(days=7), "b")
        []
        >>> _rewind([a], timedelta(days=8), "b")
        Traceback (most recent call last):
            ...
        ValueError: Window 'b' looks back 8 days, reaching before the prediction time.
        >>> e = Edge("e", ToEventWindowBounds(False, "discharge", True))
        >>> _rewind([a, e], timedelta(days=1), "b")
        Traceback (most recent call last):
            ...
        ValueError: Window 'b' looks back 1 day across the event-bounded window 'e', so it cannot be
        written as a start-rooted chain.
    """
    out = list(stack)
    remaining = days
    while remaining > timedelta(0):
        if not out:
            raise ValueError(
                f"Window '{window_name}' looks back {_render_timedelta(days)}, reaching before the "
                "prediction time."
            )
        top = out.pop()
        if not top.is_temporal or top.is_end_rooted:
            raise ValueError(
                f"Window '{window_name}' looks back {_render_timedelta(days)} across the event-bounded "
                f"window '{top.name}', so it cannot be written as a start-rooted chain."
            )
        if top.bound.window_size > remaining:
            shortened = TemporalWindowBounds(
                top.bound.left_inclusive, top.bound.window_size - remaining, False
            )
            out.append(Edge(f"{top.name}__pre_{window_name}", shortened))
            remaining = timedelta(0)
        else:
            remaining -= top.bound.window_size
    return out


def _walk(
    task_cfg: TaskExtractorConfig, root: Node, leaf: Node, allow_end_rooted: bool = False
) -> Iterator[Branch]:
    """Yields the branches contributed by one root-to-leaf path of the prediction-time subtree.

    The walk keeps a chain of unconstrained windows from the prediction time to the current position.
    Forward edges extend it; a window carrying criteria contributes a branch ending in that window; the
    label window contributes the label branch; an unconstrained leaf contributes its (realizability)
    branch. A temporal look-back rewinds the position when the chain's trailing temporal windows cover it
    (see ``_rewind``); otherwise, and for look-backs to a prior event, the window is kept *end-rooted* if
    ``allow_end_rooted`` and the look-back is guaranteed not to reach before the prediction time (see
    ``aces_relaxations.guarantees``), and is an error if not.

    Examples:
        >>> from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "week": WindowConfig("input.end", "start + 7d", False, True, label="B"),
        ...         "last_day": WindowConfig("end - 1d", "week.end", False, True, has={"A": "(1, None)"}),
        ...         "after": WindowConfig(
        ...             "last_day.start", "start + 2d", False, True, has={"B": "(None, 0)"}
        ...         ),
        ...     },
        ... )
        >>> root = cfg.window_nodes["input.end"]
        >>> for leaf in root.leaves:
        ...     for b in _walk(cfg, root, leaf): print(b)
        week: start + 7 days; label=B
        week__pre_last_day: start + 6 days -> last_day: start + 1 day (A: (1, None))
        week__pre_last_day: start + 6 days -> after: start + 2 days (B: (None, 0))
    """
    label_window = task_cfg.label_window
    label_own = _own_node_name(label_window, task_cfg.windows[label_window])
    label = task_cfg.windows[label_window].label

    path = list(leaf.node_path)
    steps = path[path.index(root) + 1 :]
    stack: list[Edge] = []
    state = _initial_state(task_cfg, root)

    def emit(edge: Edge, is_leaf: bool) -> Iterator[Branch]:
        if edge.is_constrained:
            yield Branch((*stack, replace(edge, label=None)))
        if edge.label is not None or (is_leaf and not edge.is_constrained):
            yield Branch((*stack, edge.unconstrained()))

    for i, node in enumerate(steps):
        window_name, side = node.node_name.rsplit(".", 1)
        window = task_cfg.windows[window_name]
        is_own = node.node_name == _own_node_name(window_name, window)
        raw_constraints = getattr(node, "constraints", None) or {}
        constraints = tuple(sorted((p, tuple(c)) for p, c in raw_constraints.items()))
        node_label = label if node.node_name == label_own else None
        is_leaf = i == len(steps) - 1
        bound = node.endpoint_expr
        before = state.pos

        counts = anchor_counts(window.start_inclusive, window.end_inclusive, side)
        if backward_edge_reason(node) is None:
            name = window_name if is_own else f"{window_name}_{side}_gap"
            edge = Edge(name, bound, constraints, node_label)
            if _is_identity_hop(state, edge, counts):
                # The search finds the event at the current position, so the chain does not move: the
                # window is emitted (as a point or empty window) but contributes nothing to the chain.
                state.add_criteria(raw_constraints, before, state.pos)
                yield from emit(edge, is_leaf)
                continue
            if edge.is_temporal:
                state.advance_temporal(bound.window_size)
            else:
                state.advance_event(bound.end_event, strict=not counts)
            state.add_criteria(raw_constraints, before, state.pos)
            yield from emit(edge, is_leaf)
            stack.append(edge.unconstrained())
            continue

        # A backward edge. Its window (if this is the window's own node) spans from the new position back
        # up to the position before the move, so criteria guarantees run from the new position to `before`.
        if isinstance(bound, TemporalWindowBounds):
            days = -bound.window_size
            try:
                stack = _rewind(stack, days, window_name)
                state.look_back_temporal(days)
                if is_own:
                    edge = Edge(
                        window_name,
                        TemporalWindowBounds(window.start_inclusive, days, window.end_inclusive),
                        constraints,
                        node_label,
                    )
                    state.add_criteria(raw_constraints, state.pos, before)
                    yield from emit(edge, False)
                continue
            except ValueError:
                if not allow_end_rooted or not state.can_look_back(days):
                    raise
            state.look_back_temporal(days)
        else:
            guarantee = state.guarantee_for(bound.end_event.lstrip("-"), counts)
            if not allow_end_rooted:
                raise ValueError(
                    f"Window '{window_name}' looks back to a prior '{bound.end_event.lstrip('-')}' event, "
                    "which cannot be written as a start-rooted chain."
                )
            if guarantee is None:
                raise ValueError(
                    f"Window '{window_name}' looks back to a prior '{bound.end_event.lstrip('-')}' event, "
                    "which is not guaranteed to occur after the prediction time."
                )
            state.look_back_event(guarantee)

        # Kept end-rooted: the window's own node carries its criteria; a gap node carries nothing.
        edge = Edge(window_name if is_own else f"{window_name}_{side}_gap", bound, constraints, node_label)
        state.add_criteria(raw_constraints, state.pos, before)
        yield from emit(edge, is_leaf)
        stack.append(edge.unconstrained())


def _is_identity_hop(state: PathState, edge: Edge, counts: bool) -> bool:
    """Whether a forward hop lands where it starts: an inclusive search for an event the position already is.

    Examples:
        >>> state = PathState(lambda g, p: g == p)
        >>> state.advance_event("B")
        >>> again = Edge("again", ToEventWindowBounds(False, "B", True))
        >>> _is_identity_hop(state, again, counts=True), _is_identity_hop(state, again, counts=False)
        (True, False)
        >>> _is_identity_hop(state, Edge("c", ToEventWindowBounds(False, "C", True)), counts=True)
        False
    """
    if edge.is_temporal or edge.is_end_rooted or edge.event == RECORD_END or not counts:
        return False
    return any(
        g.start == g.end == state.pos
        and not g.predicate.startswith("_RECORD_")
        and (edge.event == "_ANY_EVENT" or state.implies(g.predicate, edge.event))
        for g in state.guarantees
    )


def _edge_interval(analysis: Analysis, edge: Edge) -> Interval | None:
    """The interval of an edge that is one of the analyzed configuration's forward windows (else ``None``)."""
    node = analysis.task_cfg.window_nodes.get(f"{edge.name}.end")
    if edge.is_end_rooted or node is None or _bound_key(node.endpoint_expr) != _bound_key(edge.bound):
        return None
    return analysis.window(edge.name)


def _forbidden(analysis: Analysis, predicate: str, region: Interval) -> bool:
    """Whether some criterion or hop provably forbids ``predicate`` over all of ``region``."""
    if analysis.is_empty(region):
        return True
    return any(
        b.at_most == 0 and analysis.implies(predicate, b.predicate) and analysis.contains(b.interval, region)
        for b in analysis.bounds
    )


def _mergeable_hop(analysis: Analysis, prev: Edge, region: Interval, edge: Edge) -> bool | None:
    """Whether ``prev`` (spanning ``region``) then a hop to the next ``Q`` is one hop to the next ``Q``.

    That holds when no ``Q`` can occur where the longer search looks but the shorter one does not, which
    some criterion or hop of the configuration must forbid: strictly inside ``prev``'s window, at ``prev``'s
    end when the hop is strict, and at ``prev``'s start when the merged search includes it (so either the
    start is covered, or the merged search is made exclusive, which is only safe when ``prev``'s end is
    certainly after its start or the original hop was strict).

    Returns:
        The inclusivity of the merged hop's found endpoint, or ``None`` if the windows cannot be merged.
    """
    q = edge.event
    inclusive = edge.bound.right_inclusive
    interior = Interval(region.lo, region.hi, False, False)
    if not _forbidden(analysis, q, interior):
        return None
    if not inclusive and not _forbidden(analysis, q, Interval.point(region.hi)):
        return None
    # A ``Q`` at the start is harmless when the previous hop, searching inclusively for something ``Q``
    # implies, would itself have stopped there, so that the original chain sees it too.
    stops_there = (
        not prev.is_temporal
        and prev.bound.right_inclusive
        and prev.event != RECORD_END
        and analysis.implies(q, prev.event)
    )
    if stops_there or _forbidden(analysis, q, Interval.point(region.lo)):
        return inclusive
    if not inclusive or analysis.leq(region.lo, region.hi, strict=True):
        return False
    return None


def _collapse(branch: Branch, analysis: Analysis | None = None) -> Branch:
    """Merges runs of unconstrained, unlabeled windows whose composition is a single window.

    Two consecutive temporal windows are one temporal window of the summed size. A window followed by a hop
    to the next ``Q`` is one hop to the next ``Q`` from the first window's start when the configuration
    provably forbids ``Q`` (or a predicate ``Q`` implies) where the longer search would additionally look:
    by a criterion on any window covering it, or by an earlier hop's gap (see ``_mergeable_hop``). The
    normalized form keeps every criterion on its own branch and every hop, so the merge preserves labels.

    Examples:
        >>> a = Edge("a", TemporalWindowBounds(True, timedelta(days=7), True))
        >>> b = Edge("b", TemporalWindowBounds(False, timedelta(days=3), False))
        >>> c = Edge("c", ToEventWindowBounds(False, "discharge", True), (("death", (None, 0)),))
        >>> print(_collapse(Branch((a, b, c))))
        a__b: start + 10 days -> c: start -> discharge (death: (None, 0))
        >>> _collapse(Branch((a, b, c))).edges[0].bound
        TemporalWindowBounds(left_inclusive=True, window_size=datetime.timedelta(days=10),
                             right_inclusive=False, offset=datetime.timedelta(0))

    A constrained or labeled temporal window is never merged into:

        >>> d = Edge("d", TemporalWindowBounds(False, timedelta(days=3), True), label="death")
        >>> print(_collapse(Branch((a, d))))
        a: start + 7 days -> d: start + 3 days; label=death

    Merging a hop needs the configuration's analysis, to know what is forbidden where. Here the stay (from
    its start up to the next discharge) forbids admissions, so the next admission after the discharge is
    the next admission after the stay's start:

        >>> from aces.config import EventConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={p: PlainPredicateConfig(p) for p in ("admission", "discharge", "death")},
        ...     trigger=EventConfig("admission"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "stay": WindowConfig(
        ...             "input.end", "start -> discharge", True, True, has={"admission": "(None, 0)"}
        ...         ),
        ...         "readmission": WindowConfig("stay.end", "start -> admission", False, True),
        ...         "target": WindowConfig("readmission.end", "start + 30d", False, True, label="death"),
        ...     },
        ... )
        >>> stay = Edge("stay", ToEventWindowBounds(True, "discharge", True))
        >>> readmission = Edge("readmission", ToEventWindowBounds(False, "admission", True))
        >>> target = Edge("target", TemporalWindowBounds(False, timedelta(days=30), True), label="death")
        >>> print(_collapse(Branch((stay, readmission, target)), Analysis.of(cfg)))
        stay__readmission: start -> admission -> target: start + 30 days; label=death

    The merged hop keeps the original hop's inclusivity. Without the criterion nothing merges:

        >>> cfg.windows["stay"].has = {}
        >>> cfg = TaskExtractorConfig(predicates=cfg.predicates, trigger=cfg.trigger, windows=cfg.windows)
        >>> print(_collapse(Branch((stay, readmission, target)), Analysis.of(cfg)))
        stay: start -> discharge -> readmission: start -> admission -> target: start + 30 days; label=death

    The criterion may sit on any window provably covering the first one, not only on that window itself.
    Here a week-long wait is followed by a hop to the next admission, and a month-long window forbids
    admissions:

        >>> cfg.windows["stay"] = WindowConfig("input.end", "start + 7d", True, True)
        >>> cfg.windows["cover"] = WindowConfig(
        ...     "input.end", "start + 30d", True, True, has={"admission": "(None, 0)"}
        ... )
        >>> cfg = TaskExtractorConfig(predicates=cfg.predicates, trigger=cfg.trigger, windows=cfg.windows)
        >>> week = Edge("stay", TemporalWindowBounds(True, timedelta(days=7), True))
        >>> print(_collapse(Branch((week, readmission, target)), Analysis.of(cfg)))
        stay__readmission: start -> admission -> target: start + 30 days; label=death

    A hop to a predicate implied by the previous hop's event also merges, since that hop's gap forbids it:

        >>> from aces.config import DerivedPredicateConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={
        ...         "A": PlainPredicateConfig("A"),
        ...         "B": PlainPredicateConfig("B"),
        ...         "C": PlainPredicateConfig("C"),
        ...         "A_or_B": DerivedPredicateConfig("or(A, B)"),
        ...     },
        ...     trigger=EventConfig("C"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "either": WindowConfig("input.end", "start -> A_or_B", True, True),
        ...         "a": WindowConfig("either.end", "start -> A", False, True),
        ...         "target": WindowConfig("a.end", "start + 3d", False, True, label="C"),
        ...     },
        ... )
        >>> either = Edge("either", ToEventWindowBounds(True, "A_or_B", True))
        >>> a = Edge("a", ToEventWindowBounds(False, "A", True))
        >>> t = Edge("target", TemporalWindowBounds(False, timedelta(days=3), True), label="C")
        >>> print(_collapse(Branch((either, a, t)), Analysis.of(cfg)))
        either__a: start -> A -> target: start + 3 days; label=C

    Nor does a stay that excludes its start but includes its (event) end, ACES's usual ``(start, next
    discharge]``: an admission exactly at the stay's start is not forbidden, and whether the original chain
    lands on it depends on whether a discharge is there too (then the stay is zero-length and the hop finds
    the admission), which a single hop from the start cannot reproduce either way:

        >>> cfg = TaskExtractorConfig(
        ...     predicates={p: PlainPredicateConfig(p) for p in ("admission", "discharge", "death")},
        ...     trigger=EventConfig("admission"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "stay": WindowConfig(
        ...             "input.end", "start -> discharge", False, True, has={"admission": "(None, 0)"}
        ...         ),
        ...         "readmission": WindowConfig("stay.end", "start -> admission", False, True),
        ...         "target": WindowConfig("readmission.end", "start + 30d", False, True, label="death"),
        ...     },
        ... )
        >>> usual = Edge("stay", ToEventWindowBounds(False, "discharge", True))
        >>> print(_collapse(Branch((usual, readmission, target)), Analysis.of(cfg)))
        stay: start -> discharge -> readmission: start -> admission -> target: start + 30 days; label=death
    """
    merged: list[Edge] = []
    regions: list[Interval | None] = []
    for edge in branch.edges:
        prev = merged[-1] if merged else None
        plain = (
            prev is not None
            and not prev.is_end_rooted
            and not edge.is_end_rooted
            and not prev.is_constrained
            and not edge.is_constrained
            and prev.label is None
            and edge.label is None
        )
        region = None if analysis is None else _edge_interval(analysis, edge)
        if plain and prev.is_temporal and edge.is_temporal:
            total = prev.bound.window_size + edge.bound.window_size
            bound = TemporalWindowBounds(prev.bound.left_inclusive, total, edge.bound.right_inclusive)
            merged[-1] = Edge(f"{prev.name}__{edge.name}", bound)
            regions[-1] = _span(regions[-1], region)
            continue
        if plain and not edge.is_temporal and edge.event != RECORD_END and regions[-1] is not None:
            inclusive = _mergeable_hop(analysis, prev, regions[-1], edge)
            if inclusive is not None:
                bound = ToEventWindowBounds(prev.bound.left_inclusive, edge.event, inclusive)
                merged[-1] = Edge(f"{prev.name}__{edge.name}", bound)
                regions[-1] = _span(regions[-1], region)
                continue
        merged.append(edge)
        regions.append(region)
    return Branch(tuple(merged))


def _span(a: Interval | None, b: Interval | None) -> Interval | None:
    """The interval from the first of two consecutive windows' start to the second's end."""
    return None if a is None or b is None else Interval(a.lo, b.hi, a.lo_inclusive, b.hi_inclusive)


@dataclass
class _Trie:
    edge: Edge | None = None
    children: dict[tuple, "_Trie"] = field(default_factory=dict)
    on_label_path: bool = False

    def insert(self, branch: Branch) -> None:
        node = self
        for edge in branch.edges:
            key = (_bound_key(edge.bound), edge.constraints, edge.label)
            if key not in node.children:
                node.children[key] = _Trie(edge)
            node = node.children[key]
            node.on_label_path = node.on_label_path or branch.is_label_branch


def _future_leaves(task_cfg: TaskExtractorConfig, root: Node) -> list[Node]:
    """The leaves of the prediction-time subtree, excluding branches that start by moving backward.

    A child of the prediction-time node reached by a backward edge lies (at least partly) before the
    prediction time: it is cohort selection, not a condition on the future, and is discarded with a warning.
    The one exception is a look-back to a prior event that is guaranteed to be the prediction time itself
    (the trigger event, when the prediction time is the trigger and the window includes its end), which is
    a zero-length window at the prediction time.
    """
    leaves: list[Node] = []
    for child in root.children:
        reason = backward_edge_reason(child)
        if reason is not None:
            bound = child.endpoint_expr
            window = task_cfg.windows[child.node_name.rsplit(".", 1)[0]]
            side = child.node_name.rsplit(".", 1)[1]
            guaranteed = isinstance(bound, ToEventWindowBounds) and (
                _initial_state(task_cfg, root).guarantee_for(
                    bound.end_event[1:], anchor_counts(window.start_inclusive, window.end_inclusive, side)
                )
                is not None
            )
            if not guaranteed:
                logger.warning(
                    f"Discarding window branch at {child.node_name}, which starts before the prediction "
                    f"time ({reason})."
                )
                continue
        leaves.extend([child] if child.is_leaf else child.leaves)
    return leaves


def branches(task_cfg: TaskExtractorConfig, allow_end_rooted: bool = False) -> tuple[Branch, ...]:
    """Computes the normalized branches of a future-only task configuration, without building a config.

    Args:
        task_cfg: The task configuration.
        allow_end_rooted: Whether a look-back window that cannot be rewritten as a start-rooted chain may be
            kept end-rooted, provided it is guaranteed not to reach before the prediction time.

    Returns:
        The branches, label branch first, then the others in tree order.

    Raises:
        ValueError: If the configuration is not future-only, a look-back after the prediction time is not
            guaranteed to stay after it, or (without ``allow_end_rooted``) a look-back cannot be written as a
            start-rooted chain.

    Examples:
        >>> for b in branches(in_hosp_mortality_cfg): print(b)
        gap: start + 1 day -> target: start -> discharge_or_death; label=death
        gap: start + 1 day (admission: (None, 0), discharge_or_death: (None, 0))
        gap: start + 1 day -> target_bounding_event: start -> discharge_or_death (discharge_or_death: (1,
        None))
        >>> for b in branches(post_discharge_mortality_cfg): print(b)
        hospitalization: start -> discharge -> gap: start + 1 day -> target: start + 29 days; label=death
        post_input: start + 1 day (admission: (None, 0), discharge_or_death: (None, 0))
        hospitalization: start -> discharge (death: (None, 0))
        hospitalization: start -> discharge -> gap: start + 1 day (admission: (None, 0), death: (None, 0))
        hospitalization_bounding_event: start -> discharge (discharge: (1, None))
        >>> for b in branches(readmission_cfg): print(b)
        gap: start + 1 day -> target: start + 29 days; label=admission
        gap: start + 1 day (admission: (None, 0), death: (None, 0))
        gap: start + 1 day -> target: start + 29 days; label=admission -> censoring_protection: start ->
        end of record (_ANY_EVENT: (1, None))
        >>> for b in branches(two_stage_cfg): print(b)
        1st_infusion: start -> infusion_end -> 2nd_infusion_start_gap: start -> infusion_start ->
        2nd_infusion: start -> infusion_end; label=adverse_event
        1st_infusion: start -> infusion_end (adverse_event: (None, 0), infusion_end: (1, None))
        1st_infusion: start -> infusion_end -> 2nd_infusion_start_gap: start -> infusion_start ->
        2nd_infusion: start -> infusion_end (infusion_end: (1, None), infusion_start: (1, None))
    """
    validate_future_only(task_cfg)
    task_cfg = require_bounding_events(task_cfg)
    analysis = Analysis.of(task_cfg)
    root = task_cfg.window_nodes[prediction_time_node(task_cfg).node_name]

    seen: dict[Branch, None] = {}
    for leaf in _future_leaves(task_cfg, root):
        for branch in _walk(task_cfg, root, leaf, allow_end_rooted):
            seen[_collapse(branch, analysis)] = None

    from .simplification import simplify_branches  # circular at module level

    at_pt = [g.predicate for g in _initial_state(task_cfg, root).guarantees]
    ordered = list(simplify_branches(seen, _implies(task_cfg), at_pt, task_cfg))
    return tuple(sorted(ordered, key=lambda b: (not b.is_label_branch, ordered.index(b))))


def normalize(
    task_cfg: TaskExtractorConfig,
    allow_end_rooted: bool = False,
    index_predicate: str = INDEX_PREDICATE,
    index_code: str = INDEX_CODE,
) -> NormalizedTaskConfig:
    """Converts a future-only task configuration into normalized form.

    Args:
        task_cfg: The task configuration. Its label window must be reachable from the prediction time, and
            every window branching from the prediction time must be guaranteeably future-only.
        allow_end_rooted: Whether to produce the *universal* normalized form, in which a look-back window
            that cannot be rewritten as a start-rooted chain is kept end-rooted (``start: end - 1d`` or
            ``start: end <- P``) as long as it is guaranteed not to reach before the prediction time. The
            default, strict form only ever emits start-rooted windows and therefore rejects such
            configurations.
        index_predicate: The name of the synthetic predicate marking the prediction time, used as the
            trigger of the normalized configuration.
        index_code: The code that predicate matches.

    Returns:
        The normalized configuration.

    Raises:
        ValueError: If the configuration cannot be normalized.

    Examples:
        >>> normalized = normalize(in_hosp_mortality_cfg)
        >>> print_ACES(normalized)
        trigger; **Prediction Time**
        ├── (+1 day, 0:00:00) gap.end
        │   ├── (next discharge_or_death) target.end; **Label: Presence of death**
        │   └── (next discharge_or_death) target_bounding_event.end (at least 1 discharge_or_death)
        └── (+1 day, 0:00:00) gap_criteria.end (no admission, discharge_or_death)
        >>> print(normalized.label_branch)
        gap: start + 1 day -> target: start -> discharge_or_death; label=death
        >>> normalized.windows["gap_criteria"].start, normalized.windows["gap_criteria"].end
        ('trigger', 'start + 1 day')
        >>> normalized.trigger
        EventConfig(predicate='index_event')

    Windows sharing an unconstrained prefix share those windows in the tree:

        >>> print_ACES(normalize(post_discharge_mortality_cfg))
        trigger; **Prediction Time**
        ├── (next discharge) hospitalization.end
        │   ├── (+1 day, 0:00:00) gap.end
        │   │   └── (+29 days, 0:00:00) target.end; **Label: Presence of death**
        │   └── (+1 day, 0:00:00) gap_criteria.end (no admission, death)
        ├── (+1 day, 0:00:00) post_input.end (no admission, discharge_or_death)
        ├── (next discharge) hospitalization_criteria.end (no death)
        └── (next discharge) hospitalization_bounding_event.end (at least 1 discharge)

    Unconstrained temporal windows collapse; the label window and constrained windows never do:

        >>> from aces.config import DerivedPredicateConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "gap1": WindowConfig("input.end", "start + 7d", False, True),
        ...         "gap2": WindowConfig("gap1.end", "start + 3d", False, True),
        ...         "watch": WindowConfig("gap2.end", "start -> B", False, True, has={"A": "(None, 0)"}),
        ...         "target": WindowConfig("gap2.end", "start + 5d", False, True, label="B"),
        ...     },
        ... )
        >>> print_ACES(normalize(cfg))
        trigger; **Prediction Time**
        └── (+10 days, 0:00:00) gap1__gap2.end
            ├── (+5 days, 0:00:00) target.end; **Label: Presence of B**
            ├── (next B) watch.end (no A)
            └── (next B) watch_bounding_event.end (at least 1 B)

    A hop that lands where it starts (an inclusive search for an event the position already is) drops out
    of the chain, and a window followed by a hop to an event that is provably absent in between collapses
    into one hop (see ``_collapse``). Here ``again`` is an identity hop, and ``either``'s gap forbids ``A``,
    so the hop to the next ``A`` merges into it:

        >>> cfg = TaskExtractorConfig(
        ...     predicates={
        ...         "A": PlainPredicateConfig("A"),
        ...         "B": PlainPredicateConfig("B"),
        ...         "A_or_B": DerivedPredicateConfig("or(A, B)"),
        ...     },
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "either": WindowConfig("input.end", "start -> A_or_B", True, True),
        ...         "again": WindowConfig("either.end", "start -> A_or_B", False, True),
        ...         "a": WindowConfig("again.end", "start -> A", False, True),
        ...         "target": WindowConfig("a.end", "start + 3d", False, True, label="B"),
        ...     },
        ... )
        >>> print_ACES(normalize(cfg))
        trigger; **Prediction Time**
        ├── (next A) either__a.end
        │   └── (+3 days, 0:00:00) target.end; **Label: Presence of B**
        ├── (next A_or_B) either.end
        │   └── (next A) a_bounding_event.end (at least 1 A)
        └── (next A_or_B) either_criteria.end (at least 1 A_or_B)

    (``again``'s own "bounding event occurs" criterion was implied by ``either``'s landing, which
    ``either_criteria`` asserts, and was removed; see ``aces_relaxations.simplification``.)

    Windows defined backward from a later endpoint are fine as long as they cannot reach before the
    prediction time; they are rewritten as forward chains. Here ``last_day`` is the final day of ``week``:

        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "week": WindowConfig("input.end", "start + 7d", False, True, label="B"),
        ...         "last_day": WindowConfig("end - 1d", "week.end", False, True, has={"A": "(1, None)"}),
        ...     },
        ... )
        >>> print_ACES(normalize(cfg))
        trigger; **Prediction Time**
        ├── (+7 days, 0:00:00) week.end; **Label: Presence of B**
        └── (+6 days, 0:00:00) week__pre_last_day.end
            └── (+1 day, 0:00:00) last_day.end (at least 1 A)

    Branches that start by moving backward from the prediction time lie before it and are discarded (with
    a warning); a window that could reach before the prediction time is rejected:

        >>> print_ACES(normalize(readmission_cfg))
        trigger; **Prediction Time**
        ├── (+1 day, 0:00:00) gap.end
        │   └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
        │       └── (end of record) censoring_protection.end (at least 1 event(s))
        └── (+1 day, 0:00:00) gap_criteria.end (no admission, death)
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "lookback": WindowConfig(
        ...             "end - 3d", "input.end + 2d", False, True, has={"A": "(None, 0)"}
        ...         ),
        ...         "target": WindowConfig("input.end", "start + 5d", False, True, label="B"),
        ...     },
        ... )
        >>> normalize(cfg)
        Traceback (most recent call last):
            ...
        ValueError: Window 'lookback' looks back 3 days, reaching before the prediction time.

    The strict form cannot express a look-back from an event-bounded endpoint, even when it is provably in
    the future; the universal form (``allow_end_rooted=True``) keeps it as an end-rooted window. Here
    ``last_day`` is the day before a discharge that is itself at least five days after the prediction time,
    and ``since_admission`` looks back to an admission that a criterion guarantees occurred after the
    prediction time (its ``start_inclusive`` flag lets an admission exactly at the discharge count, which
    the guarantee needs since the discharge may coincide with the end of the guaranteeing window):

        >>> cfg = TaskExtractorConfig(
        ...     predicates={
        ...         "A": PlainPredicateConfig("A"),
        ...         "admission": PlainPredicateConfig("ADMISSION"),
        ...         "discharge": PlainPredicateConfig("DISCHARGE"),
        ...     },
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "wait": WindowConfig(
        ...             "input.end", "start + 5d", False, True, has={"admission": "(1, None)"}
        ...         ),
        ...         "stay": WindowConfig("wait.end", "start -> discharge", False, True),
        ...         "last_day": WindowConfig("end - 1d", "stay.end", False, True, has={"A": "(1, None)"}),
        ...         "since_admission": WindowConfig(
        ...             "end <- admission", "stay.end", True, True, has={"A": "(None, 0)"}
        ...         ),
        ...         "target": WindowConfig("stay.end", "start + 30d", False, True, label="A"),
        ...     },
        ... )
        >>> normalize(cfg)
        Traceback (most recent call last):
            ...
        ValueError: Window 'last_day' looks back 1 day across the event-bounded window 'stay', so it cannot
        be written as a start-rooted chain.
        >>> universal = normalize(cfg, allow_end_rooted=True)
        >>> print_ACES(universal)
        trigger; **Prediction Time**
        ├── (+5 days, 0:00:00) wait.end
        │   ├── (next discharge) stay.end
        │   │   ├── (+30 days, 0:00:00) target.end; **Label: Presence of A**
        │   │   ├── (-1 day, 0:00:00) last_day.start (at least 1 A)
        │   │   └── (prior admission) since_admission.start (no A; at least 1 admission)
        │   └── (next discharge) stay_bounding_event.end (at least 1 discharge)
        └── (+5 days, 0:00:00) wait_criteria.end (at least 1 admission)
        >>> for b in universal.criteria_branches: print(b)
        wait: start + 5 days (admission: (1, None))
        wait: start + 5 days -> stay: start -> discharge -> last_day: end - 1 day (A: (1, None))
        wait: start + 5 days -> stay: start -> discharge -> since_admission: end <- admission (A: (None,
        0), admission: (1, None))
        wait: start + 5 days -> stay_bounding_event: start -> discharge (discharge: (1, None))
        >>> universal.windows["last_day"].start, universal.windows["last_day"].end
        ('end - 1 day', 'stay.end')

    Without the criterion guaranteeing an admission after the prediction time, the look-back to a prior
    admission could reach before it and is rejected even in the universal form:

        >>> cfg.windows["wait"].has = {}
        >>> cfg = TaskExtractorConfig(predicates=cfg.predicates, trigger=cfg.trigger, windows=cfg.windows)
        >>> normalize(cfg, allow_end_rooted=True)
        Traceback (most recent call last):
            ...
        ValueError: Window 'since_admission' looks back to a prior 'admission' event, which is not
        guaranteed to occur after the prediction time.
    """
    normalized_branches = branches(task_cfg, allow_end_rooted)

    trie = _Trie()
    for branch in normalized_branches:
        trie.insert(branch)

    windows: dict[str, WindowConfig] = {}
    used_names: set[str] = set()

    def unique(name: str, edge: Edge) -> str:
        candidate = f"{name}_criteria" if edge.is_constrained and name in used_names else name
        base, i = candidate, 2
        while candidate in used_names:
            candidate = f"{base}_{i}"
            i += 1
        used_names.add(candidate)
        return candidate

    def emit(node: _Trie, start: str, first: bool) -> None:
        # Unconstrained children first, so a shared prefix claims the original window name.
        children = sorted(node.children.values(), key=lambda c: c.edge.is_constrained)
        for child in children:
            edge = child.edge
            name = unique(edge.name, edge)
            if edge.is_end_rooted:
                start_str, end_str = edge.start_str(), start
            else:
                start_str, end_str = start, edge.end_str()
            windows[name] = WindowConfig(
                start=start_str,
                end=end_str,
                start_inclusive=edge.bound.left_inclusive,
                end_inclusive=edge.bound.right_inclusive,
                has={p: f"({lo}, {hi})" for p, (lo, hi) in edge.constraints},
                label=edge.label,
                index_timestamp="start" if first and child.on_label_path else None,
            )
            emit(child, f"{name}.{edge.exit_side}", False)

    emit(trie, "trigger", True)

    predicates: dict[str, Any] = copy.deepcopy(task_cfg.predicates)
    if index_predicate in predicates:
        raise ValueError(f"Predicate '{index_predicate}' already exists; choose another index_predicate.")
    predicates[index_predicate] = PlainPredicateConfig(index_code)

    normalized = NormalizedTaskConfig(
        predicates=predicates, trigger=EventConfig(index_predicate), windows=windows
    )
    normalized.branches = normalized_branches
    return normalized
