"""The normalized form of a future-only ACES task configuration.

A future-only task configuration describes, from the prediction time onward, a *target* (the label window)
and a set of *conditions* (inclusion / exclusion criteria on windows, and the requirement that event-bounded
windows be realizable). ACES lets these be tangled together: a single window can both carry criteria and be
the parent of the label window, and criteria can sit anywhere along a chain. The **normalized form** untangles
them without changing the label of any prediction-time row:

1. Every event-bounded window is given an explicit criterion that its bounding event occurs
   ([`require_bounding_events`][aces_relaxations.bounding_events.require_bounding_events]), so that nothing
   downstream has to know that ACES drops rows whose event-bounded windows cannot be realized.
2. The tree is re-rooted at the prediction time. Everything before it (the trigger, input-window criteria,
   look-back windows) selects the cohort but cannot change the label of a row in it, so it is discarded;
   branches that start by moving backward from the prediction time are discarded with a warning.
3. Every window that carries criteria is pulled out into its own *branch*: a chain of unconstrained copies of
   the windows leading to it, ending in the constrained window. The label window's branch carries no
   criteria, and unconstrained event-bounded leaf windows are kept as branches of their own.
4. Along each branch, runs of unconstrained, unlabeled windows whose composition is a single window collapse
   into it: two temporal windows into one, and a window followed by a hop to the next ``Q`` into one hop
   whenever ``Q`` is provably absent wherever the longer search would additionally look. A hop that lands
   where it starts drops out of the chain.
5. Redundant criteria and branches are removed (see ``aces_relaxations.redundancy``).
6. Each branch becomes its own chain of windows from the prediction time, copying any windows it has in
   common with other branches. The window tree is therefore a set of *sticks* hanging off the prediction
   time: each is a chain whose windows start where the previous one exits, and only its last window carries
   criteria or the label.

Windows defined backward from a later endpoint (``start: end - 1d``) are rewritten as forward chains when the
temporal windows before them cover the look-back (the *strict* form). The *universal* form
(``allow_end_rooted=True``) additionally keeps a look-back end-rooted when no forward rewrite exists but it is
guaranteed not to reach before the prediction time (see ``aces_relaxations.guarantees``).

Why this is label-preserving: with every edge after the prediction time moving forward in time (or looking
back by a guaranteed-safe amount), the timestamp of every window endpoint is a deterministic function of the
data and the prediction time, computed along its path from the root. Copying a path therefore realizes exactly
the same endpoints, so a criterion evaluated on a copy has exactly the same truth value, and the conjunction
of criteria across branches equals the original conjunction. Each merge and removal is justified where it is
made. The property tests check the whole against ACES on random configurations and data.
"""

import copy
import logging
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
from aces.types import END_OF_RECORD_KEY, TemporalWindowBounds, ToEventWindowBounds
from bigtree import Node

from .bounding_events import require_bounding_events
from .branches import Branch, Edge, bound_key, render_timedelta
from .degeneracy import Analysis
from .future_only import validate_future_only
from .guarantees import PathState
from .ordering import Interval
from .predicates import implication
from .redundancy import remove_redundancy
from .tree import backward_edge_reason, constraint_node_name, prediction_time_node, search_includes_anchor

logger = logging.getLogger(__name__)

INDEX_PREDICATE = "index_event"
INDEX_CODE = "_INDEX_EVENT_"


@dataclass
class NormalizedTaskConfig(TaskExtractorConfig):
    """A task configuration in normalized form.

    This is a genuine ACES configuration (its trigger is the synthetic index-event predicate marking the
    prediction time), with the normalized branches it was built from exposed on ``branches``.

    Attributes:
        branches: The normalized branches, label branch first.
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
                f"Window '{window_name}' looks back {render_timedelta(days)}, reaching before the "
                "prediction time."
            )
        top = out.pop()
        if not top.is_temporal or top.is_end_rooted:
            raise ValueError(
                f"Window '{window_name}' looks back {render_timedelta(days)} across the event-bounded "
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
    label_own = constraint_node_name(label_window, task_cfg.windows[label_window])
    label = task_cfg.windows[label_window].label

    path = list(leaf.node_path)
    steps = path[path.index(root) + 1 :]
    stack: list[Edge] = []
    state = PathState.at(task_cfg, root)

    def emit(edge: Edge, is_leaf: bool) -> Iterator[Branch]:
        if edge.is_constrained:
            yield Branch((*stack, replace(edge, label=None)))
        if edge.label is not None or (is_leaf and not edge.is_constrained):
            yield Branch((*stack, edge.unconstrained()))

    for i, node in enumerate(steps):
        window_name, side = node.node_name.rsplit(".", 1)
        window = task_cfg.windows[window_name]
        is_own = node.node_name == constraint_node_name(window_name, window)
        name = window_name if is_own else f"{window_name}_{side}_gap"
        raw_constraints = getattr(node, "constraints", None) or {}
        constraints = tuple(sorted((p, tuple(c)) for p, c in raw_constraints.items()))
        node_label = label if node.node_name == label_own else None
        is_leaf = i == len(steps) - 1
        bound = node.endpoint_expr
        before = state.pos

        counts = search_includes_anchor(window, side)
        if backward_edge_reason(node) is None:
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
            stack.append(replace(edge, constraints=(), label=None))
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
        edge = Edge(name, bound, constraints, node_label)
        state.add_criteria(raw_constraints, state.pos, before)
        yield from emit(edge, is_leaf)
        stack.append(replace(edge, constraints=(), label=None))


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
    if edge.is_temporal or edge.is_end_rooted or not counts:
        return False
    return any(
        g.interval == Interval.point(state.pos) and state.implies(g.predicate, edge.event)
        for g in state.guarantees
    )


def _edge_interval(analysis: Analysis, edge: Edge) -> Interval | None:
    """The interval of an edge that is one of the analyzed configuration's forward windows (else ``None``)."""
    node = analysis.task_cfg.window_nodes.get(f"{edge.name}.end")
    if edge.is_end_rooted or node is None or bound_key(node.endpoint_expr) != bound_key(edge.bound):
        return None
    return analysis.window(edge.name)


def _forbidden(analysis: Analysis, predicate: str, region: Interval) -> bool:
    """Whether some criterion or hop provably forbids ``predicate`` over all of ``region``.

    Examples:
        >>> analysis = Analysis.of(in_hosp_mortality_cfg)
        >>> gap = analysis.window("gap")
        >>> _forbidden(analysis, "admission", gap), _forbidden(analysis, "death", gap)
        (True, True)
        >>> _forbidden(analysis, "death", analysis.window("target"))
        False

    Nothing occurs in an empty region:

        >>> _forbidden(analysis, "death", Interval(gap.lo, gap.lo, False, False))
        True
    """
    return analysis.ordering.is_empty(region) or analysis.forbidden_over(predicate, region) is not None


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
    stops_there = not prev.is_temporal and prev.bound.right_inclusive and analysis.implies(q, prev.event)
    if stops_there or _forbidden(analysis, q, Interval.point(region.lo)):
        return inclusive
    if not inclusive or analysis.ordering.leq(region.lo, region.hi, strict=True):
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
        if plain and not edge.is_temporal and edge.event != END_OF_RECORD_KEY and regions[-1] is not None:
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
            window_name, side = child.node_name.rsplit(".", 1)
            counts = search_includes_anchor(task_cfg.windows[window_name], side)
            guaranteed = isinstance(bound, ToEventWindowBounds) and (
                PathState.at(task_cfg, root).guarantee_for(bound.end_event[1:], counts) is not None
            )
            if not guaranteed:
                logger.warning(
                    f"Discarding window branch at {child.node_name}, which starts before the prediction "
                    f"time ({reason})."
                )
                continue
        leaves.extend([child] if child.is_leaf else child.leaves)
    return leaves


def _branches(task_cfg: TaskExtractorConfig, allow_end_rooted: bool = False) -> tuple[Branch, ...]:
    """Computes the normalized branches of a future-only task configuration.

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
        >>> for b in _branches(in_hosp_mortality_cfg): print(b)
        gap: start + 1 day -> target: start -> discharge_or_death; label=death
        gap: start + 1 day (admission: (None, 0), discharge_or_death: (None, 0))
        gap: start + 1 day -> target_bounding_event: start -> discharge_or_death (discharge_or_death: (1,
        None))
        >>> for b in _branches(post_discharge_mortality_cfg): print(b)
        hospitalization: start -> discharge -> gap: start + 1 day -> target: start + 29 days; label=death
        post_input: start + 1 day (admission: (None, 0), discharge_or_death: (None, 0))
        hospitalization: start -> discharge (death: (None, 0))
        hospitalization: start -> discharge -> gap: start + 1 day (admission: (None, 0), death: (None, 0))
        hospitalization_bounding_event: start -> discharge (discharge: (1, None))
        >>> for b in _branches(readmission_cfg): print(b)
        gap: start + 1 day -> target: start + 29 days; label=admission
        gap: start + 1 day (admission: (None, 0), death: (None, 0))
        gap__target: start + 30 days -> censoring_protection: start -> end of record (_ANY_EVENT: (1, None))
        >>> for b in _branches(two_stage_cfg): print(b)
        1st_infusion: start -> infusion_end -> 2nd_infusion_start_gap: start -> infusion_start ->
        2nd_infusion: start -> infusion_end; label=adverse_event
        1st_infusion: start -> infusion_end (adverse_event: (None, 0), infusion_end: (1, None))
        1st_infusion: start -> infusion_end -> 2nd_infusion_start_gap: start -> infusion_start ->
        2nd_infusion: start -> infusion_end (infusion_end: (1, None), infusion_start: (1, None))
    """
    validate_future_only(task_cfg)
    task_cfg = require_bounding_events(task_cfg)
    analysis = Analysis.of(task_cfg)
    root = prediction_time_node(task_cfg)

    seen: dict[Branch, None] = {}
    for leaf in _future_leaves(task_cfg, root):
        for branch in _walk(task_cfg, root, leaf, allow_end_rooted):
            seen[_collapse(branch, analysis)] = None

    at_pt = [g.predicate for g in PathState.at(task_cfg, root).guarantees]
    ordered = list(remove_redundancy(seen, implication(task_cfg), at_pt))
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
        │   └── (next discharge_or_death) target.end; **Label: Presence of death**
        ├── (+1 day, 0:00:00) gap_criteria.end (no admission, discharge_or_death)
        └── (+1 day, 0:00:00) target_bounding_event__gap.end
            └── (next discharge_or_death) target_bounding_event.end (at least 1 discharge_or_death)
        >>> print(normalized.label_branch)
        gap: start + 1 day -> target: start -> discharge_or_death; label=death
        >>> normalized.windows["gap_criteria"].start, normalized.windows["gap_criteria"].end
        ('trigger', 'start + 1 day')
        >>> normalized.trigger
        EventConfig(predicate='index_event')

    Every branch is its own chain of windows, even where it starts like another; windows copied into a
    condition branch are prefixed with the name of the branch's last window:

        >>> print_ACES(normalize(post_discharge_mortality_cfg))
        trigger; **Prediction Time**
        ├── (next discharge) hospitalization.end
        │   └── (+1 day, 0:00:00) gap.end
        │       └── (+29 days, 0:00:00) target.end; **Label: Presence of death**
        ├── (+1 day, 0:00:00) post_input.end (no admission, discharge_or_death)
        ├── (next discharge) hospitalization_criteria.end (no death)
        ├── (next discharge) gap_criteria__hospitalization.end
        │   └── (+1 day, 0:00:00) gap_criteria.end (no admission, death)
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
        ├── (+10 days, 0:00:00) gap1__gap2.end
        │   └── (+5 days, 0:00:00) target.end; **Label: Presence of B**
        ├── (+10 days, 0:00:00) watch__gap1__gap2.end
        │   └── (next B) watch.end (no A)
        └── (+10 days, 0:00:00) watch_bounding_event__gap1__gap2.end
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
        ├── (next A_or_B) either.end (at least 1 A_or_B)
        └── (next A_or_B) a_bounding_event__either.end
            └── (next A) a_bounding_event.end (at least 1 A)

    (``again``'s own "bounding event occurs" criterion was implied by ``either``'s landing, which
    the ``either`` branch asserts, and was removed; see ``aces_relaxations.redundancy``.)

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
        └── (+6 days, 0:00:00) last_day__week__pre_last_day.end
            └── (+1 day, 0:00:00) last_day.end (at least 1 A)

    Window names are kept where possible; a criteria copy of window ``w`` is named ``w_criteria``, and names
    that are nonetheless taken get a numeric suffix:

        >>> cfg = TaskExtractorConfig(
        ...     predicates={c: PlainPredicateConfig(c) for c in "ABC"},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "w": WindowConfig("input.end", "start -> B", False, True, has={"C": "(None, 0)"}),
        ...         "w_criteria": WindowConfig("w.end", "start + 3d", False, True, label="C"),
        ...     },
        ... )
        >>> print_ACES(normalize(cfg))
        trigger; **Prediction Time**
        ├── (next B) w.end
        │   └── (+3 days, 0:00:00) w_criteria.end; **Label: Presence of C**
        ├── (next B) w_criteria_2.end (no C)
        └── (next B) w_bounding_event.end (at least 1 B)

    The synthetic index predicate must not collide with an existing one:

        >>> normalize(in_hosp_mortality_cfg, index_predicate="death")
        Traceback (most recent call last):
            ...
        ValueError: Predicate 'death' already exists; choose another index_predicate.

    Branches that start by moving backward from the prediction time lie before it and are discarded (with
    a warning); a window that could reach before the prediction time is rejected:

        >>> print_ACES(normalize(readmission_cfg))
        trigger; **Prediction Time**
        ├── (+1 day, 0:00:00) gap.end
        │   └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
        ├── (+1 day, 0:00:00) gap_criteria.end (no admission, death)
        └── (+30 days, 0:00:00) censoring_protection__gap__target.end
            └── (end of record) censoring_protection.end (at least 1 event(s))
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
        │   └── (next discharge) stay.end
        │       └── (+30 days, 0:00:00) target.end; **Label: Presence of A**
        ├── (+5 days, 0:00:00) wait_criteria.end (at least 1 admission)
        ├── (+5 days, 0:00:00) last_day__wait.end
        │   └── (next discharge) last_day__stay.end
        │       └── (-1 day, 0:00:00) last_day.start (at least 1 A)
        ├── (+5 days, 0:00:00) since_admission__wait.end
        │   └── (next discharge) since_admission__stay.end
        │       └── (prior admission) since_admission.start (no A; at least 1 admission)
        └── (+5 days, 0:00:00) stay_bounding_event__wait.end
            └── (next discharge) stay_bounding_event.end (at least 1 discharge)
        >>> for b in universal.criteria_branches: print(b)
        wait: start + 5 days (admission: (1, None))
        wait: start + 5 days -> stay: start -> discharge -> last_day: end - 1 day (A: (1, None))
        wait: start + 5 days -> stay: start -> discharge -> since_admission: end <- admission (A: (None,
        0), admission: (1, None))
        wait: start + 5 days -> stay_bounding_event: start -> discharge (discharge: (1, None))
        >>> universal.windows["last_day"].start, universal.windows["last_day"].end
        ('end - 1 day', 'last_day__stay.end')

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
    normalized_branches = _branches(task_cfg, allow_end_rooted)

    windows: dict[str, WindowConfig] = {}

    def unique(name: str) -> str:
        candidate, i = name, 2
        while candidate in windows:
            candidate = f"{name}_{i}"
            i += 1
        return candidate

    for branch in normalized_branches:
        # The label branch claims the original window names. A condition branch is named for its last window
        # (``<window>_criteria`` when the label branch already has that name), and the copies of the windows
        # leading to it are prefixed with that name.
        terminal = branch.terminal
        if branch.is_label_branch:
            names = [unique(edge.name) for edge in branch.edges]
        else:
            last = terminal.name
            if last in windows and terminal.is_constrained:
                last = f"{last}_criteria"
            last = unique(last)
            names = [unique(f"{last}__{edge.name}") for edge in branch.edges[:-1]] + [last]
        start = "trigger"
        for i, (name, edge) in enumerate(zip(names, branch.edges, strict=True)):
            start_str, end_str = (edge.start_str(), start) if edge.is_end_rooted else (start, edge.end_str())
            windows[name] = WindowConfig(
                start=start_str,
                end=end_str,
                start_inclusive=edge.bound.left_inclusive,
                end_inclusive=edge.bound.right_inclusive,
                has={p: f"({lo}, {hi})" for p, (lo, hi) in edge.constraints},
                label=edge.label,
                index_timestamp="start" if i == 0 and branch.is_label_branch else None,
            )
            start = f"{name}.{edge.exit_side}"

    predicates: dict[str, Any] = copy.deepcopy(task_cfg.predicates)
    if index_predicate in predicates:
        raise ValueError(f"Predicate '{index_predicate}' already exists; choose another index_predicate.")
    predicates[index_predicate] = PlainPredicateConfig(index_code)

    return NormalizedTaskConfig(
        predicates=predicates,
        trigger=EventConfig(index_predicate),
        windows=windows,
        branches=normalized_branches,
    )
