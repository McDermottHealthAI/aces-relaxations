"""Normalized form for future-only ACES task configurations.

A future-only task configuration describes, from the prediction time onward, a *target* (the label window)
and a set of *conditions* (inclusion / exclusion criteria on windows, and the requirement that event-bounded
windows be realizable). ACES lets these be tangled together: a single window can both carry criteria and be
the parent of the label window, and criteria can sit anywhere along a chain. The **normalized form** untangles
them without changing the label of any prediction-time row:

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

from .future_only import backward_edge_reason, validate_future_only
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
    """

    name: str
    bound: Bound
    constraints: tuple[tuple[str, Constraint], ...] = ()
    label: str | None = None

    @property
    def is_temporal(self) -> bool:
        """Whether the end is a fixed offset from the start."""
        return isinstance(self.bound, TemporalWindowBounds)

    @property
    def is_constrained(self) -> bool:
        """Whether the window carries criteria."""
        return bool(self.constraints)

    @property
    def days(self) -> float | None:
        """The window's duration in days, for temporal windows."""
        if not self.is_temporal:
            return None
        return self.bound.window_size.total_seconds() / 86400

    @property
    def event(self) -> str | None:
        """The predicate whose next occurrence ends the window, for event-bounded windows."""
        return None if self.is_temporal else self.bound.end_event

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
        """The window's ``end`` in ACES syntax (``None`` for "end of record")."""
        if self.is_temporal:
            return f"start + {_render_timedelta(self.bound.window_size)}"
        if self.event == RECORD_END:
            return None
        return f"start -> {self.event}"

    def __str__(self) -> str:
        """Renders the edge in ACES-like syntax."""
        end = self.end_str() or "start -> end of record"
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
        ValueError: Window 'b' looks back 1 day across the event-bounded window 'e', so it is not
        guaranteeably future-only.
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
        if not top.is_temporal:
            raise ValueError(
                f"Window '{window_name}' looks back {_render_timedelta(days)} across the event-bounded "
                f"window '{top.name}', so it is not guaranteeably future-only."
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


def _walk(task_cfg: TaskExtractorConfig, root: Node, leaf: Node) -> Iterator[Branch]:
    """Yields the branches contributed by one root-to-leaf path of the prediction-time subtree.

    The walk keeps a chain of unconstrained windows from the prediction time to the current position.
    Forward edges extend it; a window carrying criteria contributes a branch ending in that window; the
    label window contributes the label branch; an unconstrained leaf contributes its (realizability)
    branch. A backward temporal edge rewinds the position instead (see ``_rewind``): a window defined from
    its end backward is emitted as a forward window from the rewound position, and its descendants continue
    from there.

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
    for i, node in enumerate(steps):
        window_name, side = node.node_name.rsplit(".", 1)
        window = task_cfg.windows[window_name]
        is_own = node.node_name == _own_node_name(window_name, window)
        constraints = tuple(
            sorted((p, tuple(c)) for p, c in (getattr(node, "constraints", None) or {}).items())
        )
        node_label = label if node.node_name == label_own else None
        is_leaf = i == len(steps) - 1
        bound = node.endpoint_expr

        if backward_edge_reason(node) is None:
            name = window_name if is_own else f"{window_name}_{side}_gap"
            edge = Edge(name, bound, constraints, node_label)
            if edge.is_constrained:
                yield Branch((*stack, replace(edge, label=None)))
            if node_label is not None or (is_leaf and not edge.is_constrained):
                yield Branch((*stack, edge.unconstrained()))
            stack.append(edge.unconstrained())
            continue

        if not isinstance(bound, TemporalWindowBounds):
            raise ValueError(
                f"Window '{window_name}' is bound to a prior event after the prediction time, so it is not "
                "guaranteeably future-only."
            )
        days = -bound.window_size
        stack = _rewind(stack, days, window_name)
        if is_own:
            own_bound = TemporalWindowBounds(window.start_inclusive, days, window.end_inclusive)
            edge = Edge(window_name, own_bound, constraints, node_label)
            if edge.is_constrained:
                yield Branch((*stack, replace(edge, label=None)))
            if node_label is not None:
                yield Branch((*stack, edge.unconstrained()))


def _collapse(branch: Branch) -> Branch:
    """Merges runs of unconstrained, unlabeled, temporal windows.

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
    """
    merged: list[Edge] = []
    for edge in branch.edges:
        prev = merged[-1] if merged else None
        mergeable = (
            prev is not None
            and prev.is_temporal
            and edge.is_temporal
            and not prev.is_constrained
            and not edge.is_constrained
            and prev.label is None
            and edge.label is None
        )
        if mergeable:
            total = prev.bound.window_size + edge.bound.window_size
            bound = TemporalWindowBounds(prev.bound.left_inclusive, total, edge.bound.right_inclusive)
            merged[-1] = Edge(f"{prev.name}__{edge.name}", bound)
        else:
            merged.append(edge)
    return Branch(tuple(merged))


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


def _future_leaves(root: Node) -> list[Node]:
    """The leaves of the prediction-time subtree, excluding branches that start by moving backward.

    A child of the prediction-time node reached by a backward edge lies (at least partly) before the
    prediction time: it is cohort selection, not a condition on the future, and is discarded with a warning.
    """
    leaves: list[Node] = []
    for child in root.children:
        reason = backward_edge_reason(child)
        if reason is not None:
            logger.warning(
                f"Discarding window branch at {child.node_name}, which starts before the prediction time "
                f"({reason})."
            )
            continue
        leaves.extend([child] if child.is_leaf else child.leaves)
    return leaves


def branches(task_cfg: TaskExtractorConfig) -> tuple[Branch, ...]:
    """Computes the normalized branches of a future-only task configuration, without building a config.

    Args:
        task_cfg: The task configuration.

    Returns:
        The branches, label branch first, then the others in tree order.

    Raises:
        ValueError: If the configuration is not future-only, or a window after the prediction time is not
            guaranteeably future-only (it looks back further than the guaranteed elapsed time, or across an
            event-bounded window).

    Examples:
        >>> for b in branches(in_hosp_mortality_cfg): print(b)
        gap: start + 1 day -> target: start -> discharge_or_death; label=death
        gap: start + 1 day (admission: (None, 0), discharge_or_death: (None, 0))
        >>> for b in branches(post_discharge_mortality_cfg): print(b)
        hospitalization: start -> discharge -> gap: start + 1 day -> target: start + 29 days; label=death
        post_input: start + 1 day (admission: (None, 0), discharge_or_death: (None, 0))
        hospitalization: start -> discharge (death: (None, 0))
        hospitalization: start -> discharge -> gap: start + 1 day (admission: (None, 0), death: (None, 0))
        >>> for b in branches(readmission_cfg): print(b)
        gap: start + 1 day -> target: start + 29 days; label=admission
        gap: start + 1 day (admission: (None, 0), death: (None, 0))
        gap: start + 1 day -> target: start + 29 days; label=admission -> censoring_protection: start ->
        end of record (_ANY_EVENT: (1, None))
        >>> for b in branches(two_stage_cfg): print(b)
        1st_infusion: start -> infusion_end -> 2nd_infusion_start_gap: start -> infusion_start ->
        2nd_infusion: start -> infusion_end; label=adverse_event
        1st_infusion: start -> infusion_end (adverse_event: (None, 0))
    """
    validate_future_only(task_cfg)
    root = task_cfg.window_nodes[prediction_time_node(task_cfg).node_name]

    seen: dict[Branch, None] = {}
    for leaf in _future_leaves(root):
        for branch in _walk(task_cfg, root, leaf):
            seen[_collapse(branch)] = None

    ordered = list(seen)
    return tuple(sorted(ordered, key=lambda b: (not b.is_label_branch, ordered.index(b))))


def normalize(
    task_cfg: TaskExtractorConfig, index_predicate: str = INDEX_PREDICATE, index_code: str = INDEX_CODE
) -> NormalizedTaskConfig:
    """Converts a future-only task configuration into normalized form.

    Args:
        task_cfg: The task configuration. Its label window must be reachable from the prediction time
            through forward-only edges, and every window branching from the prediction time must be
            future-only.
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
        └── (next discharge) hospitalization_criteria.end (no death)

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
            └── (next B) watch.end (no A)

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
    """
    normalized_branches = branches(task_cfg)

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
            windows[name] = WindowConfig(
                start=start,
                end=edge.end_str(),
                start_inclusive=edge.bound.left_inclusive,
                end_inclusive=edge.bound.right_inclusive,
                has={p: f"({lo}, {hi})" for p, (lo, hi) in edge.constraints},
                label=edge.label,
                index_timestamp="start" if first and child.on_label_path else None,
            )
            emit(child, f"{name}.end", False)

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
