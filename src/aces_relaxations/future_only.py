"""The "guaranteeably future-only" property of ACES task configurations.

A task configuration is *guaranteeably future-only* when its label can only depend on data observed strictly
after (or at) the prediction time. Structurally, this means that both endpoints of the label window are
reachable from the prediction-time node in the ACES window tree, and that every edge on those paths moves
forward in time (a non-negative temporal offset, or a bound to a *next* rather than *prior* event).

This is a conservative, purely structural check: it can reject configurations that happen to be future-only
in practice (e.g. a label window whose start is ``end - 1d`` where the end is ``prediction_time + 1d``), but
it never accepts a configuration whose label could reach into the past of the prediction time.

Windows hanging off the prediction-time-to-label path (rather than lying on it, or after the label) are not
inspected, because [`strip_to_relevant_windows`][aces_relaxations.relaxations.strip_to_relevant_windows]
prunes them from the relaxed configuration.
"""

from contextlib import redirect_stdout
from datetime import timedelta
from io import StringIO

from aces.config import TaskExtractorConfig
from aces.types import TemporalWindowBounds, ToEventWindowBounds
from bigtree import Node, print_tree

from .tree import WindowNode, label_node, prediction_time_node, resolve_node


def backward_edge_reason(node: Node) -> str | None:
    """Explains why the edge into ``node`` from its parent can move backward in time, if it can.

    Args:
        node: A node in an ACES window tree.

    Returns:
        A human-readable reason if the edge can move backward in time, otherwise ``None``. The root (with no
        ``endpoint_expr``) never moves backward.

    Examples:
        >>> from bigtree import Node
        >>> print(backward_edge_reason(Node("trigger")))
        None
        >>> print(backward_edge_reason(
        ...     Node("a", endpoint_expr=TemporalWindowBounds(True, timedelta(days=1), True, None))
        ... ))
        None
        >>> print(backward_edge_reason(
        ...     Node("a", endpoint_expr=TemporalWindowBounds(True, timedelta(days=-1), True, None))
        ... ))
        temporal bound -1 day, 0:00:00 is negative
        >>> bound = TemporalWindowBounds(True, timedelta(days=1), True, timedelta(days=-2))
        >>> print(backward_edge_reason(Node("a", endpoint_expr=bound)))
        temporal offset -2 days, 0:00:00 is negative
        >>> bound = ToEventWindowBounds(True, "death", True, None)
        >>> print(backward_edge_reason(Node("a", endpoint_expr=bound)))
        None
        >>> bound = ToEventWindowBounds(True, "-death", True, None)
        >>> print(backward_edge_reason(Node("a", endpoint_expr=bound)))
        event bound targets the prior 'death' event
        >>> print(backward_edge_reason(
        ...     Node("a", endpoint_expr=ToEventWindowBounds(True, "-_RECORD_START", True, None))
        ... ))
        event bound targets the start of the record
        >>> print(backward_edge_reason(
        ...     Node("a", endpoint_expr=ToEventWindowBounds(True, "death", True, timedelta(hours=-1)))
        ... ))
        event offset -1 day, 23:00:00 is negative
        >>> backward_edge_reason(Node("a", endpoint_expr="foo"))
        Traceback (most recent call last):
            ...
        ValueError: Unexpected type <class 'str'> for endpoint expr: foo.
    """
    match getattr(node, "endpoint_expr", None):
        case None:
            return None
        case TemporalWindowBounds() as time_bound:
            if time_bound.window_size < timedelta(0):
                return f"temporal bound {time_bound.window_size} is negative"
            if time_bound.offset is not None and time_bound.offset < timedelta(0):
                return f"temporal offset {time_bound.offset} is negative"
            return None
        case ToEventWindowBounds() as event_bound:
            if event_bound.end_event == "-_RECORD_START":
                return "event bound targets the start of the record"
            if event_bound.end_event.startswith("-"):
                return f"event bound targets the prior '{event_bound.end_event[1:]}' event"
            if event_bound.offset is not None and event_bound.offset < timedelta(0):
                return f"event offset {event_bound.offset} is negative"
            return None
        case _ as other:
            raise ValueError(f"Unexpected type {type(other)} for endpoint expr: {other}.")


def future_only_violations(task_cfg: TaskExtractorConfig) -> list[str]:
    """Lists the reasons a task configuration is not guaranteeably future-only.

    Args:
        task_cfg: The ACES task configuration to inspect.

    Returns:
        A (possibly empty) list of human-readable violations. An empty list means the configuration is
        guaranteeably future-only.

    Examples:
        >>> future_only_violations(sample_ACES_cfg)
        []
        >>> future_only_violations(readmission_cfg)
        []

    A configuration lacking a label or prediction time is not future-only:

        >>> from aces.config import EventConfig, PlainPredicateConfig, WindowConfig
        >>> predicates = {"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")}
        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={"w": WindowConfig("trigger", "start + 1d", True, True)},
        ... )
        >>> for v in future_only_violations(cfg): print(v)
        The task configuration must have a label_window defined.
        The task configuration must have a index_timestamp_window defined.

    Nor is one where the label window is not reachable from the prediction time:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger + 1h", "start + 1h", True, True, index_timestamp="start"),
        ...         "target": WindowConfig("end - 1h", "trigger + 4h", False, True, label="B"),
        ...     },
        ... )
        >>> for v in future_only_violations(cfg): print(v)
        The prediction time window node (input.start) is not an ancestor of the label window end node
        (target.end).
        The prediction time window node (input.start) is not an ancestor of the label window start node
        (target.start).

    Nor is one where the path from the prediction time to the label can move backward in time. Here, the
    target window starts one day *before* the end of the input window:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "target": WindowConfig("input.end - 1d", "start + 3d", False, True, label="B"),
        ...     },
        ... )
        >>> for v in future_only_violations(cfg): print(v)
        The edge into target.start on the path from the prediction time to the label window can move
        backward in time: temporal bound -1 day, 0:00:00 is negative.

    And here the label window ends at the *prior* admission relative to the prediction time:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "target": WindowConfig("end <- A", "input.end", False, True, label="B"),
        ...     },
        ... )
        >>> for v in future_only_violations(cfg): print(v)
        The edge into target.start on the path from the prediction time to the label window can move
        backward in time: event bound targets the prior 'A' event.
    """
    violations: dict[str, None] = {}

    for k in ("label_window", "index_timestamp_window"):
        val = getattr(task_cfg, k)
        if val is None:
            violations[f"The task configuration must have a {k} defined."] = None
        elif val == "trigger":
            violations[f"The task configuration must not have a {k} set to 'trigger'."] = None

    if violations:
        return list(violations)

    pt = task_cfg.window_nodes[prediction_time_node(task_cfg).node_name]
    label_end = task_cfg.window_nodes[label_node(task_cfg).node_name]
    label_start_ref = resolve_node(task_cfg, root_node=WindowNode(task_cfg.label_window, "start"))
    label_start = task_cfg.window_nodes[label_start_ref.node_name]

    for side, node in (("end", label_end), ("start", label_start)):
        if node is not pt and pt not in node.ancestors:
            violations[
                f"The prediction time window node ({pt.node_name}) is not an ancestor of the label window "
                f"{side} node ({node.node_name})."
            ] = None
            continue

        path = list(node.node_path)
        for step in path[path.index(pt) + 1 :]:
            reason = backward_edge_reason(step)
            if reason is not None:
                violations[
                    f"The edge into {step.node_name} on the path from the prediction time to the label "
                    f"window can move backward in time: {reason}."
                ] = None

    return list(violations)


def is_future_only(task_cfg: TaskExtractorConfig) -> bool:
    """Returns whether the task configuration is guaranteeably future-only.

    See [`future_only_violations`][aces_relaxations.future_only.future_only_violations] for the definition.

    Args:
        task_cfg: The ACES task configuration to inspect.

    Returns:
        ``True`` if the label can only depend on data at or after the prediction time.

    Examples:
        >>> is_future_only(sample_ACES_cfg)
        True
        >>> is_future_only(in_hosp_mortality_cfg)
        True
        >>> is_future_only(post_discharge_mortality_cfg)
        True
        >>> is_future_only(readmission_cfg)
        True
        >>> is_future_only(two_stage_cfg)
        True
        >>> from aces.config import EventConfig, PlainPredicateConfig, WindowConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "target": WindowConfig("end - 7d", "trigger", False, True, label="B"),
        ...     },
        ... )
        >>> is_future_only(cfg)
        False
    """
    return not future_only_violations(task_cfg)


def validate_future_only(task_cfg: TaskExtractorConfig) -> None:
    """Raises if the task configuration is not guaranteeably future-only.

    Args:
        task_cfg: The ACES task configuration to validate.

    Raises:
        ValueError: If the configuration is not future-only. The message lists every violation and includes
            the window tree for debugging.

    Examples:
        >>> validate_future_only(sample_ACES_cfg)
        >>> from aces.config import EventConfig, PlainPredicateConfig, WindowConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "target": WindowConfig("end - 7d", "trigger", False, True, label="B"),
        ...     },
        ... )
        >>> validate_future_only(cfg)
        Traceback (most recent call last):
            ...
        ValueError: The task configuration is not guaranteeably future-only:
          - The prediction time window node (input.end) is not an ancestor of the label window end node
            (trigger).
          - The prediction time window node (input.end) is not an ancestor of the label window start node
            (target.start).
        Got tree:
        trigger
        ├── input.end
        └── target.start
        <BLANKLINE>
    """
    violations = future_only_violations(task_cfg)
    if not violations:
        return

    strio = StringIO()
    with redirect_stdout(strio):
        print_tree(task_cfg.window_tree)

    lines = "\n".join(f"  - {v}" for v in violations)
    raise ValueError(
        f"The task configuration is not guaranteeably future-only:\n{lines}\nGot tree:\n{strio.getvalue()}"
    )
