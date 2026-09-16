"""The "guaranteeably future-only" property of ACES task configurations.

A task configuration is *guaranteeably future-only* when its label can only depend on data observed strictly
after (or at) the prediction time. Structurally, this means that both endpoints of the label window are
reachable from the prediction-time node in the ACES window tree, and that every edge on those paths moves
forward in time (a non-negative temporal offset, or a bound to a *next* rather than *prior* event).

This is a conservative, purely structural check (see ``aces_relaxations.guarantees``). A look-back by a
fixed offset is accepted when the position it reaches is guaranteed not to precede the prediction time; a
look-back to a prior event is accepted when that event is guaranteed to occur between the prediction time
and the look-back's end, because the path hopped to its next occurrence or a criterion on the path requires
it. The check never accepts a configuration whose label could reach into the past of the prediction time.

Windows hanging off the prediction-time-to-label path (rather than lying on it, or after the label) are not
inspected here; [`normalize`][aces_relaxations.normalization.normalize] checks each such branch on its own
when it re-roots the configuration at the prediction time.
"""

from contextlib import redirect_stdout
from io import StringIO

from aces.config import TaskExtractorConfig
from aces.types import TemporalWindowBounds, ToEventWindowBounds
from bigtree import print_tree

from .guarantees import PathState
from .tree import (
    backward_edge_reason,
    endpoint_node,
    label_node,
    prediction_time_node,
    search_includes_anchor,
)


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

    A configuration lacking a label or prediction time is not future-only, nor is one whose label or
    prediction-time window is named ``trigger`` (a name that collides with the trigger node):

        >>> from aces.config import EventConfig, PlainPredicateConfig, WindowConfig
        >>> predicates = {"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")}
        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={"w": WindowConfig("trigger", "start + 1d", True, True)},
        ... )
        >>> for v in future_only_violations(cfg): print(v)
        The task configuration must have a label_window defined.
        The task configuration must have a index_timestamp_window defined.
        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "trigger": WindowConfig(
        ...             "trigger", "start + 1d", True, True, label="B", index_timestamp="start"
        ...         )
        ...     },
        ... )
        >>> for v in future_only_violations(cfg): print(v)
        The task configuration must not have a label_window set to 'trigger'.
        The task configuration must not have a index_timestamp_window set to 'trigger'.

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

    pt = prediction_time_node(task_cfg)
    label_ends = (
        ("end", label_node(task_cfg)),
        ("start", endpoint_node(task_cfg, task_cfg.label_window, "start")),
    )
    for label_side, node in label_ends:
        if node is not pt and pt not in node.ancestors:
            violations[
                f"The prediction time window node ({pt.node_name}) is not an ancestor of the label window "
                f"{label_side} node ({node.node_name})."
            ] = None
            continue

        path = list(node.node_path)
        state = PathState.at(task_cfg, pt)
        for step in path[path.index(pt) + 1 :]:
            bound = step.endpoint_expr
            reason = backward_edge_reason(step)
            before = state.pos
            window_name, side = step.node_name.rsplit(".", 1)
            counts = search_includes_anchor(task_cfg.windows[window_name], side)
            criteria = getattr(step, "constraints", None) or {}
            if reason is None:
                if isinstance(bound, TemporalWindowBounds):
                    state.advance_temporal(bound.window_size)
                else:
                    state.advance_event(bound.end_event, strict=not counts)
                state.add_criteria(criteria, before, state.pos)
                continue
            if isinstance(bound, TemporalWindowBounds) and state.can_look_back(-bound.window_size):
                state.look_back_temporal(-bound.window_size)
                state.add_criteria(criteria, state.pos, before)
                continue
            if isinstance(bound, ToEventWindowBounds) and bound.end_event.startswith("-"):
                guarantee = state.guarantee_for(bound.end_event[1:], counts)
                if guarantee is not None:
                    state.look_back_event(guarantee)
                    state.add_criteria(criteria, state.pos, before)
                    continue
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
