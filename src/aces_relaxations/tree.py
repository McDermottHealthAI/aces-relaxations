"""Helpers for the window tree of an ACES task configuration.

ACES compiles a task configuration into a tree of window *endpoint nodes* (a [`bigtree`](
https://bigtree.readthedocs.io/) tree rooted at the ``trigger`` node). Each non-root node carries an
``endpoint_expr`` describing how it is reached from its parent (a temporal offset or an event bound) and a
``constraints`` dictionary of the criteria of the window it closes.

Not every window endpoint is present in that tree: an endpoint that is simply *equal* to another node (e.g.
``start: trigger`` or ``start: gap.end``) is elided. The helpers here resolve ``<window>.<start|end>``
references onto the nodes that actually exist, and state the semantics of a tree edge (its direction, and
whether an event search counts an event at its own starting point) that the analyses in this package rely on.
Everything in this module is a property of ACES configurations alone, and is proposed for ACES itself in
[ACES#213](https://github.com/justin13601/ACES/issues/213).
"""

from datetime import timedelta

from aces.config import TaskExtractorConfig, WindowConfig
from aces.types import END_OF_RECORD_KEY, START_OF_RECORD_KEY, TemporalWindowBounds, ToEventWindowBounds
from bigtree import Node

RECORD_BOUNDARIES = (START_OF_RECORD_KEY, END_OF_RECORD_KEY)


def endpoint_node(task_cfg: TaskExtractorConfig, window: str, side: str) -> Node:
    """Returns the node of the window tree at which a window endpoint is realized.

    Args:
        task_cfg: The task configuration.
        window: The window name.
        side: ``"start"`` or ``"end"``.

    Returns:
        The tree node the endpoint resolves to, following references through elided endpoints.

    Examples:
        >>> from bigtree import print_tree
        >>> print_tree(sample_ACES_cfg.window_tree)
        trigger
        └── input.end
            └── gap.end
                └── target.end
        >>> endpoint_node(sample_ACES_cfg, "input", "start").node_name
        'trigger'
        >>> endpoint_node(sample_ACES_cfg, "gap", "start").node_name
        'input.end'
        >>> endpoint_node(sample_ACES_cfg, "target", "end").node_name
        'target.end'

    References chain through any number of elided endpoints, including a window's own other side:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=sample_ACES_cfg.predicates,
        ...     trigger=sample_ACES_cfg.trigger,
        ...     windows={
        ...         "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...         "target": WindowConfig("gap.end", "start + 24h", False, True),
        ...         "post": WindowConfig("target.start", "start + 24h", False, True),
        ...         "back": WindowConfig("end - 24h", "post.start", False, False),
        ...     },
        ... )
        >>> [endpoint_node(cfg, w, s).node_name for w, s in [("back", "end"), ("back", "start")]]
        ['gap.end', 'back.start']
    """
    while side is not None and f"{window}.{side}" not in task_cfg.window_nodes:
        ref = (
            task_cfg.windows[window]._parsed_start
            if side == "start"
            else task_cfg.windows[window]._parsed_end
        )
        ref = ref["referenced"]
        if ref == "trigger":
            window, side = "trigger", None
        elif ref in ("start", "end"):
            side = ref
        else:
            window, side = ref.split(".")
    return task_cfg.window_tree if side is None else task_cfg.window_nodes[f"{window}.{side}"]


def prediction_time_node(task_cfg: TaskExtractorConfig) -> Node:
    """Returns the tree node at which the prediction time (``index_timestamp``) of the task is realized.

    Raises:
        ValueError: If the configuration has no ``index_timestamp`` window.

    Examples:
        >>> prediction_time_node(sample_ACES_cfg).node_name
        'input.end'
        >>> prediction_time_node(readmission_cfg).node_name
        'trigger'
        >>> cfg = TaskExtractorConfig(
        ...     predicates=sample_ACES_cfg.predicates,
        ...     trigger=sample_ACES_cfg.trigger,
        ...     windows={"w": WindowConfig("trigger", "start + 1d", True, True)},
        ... )
        >>> prediction_time_node(cfg)
        Traceback (most recent call last):
            ...
        ValueError: The task configuration has no index_timestamp window.
    """
    window = task_cfg.index_timestamp_window
    if window is None:
        raise ValueError("The task configuration has no index_timestamp window.")
    return endpoint_node(task_cfg, window, task_cfg.windows[window].index_timestamp)


def label_node(task_cfg: TaskExtractorConfig) -> Node:
    """Returns the tree node that closes the label window of the task.

    Raises:
        ValueError: If the configuration has no ``label`` window.

    Examples:
        >>> label_node(sample_ACES_cfg).node_name
        'target.end'
        >>> cfg = TaskExtractorConfig(
        ...     predicates=sample_ACES_cfg.predicates,
        ...     trigger=sample_ACES_cfg.trigger,
        ...     windows={"w": WindowConfig("trigger", "start + 1d", True, True)},
        ... )
        >>> label_node(cfg)
        Traceback (most recent call last):
            ...
        ValueError: The task configuration has no label window.
    """
    if task_cfg.label_window is None:
        raise ValueError("The task configuration has no label window.")
    return endpoint_node(task_cfg, task_cfg.label_window, "end")


def constraint_node_name(name: str, window: WindowConfig) -> str:
    """The name of the node that carries a window's criteria and label: the endpoint not at its root.

    Examples:
        >>> constraint_node_name("gap", WindowConfig("trigger", "start + 1d", False, True))
        'gap.end'
        >>> constraint_node_name("before", WindowConfig("end - 1d", "trigger", False, True))
        'before.start'
    """
    return f"{name}.{'start' if window.root_node == 'end' else 'end'}"


def search_includes_anchor(window: WindowConfig, found_side: str) -> bool:
    """Whether an event search that finds a window's ``found_side`` endpoint counts an event at its anchor.

    In ACES, ``end: start -> P`` includes an event at the start itself exactly when ``end_inclusive`` is set,
    and ``start: end <- P`` includes an event at the end exactly when ``start_inclusive`` is set: the found
    endpoint's own inclusivity decides.

    Examples:
        >>> w = WindowConfig("trigger", "start -> A", True, False)
        >>> search_includes_anchor(w, "start"), search_includes_anchor(w, "end")
        (True, False)
    """
    return window.start_inclusive if found_side == "start" else window.end_inclusive


def backward_edge_reason(node: Node) -> str | None:
    """Explains why the edge into ``node`` from its parent can move backward in time, if it can.

    Args:
        node: A node in an ACES window tree.

    Returns:
        A human-readable reason if the edge can move backward in time, otherwise ``None``. The root (with no
        ``endpoint_expr``) never moves backward.

    Raises:
        ValueError: If the node's endpoint expression is not an ACES bound.

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
            if event_bound.end_event == f"-{START_OF_RECORD_KEY}":
                return "event bound targets the start of the record"
            if event_bound.end_event.startswith("-"):
                return f"event bound targets the prior '{event_bound.end_event[1:]}' event"
            if event_bound.offset is not None and event_bound.offset < timedelta(0):
                return f"event offset {event_bound.offset} is negative"
            return None
        case _ as other:
            raise ValueError(f"Unexpected type {type(other)} for endpoint expr: {other}.")
