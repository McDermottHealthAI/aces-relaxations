"""Utilities for navigating the window tree of an ACES task configuration.

ACES compiles a task configuration into a tree of window *endpoint nodes* (a [`bigtree`](
https://bigtree.readthedocs.io/)
tree rooted at the ``trigger`` node). Each non-root node carries an ``endpoint_expr`` describing how it is
reached from its parent (a temporal offset or an event bound) and a ``constraints`` dictionary of the
inclusion / exclusion criteria that apply to the window it closes.

Not every window endpoint is present in that tree: an endpoint that is simply *equal* to another node (e.g.
``start: trigger`` or ``start: gap.end``) is elided. The helpers here resolve the ``<window>.<start|end>``
syntax used in ACES configs onto the nodes that actually exist in the tree.
"""

from typing import Literal, NamedTuple

from aces.config import TaskExtractorConfig, WindowConfig
from bigtree import Node


class RelaxedTaskConfig(TaskExtractorConfig):
    """An ACES task configuration whose window tree can be re-rooted.

    Plain ACES configurations are always rooted at the ``trigger`` node. Relaxed configurations are typically
    re-rooted at the prediction-time node (so that only the future of the prediction time is described) by
    [`strip_to_relevant_windows`][aces_relaxations.relaxations.strip_to_relevant_windows]. This subclass
    stores that alternate root and exposes it through the usual ``window_tree`` property.

    Examples:
        >>> cfg = RelaxedTaskConfig(
        ...     predicates=sample_ACES_cfg.predicates,
        ...     trigger=sample_ACES_cfg.trigger,
        ...     windows=sample_ACES_cfg.windows,
        ... )
        >>> cfg.window_tree.node_name
        'trigger'
        >>> cfg.window_tree = cfg.window_nodes["input.end"]
        >>> cfg.window_tree.node_name
        'input.end'
    """

    @property
    def window_tree(self) -> Node:
        """The root node of the (possibly re-rooted) window tree."""
        if not hasattr(self, "_root_node"):
            return self.window_nodes["trigger"]
        else:
            return self.window_nodes[self._root_node]

    @window_tree.setter
    def window_tree(self, node: Node):
        self._root_node = node.node_name


class WindowNode(NamedTuple):
    """A window endpoint node in the ACES configuration file.

    You can reference `node_name` to retrieve the resolved name of the node in the ACES tree.

    Attributes:
        name: The name of the window.
        root: The root node of the window, which can be "start", "end", or None. None is only allowed in the
            case of the trigger node.

    Examples:
        >>> N = WindowNode("foo", "end")
        >>> print(f"name: {N.name}, root: {N.root}, node_name: {N.node_name}")
        name: foo, root: end, node_name: foo.end
        >>> N = WindowNode("trigger", None)
        >>> print(f"name: {N.name}, root: {N.root}, node_name: {N.node_name}")
        name: trigger, root: None, node_name: trigger
    """

    name: str | Literal["trigger"]
    root: Literal["start", "end"] | None

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return self.name if self.root is None else f"{self.name}.{self.root}"


def get_referenced_node(windows: dict[str, WindowConfig], window: WindowNode) -> WindowNode:
    """Identifies the node in the windows mapping that the given window node refers to.

    Args:
        windows: A dictionary mapping window name to ACES window configurations.
        window: The query window node (tuple of window name and endpoint).

    Returns:
        All (non-trigger) nodes in the ACES windows tree refer to another node. This function returns the
        identifier of the window node to which the passed window node refers. This returned node need not be
        preserved in the final tree; this merely resolves the start vs. end syntax used in the ACES task
        configuration.

    Examples:
        >>> windows = {
        ...     "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...     "input": WindowConfig(None, "trigger + 24h", True, True),
        ...     "target": WindowConfig("gap.end", "start -> discharge_or_death", False, True),
        ...     "post_target": WindowConfig("target.end", "start + 24h", False, True),
        ...     "weird_window": WindowConfig("end - 24h", "post_target.start", False, False),
        ... }
        >>> get_referenced_node(windows, WindowNode("weird_window", "start"))
        WindowNode(name='weird_window', root='end')
        >>> get_referenced_node(windows, WindowNode("weird_window", "end"))
        WindowNode(name='post_target', root='start')
        >>> get_referenced_node(windows, WindowNode("post_target", "start"))
        WindowNode(name='target', root='end')
        >>> get_referenced_node(windows, WindowNode("input", "start"))
        WindowNode(name='input', root='end')
        >>> get_referenced_node(windows, WindowNode("gap", "end"))
        WindowNode(name='gap', root='start')
        >>> get_referenced_node(windows, WindowNode("target", "start"))
        WindowNode(name='gap', root='end')
        >>> get_referenced_node(windows, WindowNode("gap", "start"))
        WindowNode(name='trigger', root=None)
        >>> get_referenced_node(windows, WindowNode("post_target", "end"))
        WindowNode(name='post_target', root='start')
    """
    window_cfg = windows[window.name]

    bound = window_cfg._parsed_start if window.root == "start" else window_cfg._parsed_end

    ref = bound["referenced"]

    if ref == "trigger":
        return WindowNode("trigger", None)
    elif ref in {"start", "end"}:
        return WindowNode(window.name, ref)
    else:
        return WindowNode(*ref.split("."))


def node_in_tree(windows: dict[str, WindowConfig], window: WindowNode) -> bool:
    """Checks if the node will be in the ACES final tree.

    ACES deletes nodes from the tree if they are "equality" nodes that point directly to other nodes in the
    tree. This checks if the given node is such a node or not.

    Args:
        windows: A dictionary mapping window name to ACES window configurations.
        window: The query window node (tuple of window name and endpoint).

    Returns:
        True if the node is in the tree, False otherwise.

    Examples:
        >>> windows = {
        ...     "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...     "input": WindowConfig(None, "trigger + 24h", True, True),
        ...     "target": WindowConfig("gap.end", "start -> discharge_or_death", False, True),
        ...     "post_target": WindowConfig("target.end", "start + 24h", False, True),
        ...     "weird_window": WindowConfig("end - 24h", "post_target.start", False, False),
        ... }
        >>> node_in_tree(windows, WindowNode("weird_window", "start"))
        True
        >>> node_in_tree(windows, WindowNode("weird_window", "end"))
        False
        >>> node_in_tree(windows, WindowNode("post_target", "start"))
        False
        >>> node_in_tree(windows, WindowNode("input", "start"))
        True
        >>> node_in_tree(windows, WindowNode("gap", "end"))
        True
        >>> node_in_tree(windows, WindowNode("target", "start"))
        False
        >>> node_in_tree(windows, WindowNode("gap", "start"))
        False
        >>> node_in_tree(windows, WindowNode("post_target", "end"))
        True
    """
    if window.root is None:
        return True

    if window.root == "start":
        return windows[window.name].start_endpoint_expr is not None
    else:
        return windows[window.name].end_endpoint_expr is not None


def resolve_node(
    task_cfg: TaskExtractorConfig,
    window_name: str | None = None,
    root_node: WindowNode | None = None,
) -> WindowNode:
    """Resolves a node in the task configuration based on the window name or an input root node.

    Args:
        task_cfg: The task configuration to resolve the node from.
        window_name: The name of the window to resolve. If None, the root_node will be used.
        root_node: The root node to resolve. If None, the window_name will be used.

    Returns:
        The node that exists in the final output tree that corresponds to the passed node.

    Raises:
        ValueError: If the window name is not found in the task configuration.

    Examples:
        >>> from bigtree import print_tree
        >>> print_tree(sample_ACES_cfg.window_tree)
        trigger
        └── input.end
            └── gap.end
                └── target.end

    On this tree, the windows depend on the following nodes:

        >>> resolve_node(sample_ACES_cfg, window_name="gap")
        WindowNode(name='input', root='end')
        >>> resolve_node(sample_ACES_cfg, window_name="input")
        WindowNode(name='trigger', root=None)
        >>> resolve_node(sample_ACES_cfg, window_name="target")
        WindowNode(name='gap', root='end')

    If we pass a non-existent window name, it raises a ValueError:

        >>> resolve_node(sample_ACES_cfg, window_name="nonexistent")
        Traceback (most recent call last):
            ...
        ValueError: Window 'nonexistent' not found in task configuration.

    We can also pass a node directly, rather than resolving the window to the root node:

        >>> resolve_node(sample_ACES_cfg, root_node=WindowNode("input", "start"))
        WindowNode(name='trigger', root=None)
        >>> resolve_node(sample_ACES_cfg, root_node=WindowNode("input", "end"))
        WindowNode(name='input', root='end')
        >>> resolve_node(sample_ACES_cfg, root_node=WindowNode("target", "start"))
        WindowNode(name='gap', root='end')
        >>> resolve_node(sample_ACES_cfg, root_node=WindowNode("target", "end"))
        WindowNode(name='target', root='end')

    We must pass exactly one of window_name or root_node:

        >>> resolve_node(sample_ACES_cfg, window_name="input", root_node=WindowNode("target", "start"))
        Traceback (most recent call last):
            ...
        ValueError: Exactly one of window_name or root_node must be provided.
        >>> resolve_node(sample_ACES_cfg, window_name=None, root_node=None)
        Traceback (most recent call last):
            ...
        ValueError: Exactly one of window_name or root_node must be provided.
    """
    if (window_name is None and root_node is None) or (window_name is not None and root_node is not None):
        raise ValueError("Exactly one of window_name or root_node must be provided.")

    if window_name is not None:
        if window_name not in task_cfg.windows:
            raise ValueError(f"Window '{window_name}' not found in task configuration.")

        root_node = WindowNode(window_name, task_cfg.windows[window_name].root_node)

    if root_node.node_name == task_cfg.window_tree.node_name:
        return root_node

    while not node_in_tree(task_cfg.windows, root_node):
        root_node = get_referenced_node(task_cfg.windows, root_node)

    return root_node


def prediction_time_node(task_cfg: TaskExtractorConfig) -> WindowNode:
    """Returns the tree node at which the prediction time (``index_timestamp``) of the task is realized.

    Args:
        task_cfg: The task configuration.

    Returns:
        The resolved window node marking the prediction time.

    Raises:
        ValueError: If the configuration has no ``index_timestamp`` window.

    Examples:
        >>> prediction_time_node(sample_ACES_cfg)
        WindowNode(name='input', root='end')
        >>> prediction_time_node(readmission_cfg)
        WindowNode(name='trigger', root=None)
        >>> cfg = RelaxedTaskConfig(
        ...     predicates=sample_ACES_cfg.predicates,
        ...     trigger=sample_ACES_cfg.trigger,
        ...     windows={"w": WindowConfig("trigger", "start + 1d", True, True)},
        ... )
        >>> prediction_time_node(cfg)
        Traceback (most recent call last):
            ...
        ValueError: The task configuration has no index_timestamp window.
    """
    window_name = task_cfg.index_timestamp_window
    if window_name is None:
        raise ValueError("The task configuration has no index_timestamp window.")
    side = task_cfg.windows[window_name].index_timestamp
    return resolve_node(task_cfg, root_node=WindowNode(window_name, side))


def label_node(task_cfg: TaskExtractorConfig) -> WindowNode:
    """Returns the tree node that closes the label window of the task.

    Args:
        task_cfg: The task configuration.

    Returns:
        The resolved window node for the ``end`` of the label window.

    Raises:
        ValueError: If the configuration has no ``label`` window.

    Examples:
        >>> label_node(sample_ACES_cfg)
        WindowNode(name='target', root='end')
        >>> cfg = RelaxedTaskConfig(
        ...     predicates=sample_ACES_cfg.predicates,
        ...     trigger=sample_ACES_cfg.trigger,
        ...     windows={"w": WindowConfig("trigger", "start + 1d", True, True)},
        ... )
        >>> label_node(cfg)
        Traceback (most recent call last):
            ...
        ValueError: The task configuration has no label window.
    """
    window_name = task_cfg.label_window
    if window_name is None:
        raise ValueError("The task configuration has no label window.")
    return resolve_node(task_cfg, root_node=WindowNode(window_name, "end"))
