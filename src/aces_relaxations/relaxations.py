"""Relaxations of ACES task configurations.

A *relaxation* is a transformation of an ACES task configuration that removes constraints or simplifies the
relationships between window endpoints, so that the (future-only) task can be posed to a system with a more
limited query language than ACES itself -- for example, a generative model whose sampled trajectories are
labeled with the relaxed configuration, or a query-answering foundation model that can only answer
"does X ever occur within region R?".

Every relaxation here operates *in place* on a
[`RelaxedTaskConfig`][aces_relaxations.tree.RelaxedTaskConfig] and returns it for chaining. The
[`relax`][aces_relaxations.relaxations.relax] entry point first copies and prunes the configuration via
[`strip_to_relevant_windows`][aces_relaxations.relaxations.strip_to_relevant_windows], so the caller's
configuration is never modified.
"""

import copy
from collections.abc import Callable, Mapping
from typing import Any

from aces.config import TaskExtractorConfig
from aces.types import TemporalWindowBounds, ToEventWindowBounds

from .constraint_simplification import simplify_constraints
from .future_only import validate_future_only
from .tree import RelaxedTaskConfig, WindowNode, label_node, prediction_time_node


def strip_to_relevant_windows(task_cfg: TaskExtractorConfig) -> RelaxedTaskConfig:
    """Re-roots the configuration at the prediction time and prunes windows irrelevant to the label.

    The returned configuration is a deep copy of the input, re-rooted at the prediction-time node, and
    containing only the nodes that are both descendants of the prediction time and either ancestors of,
    equal to, or descendants of the label node. This ensures the configuration does not rely on any window
    outside the future of the prediction time.

    Args:
        task_cfg: The task configuration to convert.

    Returns:
        A relaxed task configuration with only the relevant windows for future-only labeling.

    Examples:
        >>> print_ACES(in_hosp_mortality_cfg)
        trigger
        ├── (start of record) sufficient_history.start (at least 5 event(s))
        └── (+1 day, 0:00:00) input.end (no admission, discharge_or_death); **Prediction Time**
            └── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
                └── (next discharge_or_death) target.end; **Label: Presence of death**
        >>> print_ACES(strip_to_relevant_windows(in_hosp_mortality_cfg))
        input.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
            └── (next discharge_or_death) target.end; **Label: Presence of death**

    Windows after the label are retained, while windows off the path to the label are dropped:

        >>> print_ACES(readmission_cfg)
        trigger; **Prediction Time**
        ├── (prior admission) hospitalization.start (at least 10 event(s))
        │   └── (start of record) sufficient_history.start (at least 5 event(s))
        └── (+1 day, 0:00:00) gap.end (no admission, death)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
                └── (end of record) censoring_protection.end (at least 1 event(s))
        >>> print_ACES(strip_to_relevant_windows(readmission_cfg))
        hospitalization.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end (no admission, death)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
                └── (end of record) censoring_protection.end (at least 1 event(s))

    The input configuration is not modified:

        >>> print_ACES(readmission_cfg)
        trigger; **Prediction Time**
        ├── (prior admission) hospitalization.start (at least 10 event(s))
        │   └── (start of record) sufficient_history.start (at least 5 event(s))
        └── (+1 day, 0:00:00) gap.end (no admission, death)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
                └── (end of record) censoring_protection.end (at least 1 event(s))
    """
    prediction_time_window_name = task_cfg.index_timestamp_window
    prediction_time_window_cfg = task_cfg.windows[prediction_time_window_name]
    prediction_time_window = WindowNode(
        prediction_time_window_name, prediction_time_window_cfg.index_timestamp
    )
    new_root = prediction_time_node(task_cfg)

    label_window_node = label_node(task_cfg)

    new_task_cfg = RelaxedTaskConfig(
        predicates=copy.deepcopy(task_cfg.predicates),
        windows=copy.deepcopy(task_cfg.windows),
        trigger=copy.deepcopy(task_cfg.trigger),
        label_window=copy.deepcopy(task_cfg.label_window),
        index_timestamp_window=copy.deepcopy(task_cfg.index_timestamp_window),
    )

    new_root_node = new_task_cfg.window_nodes[new_root.node_name]
    new_root_node.name = prediction_time_window.node_name
    new_root_node.endpoint_expr = None
    new_root_node.constraints = {}

    if prediction_time_window.root == "end":
        new_task_cfg.windows[prediction_time_window_name].has = {}

    new_task_cfg.window_tree = new_root_node

    new_label_node = new_task_cfg.window_nodes[label_window_node.node_name]

    label_rel_nodes = set(new_label_node.ancestors) | {new_label_node} | set(new_label_node.descendants)
    prediction_time_rel_nodes = set(new_root_node.descendants) | {new_root_node}

    allowed_nodes = prediction_time_rel_nodes & label_rel_nodes

    new_task_cfg.window_nodes = {
        v.node_name: v for v in new_task_cfg.window_nodes.values() if v in allowed_nodes
    }

    for node in new_task_cfg.window_nodes.values():
        node.parent = node.parent if node.parent in allowed_nodes else None
        node.children = [n for n in node.children if n in allowed_nodes]

    return new_task_cfg


def remove_all_criteria(task_cfg: RelaxedTaskConfig) -> RelaxedTaskConfig:
    """Removes all inclusion / exclusion criteria from every window, leaving the window boundaries intact.

    Args:
        task_cfg: The task configuration to modify in place.

    Returns:
        The modified configuration.

    Examples:
        >>> cfg = strip_to_relevant_windows(in_hosp_mortality_cfg)
        >>> print_ACES(remove_all_criteria(cfg))
        input.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end
            └── (next discharge_or_death) target.end; **Label: Presence of death**
    """
    for window in task_cfg.windows.values():
        window.has = {}
    for node in task_cfg.window_nodes.values():
        node.constraints = {}
    return task_cfg


def collapse_temporal_gap_windows(task_cfg: RelaxedTaskConfig) -> RelaxedTaskConfig:
    """Absorbs unconstrained temporal windows between the prediction time and the label into the label window.

    Walking from the label node back towards the root, any node that is reached by a temporal bound and has
    no constraints is removed and its duration is added to the label-side node. The walk stops at the first
    constrained node or event-bound node.

    Args:
        task_cfg: The task configuration to modify in place.

    Returns:
        The modified configuration.

    Examples:
        >>> cfg = remove_all_criteria(strip_to_relevant_windows(post_discharge_mortality_cfg))
        >>> print_ACES(cfg)
        input.end; **Prediction Time**
        └── (next discharge) hospitalization.end
            └── (+1 day, 0:00:00) gap.end
                └── (+29 days, 0:00:00) target.end; **Label: Presence of death**
        >>> print_ACES(collapse_temporal_gap_windows(cfg))
        input.end; **Prediction Time**
        └── (next discharge) hospitalization.end
            └── (+30 days, 0:00:00) target.end; **Label: Presence of death**

    Constrained gap windows are not collapsed:

        >>> cfg = strip_to_relevant_windows(post_discharge_mortality_cfg)
        >>> print_ACES(collapse_temporal_gap_windows(cfg))
        input.end; **Prediction Time**
        └── (next discharge) hospitalization.end (no death)
            └── (+1 day, 0:00:00) gap.end (no admission, death)
                └── (+29 days, 0:00:00) target.end; **Label: Presence of death**
    """
    label_window_node = label_node(task_cfg)
    root_to_label = list(task_cfg.window_nodes[label_window_node.node_name].node_path)

    label_to_root = root_to_label[::-1]

    new_label_to_root = [label_to_root.pop(0)]

    while label_to_root:
        node = label_to_root[0]

        if node.constraints:
            break

        match getattr(node, "endpoint_expr", None):
            case ToEventWindowBounds() | None:
                break
            case TemporalWindowBounds() as time_bound:
                label_to_root.pop(0)
                new_label_to_root[-1].parent = node.parent
                node.parent.children = tuple(n for n in node.parent.children if n != node)
                task_cfg.window_nodes.pop(node.node_name, None)
                if isinstance(new_label_to_root[-1].endpoint_expr, TemporalWindowBounds):
                    new_label_to_root[-1].endpoint_expr.window_size += time_bound.window_size
            case _ as other:
                raise ValueError(f"Unexpected type {type(other)} for endpoint expr: {other}.")

    new_label_to_root.extend(label_to_root)

    return task_cfg


def remove_post_label_windows(task_cfg: RelaxedTaskConfig) -> RelaxedTaskConfig:
    """Removes all windows that are descendants of the label window end node.

    This is useful for removing censoring protection windows that are not needed for future-only labeling.

    Args:
        task_cfg: The task configuration to modify in place.

    Returns:
        The modified configuration.

    Examples:
        >>> cfg = strip_to_relevant_windows(readmission_cfg)
        >>> print_ACES(remove_post_label_windows(cfg))
        hospitalization.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end (no admission, death)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**
    """
    label_window_node = task_cfg.window_nodes[label_node(task_cfg).node_name]

    descendant_names = [n.node_name for n in label_window_node.descendants]

    label_window_node.children = ()
    for n in descendant_names:
        task_cfg.window_nodes.pop(n, None)

    return task_cfg


Relaxation = Callable[..., RelaxedTaskConfig]

# The registry of named relaxations, in the order `relax` applies them. Constraint-modifying relaxations
# run before structural ones so that, e.g., a window whose constraints were just removed can be collapsed.
RELAXATIONS: dict[str, Relaxation] = {
    "remove_all_criteria": remove_all_criteria,
    "simplify_constraints": simplify_constraints,
    "collapse_temporal_gap_windows": collapse_temporal_gap_windows,
    "remove_post_label_windows": remove_post_label_windows,
}


def relax(
    task_cfg: TaskExtractorConfig,
    relaxations: Mapping[str, bool | Mapping[str, Any] | None] | None = None,
    *,
    validate: bool = True,
) -> RelaxedTaskConfig:
    """Converts a task configuration into a future-only, relaxed configuration.

    The configuration is first (optionally) validated to be guaranteeably future-only, then copied and pruned
    with [`strip_to_relevant_windows`][aces_relaxations.relaxations.strip_to_relevant_windows], and finally
    each requested relaxation is applied in the order given by
    [`RELAXATIONS`][aces_relaxations.relaxations.RELAXATIONS].

    Args:
        task_cfg: The task configuration to convert. It is not modified.
        relaxations: A mapping from relaxation name to either a boolean (apply with default parameters or
            skip), ``None`` (skip), or a mapping of keyword arguments for that relaxation.
        validate: Whether to first validate that the configuration is guaranteeably future-only.

    Returns:
        A relaxed task configuration.

    Raises:
        ValueError: If an unknown relaxation is requested, a relaxation specification has an unsupported
            type, or (when ``validate`` is set) the configuration is not future-only.

    Examples:
        >>> print_ACES(relax(in_hosp_mortality_cfg))
        input.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
            └── (next discharge_or_death) target.end; **Label: Presence of death**
        >>> print_ACES(relax(in_hosp_mortality_cfg, {"remove_all_criteria": True}))
        input.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end
            └── (next discharge_or_death) target.end; **Label: Presence of death**
        >>> relaxations = {"remove_all_criteria": True, "collapse_temporal_gap_windows": True}
        >>> print_ACES(relax(in_hosp_mortality_cfg, relaxations))
        input.end; **Prediction Time**
        └── (next discharge_or_death) target.end; **Label: Presence of death**

    Parameterized relaxations take a mapping of keyword arguments:

        >>> print_ACES(relax(
        ...     readmission_cfg,
        ...     {
        ...         "simplify_constraints": {"predicate_overrides": {"admission": "null"}},
        ...         "remove_post_label_windows": True,
        ...     },
        ... ))
        hospitalization.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end (no death)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**

    Unknown or malformed relaxations are rejected:

        >>> relax(in_hosp_mortality_cfg, {"make_it_easy": True})
        Traceback (most recent call last):
            ...
        ValueError: Unknown relaxations: ['make_it_easy']. Known relaxations: ['remove_all_criteria',
        'simplify_constraints', 'collapse_temporal_gap_windows', 'remove_post_label_windows'].
        >>> relax(in_hosp_mortality_cfg, {"remove_all_criteria": 3})
        Traceback (most recent call last):
            ...
        ValueError: Relaxation 'remove_all_criteria' must be a boolean, None, or a mapping of keyword
        arguments; got 3.

    Configurations that are not future-only are rejected unless validation is disabled:

        >>> from aces.config import EventConfig, PlainPredicateConfig, WindowConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "target": WindowConfig("input.end - 1d", "start + 3d", False, True, label="B"),
        ...     },
        ... )
        >>> relax(cfg)
        Traceback (most recent call last):
            ...
        ValueError: The task configuration is not guaranteeably future-only:
          - The edge into target.start on the path from the prediction time to the label window can move
            backward in time: temporal bound -1 day, 0:00:00 is negative.
        Got tree:
        trigger
        └── input.end
            └── target.start
                └── target.end
        <BLANKLINE>
        >>> print_ACES(relax(cfg, validate=False))
        input.end; **Prediction Time**
        └── (-1 day, 0:00:00) target.start
            └── (+3 days, 0:00:00) target.end; **Label: Presence of B**
    """
    if validate:
        validate_future_only(task_cfg)

    relaxations = dict(relaxations or {})

    unknown = [k for k in relaxations if k not in RELAXATIONS]
    if unknown:
        raise ValueError(f"Unknown relaxations: {unknown}. Known relaxations: {list(RELAXATIONS)}.")

    relaxed_cfg = strip_to_relevant_windows(task_cfg)

    for name, fn in RELAXATIONS.items():
        spec = relaxations.get(name)
        match spec:
            case None | False:
                continue
            case True:
                relaxed_cfg = fn(relaxed_cfg)
            case Mapping():
                relaxed_cfg = fn(relaxed_cfg, **spec)
            case _:
                raise ValueError(
                    f"Relaxation '{name}' must be a boolean, None, or a mapping of keyword arguments; "
                    f"got {spec!r}."
                )

    return relaxed_cfg
