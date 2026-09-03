"""Pretty-printing helpers for ACES task configurations and their window trees."""

from collections import defaultdict
from datetime import timedelta

from aces.config import TaskExtractorConfig, WindowConfig
from aces.types import TemporalWindowBounds, ToEventWindowBounds
from bigtree import yield_tree

from .tree import WindowNode, resolve_node


def get_constraint_str(window_cfg: WindowConfig) -> str:
    """Return a simple string representing the constraints of a window.

    Args:
        window_cfg: The configuration of the window whose constraints should be printed.

    Returns:
        A string representation of the constraints of a window.

    Examples:
        >>> get_constraint_str(WindowConfig("foo", "start + 1d", False, False))
        ''
        >>> get_constraint_str(
        ...     WindowConfig("b", "start + 1d", False, False, has={"foo": "(None, 0)", "bar": "(None, 0)"}),
        ... )
        'no foo, bar'
        >>> get_constraint_str(
        ...     WindowConfig("b", "start + 1d", False, False, has={"foo": "(None, 4)", "bar": "(None, 2)"}),
        ... )
        'no more than 4 foo, 2 bar'
        >>> get_constraint_str(
        ...     WindowConfig("b", "start + 1d", False, False, has={"foo": "(5, None)", "bar": "(5, None)"}),
        ... )
        'at least 5 foo, 5 bar'
        >>> get_constraint_str(
        ...     WindowConfig("b", "start + 1d", False, False, has={"foo": "(5, 6)", "bar": "(1, 3)"}),
        ... )
        'between 5-6 foo, 1-3 bar'
        >>> get_constraint_str(
        ...     WindowConfig(
        ...         "b", "start + 1d", False, False,
        ...         has={
        ...             "A": "(5, 6)",
        ...             "_ANY_EVENT": "(1, None)",
        ...             "C": "(2, 4)",
        ...             "D": "(None, 0)",
        ...             "E": "(2, None)",
        ...             "F": "(None, 0)",
        ...             "G": "(None, 2)",
        ...             "H": "(None, 0)",
        ...         },
        ...     ),
        ... )
        'between 5-6 A, 2-4 C; at least 1 event(s), 2 E; no D, F, H; no more than 2 G'
    """
    if not window_cfg.has:
        return ""

    by_prefix = defaultdict(list)
    for k_raw, constraint in window_cfg.has.items():
        left, right = constraint

        k = "event(s)" if k_raw == "_ANY_EVENT" else k_raw

        if left is None and right == 0:
            prefix = "no"
            str_rep = k
        elif left is None:
            prefix = "no more than"
            str_rep = f"{right} {k}"
        elif right is None:
            prefix = "at least"
            str_rep = f"{left} {k}"
        else:
            prefix = "between"
            str_rep = f"{left}-{right} {k}"

        by_prefix[prefix].append(str_rep)

    out_parts = [f"{prefix} {', '.join(strs)}" for prefix, strs in by_prefix.items()]
    return "; ".join(out_parts)


def print_ACES(task_cfg: TaskExtractorConfig, **kwargs):
    """A pretty printer for ACES task configurations.

    This is purely for development / debugging purposes.

    Args:
        task_cfg: The task configuration to print.
        **kwargs: Additional arguments to pass to the print function.

    Examples:
        >>> print_ACES(sample_ACES_cfg)
        trigger
        └── (+1 day, 0:00:00) input.end (no icu_admission, discharge_or_death); **Prediction Time**
            └── (+1 day, 0:00:00) gap.end (no icu_admission, discharge_or_death)
                └── (next discharge_or_death) target.end; **Label: Presence of death**

        >>> from aces.config import PlainPredicateConfig, EventConfig
        >>> predicates = {
        ...     "admission": PlainPredicateConfig("ADMISSION"),
        ...     "discharge": PlainPredicateConfig("DISCHARGE"),
        ... }
        >>> trigger = EventConfig("admission")
        >>> windows = {
        ...     "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...     "input": WindowConfig(None, "trigger + 24h", True, True),
        ...     "discharge": WindowConfig("gap.end", "start -> discharge", False, True),
        ...     "target": WindowConfig("discharge.end + 1d", "start + 29d", False, True),
        ... }
        >>> cfg = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> print_ACES(cfg)
        trigger
        ├── (+1 day, 0:00:00) input.end
        │   └── (start of record) input.start
        └── (+2 days, 0:00:00) gap.end
            └── (next discharge) discharge.end
                └── (+1 day, 0:00:00) target.start
                    └── (+29 days, 0:00:00) target.end
        >>> windows = {
        ...     "prior_hospitalization": WindowConfig("end <- admission", "trigger <- discharge", True, True),
        ...     "pre_data": WindowConfig("end - 12d", "trigger-2d", True, False),
        ...     "gap": WindowConfig("trigger", "start + 48h", False, True),
        ...     "input": WindowConfig(None, "trigger + 24h", True, True),
        ...     "discharge": WindowConfig("gap.end", "start -> discharge", False, True),
        ...     "target": WindowConfig("discharge.end + 1d", "start + 29d", False, True),
        ... }
        >>> cfg = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> print_ACES(cfg)
        trigger
        ├── (prior discharge) prior_hospitalization.end
        │   └── (prior admission) prior_hospitalization.start
        ├── (-2 days, 0:00:00) pre_data.end
        │   └── (-12 days, 0:00:00) pre_data.start
        ├── (+1 day, 0:00:00) input.end
        │   └── (start of record) input.start
        └── (+2 days, 0:00:00) gap.end
            └── (next discharge) discharge.end
                └── (+1 day, 0:00:00) target.start
                    └── (+29 days, 0:00:00) target.end
    """
    index_ts_window_name = getattr(task_cfg, "index_timestamp_window", None)

    if index_ts_window_name:
        index_ts_window = task_cfg.windows[index_ts_window_name]
        index_ts_node = resolve_node(
            task_cfg, root_node=WindowNode(index_ts_window_name, index_ts_window.index_timestamp)
        )
    else:
        index_ts_node = None

    for branch, stem, node in yield_tree(task_cfg.window_tree):
        window_name = node.node_name.split(".")[0]

        if index_ts_node and (node.node_name == index_ts_node.node_name):
            index_ts_str = "; **Prediction Time**"
        else:
            index_ts_str = ""

        if window_name == "trigger":
            print(f"{branch}{stem}{node.node_name}{index_ts_str}", **kwargs)
            continue

        match getattr(node, "endpoint_expr", None):
            case ToEventWindowBounds() as event_bound:
                if event_bound.end_event == "-_RECORD_START":
                    event_str = "start of record"
                elif event_bound.end_event == "_RECORD_END":
                    event_str = "end of record"
                elif event_bound.end_event.startswith("-"):
                    event_str = f"prior {event_bound.end_event[1:]}"
                else:
                    event_str = f"next {event_bound.end_event}"

                if event_bound.offset:
                    raise NotImplementedError("Offset not supported.")
                else:
                    bound = f"({event_str}) "
            case TemporalWindowBounds() as time_bound:
                sign = "+" if time_bound.window_size >= timedelta(0) else ""
                if time_bound.offset:
                    raise NotImplementedError("Offset not supported.")
                else:
                    bound = f"({sign}{time_bound.window_size}) "
            case None:
                bound = ""
            case _ as other:
                raise ValueError(f"Unexpected type {type(other)} for endpoint expr: {other}.")

        window_cfg = task_cfg.windows[window_name]

        side = node.node_name.split(".")[1]

        is_end_node = side != window_cfg.root_node

        if not is_end_node:
            print(f"{branch}{stem}{bound}{node.node_name}{index_ts_str}", **kwargs)
            continue

        c_str = get_constraint_str(window_cfg)
        constraint_str = f" ({get_constraint_str(window_cfg)})" if c_str else ""
        label_str = f"; **Label: Presence of {window_cfg.label}**" if window_cfg.label else ""
        print(f"{branch}{stem}{bound}{node.node_name}{constraint_str}{label_str}{index_ts_str}", **kwargs)
