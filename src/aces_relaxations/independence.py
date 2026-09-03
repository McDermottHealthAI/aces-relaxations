"""The "independent codes" relaxation.

Time-to-event models such as MOTOR predict, for each code, the distribution of the time until that code next
occurs, *independently* of every other code. Under that modeling assumption, any inclusion / exclusion
criterion over codes unrelated to the label cannot change the label's distribution, so it can be dropped
without further loss, and any window after the label (a censoring-protection window, say) is likewise
irrelevant. What remains are the criteria that share plain predicates with the label, which such a model *can*
honor: "no death in the gap window" is a condition on the very time-to-event distribution being queried.
"""

from .predicates import PredicateExpr, label_predicate_expr, predicate_expr
from .relaxations import remove_post_label_windows
from .tree import RelaxedTaskConfig

NEVER = (None, 0)


def _narrow_constraint(
    expr: PredicateExpr, constraint: tuple[int | None, int | None], label_leaves: set[str]
) -> dict[str, tuple[int | None, int | None]]:
    """Restricts one constraint to the label-related plain predicates it depends on.

    A "never" constraint on a disjunction is equivalent to "never" on each of its leaves, so it is narrowed
    to the label-related leaves. Any other constraint that touches a label-related leaf is kept as is; a
    constraint touching none is dropped.

    Examples:
        >>> a, b = PredicateExpr.leaf("a"), PredicateExpr.leaf("b")
        >>> _narrow_constraint(PredicateExpr.or_(a, b), (None, 0), {"a"})
        {'a': (None, 0)}
        >>> _narrow_constraint(PredicateExpr.or_(a, b), (None, 0), {"c"})
        {}
        >>> _narrow_constraint(PredicateExpr.and_(a, b), (None, 0), {"a"})
        {'and(a, b)': (None, 0)}
        >>> _narrow_constraint(PredicateExpr.or_(a, b), (2, None), {"a"})
        {'or(a, b)': (2, None)}
    """
    related = expr.leaves & label_leaves
    if not related:
        return {}
    if tuple(constraint) == NEVER and expr.is_or_only:
        return dict.fromkeys(sorted(related), NEVER)
    return {str(expr): constraint}


def assume_independent_codes(task_cfg: RelaxedTaskConfig) -> RelaxedTaskConfig:
    """Drops every criterion and window that cannot affect the label if code occurrences are independent.

    Concretely, this relaxation:

    1. Drops all window criteria whose predicates share no plain predicate with the label predicate
       (including ``_ANY_EVENT`` criteria).
    2. Narrows "never" criteria on disjunctions to the label-related plain predicates (``no
       discharge_or_death`` becomes ``no death`` when the label is ``death``), and keeps any other
       label-related criterion unchanged.
    3. Removes all windows after the label window.

    Windows off the path from the prediction time to the label are already removed by
    [`strip_to_relevant_windows`][aces_relaxations.relaxations.strip_to_relevant_windows].

    Args:
        task_cfg: The task configuration to modify in place.

    Returns:
        The modified configuration.

    Examples:
        >>> cfg = strip_to_relevant_windows(in_hosp_mortality_cfg)
        >>> print_ACES(cfg)
        input.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
            └── (next discharge_or_death) target.end; **Label: Presence of death**
        >>> print_ACES(assume_independent_codes(cfg))
        input.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end (no death)
            └── (next discharge_or_death) target.end; **Label: Presence of death**

    The readmission task's censoring window and its ``no death`` criterion are both dropped, while ``no
    admission`` is kept because ``admission`` is the label:

        >>> print_ACES(assume_independent_codes(strip_to_relevant_windows(readmission_cfg)))
        hospitalization.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end (no admission)
            └── (+29 days, 0:00:00) target.end; **Label: Presence of admission**

    A "never" criterion on the label predicate itself is kept:

        >>> cfg = assume_independent_codes(strip_to_relevant_windows(two_stage_cfg))
        >>> print_ACES(cfg)
        1st_infusion.start; **Prediction Time**
        └── (next infusion_end) 1st_infusion.end (no adverse_event)
            └── (next infusion_start) 2nd_infusion.start
                └── (next infusion_end) 2nd_infusion.end; **Label: Presence of adverse_event**

    Non-"never" label-related criteria are kept as they are (here on a derived predicate):

        >>> from aces.config import (
        ...     DerivedPredicateConfig, EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
        ... )
        >>> cfg = TaskExtractorConfig(
        ...     predicates={
        ...         "a": PlainPredicateConfig("A"),
        ...         "b": PlainPredicateConfig("B"),
        ...         "c": PlainPredicateConfig("C"),
        ...         "ab": DerivedPredicateConfig("or(a, b)"),
        ...     },
        ...     trigger=EventConfig("a"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "gap": WindowConfig(
        ...             "input.end", "start + 1d", False, True, has={"ab": "(2, None)", "c": "(1, None)"}
        ...         ),
        ...         "target": WindowConfig("gap.end", "start + 7d", False, True, label="a"),
        ...     },
        ... )
        >>> print_ACES(assume_independent_codes(strip_to_relevant_windows(cfg)))
        input.end; **Prediction Time**
        └── (+1 day, 0:00:00) gap.end (at least 2 ab)
            └── (+7 days, 0:00:00) target.end; **Label: Presence of a**
    """
    label_leaves = label_predicate_expr(task_cfg).leaves

    for name, window in task_cfg.windows.items():
        if not window.has:
            continue

        new_has: dict[str, tuple[int | None, int | None]] = {}
        for predicate, constraint in window.has.items():
            expr = predicate_expr(task_cfg, predicate)
            for narrowed_predicate, narrowed_constraint in _narrow_constraint(
                expr, constraint, label_leaves
            ).items():
                key = narrowed_predicate if narrowed_predicate in task_cfg.predicates else predicate
                new_has[key] = narrowed_constraint

        window.has.clear()
        window.has.update(new_has)

        constrained_side = "start" if window.root_node == "end" else "end"
        node = task_cfg.window_nodes.get(f"{name}.{constrained_side}")
        if node is not None:
            node.constraints = window.has

    return remove_post_label_windows(task_cfg)
