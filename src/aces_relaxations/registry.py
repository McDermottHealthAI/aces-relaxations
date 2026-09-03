"""The registry of named relaxations and the ``relax`` entry point."""

from collections.abc import Callable, Mapping
from typing import Any

from aces.config import TaskExtractorConfig

from .constraint_simplification import simplify_constraints
from .future_only import validate_future_only
from .independence import assume_independent_codes
from .relaxations import (
    collapse_temporal_gap_windows,
    remove_all_criteria,
    remove_post_label_windows,
    strip_to_relevant_windows,
)
from .tree import RelaxedTaskConfig
from .vocabulary import restrict_to_vocabulary

Relaxation = Callable[..., RelaxedTaskConfig]

# The registry of named relaxations, in the order `relax` applies them: predicate-level, then criteria-level,
# then structural, so that, e.g., a window whose constraints were just removed can be collapsed.
RELAXATIONS: dict[str, Relaxation] = {
    "restrict_to_vocabulary": restrict_to_vocabulary,
    "remove_all_criteria": remove_all_criteria,
    "assume_independent_codes": assume_independent_codes,
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
    [`RELAXATIONS`][aces_relaxations.registry.RELAXATIONS].

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
        ValueError: Unknown relaxations: ['make_it_easy']. Known relaxations: ['restrict_to_vocabulary',
        'remove_all_criteria', 'assume_independent_codes', 'simplify_constraints',
        'collapse_temporal_gap_windows', 'remove_post_label_windows'].
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
