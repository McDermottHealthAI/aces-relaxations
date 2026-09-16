"""Reduce the label branch of a normalized task to a single window re-rooted at its own start.

Many query languages can only ask about one window that starts at the prediction time: "does ``P`` occur
within ``d`` days?", "before the next ``Q``?", "before the end of the record?". ``last_link_target`` makes the
corresponding approximation: it assumes the model's prediction at the prediction time is as good as the
prediction it would make at the start of the label window, drops every link of the label branch but the last,
and re-roots that window at its own start. A look-back window is inverted into a forward one; this is always
possible for a fixed look-back, and possible for a look-back to a prior event when the window's end was itself
reached by an event hop.

Whether the approximation is justified for a given model is a question for the code that uses it. The
condition branches (``NormalizedTaskConfig.criteria_branches``) are not touched: dropping them, too, is
another model-specific approximation.
"""

from dataclasses import dataclass

from aces.types import END_OF_RECORD_KEY

from .normalization import NormalizedTaskConfig
from .predicates import PredicateExpr, predicate_expr


@dataclass(frozen=True)
class TargetWindow:
    """A single forward window from a (re-rooted) prediction time, and the label predicate counted in it.

    Attributes:
        label: The label predicate.
        duration_days: The window's length, for a fixed-duration window.
        bounding_event: The predicate whose next occurrence ends the window, for an event-bounded window.
        to_record_end: Whether the window extends to the end of the record.
        start_inclusive: Whether an occurrence exactly at the window's start counts.
        end_inclusive: Whether an occurrence exactly at the window's end counts.
        note: How the window relates to the original task: ``"exact"`` when it *is* the original label window
            re-rooted at its own start, otherwise a description of the approximation made.

    Examples:
        >>> from datetime import timedelta
        >>> TargetWindow(PredicateExpr.leaf("death"), duration_days=30.0).describe()
        'death within 30.0 days of the window start (start exclusive, end inclusive)'
        >>> discharge = PredicateExpr.leaf("discharge")
        >>> TargetWindow(PredicateExpr.leaf("death"), bounding_event=discharge).describe()
        'death before the next discharge after the window start (start exclusive, end inclusive)'
        >>> TargetWindow(PredicateExpr.leaf("death"), to_record_end=True, end_inclusive=False).describe()
        'death before the end of the record (start exclusive, end exclusive)'
    """

    label: PredicateExpr
    duration_days: float | None = None
    bounding_event: PredicateExpr | None = None
    to_record_end: bool = False
    start_inclusive: bool = False
    end_inclusive: bool = True
    note: str = "exact"

    @property
    def is_temporal(self) -> bool:
        """Whether the window has a fixed duration."""
        return self.duration_days is not None

    def describe(self) -> str:
        """A one-line description."""
        if self.is_temporal:
            span = f"within {self.duration_days} days of the window start"
        elif self.to_record_end:
            span = "before the end of the record"
        else:
            span = f"before the next {self.bounding_event} after the window start"
        flags = (
            f"start {'inclusive' if self.start_inclusive else 'exclusive'}, "
            f"end {'inclusive' if self.end_inclusive else 'exclusive'}"
        )
        return f"{self.label} {span} ({flags})"


def last_link_target(normalized: NormalizedTaskConfig) -> TargetWindow:
    """Reduces the label branch to its last window, re-rooted at that window's start.

    This assumes that the model's prediction at the prediction time stands in for the prediction it would
    make at the start of the label window. For a forward last window nothing else changes. A look-back
    window is inverted: a fixed look-back of ``d`` becomes "``d`` after the start"; a look-back to a prior
    event becomes "until the next occurrence of the event that ended the original window", which is exact
    only when no such event occurs in between, and is impossible when the original end was reached by a
    fixed offset.

    Args:
        normalized: A normalized configuration.

    Returns:
        The target window.

    Raises:
        ValueError: If the last window looks back to a prior event from an end that was reached by a fixed
            offset, which has no forward rewriting.

    Examples:
        >>> last_link_target(normalize(in_hosp_mortality_cfg)).describe()
        'death before the next or(discharge, death) after the window start (start exclusive, end inclusive)'
        >>> last_link_target(normalize(post_discharge_mortality_cfg)).describe()
        'death within 29.0 days of the window start (start exclusive, end inclusive)'
        >>> last_link_target(normalize(two_stage_cfg)).describe()
        'adverse_event before the next infusion_end after the window start (start inclusive, end inclusive)'

    A fixed look-back label window inverts exactly:

        >>> from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig
        >>> predicates = {
        ...     "A": PlainPredicateConfig("A"),
        ...     "B": PlainPredicateConfig("B"),
        ...     "discharge": PlainPredicateConfig("DISCHARGE"),
        ... }
        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "stay": WindowConfig("input.end + 2d", "start -> discharge", False, True),
        ...         "last_day": WindowConfig("end - 1d", "stay.end", False, True, label="B"),
        ...     },
        ... )
        >>> target = last_link_target(normalize(cfg, allow_end_rooted=True))
        >>> target.describe(), target.note
        ('B within 1.0 days of the window start (start exclusive, end inclusive)', 'exact')

    A label window running to the end of the record stays so:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "rest": WindowConfig("input.end", None, False, True, label="B"),
        ...     },
        ... )
        >>> last_link_target(normalize(cfg)).describe()
        'B before the end of the record (start exclusive, end inclusive)'

    A look-back to a prior event from an event-bounded end is inverted approximately, except from the end
    of the record, which is unique:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "wait": WindowConfig("input.end", "start + 3d", False, True, has={"B": "(1, None)"}),
        ...         "stay": WindowConfig("wait.end", "start -> discharge", False, True),
        ...         "since_b": WindowConfig("end <- B", "stay.end", True, True, label="A"),
        ...     },
        ... )
        >>> target = last_link_target(normalize(cfg, allow_end_rooted=True))
        >>> target.describe()
        'A before the next discharge after the window start (start inclusive, end inclusive)'
        >>> print(target.note)
        the window originally ran from the last B before a discharge to that discharge; re-rooted at the B,
        it runs to the next discharge, which is the same discharge only if no other discharge intervenes
        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "wait": WindowConfig("input.end", "start + 3d", False, True, has={"B": "(1, None)"}),
        ...         "rest": WindowConfig("wait.end", None, False, True),
        ...         "since_b": WindowConfig("end <- B", "rest.end", True, True, label="A"),
        ...     },
        ... )
        >>> target = last_link_target(normalize(cfg, allow_end_rooted=True))
        >>> target.describe(), target.note
        ('A before the end of the record (start inclusive, end inclusive)', 'exact')

    And from a fixed-offset end it cannot be inverted:

        >>> cfg = TaskExtractorConfig(
        ...     predicates=predicates, trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...         "wait": WindowConfig("input.end", "start + 3d", False, True, has={"B": "(1, None)"}),
        ...         "since_b": WindowConfig("end <- B", "wait.end", True, True, label="A"),
        ...     },
        ... )
        >>> last_link_target(normalize(cfg, allow_end_rooted=True))
        Traceback (most recent call last):
            ...
        ValueError: The label window 'since_b' looks back to a prior B from an end reached by a fixed offset,
        so it cannot be re-rooted at its start.
    """
    branch = normalized.label_branch
    last = branch.terminal
    label = predicate_expr(normalized, last.label)
    flags = {"start_inclusive": last.bound.left_inclusive, "end_inclusive": last.bound.right_inclusive}

    if last.is_temporal:
        return TargetWindow(label, duration_days=abs(last.days), **flags)

    end_event, note = last.event, "exact"
    if last.is_end_rooted:
        previous = branch.edges[-2] if len(branch.edges) > 1 else None
        if previous is None or previous.is_temporal:
            raise ValueError(
                f"The label window '{last.name}' looks back to a prior {last.event} from an end reached by a "
                "fixed offset, so it cannot be re-rooted at its start."
            )
        end_event = previous.event
        if end_event != END_OF_RECORD_KEY:  # There is exactly one record end, so that case stays exact.
            note = (
                f"the window originally ran from the last {last.event} before a {end_event} to that "
                f"{end_event}; re-rooted at the {last.event}, it runs to the next {end_event}, which is the "
                f"same {end_event} only if no other {end_event} intervenes"
            )

    if end_event == END_OF_RECORD_KEY:
        return TargetWindow(label, to_record_end=True, note=note, **flags)
    return TargetWindow(label, bounding_event=predicate_expr(normalized, end_event), note=note, **flags)
