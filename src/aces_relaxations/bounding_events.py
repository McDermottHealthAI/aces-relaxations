"""Make the existence of every event-bounded window's bounding event an explicit criterion.

A window such as ``end: start -> discharge`` only exists if a discharge occurs after its start. ACES enforces
that implicitly, by dropping rows for which the window cannot be realized, but another interpreter could
just as reasonably read the window as extending to infinity when no discharge occurs. To leave no room for
that ambiguity, ``require_bounding_events`` states the requirement as an ordinary criterion.

Where the criterion goes depends on whether the bounding event is certainly counted by the window:

- If *both* endpoints are inclusive, the bounding event is always counted by the window itself, so the
  window gets the criterion ``P: (1, None)`` directly (merged with any existing constraint on ``P``).
- Otherwise it may not be: an exclusive bounding endpoint never counts it, and even an inclusive one does
  not when the window is zero-length with an exclusive other endpoint (``[t, t)`` is empty), which happens
  whenever the bounding event coincides with the other endpoint. Then an auxiliary window carries the
  criterion, in whichever of two forms is exact:

  - a forward bound with an inclusive end (``end: start -> P``, ``end: x.end -> P``) gets a *sibling*
    anchored where the original search runs from (``start: <the window's start>`` or the referenced node),
    ``end: start -> P``, both endpoints inclusive. ACES resolves ``anchor -> P`` at or after the anchor, so
    the sibling resolves to the same event as the original search and counts it; under infinity semantics
    it reads directly as an infinite window containing no ``P``, failing the criterion.
  - any other bound (a strict forward search, which skips a ``P`` at its anchor; a look-back ``<- P``,
    which ACES resolves strictly before its anchor when the anchor is external) gets a zero-length
    *singleton* anchored at the bounding endpoint itself: ``start: x.<side>``, ``end: start -> P``, both
    inclusive. ACES resolves it to ``[t, t]`` at the bounding event, so the criterion holds exactly when
    the original search lands. A sibling repeating such a search cannot be exact: made inclusive it admits
    a ``P`` at the anchor the original rejects, and left exclusive its window excludes the event it found.

Either way the transformation is a no-op for ACES, which the property tests check on arbitrary
configurations, and it is idempotent.
"""

import copy
from typing import NamedTuple

from aces.config import TaskExtractorConfig, WindowConfig

RECORD_BOUNDARIES = ("_RECORD_START", "_RECORD_END")
AUXILIARY_SUFFIX = "_bounding_event"


class Bounding(NamedTuple):
    """One event bound of a window."""

    predicate: str
    side: str
    inclusive: bool
    referenced: str
    backward: bool


def _boundings(window: WindowConfig) -> list[Bounding]:
    """Each event bound of a window: predicate, bounded side, that side's inclusivity, anchor, direction.

    The anchor (``referenced``) is the reference the bound's search runs from: ``"start"`` / ``"end"`` for
    an internal reference (the window's other side), or the referenced node otherwise (``"trigger"``,
    ``"w.end"``). A window can be event-bounded on both sides (``start: a.end -> admission``, ``end: start
    -> discharge``).
    """
    out = []
    for side, parsed in (("start", window._parsed_start), ("end", window._parsed_end)):
        event = parsed["event_bound"]
        if not event or event.lstrip("-") in RECORD_BOUNDARIES:
            continue
        inclusive = window.start_inclusive if side == "start" else window.end_inclusive
        out.append(
            Bounding(event.lstrip("-"), side, inclusive, parsed["referenced"], parsed["occurs_before"])
        )
    return out


def _requires(window: WindowConfig, predicate: str) -> bool:
    lo = window.has.get(predicate, (None, None))[0]
    return lo is not None and lo >= 1


def require_bounding_events(task_cfg: TaskExtractorConfig) -> TaskExtractorConfig:
    """Adds an explicit criterion that every event-bounded window's bounding event occurs.

    The input is not modified. Criteria that are already implied are not added, so the function is
    idempotent.

    Args:
        task_cfg: Any ACES task configuration.

    Returns:
        A new configuration in which every window bounded by ``start -> P`` or ``end <- P`` either carries
        ``P: (1, None)`` itself (both endpoints inclusive) or is accompanied by an auxiliary window
        ``<name>_bounding_event`` that carries it (see the module docstring for its two forms).

    Examples:
        >>> print_ACES(in_hosp_mortality_cfg)
        trigger
        ├── (start of record) sufficient_history.start (at least 5 event(s))
        └── (+1 day, 0:00:00) input.end (no admission, discharge_or_death); **Prediction Time**
            └── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
                └── (next discharge_or_death) target.end; **Label: Presence of death**
        >>> explicit = require_bounding_events(in_hosp_mortality_cfg)
        >>> print_ACES(explicit)
        trigger
        ├── (start of record) sufficient_history.start (at least 5 event(s))
        └── (+1 day, 0:00:00) input.end (no admission, discharge_or_death); **Prediction Time**
            └── (+1 day, 0:00:00) gap.end (no admission, discharge_or_death)
                ├── (next discharge_or_death) target.end; **Label: Presence of death**
                └── (next discharge_or_death) target_bounding_event.end (at least 1 discharge_or_death)
        >>> require_bounding_events(explicit).windows == explicit.windows
        True

    Here ``target`` has an exclusive start, so the criterion went on an auxiliary sibling window repeating
    the same search from ``gap.end`` with an inclusive end (``sufficient_history`` above is bounded by the
    record start, which is not an event). Below, the strict search ``until_b`` and the look-back
    ``since_b`` get zero-length singletons anchored at their bounding events, and a window with both
    endpoints inclusive carries the criterion itself:

        >>> from aces.config import EventConfig, PlainPredicateConfig, WindowConfig
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", False, True, index_timestamp="end"),
        ...         "until_b": WindowConfig("input.end", "start -> B", False, False, label="A"),
        ...         "since_b": WindowConfig("end <- B", "until_b.end", False, True, has={"A": "(None, 0)"}),
        ...     },
        ... )
        >>> explicit = require_bounding_events(cfg)
        >>> print_ACES(explicit)
        trigger
        └── (+1 day, 0:00:00) input.end; **Prediction Time**
            └── (next B) until_b.end; **Label: Presence of A**
                ├── (prior B) since_b.start (no A)
                │   └── (next B) since_b_bounding_event.end (at least 1 B)
                └── (next B) until_b_bounding_event.end (at least 1 B)
        >>> w = explicit.windows["until_b_bounding_event"]
        >>> w.start, w.end, w.start_inclusive, w.end_inclusive
        ('until_b.end', 'start -> B', True, True)
        >>> require_bounding_events(explicit).windows == explicit.windows
        True
        >>> both = TaskExtractorConfig(
        ...     predicates=cfg.predicates, trigger=cfg.trigger,
        ...     windows={
        ...         "input": WindowConfig("trigger", "start + 1d", False, True, index_timestamp="end"),
        ...         "until_b": WindowConfig("input.end", "start -> B", True, True, label="A"),
        ...     },
        ... )
        >>> print_ACES(require_bounding_events(both))
        trigger
        └── (+1 day, 0:00:00) input.end; **Prediction Time**
            └── (next B) until_b.end (at least 1 B); **Label: Presence of A**

    A backward search defining a window's *end* (``end: trigger <- B``) lands strictly before its anchor,
    which no sibling search can reproduce, so it also gets the singleton:

        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")},
        ...     trigger=EventConfig("A"),
        ...     windows={
        ...         "before": WindowConfig("end - 1d", "trigger <- B", False, False, index_timestamp="start"),
        ...         "target": WindowConfig("trigger", "start + 1d", True, True, label="B"),
        ...     },
        ... )
        >>> w = require_bounding_events(cfg).windows["before_bounding_event"]
        >>> w.start, w.end, w.start_inclusive, w.has
        ('before.end', 'start -> B', True, {'B': (1, None)})

    It is a no-op for ACES:

        >>> extract(require_bounding_events(sample_ACES_cfg), sample_MEDS_df).equals(
        ...     extract(sample_ACES_cfg, sample_MEDS_df)
        ... )
        True
    """
    windows: dict[str, WindowConfig] = copy.deepcopy(task_cfg.windows)

    for name, window in list(windows.items()):
        for predicate, side, inclusive, referenced, backward in _boundings(window):
            other_inclusive = window.end_inclusive if side == "start" else window.start_inclusive
            existing = window.has.get(predicate)
            compatible = existing is None or existing[1] is None or existing[1] >= 1
            if inclusive and other_inclusive and compatible:
                if not _requires(window, predicate):
                    lo = 1 if existing is None or existing[0] is None else max(existing[0], 1)
                    window.has[predicate] = (lo, None if existing is None else existing[1])
                continue
            if inclusive and _requires(window, predicate):
                continue  # an inclusive bounding endpoint with a >=1 criterion already asserts existence

            # A forward, end-inclusive search from its anchor lands where the original does, so a sibling
            # repeating it is exact; anything else is exact only as a singleton at the bounding event.
            if side == "end" and inclusive and not backward:
                start = f"{name}.start" if referenced == "start" else referenced
            else:
                start = f"{name}.{side}"
            aux = WindowConfig(start, f"start -> {predicate}", True, True, has={predicate: "(1, None)"})
            already = any(
                (w.start, w.end, w.start_inclusive, w.end_inclusive) == (aux.start, aux.end, True, True)
                and _requires(w, predicate)
                for w in windows.values()
            )
            if already:
                continue
            aux_name, i = f"{name}{AUXILIARY_SUFFIX}", 2
            while aux_name in windows:
                aux_name = f"{name}{AUXILIARY_SUFFIX}_{i}"
                i += 1
            windows[aux_name] = aux

    return type(task_cfg)(
        predicates=copy.deepcopy(task_cfg.predicates),
        trigger=copy.deepcopy(task_cfg.trigger),
        windows=windows,
    )
