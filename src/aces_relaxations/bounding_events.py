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
  whenever the bounding event coincides with the other endpoint. Then a zero-length auxiliary window is
  anchored at the bounding event: ``start: x.end``, ``end: start -> P``, both endpoints inclusive, with
  ``P: (1, None)``. ACES resolves ``start -> P`` with an inclusive end to the start event itself when that
  event is a ``P``, so the window is ``[x.end, x.end]`` and its criterion always holds when ``x`` is
  realizable; under infinity semantics it fails exactly when ``P`` never occurs.

Either way the transformation is a no-op for ACES, which the property tests check on arbitrary
configurations, and it is idempotent.
"""

import copy

from aces.config import TaskExtractorConfig, WindowConfig

RECORD_BOUNDARIES = ("_RECORD_START", "_RECORD_END")
AUXILIARY_SUFFIX = "_bounding_event"


def _boundings(window: WindowConfig) -> list[tuple[str, str, bool]]:
    """Each event bound of a window: its predicate, the bounded side, and whether that side is inclusive.

    A window can be event-bounded on both sides (``start: a.end -> admission``, ``end: start -> discharge``).
    """
    out = []
    for side, parsed in (("start", window._parsed_start), ("end", window._parsed_end)):
        event = parsed["event_bound"]
        if not event or event.lstrip("-") in RECORD_BOUNDARIES:
            continue
        inclusive = window.start_inclusive if side == "start" else window.end_inclusive
        out.append((event.lstrip("-"), side, inclusive))
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
        ``P: (1, None)`` itself (both endpoints inclusive) or is accompanied by a zero-length auxiliary
        window ``<name>_bounding_event`` anchored at the bounding event that carries it.

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
                └── (next discharge_or_death) target.end; **Label: Presence of death**
                    └── (next discharge_or_death) target_bounding_event.end (at least 1 discharge_or_death)
        >>> require_bounding_events(explicit).windows == explicit.windows
        True

    Here ``target`` has an exclusive start, so the criterion went on a zero-length auxiliary window anchored
    at the bounding event; the same happens below (``sufficient_history`` above is bounded by the record
    start, which is not an event). A window with both endpoints inclusive carries the criterion itself:

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

    It is a no-op for ACES:

        >>> extract(require_bounding_events(sample_ACES_cfg), sample_MEDS_df).equals(
        ...     extract(sample_ACES_cfg, sample_MEDS_df)
        ... )
        True
    """
    windows: dict[str, WindowConfig] = copy.deepcopy(task_cfg.windows)

    for name, window in list(windows.items()):
        for predicate, side, inclusive in _boundings(window):
            other_inclusive = window.end_inclusive if side == "start" else window.start_inclusive
            existing = window.has.get(predicate)
            compatible = existing is None or existing[1] is None or existing[1] >= 1
            if inclusive and other_inclusive and compatible:
                if not _requires(window, predicate):
                    lo = 1 if existing is None or existing[0] is None else max(existing[0], 1)
                    window.has[predicate] = (lo, None if existing is None else existing[1])
                continue

            anchor = f"{name}.{side}"
            already = any(
                w.start == anchor
                and w.end == f"start -> {predicate}"
                and w.start_inclusive
                and w.end_inclusive
                and _requires(w, predicate)
                for w in windows.values()
            )
            if already:
                continue
            aux_name, i = f"{name}{AUXILIARY_SUFFIX}", 2
            while aux_name in windows:
                aux_name = f"{name}{AUXILIARY_SUFFIX}_{i}"
                i += 1
            windows[aux_name] = WindowConfig(
                anchor, f"start -> {predicate}", True, True, has={predicate: "(1, None)"}
            )

    return type(task_cfg)(
        predicates=copy.deepcopy(task_cfg.predicates),
        trigger=copy.deepcopy(task_cfg.trigger),
        windows=windows,
    )
