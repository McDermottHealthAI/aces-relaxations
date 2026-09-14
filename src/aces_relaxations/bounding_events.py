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
  whenever the bounding event coincides with the other endpoint. Then an auxiliary window is anchored at
  the window's *other* endpoint — the same anchor the original search runs from — repeating the search
  with the bounding endpoint made inclusive: for ``end: start -> P``, the auxiliary window is
  ``start: <x's start>`` (same reference, same inclusivity), ``end: start -> P``, end inclusive, with
  ``P: (1, None)``. It resolves to the same bounding event the original window resolves to and counts it,
  so the criterion holds exactly when ``x`` is realizable — and, unlike a window anchored at the (possibly
  unrealized) bounding endpoint itself, it needs no interpretation of searches launched from an unrealized
  position: under infinity semantics the window simply runs to infinity containing no ``P``, and the
  criterion fails.

Either way the transformation is a no-op for ACES, which the property tests check on arbitrary
configurations, and it is idempotent.
"""

import copy

from aces.config import TaskExtractorConfig, WindowConfig

RECORD_BOUNDARIES = ("_RECORD_START", "_RECORD_END")
AUXILIARY_SUFFIX = "_bounding_event"


def _boundings(window: WindowConfig) -> list[tuple[str, str, bool, str, bool]]:
    """Each event bound of a window: predicate, bounded side, that side's inclusivity, anchor, direction.

    The anchor is the reference the bound's search runs from: ``"start"`` / ``"end"`` for an internal
    reference (the window's other side), or the referenced node otherwise (``"trigger"``, ``"w.end"``). The
    direction is whether the search runs backward (``<- P``). A window can be event-bounded on both sides
    (``start: a.end -> admission``, ``end: start -> discharge``).
    """
    out = []
    for side, parsed in (("start", window._parsed_start), ("end", window._parsed_end)):
        event = parsed["event_bound"]
        if not event or event.lstrip("-") in RECORD_BOUNDARIES:
            continue
        inclusive = window.start_inclusive if side == "start" else window.end_inclusive
        out.append((event.lstrip("-"), side, inclusive, parsed["referenced"], event.startswith("-")))
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
        ``<name>_bounding_event`` — a sibling repeating the same search from the same anchor with the
        bounding endpoint inclusive — that carries it.

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
    the same search from ``gap.end`` with an inclusive end; the look-back bound below keeps the zero-length
    singleton anchored at the bounding event (``sufficient_history`` above is bounded by the record start,
    which is not an event). A window with both endpoints inclusive carries the criterion itself:

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
            ├── (next B) until_b.end; **Label: Presence of A**
            │   └── (prior B) since_b.start (no A)
            │       └── (next B) since_b_bounding_event.end (at least 1 B)
            └── (next B) until_b_bounding_event.end (at least 1 B)
        >>> w = explicit.windows["until_b_bounding_event"]
        >>> w.start, w.end, w.start_inclusive, w.end_inclusive
        ('input.end', 'start -> B', True, True)
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

    A backward search defining a window's *end* (``end: trigger <- B``) gets a sibling look-back window
    from the same anchor (``start: end <- B``, ``end: trigger``, start inclusive), which lands exactly when
    some ``B`` is at or before the anchor, as the original search does:

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
        ('end <- B', 'trigger', True, {'B': (1, None)})

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

            # A forward (end-side) bound whose search anchor is a plain reference gets its auxiliary
            # window anchored at that same reference — a sibling repeating the search with an inclusive
            # end, so it resolves to the same bounding event and counts it, and reads correctly under
            # infinity semantics (no bounding event: an infinite window containing no ``P``). Look-back
            # bounds and hop-valued anchors keep the zero-length singleton anchored at the bounding
            # event, whose counting behavior ACES pins down (see the module docstring).
            if inclusive and _requires(window, predicate):
                continue  # an inclusive bounding endpoint with a >=1 criterion already asserts existence
            aux_args = None
            if side == "end":
                anchor_expr = window.start if referenced == "start" else referenced
                plain = anchor_expr is not None and "->" not in anchor_expr and "<-" not in anchor_expr
                if plain and backward:
                    # A look-back from the anchor to the prior ``P``; start inclusive so that a ``P`` at
                    # the anchor is found (ACES finds it for the original bound when no earlier one exists).
                    aux_args = {
                        "start": f"end <- {predicate}",
                        "end": anchor_expr,
                        "start_inclusive": True,
                        "end_inclusive": True,
                    }
                elif plain:
                    # both endpoints inclusive: ACES resolves ``anchor -> P`` at-or-after the anchor, so
                    # a bounding event coincident with the anchor must be counted ([t, t], not (t, t])
                    aux_args = {
                        "start": anchor_expr,
                        "end": f"start -> {predicate}",
                        "start_inclusive": True,
                        "end_inclusive": True,
                    }
            if aux_args is None:
                aux_args = {
                    "start": f"{name}.{side}",
                    "end": f"start -> {predicate}",
                    "start_inclusive": True,
                    "end_inclusive": True,
                }
            already = any(
                w.start == aux_args["start"]
                and w.end == aux_args["end"]
                and w.start_inclusive == aux_args["start_inclusive"]
                and w.end_inclusive == aux_args["end_inclusive"]
                and _requires(w, predicate)
                for w in windows.values()
            )
            if already:
                continue
            aux_name, i = f"{name}{AUXILIARY_SUFFIX}", 2
            while aux_name in windows:
                aux_name = f"{name}{AUXILIARY_SUFFIX}_{i}"
                i += 1
            windows[aux_name] = WindowConfig(has={predicate: "(1, None)"}, **aux_args)

    return type(task_cfg)(
        predicates=copy.deepcopy(task_cfg.predicates),
        trigger=copy.deepcopy(task_cfg.trigger),
        windows=windows,
    )
