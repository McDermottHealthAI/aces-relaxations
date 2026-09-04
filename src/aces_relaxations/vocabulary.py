"""Restrict the plain predicates of a task configuration to a model's code vocabulary.

Tokenized and time-to-event foundation models can only reason about a fixed set of codes. This relaxation
rewrites each plain predicate into the subset of vocabulary codes it matches (an exact code, a regex, or an
``any`` list), drops value constraints such models cannot express, and replaces predicates that match nothing
with a designated *marginal* sentinel code that a downstream converter can interpret as "behaves like the
average expressible code".
"""

import re
from collections.abc import Iterable
from typing import Literal

from aces.config import PlainPredicateConfig, TaskExtractorConfig

MARGINAL_CODE = "_MARGINAL_"

UnmatchedPolicy = Literal["marginal", "keep", "error"]


def matching_codes(predicate: PlainPredicateConfig, vocabulary: Iterable[str]) -> list[str]:
    r"""Returns the vocabulary codes a plain predicate's code specification matches, sorted.

    Args:
        predicate: The plain predicate.
        vocabulary: The codes the model can express.

    Returns:
        The matching codes. An exact code matches itself; a ``regex`` is searched (not anchored) within each
        code, as ACES does; an ``any`` list matches its members.

    Examples:
        >>> vocab = ["ICD9//250.1", "ICD9//250.2", "ICD9//400", "MEDS_DEATH"]
        >>> matching_codes(PlainPredicateConfig("MEDS_DEATH"), vocab)
        ['MEDS_DEATH']
        >>> matching_codes(PlainPredicateConfig("MEDS_BIRTH"), vocab)
        []
        >>> matching_codes(PlainPredicateConfig(code={"regex": r"ICD9//250\..*"}), vocab)
        ['ICD9//250.1', 'ICD9//250.2']
        >>> matching_codes(PlainPredicateConfig(code={"any": ["ICD9//400", "ICD9//401"]}), vocab)
        ['ICD9//400']
        >>> matching_codes(PlainPredicateConfig(code={"all": ["ICD9//400"]}), vocab)
        Traceback (most recent call last):
            ...
        ValueError: Unsupported code specification: {'all': ['ICD9//400']}.
    """
    vocabulary = set(vocabulary)
    match predicate.code:
        case str() as code:
            matches = [code] if code in vocabulary else []
        case {"regex": pattern}:
            regex = re.compile(pattern)
            matches = [c for c in vocabulary if regex.search(c)]
        case {"any": codes}:
            matches = [c for c in codes if c in vocabulary]
        case other:
            raise ValueError(f"Unsupported code specification: {other}.")
    return sorted(matches)


VALUE_BIN = re.compile(r"^(?P<base>.*)//value_(?P<lo_br>[\[(])(?P<lo>[^,]+),(?P<hi>[^\])]+)(?P<hi_br>[\])])$")


def _parse_bound(text: str) -> float:
    text = text.strip().lower()
    if text in ("inf", "+inf"):
        return float("inf")
    if text == "-inf":
        return float("-inf")
    return float(text)


def quantized_value_codes(
    predicate: PlainPredicateConfig,
    vocabulary: Iterable[str],
    partial: Literal["include", "exclude"] = "include",
) -> list[str]:
    r"""Returns the quantized value-bin codes a value-constrained plain predicate matches.

    Everything-is-code tokenizations join a code with a bin of its numeric value into one code, named like
    ``HR//value_[102.6,105.1)``. A predicate such as "``HR`` with value at least 100" then corresponds to the
    set of ``HR`` bin codes whose interval lies in ``[100, inf)``.

    Args:
        predicate: A plain predicate with ``value_min`` and / or ``value_max``.
        vocabulary: The codes the model can express.
        partial: What to do with a bin that only partly overlaps the constraint's range: include it (a
            weakening) or exclude it (a strengthening).

    Returns:
        The matching bin codes, sorted. A predicate without value constraints matches nothing here (use
        ``matching_codes``).

    Examples:
        >>> vocab = [
        ...     "HR//value_[-inf,60.0)", "HR//value_[60.0,100.0)", "HR//value_[100.0,120.0)",
        ...     "HR//value_[120.0,inf)", "HR", "SBP//value_[90.0,140.0)",
        ... ]
        >>> quantized_value_codes(PlainPredicateConfig("HR", value_min=100), vocab)
        ['HR//value_[100.0,120.0)', 'HR//value_[120.0,inf)']
        >>> quantized_value_codes(PlainPredicateConfig("HR", value_min=110), vocab)
        ['HR//value_[100.0,120.0)', 'HR//value_[120.0,inf)']
        >>> quantized_value_codes(PlainPredicateConfig("HR", value_min=110), vocab, partial="exclude")
        ['HR//value_[120.0,inf)']
        >>> quantized_value_codes(PlainPredicateConfig("HR", value_max=60, value_max_inclusive=False), vocab)
        ['HR//value_[-inf,60.0)']
        >>> quantized_value_codes(PlainPredicateConfig(code={"regex": "^HR$"}, value_min=100), vocab)
        ['HR//value_[100.0,120.0)', 'HR//value_[120.0,inf)']
        >>> quantized_value_codes(PlainPredicateConfig("HR"), vocab)
        []
    """
    if predicate.value_min is None and predicate.value_max is None:
        return []

    lo = float("-inf") if predicate.value_min is None else float(predicate.value_min)
    hi = float("inf") if predicate.value_max is None else float(predicate.value_max)

    bins: dict[str, list[tuple[str, float, float]]] = {}
    for code in vocabulary:
        m = VALUE_BIN.match(code)
        if m:
            bins.setdefault(m["base"], []).append((code, _parse_bound(m["lo"]), _parse_bound(m["hi"])))

    base_predicate = PlainPredicateConfig(predicate.code)
    bases = matching_codes(base_predicate, bins)

    out = []
    for base in bases:
        for code, b_lo, b_hi in bins[base]:
            inside = lo <= b_lo and b_hi <= hi
            overlaps = b_lo < hi and lo < b_hi
            if inside or (partial == "include" and overlaps):
                out.append(code)
    return sorted(out)


def vocabulary_matches(task_cfg: TaskExtractorConfig, vocabulary: Iterable[str]) -> dict[str, list[str]]:
    """Returns, for each plain predicate, the vocabulary codes it matches.

    Examples:
        >>> vocabulary_matches(sample_ACES_cfg, ["ICU_ADMISSION", "MEDS_DEATH", "LAB//A"])
        {'icu_admission': ['ICU_ADMISSION'], 'icu_discharge': [], 'death': ['MEDS_DEATH']}
    """
    vocabulary = set(vocabulary)
    return {name: matching_codes(pred, vocabulary) for name, pred in task_cfg.plain_predicates.items()}


def restrict_to_vocabulary(
    task_cfg: TaskExtractorConfig,
    vocabulary: Iterable[str],
    unmatched: UnmatchedPolicy = "marginal",
    marginal_code: str = MARGINAL_CODE,
    drop_value_constraints: bool = True,
    quantized_values: bool = False,
) -> TaskExtractorConfig:
    r"""Rewrites every plain predicate into the vocabulary codes it matches.

    The configuration is modified in place and returned. Predicates matching one code become that code;
    predicates matching several become an ``any`` list; predicates matching none are handled per
    ``unmatched``.

    Args:
        task_cfg: The task configuration to modify.
        vocabulary: The codes the model can express.
        unmatched: What to do with predicates matching no vocabulary code: replace their code with
            ``marginal_code`` (``"marginal"``), leave them untouched (``"keep"``), or raise (``"error"``).
        marginal_code: The sentinel code standing in for unmatched predicates.
        drop_value_constraints: Whether to drop ``value_min`` / ``value_max`` bounds (a weakening). If
            ``False``, predicates with value bounds raise.
        quantized_values: Whether to resolve value-constrained predicates onto the vocabulary's quantized
            value-bin codes (see ``quantized_value_codes``) instead of dropping the constraint. Such a
            predicate then matches its bin codes and carries no value constraint.

    Returns:
        The modified configuration.

    Raises:
        ValueError: Per ``unmatched`` and ``drop_value_constraints``.

    Examples:
        >>> from aces.config import DerivedPredicateConfig, EventConfig, WindowConfig
        >>> def make_cfg():
        ...     return TaskExtractorConfig(
        ...         predicates={
        ...             "admission": PlainPredicateConfig("ADMISSION"),
        ...             "dm": PlainPredicateConfig(code={"regex": r"ICD9//250\..*"}),
        ...             "high_glucose": PlainPredicateConfig("LAB//GLUCOSE", value_min=200),
        ...             "death": PlainPredicateConfig("MEDS_DEATH"),
        ...             "dm_or_death": DerivedPredicateConfig("or(dm, death)"),
        ...         },
        ...         trigger=EventConfig("admission"),
        ...         windows={
        ...             "input": WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end"),
        ...             "target": WindowConfig(
        ...                 "input.end", "start + 30d", False, True, label="dm_or_death",
        ...                 has={"high_glucose": "(None, 0)"},
        ...             ),
        ...         },
        ...     )
        >>> vocab = ["ADMISSION", "ICD9//250.1", "ICD9//250.2", "LAB//GLUCOSE"]
        >>> cfg = restrict_to_vocabulary(make_cfg(), vocab)
        >>> for name, pred in cfg.plain_predicates.items(): print(f"{name}: {pred.code} {pred.value_min}")
        admission: ADMISSION None
        dm: {'any': ['ICD9//250.1', 'ICD9//250.2']} None
        high_glucose: LAB//GLUCOSE None
        death: _MARGINAL_ None

    The derived predicates and windows are untouched, so the configuration remains valid:

        >>> print_ACES(cfg)
        trigger
        └── (+1 day, 0:00:00) input.end; **Prediction Time**
            └── (+30 days, 0:00:00) target.end (no high_glucose); **Label: Presence of dm_or_death**

    Other policies:

        >>> cfg = restrict_to_vocabulary(make_cfg(), vocab, unmatched="keep")
        >>> cfg.plain_predicates["death"].code
        'MEDS_DEATH'
        >>> restrict_to_vocabulary(make_cfg(), vocab, unmatched="error")
        Traceback (most recent call last):
            ...
        ValueError: Predicates match no vocabulary code: ['death'].
        >>> restrict_to_vocabulary(make_cfg(), vocab, drop_value_constraints=False)
        Traceback (most recent call last):
            ...
        ValueError: Predicates have value constraints, which cannot be expressed: ['high_glucose'].

    With ``quantized_values``, a value-constrained predicate becomes the value-bin codes it matches:

        >>> binned = ["ADMISSION", "LAB//GLUCOSE//value_[0.0,200.0)", "LAB//GLUCOSE//value_[200.0,inf)"]
        >>> cfg = restrict_to_vocabulary(make_cfg(), binned, quantized_values=True)
        >>> cfg.plain_predicates["high_glucose"].code, cfg.plain_predicates["high_glucose"].value_min
        ('LAB//GLUCOSE//value_[200.0,inf)', None)

        >>> restrict_to_vocabulary(make_cfg(), vocab, unmatched="sometimes")
        Traceback (most recent call last):
            ...
        ValueError: Unknown unmatched policy 'sometimes'; expected one of ['marginal', 'keep', 'error'].
    """
    if unmatched not in ("marginal", "keep", "error"):
        raise ValueError(
            f"Unknown unmatched policy '{unmatched}'; expected one of ['marginal', 'keep', 'error']."
        )

    vocabulary = list(vocabulary)
    matches = vocabulary_matches(task_cfg, vocabulary)

    with_values = [
        n for n, p in task_cfg.plain_predicates.items() if p.value_min is not None or p.value_max is not None
    ]
    if quantized_values:
        for name in with_values:
            matches[name] = quantized_value_codes(task_cfg.plain_predicates[name], vocabulary)
        drop_value_constraints = True
    if with_values and not drop_value_constraints:
        raise ValueError(f"Predicates have value constraints, which cannot be expressed: {with_values}.")

    unmatched_names = [n for n, m in matches.items() if not m]
    if unmatched_names and unmatched == "error":
        raise ValueError(f"Predicates match no vocabulary code: {unmatched_names}.")

    for name, pred in task_cfg.plain_predicates.items():
        codes = matches[name]
        if codes:
            pred.code = codes[0] if len(codes) == 1 else {"any": codes}
        elif unmatched == "marginal":
            pred.code = marginal_code

        if drop_value_constraints:
            pred.value_min = None
            pred.value_max = None
            pred.value_min_inclusive = None
            pred.value_max_inclusive = None

    return task_cfg
