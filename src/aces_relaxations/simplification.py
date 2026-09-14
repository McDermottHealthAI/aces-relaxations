"""Remove redundant criteria and branches from a set of normalized branches, without changing what they mean.

The normalized form pulls every criterion onto its own branch, which can leave criteria that other parts of
the configuration already imply, and branches that assert nothing. This pass proves such redundancy with the
same symbolic bookkeeping as ``degeneracy``, applied to the branches themselves (positions along a branch are
offsets from anchors; identical searches from identical positions share an anchor, so branches sharing a
prefix share positions), and removes:

- **Same-region criteria windows.** Constrained windows with the same prefix and the same bound are one
  window whose criteria are the conjunction (the largest lower bound and the smallest upper bound of each
  predicate).
- **Implied criteria.** An upper bound of ``k`` on ``P`` over a window is implied by an upper bound of at
  most ``k`` on a predicate ``P`` implies, over a window containing it, or by a hop's gap (no ``P'`` strictly
  between a search's start and the event it finds). A lower bound of ``k`` on ``P`` is implied by a lower
  bound of at least ``k`` on a predicate implying ``P`` over a window inside it, or (for ``k = 1``) by an
  occurrence guaranteed inside it: the event at the prediction time, or the event a hop landed on *whose
  landing some other surviving criterion explicitly asserts* (a lower bound on an implying predicate over
  a window within the search's range and ending no later than the landing). A hop's landing is only a fact
  for rows on which the hop is realizable, and the normalized form makes realizability explicit through
  exactly such criteria ("bounding event occurs"), so a criterion never discharges itself and a shallow
  realizability criterion is never traded for a deeper one: the explicit criteria stay.
- **Empty branches.** A branch ending in an unconstrained temporal window asserts nothing. One ending in an
  unconstrained forward hop asserts that the hop lands, which a surviving lower-bound criterion for an
  implying predicate within the search's range, or an explicitly asserted landing, already guarantees.

Every removal is implied by what remains, so the branch set extracts exactly the same rows with the same
labels; the property tests check this against ACES.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from aces.config import TaskExtractorConfig

from .degeneracy import BOTTOM, RECORD_START, TOP, Analysis, Anchor, CountBound, Interval, Position
from .normalization import RECORD_END, Branch, Edge, _bound_key

Constraint = tuple[int | None, int | None]


@dataclass
class _Landing:
    """An occurrence at a position: the event a hop (searching from ``anchor``) landed on, or one at ``pt``.

    A landing from a hop is only a fact once the hop is known to land, so it carries the search's range.
    """

    predicate: str
    position: Position
    search: Interval | None = None


@dataclass
class _Space:
    """Positions and facts over a set of branches."""

    analysis: Analysis
    landings: list[_Landing] = field(default_factory=list)
    gaps: list[CountBound] = field(default_factory=list)
    _paths: dict[tuple, list[Position]] = field(default_factory=dict)

    def path(self, edges: Iterable[Edge]) -> list[Position]:
        """The positions before and after each edge (identical prefixes get identical positions)."""
        edges = tuple(edges)
        key = tuple((_bound_key(e.bound), e.constraints, e.label) for e in edges)
        if key in self._paths:
            return self._paths[key]
        positions = [Position(0)] if not edges else list(self.path(edges[:-1]))
        if edges:
            positions.append(self._step(positions[-1], edges[-1]))
        self._paths[key] = positions
        return positions

    def _step(self, pos: Position, edge: Edge) -> Position:
        if edge.is_temporal:
            return pos.shifted(edge.bound.window_size)
        if edge.event == RECORD_END:
            return Position(TOP)
        if edge.event == RECORD_START:
            return Position(BOTTOM)
        if edge.is_end_rooted:
            anchor = Anchor(edge.event, upper=pos, strict=not edge.bound.left_inclusive)
        else:
            anchor = Anchor(edge.event, lower=pos, strict=not edge.bound.right_inclusive)
        anchors = self.analysis.anchors
        if anchor not in anchors:
            anchors.append(anchor)
            landing = Position(len(anchors) - 1)
            if edge.is_end_rooted:
                gap = Interval(landing, pos, False, not anchor.strict)
                search = None  # a backward search's range is not bounded below by anything provable
            else:
                gap = Interval(pos, landing, not anchor.strict, False)
                search = Interval(pos, Position(TOP), not anchor.strict, True)
            self.landings.append(_Landing(edge.event, landing, search))
            self.gaps.append(CountBound(edge.event, gap, 0, 0))
        return Position(anchors.index(anchor))

    def region(self, edges: Iterable[Edge]) -> Interval:
        """The interval of a branch's last window."""
        edges = tuple(edges)
        positions = self.path(edges)
        edge, before, after = edges[-1], positions[-2], positions[-1]
        if edge.is_end_rooted:
            return Interval(after, before, edge.bound.left_inclusive, edge.bound.right_inclusive)
        return Interval(before, after, edge.bound.left_inclusive, edge.bound.right_inclusive)


@dataclass(frozen=True)
class _Criterion:
    branch: int
    predicate: str
    at_least: int | None
    at_most: int | None
    region: Interval


def _merge_constraints(
    groups: Iterable[tuple[tuple[str, Constraint], ...]],
) -> tuple[tuple[str, Constraint], ...]:
    merged: dict[str, list[int | None]] = {}
    for constraints in groups:
        for predicate, (lo, hi) in constraints:
            current = merged.setdefault(predicate, [None, None])
            if lo is not None:
                current[0] = lo if current[0] is None else max(current[0], lo)
            if hi is not None:
                current[1] = hi if current[1] is None else min(current[1], hi)
    return tuple(sorted((p, (lo, hi)) for p, (lo, hi) in merged.items()))


def _merge_same_region(branches: list[Branch]) -> list[Branch]:
    """Merges constrained, unlabeled terminals sharing a prefix and a bound into one window."""
    groups: dict[tuple, list[int]] = {}
    for i, branch in enumerate(branches):
        terminal = branch.terminal
        if not terminal.is_constrained or terminal.label is not None:
            continue
        prefix = tuple((_bound_key(e.bound), e.constraints, e.label) for e in branch.edges[:-1])
        groups.setdefault((prefix, _bound_key(terminal.bound)), []).append(i)
    out = list(branches)
    dropped: set[int] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        first = branches[members[0]]
        constraints = _merge_constraints(branches[i].terminal.constraints for i in members)
        name = "__".join(branches[i].terminal.name for i in members)
        out[members[0]] = Branch(
            (*first.edges[:-1], replace(first.terminal, name=name, constraints=constraints))
        )
        dropped.update(members[1:])
    return [b for i, b in enumerate(out) if i not in dropped]


def simplify_branches(
    branches: Iterable[Branch],
    implies: Callable[[str, str], bool],
    at_prediction_time: Iterable[str] = (),
    task_cfg: TaskExtractorConfig | None = None,
) -> tuple[Branch, ...]:
    """Removes redundant criteria and branches (see the module docstring).

    Args:
        branches: The normalized branches (label branch included).
        implies: ``implies(g, p)``: an occurrence of predicate ``g`` is also an occurrence of ``p``.
        at_prediction_time: Predicates guaranteed to occur exactly at the prediction time.
        task_cfg: The configuration, if available (only used to render positions in messages).

    Returns:
        The simplified branches, in the input order.

    Examples:
        >>> from aces.types import TemporalWindowBounds, ToEventWindowBounds
        >>> from .future_only import _implies
        >>> implies = _implies(in_hosp_mortality_cfg)
        >>> gap = Edge("gap", TemporalWindowBounds(False, timedelta(days=1), True))
        >>> target = Edge("target", ToEventWindowBounds(False, "discharge_or_death", True), label="death")
        >>> label = Branch((gap, target))
        >>> def one(name, days, has):
        ...     return Branch((Edge(name, TemporalWindowBounds(False, timedelta(days=days), True), has),))

        Two same-region windows merge, and a criterion implied by another on a containing window goes:

        >>> a = one("a", 5, (("admission", (None, 0)),))
        >>> b = one("b", 2, (("admission", (None, 0)),))
        >>> c = one("c", 2, (("death", (1, None)),))
        >>> for br in simplify_branches([label, a, b, c], implies): print(br)
        gap: start + 1 day -> target: start -> discharge_or_death; label=death
        a: start + 5 days (admission: (None, 0))
        b__c: start + 2 days (death: (1, None))

        A "bounding event occurs" criterion on a window's own hop is kept, but one implied by an earlier hop
        on the branch goes, and so does the now-empty branch behind it:

        >>> occurs = (("discharge", (1, None)),)
        >>> again = Edge("again_bounding_event", ToEventWindowBounds(True, "discharge", True), occurs)
        >>> stay = Edge("stay", ToEventWindowBounds(False, "discharge", True))
        >>> own_edge = Edge("stay_bounding_event", ToEventWindowBounds(False, "discharge", True), occurs)
        >>> own = Branch((own_edge,))
        >>> for br in simplify_branches([label, own, Branch((stay, again))], implies): print(br)
        gap: start + 1 day -> target: start -> discharge_or_death; label=death
        stay_bounding_event: start -> discharge (discharge: (1, None))

        An unconstrained hop whose landing a criterion guarantees is redundant, as is an unconstrained
        temporal window:

        >>> bare = Branch((Edge("stay", ToEventWindowBounds(False, "discharge", True)),))
        >>> week = Branch((Edge("week", TemporalWindowBounds(False, timedelta(days=7), True)),))
        >>> for br in simplify_branches([label, own, bare, week], implies): print(br)
        gap: start + 1 day -> target: start -> discharge_or_death; label=death
        stay_bounding_event: start -> discharge (discharge: (1, None))

        An unconstrained hop that nothing proves to land is kept, since ACES drops rows on which it does
        not:

        >>> for br in simplify_branches([label, bare], implies): print(br)
        gap: start + 1 day -> target: start -> discharge_or_death; label=death
        stay: start -> discharge

        Record boundaries are positions, not events: a look-back to the record's start is kept as is, and
        the record's start being at the prediction time guarantees no occurrence of anything:

        >>> since = Edge("since_start", ToEventWindowBounds(True, "-_RECORD_START", True), occurs)
        >>> for br in simplify_branches([label, Branch((since,))], implies, ["_RECORD_START"]): print(br)
        gap: start + 1 day -> target: start -> discharge_or_death; label=death
        since_start: end <- _RECORD_START (discharge: (1, None))
    """

    def extended(g: str, p: str) -> bool:
        # Record boundaries are positions, not predicates.
        if g.startswith("_RECORD_") or p.startswith("_RECORD_"):
            return False
        return p == "_ANY_EVENT" or implies(g, p)

    space = _Space(Analysis(task_cfg, extended, anchors=[Anchor(None)]))
    space.landings.extend(_Landing(p, Position(0)) for p in at_prediction_time)

    current = _merge_same_region(list(branches))
    # Positions and facts for every branch (prefix positions are shared).
    for branch in current:
        space.path(branch.edges)
    an = space.analysis

    def criteria() -> list[_Criterion]:
        out = []
        for i, branch in enumerate(current):
            if branch.terminal.label is not None:
                continue
            region = space.region(branch.edges)
            for predicate, (lo, hi) in branch.terminal.constraints:
                out.append(_Criterion(i, predicate, lo, hi, region))
        return out

    def asserted(landing: _Landing, alive: list[_Criterion], exclude: tuple) -> bool:
        """Whether a surviving criterion (other than ``exclude``) guarantees the landing's search lands."""
        return any(
            (o.branch, o.predicate) != exclude
            and o.at_least is not None
            and o.at_least >= 1
            and an.implies(o.predicate, landing.predicate)
            and an.contains(landing.search, o.region)
            and an.leq(o.region.hi, landing.position)
            for o in alive
        )

    def usable_landings(alive: list[_Criterion], exclude: tuple = ()) -> list[_Landing]:
        """Landings that are facts: at the prediction time, or explicitly asserted by another criterion."""
        return [
            ld
            for ld in space.landings
            if (ld.search is None and ld.position == Position(0))
            or (ld.search is not None and asserted(ld, alive, exclude))
        ]

    # Drop implied criteria, one at a time so that mutually implying criteria keep one copy.
    for c in criteria():
        alive = criteria()
        others = [o for o in alive if (o.branch, o.predicate) != (c.branch, c.predicate)]
        landings = usable_landings(alive, (c.branch, c.predicate))
        lo, hi = c.at_least, c.at_most
        upper_implied = hi is not None and (
            any(
                o.at_most is not None
                and o.at_most <= hi
                and an.implies(c.predicate, o.predicate)
                and an.contains(o.region, c.region)
                for o in others
            )
            or any(
                an.implies(c.predicate, g.predicate) and an.contains(g.interval, c.region) for g in space.gaps
            )
        )
        lower_implied = (
            lo is not None
            and lo >= 1
            and (
                any(
                    o.at_least is not None
                    and o.at_least >= lo
                    and an.implies(o.predicate, c.predicate)
                    and an.contains(c.region, o.region)
                    for o in others
                )
                or (
                    lo == 1
                    and any(
                        an.implies(ld.predicate, c.predicate)
                        and an.contains(c.region, Interval.point(ld.position))
                        for ld in landings
                    )
                )
            )
        )
        if upper_implied:
            hi = None
        if lower_implied:
            lo = None
        if (lo, hi) != (c.at_least, c.at_most):
            branch = current[c.branch]
            kept = tuple(
                (p, (lo, hi) if p == c.predicate else bound)
                for p, bound in branch.terminal.constraints
                if p != c.predicate or lo is not None or hi is not None
            )
            current[c.branch] = Branch((*branch.edges[:-1], replace(branch.terminal, constraints=kept)))

    # Drop branches that assert nothing.
    alive = criteria()
    landings = usable_landings(alive)
    kept_branches: list[Branch] = []
    for branch in current:
        terminal = branch.terminal
        if terminal.label is not None or terminal.is_constrained:
            kept_branches.append(branch)
            continue
        if terminal.is_temporal and not terminal.is_end_rooted:
            continue
        if terminal.is_end_rooted or terminal.event == RECORD_END:
            kept_branches.append(branch)
            continue
        positions = space.path(branch.edges)
        anchor_pos, strict = positions[-2], not terminal.bound.right_inclusive
        own = _Landing(terminal.event, positions[-1], Interval(anchor_pos, Position(TOP), not strict, True))
        realizable = asserted(own, alive, ()) or any(
            an.implies(ld.predicate, terminal.event) and an.leq(anchor_pos, ld.position, strict=strict)
            for ld in landings
        )
        if not realizable:
            kept_branches.append(branch)
    return tuple(kept_branches)
