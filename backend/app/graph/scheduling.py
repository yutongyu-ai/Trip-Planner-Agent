"""Deterministic day scheduler for the itinerary node.

`itinerary_node`'s LLM decides *which* attractions go on a day and roughly
how long to spend at each. This module decides *when*: the arithmetic --
lining each visit up with the place's opening hours, leaving room for travel
between stops, and ordering the stops to keep total travel down -- that an
LLM does unreliably. (It will cheerfully schedule a 09:00 visit to a place
that opens at 11:00, or leave 20 minutes for a 40-minute crossing, and keep
doing it after being told not to -- see the refine loop's failure mode.)

`schedule_day` takes the day's chosen stops and returns a concrete timeline,
scheduled tight -- each stop starts as early as travel and opening hours
allow. It does not pad the day out to some target end time: a day that
genuinely only needs four hours should end after four hours, and "the day
finishes early" is a signal for the itinerary node to pick more stops, not
for the scheduler to inflate the gaps. When the stops can't all fit the
window it drops the lowest-priority ones (highest `rank` -- the recommender
already decided they matter least) and reports what it dropped.

Ordering is a tiny travelling-salesman-with-time-windows search: for
`_MAX_EXACT_STOPS` stops or fewer every visit order is tried and the
feasible one with the least total travel wins; above that (never, at the
current pace caps) a nearest-neighbour heuristic seeded by opening time is
used. "Feasible" means the greedy forward schedule for that order never
arrives before a place opens or leaves after it closes, and the day ends by
`day_end_min`.

Everything here is pure and deterministic. Travel time is supplied as a
callable, so the scheduler has no opinion on how distances are estimated --
a real OSRM matrix in production, a stub in tests.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace

# 7! = 5040 orderings, each scored in O(n); the "packed" pace caps a day at
# 6 stops, so the heuristic branch below is currently unreachable in prod
# and exists only so an oversized call can't blow up.
_MAX_EXACT_STOPS = 7

TravelFn = Callable[[str, str], float]


@dataclass(frozen=True)
class ScheduleInput:
    """One attraction the LLM chose for this day, plus the facts the
    scheduler needs. `open_min`/`close_min` are clock-minutes since midnight
    on the visit day for that weekday (from `Attraction.opening_hours`);
    pass ``None`` for both when hours are unknown -- the stop is then treated
    as open all day (the same "unknown, don't penalise it" convention as
    `app.graph.nodes._shared.is_open_during`). `rank` is the drop priority
    when the day is overbooked: lower stays, higher goes first (use the
    attraction's position in the ranked list).
    """

    place_id: str
    name: str
    dwell_minutes: int
    open_min: int | None
    close_min: int | None
    rank: int


@dataclass(frozen=True)
class ScheduledStop:
    place_id: str
    name: str
    arrival_offset_min: int  # minutes after day_start_min
    depart_offset_min: int  # arrival_offset_min + dwell_minutes
    travel_from_prev_min: int  # 0 for the first stop
    # Minutes spent waiting at the door because the traveller got here before
    # the place opened. 0 unless the arrival was pushed later by open_min.
    wait_for_opening_min: int
    kind: str = "attraction"  # or "meal" -- a break inserted by schedule_day_with_lunch
    open_min: int | None = None  # carried through so _insert_lunch can re-time the tail
    close_min: int | None = None


@dataclass(frozen=True)
class ScheduledDay:
    stops: list[ScheduledStop]  # chronological, in visit order
    dropped: list[str]  # names of stops that didn't fit, in the order dropped


@dataclass(frozen=True)
class _Segment:
    arrival: int
    depart: int
    travel: int
    wait: int


def schedule_day(
    stops: Sequence[ScheduleInput],
    *,
    day_start_min: int,
    day_end_min: int,
    travel_minutes: TravelFn,
) -> ScheduledDay:
    """Schedule `stops` into ``[day_start_min, day_end_min]`` (both
    clock-minutes since midnight), tight -- each stop as early as travel and
    opening hours allow. Drops the lowest-priority stops until the rest fit.
    """
    remaining = sorted(stops, key=lambda s: s.rank)
    dropped: list[str] = []

    while remaining:
        order = _best_order(remaining, day_start_min, day_end_min, travel_minutes)
        if order is None:
            # Nothing fits with the full set -- find the one stop whose
            # removal makes the rest feasible, preferring the lowest-priority
            # one when more than one eviction would work (so priority still
            # governs the common "too many stops" case). Checking in
            # lowest-priority-first order and stopping at the first eviction
            # that works also catches the pathological case a naive
            # always-drop-the-last-one rule misses: a single stop (of any
            # rank) whose own hours can never fit this window would otherwise
            # get carried through every iteration, dragging perfectly
            # schedulable lower-priority stops down with it until nothing is
            # left, when only that one stop needed to go.
            for candidate in reversed(remaining):
                rest = [s for s in remaining if s is not candidate]
                if not rest or _best_order(rest, day_start_min, day_end_min, travel_minutes) is not None:
                    dropped.append(candidate.name)
                    remaining = rest
                    break
            else:
                dropped.append(remaining[-1].name)
                remaining = remaining[:-1]
            continue

        segments = _forward(order, day_start_min, day_end_min, travel_minutes)
        assert segments is not None  # _best_order only returns feasible orders
        return ScheduledDay(
            stops=[
                ScheduledStop(
                    place_id=stop.place_id,
                    name=stop.name,
                    arrival_offset_min=seg.arrival - day_start_min,
                    depart_offset_min=seg.depart - day_start_min,
                    travel_from_prev_min=seg.travel,
                    wait_for_opening_min=seg.wait,
                    open_min=stop.open_min,
                    close_min=stop.close_min,
                )
                for stop, seg in zip(order, segments)
            ],
            dropped=dropped,
        )

    return ScheduledDay(stops=[], dropped=dropped)


def _candidate_orders(
    stops: Sequence[ScheduleInput], day_start_min: int, travel_minutes: TravelFn
) -> Iterator[tuple[ScheduleInput, ...]]:
    if len(stops) <= _MAX_EXACT_STOPS:
        yield from itertools.permutations(stops)
        return
    # Heuristic fallback: earliest-opening first, then nearest unvisited.
    ordered = sorted(stops, key=lambda s: s.open_min if s.open_min is not None else day_start_min)
    chain = [ordered[0]]
    pool = ordered[1:]
    while pool:
        last = chain[-1]
        nxt = min(pool, key=lambda s: travel_minutes(last.place_id, s.place_id))
        chain.append(nxt)
        pool.remove(nxt)
    yield tuple(chain)


def _best_order(
    stops: Sequence[ScheduleInput],
    day_start_min: int,
    day_end_min: int,
    travel_minutes: TravelFn,
) -> tuple[ScheduleInput, ...] | None:
    best: tuple[ScheduleInput, ...] | None = None
    best_key: tuple[int, int] | None = None
    for order in _candidate_orders(stops, day_start_min, travel_minutes):
        segments = _forward(order, day_start_min, day_end_min, travel_minutes)
        if segments is None:
            continue
        # Least total travel + door-waiting, tie-broken by the earlier finish.
        cost = sum(seg.travel + seg.wait for seg in segments)
        key = (cost, segments[-1].depart)
        if best_key is None or key < best_key:
            best, best_key = order, key
    return best


def _forward(
    order: Sequence[ScheduleInput],
    day_start_min: int,
    day_end_min: int,
    travel_minutes: TravelFn,
) -> list[_Segment] | None:
    """Greedy forward schedule for a fixed visit order, or ``None`` if it
    can't be made feasible (arrives after a close, or runs past day end)."""
    segments: list[_Segment] = []
    clock = day_start_min
    prev_id: str | None = None
    for stop in order:
        travel = 0 if prev_id is None else max(0, round(travel_minutes(prev_id, stop.place_id)))
        arrival = clock + travel
        wait = 0
        if stop.open_min is not None and arrival < stop.open_min:
            wait = stop.open_min - arrival
            arrival = stop.open_min
        depart = arrival + stop.dwell_minutes
        if stop.close_min is not None and depart > stop.close_min:
            return None
        if depart > day_end_min:
            return None
        segments.append(_Segment(arrival=arrival, depart=depart, travel=travel, wait=wait))
        clock = depart
        prev_id = stop.place_id
    return segments


# -- lunch --------------------------------------------------------------------
#
# Lunch is inserted *after* the day's stops are scheduled, as a gap in an
# already-optimised route -- not reserved upfront, splitting the day into two
# independently-solved halves. An earlier version of this module tried that
# split: it guaranteed lunch but decided which stops fell before/after it by
# each half's own capacity alone, with no way to know that two stops happen
# to sit right next to each other -- a real run put two adjacent London parks
# at opposite ends of the day, with a cross-town landmark pair sandwiched
# between them, because one park landed in the morning half and the other in
# the afternoon half. Scheduling the whole day as one problem first keeps
# `schedule_day`'s own travel-minimising order intact; lunch then has to find
# a place *inside* that order rather than being allowed to fragment it.

LUNCH_PLACE_ID = "__meal_lunch__"
_LUNCH_WINDOW = (12 * 60, 14 * 60)  # a lunch break must *start* in this range
# Below this many minutes, shrinking lunch further isn't worth it -- give up
# on that gap and try the next-nearest one instead.
_LUNCH_DWELL_FLOOR = 20


def schedule_day_with_lunch(
    stops: Sequence[ScheduleInput],
    *,
    day_start_min: int,
    day_end_min: int,
    travel_minutes: TravelFn,
    lunch_dwell_minutes: int,
    lunch_label: str,
) -> ScheduledDay:
    """`schedule_day`, then `_insert_lunch`, then `_absorb_waits` on the result."""
    day = schedule_day(stops, day_start_min=day_start_min, day_end_min=day_end_min, travel_minutes=travel_minutes)
    day = _insert_lunch(
        day,
        day_start_min=day_start_min,
        day_end_min=day_end_min,
        lunch_dwell_minutes=lunch_dwell_minutes,
        lunch_label=lunch_label,
    )
    return _absorb_waits(day, day_start_min=day_start_min)


def _absorb_waits(day: ScheduledDay, *, day_start_min: int) -> ScheduledDay:
    """Final pass over the finished day (attractions and lunch alike):
    convert each stop's `wait_for_opening_min` into extra time at the
    *previous* entry instead -- same total elapsed time, just spent
    lingering somewhere already open (or over lunch) rather than standing
    at a locked door. Run last, after lunch has already claimed whatever
    gap it could: a stop that still waits after that is better served by a
    longer lunch than by extending some attraction's dwell to a length that
    has nothing to do with the place itself. Only the day's very first
    stop, with nothing earlier to linger at, can still show a real wait.
    """
    stops = list(day.stops)
    for i in range(1, len(stops)):
        stop = stops[i]
        if stop.wait_for_opening_min <= 0:
            continue
        prev = stops[i - 1]
        extended_depart = prev.depart_offset_min + stop.wait_for_opening_min
        if prev.close_min is not None and extended_depart > prev.close_min - day_start_min:
            continue  # would run the previous stop past its own close time -- leave the wait as a wait
        stops[i - 1] = replace(prev, depart_offset_min=extended_depart)
        stops[i] = replace(stop, wait_for_opening_min=0)
    return ScheduledDay(stops=stops, dropped=day.dropped)


def _insert_lunch(
    day: ScheduledDay, *, day_start_min: int, day_end_min: int, lunch_dwell_minutes: int, lunch_label: str
) -> ScheduledDay:
    """Post-pass: put a lunch break into an already-scheduled day, without
    touching the visiting order `schedule_day` chose to minimise travel.

    Tries every between-stop gap, nearest to 13:00 first (shrinking lunch
    toward `_LUNCH_DWELL_FLOOR` if the full `lunch_dwell_minutes` would push a
    stop past its close time or the day's end), and takes the first gap that
    works -- rather than only ever trying the single nearest gap and giving
    up entirely if that one happens to be infeasible while a farther one
    wasn't. The chosen gap sits right after that stop (you eat where you
    are, travel_from_prev = 0); the tail is re-timed forward from the end of
    lunch, so travel to the next stop still happens *after* eating and a
    later opening wait can absorb part of the break instead of the day
    getting longer. No-op when the day doesn't span lunch, has < 2 stops, or
    no gap can fit even a floor-length break.
    """
    stops = day.stops
    if len(stops) < 2:
        return day

    lunch_lo, lunch_hi = _LUNCH_WINDOW
    first_arr = day_start_min + stops[0].arrival_offset_min
    last_dep = day_start_min + stops[-1].depart_offset_min
    if first_arr >= lunch_hi or last_dep <= lunch_lo:
        return day  # the day is entirely before or after lunch

    target = (lunch_lo + lunch_hi) // 2
    day_len = day_end_min - day_start_min
    gaps_by_proximity = sorted(
        range(len(stops) - 1),
        key=lambda i: abs(
            ((day_start_min + stops[i].depart_offset_min) + (day_start_min + stops[i + 1].arrival_offset_min)) // 2
            - target
        ),
    )

    for i in gaps_by_proximity:
        before, tail = stops[i], stops[i + 1 :]

        def _retime_tail(meal_dwell: int) -> list[ScheduledStop] | None:
            """Forward-schedule `tail` starting from the end of a `meal_dwell`
            lunch that begins at `before.depart_offset_min`; None if infeasible."""
            out: list[ScheduledStop] = []
            clock = before.depart_offset_min + meal_dwell
            for s in tail:
                arrival = clock + s.travel_from_prev_min
                wait = 0
                if s.open_min is not None and arrival < s.open_min - day_start_min:
                    wait = (s.open_min - day_start_min) - arrival
                    arrival = s.open_min - day_start_min
                depart = arrival + (s.depart_offset_min - s.arrival_offset_min)
                if s.close_min is not None and depart > s.close_min - day_start_min:
                    return None
                if depart > day_len:
                    return None
                out.append(replace(s, arrival_offset_min=arrival, depart_offset_min=depart, wait_for_opening_min=wait))
                clock = depart
            return out

        retimed = _retime_tail(lunch_dwell_minutes)
        dwell = lunch_dwell_minutes
        if retimed is None:
            lo, hi = _LUNCH_DWELL_FLOOR, lunch_dwell_minutes - 1
            dwell, retimed = 0, None
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = _retime_tail(mid)
                if candidate is not None:
                    dwell, retimed = mid, candidate
                    lo = mid + 1
                else:
                    hi = mid - 1

        if retimed is not None:
            meal = ScheduledStop(
                place_id=LUNCH_PLACE_ID,
                name=lunch_label,
                arrival_offset_min=before.depart_offset_min,
                depart_offset_min=before.depart_offset_min + dwell,
                travel_from_prev_min=0,
                wait_for_opening_min=0,
                kind="meal",
            )
            return ScheduledDay(stops=[*stops[: i + 1], meal, *retimed], dropped=day.dropped)

    return day  # no gap could fit even a floor-length break -- leave the day alone
