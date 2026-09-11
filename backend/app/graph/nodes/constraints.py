"""Deterministic constraint check: does the generated itinerary respect the
traveller's stated pace/budget? If not, and the graph hasn't hit
MAX_REFINE_ITERATIONS yet, `should_refine` routes back to the itinerary
node with the violations attached to state so it can be told what to fix.

Violations are collected both as a flat list (`constraint_violations` --
the human-readable form finalize.py/the UI show) and bucketed by day
offset (`violations_by_day`) -- itinerary_node only re-plans days that
appear in the latter, leaving every other day exactly as it was on the
previous attempt.

This loop is intentionally bounded -- see MAX_REFINE_ITERATIONS in
app/graph/state.py -- so a stubborn LLM can't spin the graph forever.
"""

from __future__ import annotations

from typing import Literal

from app.graph.nodes._shared import (
    MAX_PRICE_LEVEL,
    PRICE_LEVEL_RANK,
    STOPS_PER_DAY_RANGE,
    attraction_lookup,
    estimate_travel_minutes,
    is_open_during,
    is_open_on,
    minutes_to_clock,
    realistic_transit_minutes,
    time_to_minutes,
)
from app.graph.state import MAX_REFINE_ITERATIONS, AgentState
from app.i18n import (
    violation_beyond_window,
    violation_budget,
    violation_closed,
    violation_day_underfilled,
    violation_duplicate_place,
    violation_gap_too_short,
    violation_outside_hours,
    violation_stop_count,
)


# The itinerary node now hands us a schedule the deterministic scheduler
# already made feasible, so the timing checks below are a backstop, not the
# primary enforcement. Travel time is a rough estimate either way (haversine
# or an OSRM matrix), so a shortfall inside this margin isn't worth a whole
# refine iteration -- it's noise, not a real conflict.
_GAP_TOLERANCE_MIN = 5

# A day that, with all its stops scheduled, still finishes before this
# fraction of the window has passed -- and has room for another stop -- is
# too thin: the fix is "pick more", handled by the refine loop, NOT by
# padding the gaps (that was the old ends_too_early check, and it was wrong).
_UNDERFILL_RATIO = 0.55


def check_constraints_node(state: AgentState) -> dict:
    trip_setup = state["trip_setup"]
    lang = trip_setup.get("lang", "en")
    itinerary = state.get("itinerary", [])
    attractions_by_id = attraction_lookup(state.get("ranked_attractions", []))
    lo, hi = STOPS_PER_DAY_RANGE[trip_setup["daily_pace"]]
    max_price = MAX_PRICE_LEVEL[trip_setup["budget_level"]]
    day_start_min = time_to_minutes(trip_setup["day_start_time"])
    window_min = time_to_minutes(trip_setup["day_end_time"]) - day_start_min

    violations: list[str] = []
    violations_by_day: dict[int, list[str]] = {}

    def add(offset: int, message: str) -> None:
        violations.append(message)
        violations_by_day.setdefault(offset, []).append(message)

    def real_stops(day: dict) -> list[dict]:
        return [s for s in day["stops"] if s.get("kind", "attraction") == "attraction"]

    # A day can't be asked to add stops that don't exist: if every ranked
    # candidate is already scheduled somewhere, an under-full day is the
    # candidate pool's fault, not the LLM's, and re-planning won't help.
    used_anywhere = {stop["place_id"] for d in itinerary for stop in real_stops(d)}
    pool_exhausted = bool(attractions_by_id) and used_anywhere >= set(attractions_by_id)

    # Likewise for budget: if there aren't `lo` candidates within the budget
    # cap, a day of `lo`+ stops *has* to include pricier ones -- that's the
    # city, not the LLM. finalize.py turns this into an advisory note.
    affordable = sum(
        1
        for a in attractions_by_id.values()
        if a.price_level is None or PRICE_LEVEL_RANK.get(a.price_level, 0) <= max_price
    )
    budget_unavoidable = bool(attractions_by_id) and affordable < lo

    first_seen_date: dict[str, str] = {}  # place_id -> date_str of its first-scheduled day
    for offset, day in enumerate(itinerary):
        date_str = str(day["date"])
        stops = real_stops(day)
        stop_count = len(stops)
        if stop_count > hi or (stop_count < lo and not pool_exhausted):
            add(offset, violation_stop_count(lang, date_str, stop_count, trip_setup["daily_pace"], lo, hi))

        price_levels = [
            PRICE_LEVEL_RANK[attractions_by_id[stop["place_id"]].price_level]
            for stop in stops
            if stop["place_id"] in attractions_by_id
            and attractions_by_id[stop["place_id"]].price_level in PRICE_LEVEL_RANK
        ]
        if price_levels:
            avg_price = sum(price_levels) / len(price_levels)
            if avg_price > max_price and not budget_unavoidable:
                add(offset, violation_budget(lang, date_str, avg_price, trip_setup["budget_level"], max_price))

        for stop in stops:
            attraction = attractions_by_id.get(stop["place_id"])
            if attraction and not is_open_on(attraction, day["date"]):
                add(offset, violation_closed(lang, date_str, attraction.name, day["date"].weekday()))
            elif attraction:
                arrival_min = day_start_min + stop["arrival_offset_min"]
                departure_min = arrival_min + stop["dwell_minutes"]
                if not is_open_during(attraction, day["date"], arrival_min, departure_min):
                    open_min, close_min = attraction.opening_hours[day["date"].weekday()]
                    add(
                        offset,
                        violation_outside_hours(
                            lang,
                            date_str,
                            attraction.name,
                            minutes_to_clock(arrival_min),
                            minutes_to_clock(departure_min),
                            minutes_to_clock(open_min),
                            minutes_to_clock(close_min),
                        ),
                    )

            place_id = stop["place_id"]
            if place_id in first_seen_date:
                # Attributed to *this* (later) day -- the earlier day that
                # first claimed the place keeps it, this one gives it up.
                name = attraction.name if attraction else place_id
                add(offset, violation_duplicate_place(lang, date_str, name, first_seen_date[place_id]))
            else:
                first_seen_date[place_id] = date_str

        ordered_stops = sorted(
            (s for s in stops if s["place_id"] in attractions_by_id),
            key=lambda s: s["arrival_offset_min"],
        )
        for prev_stop, next_stop in zip(ordered_stops, ordered_stops[1:]):
            prev_attraction = attractions_by_id[prev_stop["place_id"]]
            next_attraction = attractions_by_id[next_stop["place_id"]]
            gap_min = next_stop["arrival_offset_min"] - prev_stop["arrival_offset_min"]
            # Prefer the travel time the scheduler actually used (stored on
            # the stop); fall back to a fresh estimate for stops that predate
            # that field.
            travel_min = next_stop.get("travel_from_prev_min")
            if travel_min is None:
                travel_min = realistic_transit_minutes(estimate_travel_minutes(prev_attraction, next_attraction))
            # Gap only needs to cover finishing *prev* (its own dwell time)
            # plus the trip over -- next's dwell time is what it does once
            # it's arrived, not what it costs to get there.
            needed_min = prev_stop["dwell_minutes"] + travel_min
            if gap_min < needed_min - _GAP_TOLERANCE_MIN:
                add(
                    offset,
                    violation_gap_too_short(
                        lang, date_str, gap_min, prev_attraction.name, next_attraction.name, needed_min
                    ),
                )

        if ordered_stops:
            last_stop_end = ordered_stops[-1]["arrival_offset_min"] + ordered_stops[-1]["dwell_minutes"]
            if last_stop_end > window_min:
                add(
                    offset,
                    violation_beyond_window(
                        lang,
                        date_str,
                        ordered_stops[-1]["attraction_name"],
                        last_stop_end,
                        window_min,
                        f"{trip_setup['day_start_time']:%H:%M}",
                        f"{trip_setup['day_end_time']:%H:%M}",
                    ),
                )
            elif stop_count < hi and not pool_exhausted and last_stop_end < window_min * _UNDERFILL_RATIO:
                add(offset, violation_day_underfilled(lang, date_str, last_stop_end, window_min, hi))

    return {"constraint_violations": violations, "violations_by_day": violations_by_day}


def should_refine(state: AgentState) -> Literal["itinerary", "finalize"]:
    violations = state.get("constraint_violations", [])
    refine_iterations = state.get("refine_iterations", 0)
    if violations and refine_iterations < MAX_REFINE_ITERATIONS:
        return "itinerary"
    return "finalize"
