"""Builds the user-facing summary and advisory notes once the refine loop is
done (or capped).

Runs exactly once, after `should_refine` decides to stop looping, so it
never has to guess whether more refine attempts are still coming.

`summary` is the one-line trip headline; `notes` is the list of advisory
lines (partial forecast, low-coverage destination, unresolved constraints),
already localized, shown next to the itinerary.
"""

from __future__ import annotations

from app.graph.nodes._shared import MAX_PRICE_LEVEL, PRICE_LEVEL_RANK, STOPS_PER_DAY_RANGE, attraction_lookup
from app.graph.state import AgentState
from app.i18n import (
    budget_options_limited,
    destination_low_coverage,
    refine_limit_reached,
    stops_dropped_to_fit,
    trip_planned,
    weather_beyond_horizon,
    weather_lookup_failed,
)


def _budget_options_limited(state: AgentState, itinerary: list) -> bool:
    """True when the destination has fewer within-budget attractions than a
    day needs, so the plan unavoidably includes pricier stops (constraints.py
    suppresses the hard violation in that case; this surfaces it as a note)."""
    trip_setup = state["trip_setup"]
    attractions_by_id = attraction_lookup(state.get("ranked_attractions", []))
    if not attractions_by_id or "budget_level" not in trip_setup or "daily_pace" not in trip_setup:
        return False
    max_price = MAX_PRICE_LEVEL[trip_setup["budget_level"]]
    lo, _ = STOPS_PER_DAY_RANGE[trip_setup["daily_pace"]]
    affordable = sum(
        1
        for a in attractions_by_id.values()
        if a.price_level is None or PRICE_LEVEL_RANK.get(a.price_level, 0) <= max_price
    )
    over_budget_stop = any(
        (a := attractions_by_id.get(stop["place_id"])) is not None
        and a.price_level in PRICE_LEVEL_RANK
        and PRICE_LEVEL_RANK[a.price_level] > max_price
        for day in itinerary
        for stop in day["stops"]
    )
    return affordable < lo and over_budget_stop


def finalize_node(state: AgentState) -> dict:
    trip_setup = state["trip_setup"]
    lang = trip_setup.get("lang", "en")
    itinerary = state.get("itinerary", [])
    total_stops = sum(
        1 for day in itinerary for stop in day["stops"] if stop.get("kind", "attraction") == "attraction"
    )

    headline = trip_planned(lang, trip_setup["destination"], len(itinerary), total_stops)
    notes: list[str] = []

    errors = state.get("errors", [])
    weather_error = next((e for e in errors if e.startswith("weather:")), None)
    missing_forecast_days = [day for day in itinerary if day["weather_note"] is None]
    if weather_error:
        # A real API failure, not "trip is beyond the 5-day horizon" -- say
        # so. The "weather: " tag itself is a stable, unlocalized marker
        # (see app/i18n.py) -- only the reason text after it is real content,
        # and that's the raw exception message, which stays untranslated.
        notes.append(weather_lookup_failed(lang, weather_error.removeprefix("weather: ")))
    elif missing_forecast_days:
        notes.append(weather_beyond_horizon(lang, len(missing_forecast_days)))

    if state.get("low_coverage_destination"):
        notes.append(destination_low_coverage(lang))

    dropped = [name for day in itinerary for name in day.get("dropped_stops", [])]
    if dropped:
        notes.append(stops_dropped_to_fit(lang, dropped))

    if _budget_options_limited(state, itinerary):
        notes.append(budget_options_limited(lang))

    violations = state.get("constraint_violations") or []
    if violations:
        notes.append(refine_limit_reached(lang, state.get("refine_iterations", 0), violations))

    return {"summary": headline, "notes": notes}
