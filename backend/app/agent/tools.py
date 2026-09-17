"""Tools for the conversational itinerary-edit agent (see app.agent.graph).

Mirrors itinerary_node's division of labour (app/graph/nodes/itinerary.py):
the model only ever *picks* a place_id -- arrival time, visit order and
whether an edit even fits the day are always worked out by the same
deterministic scheduler the generation pipeline uses
(`app.graph.scheduling.schedule_day_with_lunch`). No tool here ever writes
an arrival/departure time directly; every edit re-derives the whole day from
scratch through `_reschedule_day`, the same way a refine pass re-derives a
day in itinerary_node.

Each tool validates its arguments against the actual session state (see
app.agent.guardrails) before touching anything, and returns a plain string
on invalid input rather than raising -- the model sees the problem in its
next turn and can correct course, the same "never silently corrupt state"
discipline itinerary_node applies to hallucinated place_ids.

Known scope limit: an edit only has to satisfy the scheduler's own
feasibility (opening hours, travel time, lunch) -- it is never re-run
through check_constraints_node, so a chat edit *can* leave the itinerary
short of the traveller's stated pace/budget range without saying so (e.g.
removing a stop can drop a day below its minimum stop count). Generation
enforces that; editing doesn't yet.
"""

from __future__ import annotations

import datetime as dt

from langchain_core.tools import tool

from app.agent.guardrails import (
    validate_day_offset,
    validate_known_place_id,
    validate_place_not_used_anywhere,
    validate_place_used_on_day,
)
from app.graph.nodes._shared import (
    LUNCH_DWELL_MINUTES,
    PRICE_LEVEL_RANK,
    attraction_lookup,
    category_dwell_minutes,
    estimate_travel_minutes,
    is_open_on,
    minutes_to_clock,
    ranked_centroid,
    realistic_transit_minutes,
    time_to_minutes,
)
from app.graph.scheduling import ScheduleInput, schedule_day_with_lunch
from app.graph.state import AgentState, ItineraryDay, ItineraryStop
from app.i18n import meal_lunch_label, openweather_lang
from app.tools.providers import get_weather_provider
from app.tools.weather_client import FORECAST_HORIZON_DAYS, WeatherClientError

_MAX_CANDIDATES_SHOWN = 10


def _used_place_ids(state: AgentState) -> set[str]:
    return {
        stop["place_id"]
        for day in state.get("itinerary", [])
        for stop in day["stops"]
        if stop["kind"] == "attraction"
    }


def _render_day(day: ItineraryDay, day_start_min: int) -> str:
    if not day["stops"]:
        return f"{day['date']}: (no stops)"
    lines = [f"{day['date']}:"]
    for stop in day["stops"]:
        start = minutes_to_clock(day_start_min + stop["arrival_offset_min"])
        end = minutes_to_clock(day_start_min + stop["depart_offset_min"])
        marker = " [meal]" if stop["kind"] == "meal" else f" (place_id={stop['place_id']})"
        lines.append(f"  {start}-{end} {stop['attraction_name']}{marker}")
    if day["dropped_stops"]:
        lines.append(f"  dropped (didn't fit): {', '.join(day['dropped_stops'])}")
    return "\n".join(lines)


def _availability_note(attraction, date: dt.date) -> str:
    if attraction.opening_hours is None:
        return ""
    if not is_open_on(attraction, date):
        return " | CLOSED that day"
    hours = attraction.opening_hours.get(date.weekday())
    if hours is None:
        return ""
    return f" | open {minutes_to_clock(hours[0])}-{minutes_to_clock(hours[1])} that day"


def _reschedule_day(state: AgentState, day_offset: int, place_ids: list[str]) -> str:
    """Rebuild day `day_offset` from scratch out of `place_ids` (in any
    order -- the scheduler picks the travel-minimising one) and write the
    result back into `state["itinerary"]` in place. The single choke point
    every edit tool goes through, so no edit can bypass the deterministic
    scheduler."""
    trip_setup = state["trip_setup"]
    day = state["itinerary"][day_offset]
    date = day["date"]
    attractions_by_id = attraction_lookup(state.get("ranked_attractions", []))
    rank_by_id = {r.attraction.place_id: i for i, r in enumerate(state.get("ranked_attractions", []))}
    matrix = state.get("distance_matrix")

    picks: list[ScheduleInput] = []
    for place_id in place_ids:
        attraction = attractions_by_id[place_id]
        if not is_open_on(attraction, date):
            return f"'{attraction.name}' is closed on {date} ({date.strftime('%A')}) -- can't schedule it that day."
        hours = (attraction.opening_hours or {}).get(date.weekday())
        open_min, close_min = hours if hours else (None, None)
        picks.append(
            ScheduleInput(
                place_id=place_id,
                name=attraction.name,
                dwell_minutes=category_dwell_minutes(attraction),
                open_min=open_min,
                close_min=close_min,
                rank=rank_by_id.get(place_id, len(rank_by_id)),
            )
        )

    def travel_minutes(a_id: str, b_id: str) -> float:
        raw = matrix.minutes_between(a_id, b_id) if matrix is not None else None
        if raw is None:
            raw = estimate_travel_minutes(attractions_by_id[a_id], attractions_by_id[b_id])
        return realistic_transit_minutes(raw)

    day_start_min = time_to_minutes(trip_setup["day_start_time"])
    day_end_min = time_to_minutes(trip_setup["day_end_time"])
    lang = trip_setup.get("lang", "en")
    scheduled = schedule_day_with_lunch(
        picks,
        day_start_min=day_start_min,
        day_end_min=day_end_min,
        travel_minutes=travel_minutes,
        lunch_dwell_minutes=LUNCH_DWELL_MINUTES[trip_setup["daily_pace"]],
        lunch_label=meal_lunch_label(lang),
    )

    new_stops: list[ItineraryStop] = [
        ItineraryStop(
            attraction_name=s.name,
            place_id=s.place_id,
            kind=s.kind,
            arrival_offset_min=s.arrival_offset_min,
            depart_offset_min=s.depart_offset_min,
            dwell_minutes=s.depart_offset_min - s.arrival_offset_min,
            travel_from_prev_min=s.travel_from_prev_min,
            wait_for_opening_min=s.wait_for_opening_min,
        )
        for s in scheduled.stops
    ]
    state["itinerary"][day_offset] = ItineraryDay(
        date=date, stops=new_stops, weather_note=day.get("weather_note"), dropped_stops=list(scheduled.dropped)
    )

    summary = _render_day(state["itinerary"][day_offset], day_start_min)
    if scheduled.dropped:
        summary += f"\nCouldn't fit: {', '.join(scheduled.dropped)} (opening hours or travel time didn't allow it)."
    return summary


def build_tools(state: AgentState) -> list:
    """One fresh set of closures per chat session -- each tool reads/writes
    `state` directly (the same dict object the session store holds), so an
    edit made in one turn is visible to every tool call in the next."""

    @tool
    def list_itinerary() -> str:
        """Show every day of the current itinerary with its scheduled stops and times. Always call this (or
        find_alternative) before making a change, so the edit is grounded in what's actually on the day."""
        trip_setup = state["trip_setup"]
        day_start_min = time_to_minutes(trip_setup["day_start_time"])
        itinerary = state.get("itinerary", [])
        if not itinerary:
            return "This trip has no itinerary yet."
        return "\n".join(_render_day(day, day_start_min) for day in itinerary)

    @tool
    def find_alternative(day_offset: int, category: str = "", max_price_level: str = "") -> str:
        """Search this trip's ranked candidate pool for attractions not currently used anywhere in the
        itinerary. `category` optionally filters by a Google Places category keyword (e.g. "museum", "park") --
        leave empty for no filter. `max_price_level` optionally caps price, one of PRICE_LEVEL_FREE,
        PRICE_LEVEL_INEXPENSIVE, PRICE_LEVEL_MODERATE, PRICE_LEVEL_EXPENSIVE, PRICE_LEVEL_VERY_EXPENSIVE -- leave
        empty for no cap. Returns place_id, name, category, price and same-day opening hours for each match; only
        use a place_id this tool (or list_itinerary) actually returned to you."""
        err = validate_day_offset(state, day_offset)
        if err:
            return err
        date = state["itinerary"][day_offset]["date"]
        used = _used_place_ids(state)
        max_rank = PRICE_LEVEL_RANK.get(max_price_level) if max_price_level else None

        candidates = []
        for r in state.get("ranked_attractions", []):
            a = r.attraction
            if a.place_id in used:
                continue
            if category and not any(category.lower() in c.lower() for c in a.categories):
                continue
            if max_rank is not None and a.price_level in PRICE_LEVEL_RANK and PRICE_LEVEL_RANK[a.price_level] > max_rank:
                continue
            candidates.append(a)

        if not candidates:
            return "No unused candidates match that filter."
        lines = [
            f"- place_id={a.place_id} | {a.name} | categories={', '.join(a.categories) or 'unknown'} | "
            f"price_level={a.price_level or 'unknown'} | rating={a.rating}{_availability_note(a, date)}"
            for a in candidates[:_MAX_CANDIDATES_SHOWN]
        ]
        return "\n".join(lines)

    @tool
    def swap_stop(day_offset: int, remove_place_id: str, add_place_id: str) -> str:
        """Replace one attraction stop with another on the given day (0-indexed), then rebuild that day's whole
        schedule (opening hours, travel time, lunch break) from scratch. remove_place_id must currently be
        scheduled on that day; add_place_id must come from find_alternative's results and not be used elsewhere
        in the trip. Returns the day's new timeline, or an error explaining why the swap couldn't be made."""
        err = validate_day_offset(state, day_offset)
        if err:
            return err
        err = validate_place_used_on_day(state, day_offset, remove_place_id)
        if err:
            return err
        err = validate_known_place_id(state, add_place_id) or validate_place_not_used_anywhere(state, add_place_id)
        if err:
            return err

        current_ids = [s["place_id"] for s in state["itinerary"][day_offset]["stops"] if s["kind"] == "attraction"]
        new_ids = [add_place_id if pid == remove_place_id else pid for pid in current_ids]
        return _reschedule_day(state, day_offset, new_ids)

    @tool
    def remove_stop(day_offset: int, place_id: str) -> str:
        """Drop one attraction stop from a day and reschedule the rest of that day around its removal. Use this
        instead of swap_stop when the traveller wants fewer stops, not a replacement."""
        err = validate_day_offset(state, day_offset)
        if err:
            return err
        err = validate_place_used_on_day(state, day_offset, place_id)
        if err:
            return err
        current_ids = [
            s["place_id"] for s in state["itinerary"][day_offset]["stops"] if s["kind"] == "attraction" and s["place_id"] != place_id
        ]
        return _reschedule_day(state, day_offset, current_ids)

    @tool
    def add_stop(day_offset: int, place_id: str) -> str:
        """Add one attraction (from find_alternative's results) to a day and reschedule that day to fit it in.
        May fail -- and say why -- if the day is already full or the place's opening hours don't allow it."""
        err = validate_day_offset(state, day_offset)
        if err:
            return err
        err = validate_known_place_id(state, place_id) or validate_place_not_used_anywhere(state, place_id)
        if err:
            return err
        current_ids = [s["place_id"] for s in state["itinerary"][day_offset]["stops"] if s["kind"] == "attraction"]
        return _reschedule_day(state, day_offset, [*current_ids, place_id])

    @tool
    def check_weather(date: str) -> str:
        """Look up the weather forecast for one date of this trip (format YYYY-MM-DD). Only covers roughly the
        next 5 days from today (the forecast provider's free-tier horizon); returns an explanation instead of a
        forecast for a date outside that range or if the lookup fails."""
        try:
            parsed = dt.date.fromisoformat(date)
        except ValueError:
            return f"'{date}' isn't a valid date -- use YYYY-MM-DD."
        centroid = ranked_centroid(state.get("ranked_attractions", []))
        if centroid is None:
            return "No location data available for this trip yet."
        lat, lng = centroid
        lang = state["trip_setup"].get("lang", "en")
        client = get_weather_provider(state.get("destination_country"))
        try:
            forecast = client.get_daily_forecast(lat, lng, lang=openweather_lang(lang))
        except WeatherClientError as exc:
            return f"Weather lookup failed: {exc}"
        match = next((f for f in forecast if f.date == parsed), None)
        if match is None:
            return f"No forecast available for {date} (the provider only covers ~{FORECAST_HORIZON_DAYS} days out)."
        note = " (bad weather -- consider indoor alternatives)" if match.is_bad_weather else ""
        return f"{date}: {match.description}, {match.avg_temp_c:.0f}C{note}"

    return [list_itinerary, find_alternative, swap_stop, remove_stop, add_stop, check_weather]
