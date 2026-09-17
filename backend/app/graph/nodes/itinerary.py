"""Itinerary synthesis node.

The LLM's job is narrow: for each day, pick *which* ranked candidates to
visit and roughly *how long* to spend at each. It does not decide arrival
times or visit order -- that's arithmetic (line each visit up with the
place's opening hours, leave room for travel, keep total travel down) that
an LLM does unreliably and won't fix on a refine pass, so it's done
deterministically in `app/graph/scheduling.py` instead. Everything else
user-facing (names from place_id, weather-note text, day dates) is also
computed in Python here, so those facts can't be hallucinated.

Structured output (`_LLMDayPlan`) keeps the LLM's response constrained to
the pick-and-dwell decision instead of free text we'd have to parse;
`clamped_dwell_minutes` (see _shared.py) keeps its per-stop dwell_minutes
estimate from straying too far from a category-based default; the scheduler
then places the picks in time and, if they can't all fit the day, drops the
lowest-ranked ones and reports what it dropped.

One LLM call per day, not one call for the whole trip. A single call
planning every day at once would -- when refine feedback covered
violations on more than one day simultaneously -- sometimes silently drop
an entire day's stops while trying to fix another day's, and the dropped
day wouldn't reliably come back on a later refine attempt either (caught
by recording raw structured output across a real refine run: day 1 with
several violations, day 2 clean; the very next call returned only day 1,
padded with a stop taken from day 2, and day 2 never recovered even after
being told explicitly it now had zero stops). Per-day calls make that
failure mode structurally impossible: a day check_constraints_node didn't
flag is never re-sent to the LLM at all on a refine pass, so it can't be
collaterally damaged by whatever the LLM does with a different day's
feedback. Each day's prompt only offers candidates not already claimed by
another day in the same pass -- whether that day was already regenerated
or is being kept unchanged, including ones later in the loop (see the
used_place_ids seeding pass at the top of itinerary_node) -- so "don't
repeat a place across days" is enforced by construction rather than by
asking nicely.
"""

from __future__ import annotations

import datetime as dt
import logging

import openai
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.nodes._shared import (
    LUNCH_DWELL_MINUTES,
    PACE_ACTIVITY_HOURS,
    STOPS_PER_DAY_RANGE,
    attraction_lookup,
    category_dwell_minutes,
    clamped_dwell_minutes,
    dwell_bounds,
    estimate_travel_minutes,
    is_indoor,
    is_open_on,
    minutes_to_clock,
    realistic_transit_minutes,
    time_to_minutes,
)
from app.graph.scheduling import ScheduleInput, schedule_day_with_lunch
from app.graph.state import AgentState, ItineraryDay, ItineraryStop
from app.i18n import (
    meal_lunch_label,
    weather_note_bad_adjusted,
    weather_note_bad_unadjusted,
    weather_note_ok,
)
from app.recsys.rank import RankedAttraction
from app.tools.places_client import Attraction
from app.tools.route_client import DistanceMatrix
from app.tools.weather_client import DailyForecast

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"

_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _hours_hint(attraction: Attraction) -> str:
    """`| hours: Mon 09:00-17:00, Tue closed, ...` -- per-weekday open/close
    clock times (not just which weekdays are open), so the LLM can avoid
    scheduling a visit that starts before opening or runs past closing, not
    just avoid the wrong day entirely. Empty string if we have no hours
    data at all (see Attraction.opening_hours's "unknown" convention).
    """
    if attraction.opening_hours is None:
        return ""
    parts = []
    for d in range(7):
        hours = attraction.opening_hours.get(d)
        if hours is None:
            parts.append(f"{_WEEKDAY_NAMES[d]} closed")
        else:
            open_min, close_min = hours
            parts.append(f"{_WEEKDAY_NAMES[d]} {minutes_to_clock(open_min)}-{minutes_to_clock(close_min)}")
    return " | hours: " + ", ".join(parts)


class _LLMStop(BaseModel):
    place_id: str = Field(description="Must be one of the candidate place_id values given above.")
    dwell_minutes: int = Field(
        description=(
            "Minutes to spend at this stop. Start from its typical_visit estimate, then use what "
            "you know about the specific place: a major national museum (the British Museum, the "
            "Louvre) realistically needs 2-4 hours, a small local one under an hour, a photo-stop "
            "landmark 15-20 min. Also scale to the traveller's pace. Keep it within the ok-range "
            "shown for the candidate."
        )
    )


class _LLMDayPlan(BaseModel):
    stops: list[_LLMStop]


def _default_llm() -> ChatOpenAI:
    # Explicit timeout, matching the other three tool clients (each use
    # httpx.Client(timeout=10.0)) -- without it, a hung OpenAI call would
    # block this node (and the whole /trip-plan request) indefinitely
    # instead of hitting the openai.OpenAIError handling below.
    return ChatOpenAI(model=MODEL_NAME, api_key=get_settings().openai_api_key, temperature=0, timeout=30.0)


def _forecast_by_date(state: AgentState) -> dict[dt.date, DailyForecast]:
    return {f.date: f for f in state.get("weather_forecast", [])}


_MAX_PROXIMITY_HINT_STOPS = 6
# Below this door-to-door estimate the pair is "basically the same place" --
# listing it just adds noise to the prompt (and an all-co-located pool would
# produce a screenful of lines).
_MIN_PROXIMITY_HINT_MINUTES = 12


def _proximity_hint_lines(matrix: DistanceMatrix | None, attractions_by_id: dict[str, Attraction]) -> list[str]:
    """Named pairwise door-to-door times among the top few candidates, so the
    LLM can prefer picking places that sit close together. Only the pairs far
    enough apart to matter, capped so the prompt doesn't carry a full
    N-by-N block; empty when everything's clustered."""
    if matrix is None or len(matrix.place_ids) < 2:
        return []
    ids = matrix.place_ids[:_MAX_PROXIMITY_HINT_STOPS]
    lines = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            raw = matrix.minutes_between(a, b)
            if raw is None:
                continue
            minutes = realistic_transit_minutes(raw)
            if minutes < _MIN_PROXIMITY_HINT_MINUTES:
                continue
            name_a = attractions_by_id[a].name if a in attractions_by_id else a
            name_b = attractions_by_id[b].name if b in attractions_by_id else b
            lines.append(f"- {name_a} <-> {name_b}: ~{minutes:.0f} min apart")
    return lines


def _build_day_prompt(
    trip_setup,
    date: dt.date,
    available: list[RankedAttraction],
    forecast: DailyForecast | None,
    route_lines: list[str],
    feedback: list[str],
    stops_hi: int,
    indoor_filtered: bool = False,
) -> str:
    candidate_lines = [
        f"- place_id={r.attraction.place_id} | {r.attraction.name} | "
        f"categories={', '.join(r.attraction.categories) or 'unknown'} | "
        f"rating={r.attraction.rating} ({r.attraction.rating_count or 0} reviews) | "
        f"price_level={r.attraction.price_level or 'unknown'} | "
        f"{'indoor' if is_indoor(r.attraction) else 'outdoor/unclear'} | "
        f"typical_visit≈{category_dwell_minutes(r.attraction)}min "
        f"(ok range {dwell_bounds(r.attraction)[0]}-{dwell_bounds(r.attraction)[1]})"
        f"{_hours_hint(r.attraction)}"
        for r in available
    ]

    weather_block = ""
    if forecast is not None:
        if indoor_filtered:
            # Already hard-filtered to indoor candidates below (see
            # itinerary_node) -- no need to also ask the LLM to prefer them.
            weather_block = (
                f"\nWeather: {forecast.condition}, {forecast.avg_temp_c:.0f}C -- the traveller asked to skip "
                "outdoor plans on bad-weather days, so only indoor candidates are listed below.\n"
            )
        else:
            bad_weather_hint = " (bad weather, prefer indoor stops)" if forecast.is_bad_weather else ""
            weather_block = f"\nWeather: {forecast.condition}, {forecast.avg_temp_c:.0f}C{bad_weather_hint}\n"

    feedback_block = ""
    if feedback:
        feedback_block = (
            "\nThe previous attempt at this day had these problems -- change your picks to fix them:\n"
            + "\n".join(f"- {v}" for v in feedback)
        )

    pace = trip_setup["daily_pace"]
    stops_lo, _ = STOPS_PER_DAY_RANGE[pace]
    stops_lo = min(stops_lo, stops_hi)  # a starved late day may be capped below the pace minimum
    activity_lo, activity_hi = PACE_ACTIVITY_HOURS[pace]
    window_hours = (
        time_to_minutes(trip_setup["day_end_time"]) - time_to_minutes(trip_setup["day_start_time"])
    ) / 60

    return f"""Plan {date.strftime('%A')} ({date.isoformat()}) of a trip to {trip_setup['destination']}.
Traveller pace: {pace}. Budget: {trip_setup['budget_level']}.
Interests: {', '.join(trip_setup['interests']) or 'general sightseeing'}.
The day runs from {trip_setup['day_start_time']:%H:%M} to {trip_setup['day_end_time']:%H:%M}
({window_hours:.1f} hours).

Your job is only to choose which attractions to visit and how long to spend at each
(dwell_minutes). Arrival times and visit order are worked out afterwards by a scheduler that
respects each place's opening hours and the travel time between stops -- you do not assign times.

Plan a full {pace} day: visits plus travel between them should add up to roughly
{activity_lo:.0f}-{activity_hi:.0f} hours (leaving the rest of the {window_hours:.0f}-hour window
for a meal and breaks). That's about {stops_lo}-{stops_hi} stops. You choose the mix -- fewer long
visits or more short ones -- but don't hand back a half-empty day: a moderate day that ends by
early afternoon is too thin. Prefer places open on {date.strftime('%A')} that sit close enough
together to travel between. The typical_visit per candidate is only a starting anchor -- set
dwell_minutes from what you know about the specific place (a world-class museum realistically
needs 2-3 hours, a big zoo 2-3, a quick landmark 15-20 min), within its ok-range.
{weather_block}
Candidate attractions available today, listed highest-ranked first (relevance to the traveller's
stated interests blended with rating; already excludes anything used on another day of this trip --
pick only from this list). Review count is a real popularity signal: a place with tens of thousands
of reviews is a well-known highlight most visitors would regret skipping, while a similarly-rated
place with a handful of reviews is a minor, easily-missable spot -- don't let a small travel-time
saving talk you out of a major landmark in favor of an obscure one just because it happens to sit
closer to your other picks. A candidate with an `hours` list is closed on any weekday marked
"closed" -- don't pick it if today is one; a place open only a few hours that day is a poor pick if
you want a long visit there:
{chr(10).join(candidate_lines)}

Driving time between the top candidates (use this to sequence a good route among your picks, not to
override picking the well-known highlights above):
{chr(10).join(route_lines) or 'none available'}
{feedback_block}

Return {stops_lo}-{stops_hi} stops for today (a full day, not a half day) using only the
place_id values above."""


def _weather_note(
    date: dt.date,
    stops: list[ItineraryStop],
    forecasts: dict[dt.date, DailyForecast],
    attractions_by_id: dict,
    lang: str,
) -> str | None:
    forecast = forecasts.get(date)
    if forecast is None:
        return None
    if not forecast.is_bad_weather:
        return weather_note_ok(lang, forecast.description, forecast.avg_temp_c)

    stop_attractions = [attractions_by_id[s["place_id"]] for s in stops if s["place_id"] in attractions_by_id]
    if stop_attractions and all(is_indoor(a) for a in stop_attractions):
        return weather_note_bad_adjusted(lang, forecast.description)
    return weather_note_bad_unadjusted(lang, forecast.description)


def _empty_days(trip_setup, num_days: int) -> list[ItineraryDay]:
    return [
        ItineraryDay(
            date=trip_setup["start_date"] + dt.timedelta(days=i), stops=[], weather_note=None, dropped_stops=[]
        )
        for i in range(num_days)
    ]


def itinerary_node(state: AgentState, *, llm=None) -> dict:
    trip_setup = state["trip_setup"]
    num_days = (trip_setup["end_date"] - trip_setup["start_date"]).days + 1
    ranked = state.get("ranked_attractions", [])
    attractions_by_id = attraction_lookup(ranked)
    forecasts = _forecast_by_date(state)
    matrix = state.get("distance_matrix")
    route_lines = _proximity_hint_lines(matrix, attractions_by_id)

    refine_iterations = state.get("refine_iterations", 0) + (1 if state.get("constraint_violations") else 0)

    if not ranked:
        errors = list(state.get("errors", []))
        # attractions_node already explains an empty pool caused by a
        # destination it couldn't resolve at all (see its "destination_not_found"
        # tag) -- don't pile on a second, more generic error for the traveller
        # to read past. Anything else that left the pool empty still gets this
        # fallback so it isn't silently swallowed.
        if "destination_not_found" not in errors:
            errors.append("itinerary: no ranked attractions available, produced an empty plan")
        return {"itinerary": _empty_days(trip_setup, num_days), "refine_iterations": refine_iterations, "errors": errors}

    previous_days = state.get("itinerary") or []
    violations_by_day: dict[int, list[str]] = state.get("violations_by_day") or {}
    lang = trip_setup.get("lang", "en")

    rank_by_id = {r.attraction.place_id: i for i, r in enumerate(ranked)}
    day_start_min = time_to_minutes(trip_setup["day_start_time"])
    day_end_min = time_to_minutes(trip_setup["day_end_time"])
    stops_lo, stops_hi = STOPS_PER_DAY_RANGE[trip_setup["daily_pace"]]

    def travel_minutes(a_id: str, b_id: str) -> float:
        # Raw free-flow driving minutes: the real OSRM matrix, or a
        # straight-line estimate for a pair it doesn't cover (a pick ranked
        # below MAX_MATRIX_STOPS, or no matrix at all). Then converted to a
        # realistic door-to-door time.
        raw = matrix.minutes_between(a_id, b_id) if matrix is not None else None
        if raw is None:
            raw = estimate_travel_minutes(attractions_by_id[a_id], attractions_by_id[b_id])
        return realistic_transit_minutes(raw)

    days: list[ItineraryDay] = []
    used_place_ids: set[str] = set()
    errors = list(state.get("errors", []))

    # Seed with place_ids from every day that will be kept unchanged this
    # pass, not just ones a regeneration loop below has reached yet --
    # otherwise an *earlier* day being regenerated wouldn't know about a
    # *later*, already-clean day's claimed place_ids, and could legally
    # pick a duplicate that only gets caught (costing a whole refine
    # iteration) on the next constraint check.
    for offset in range(num_days):
        if offset < len(previous_days) and offset not in violations_by_day:
            used_place_ids.update(s["place_id"] for s in previous_days[offset]["stops"])

    for offset in range(num_days):
        date = trip_setup["start_date"] + dt.timedelta(days=offset)
        day_feedback = violations_by_day.get(offset, [])

        # A day with no feedback and a prior attempt to fall back on is left
        # exactly as it was, with no LLM call at all -- see module docstring
        # for why that matters (a clean day must never be collateral damage
        # from fixing a different day).
        if not day_feedback and offset < len(previous_days):
            days.append(previous_days[offset])
            continue

        available = [r for r in ranked if r.attraction.place_id not in used_place_ids]

        # Cap this day's stop count so the days after it still have enough
        # unclaimed candidates to hit their own minimum -- otherwise the
        # per-day LLM front-loads a tight multi-day trip and the last days
        # come back empty. (Days already fixed this pass have their picks in
        # used_place_ids, so `available` and this count only concern days
        # still to be regenerated.)
        future_regen_days = sum(
            1
            for future in range(offset + 1, num_days)
            if violations_by_day.get(future) or future >= len(previous_days)
        )
        effective_hi = min(stops_hi, max(stops_lo, len(available) - future_regen_days * stops_lo))

        forecast = forecasts.get(date)
        day_available = available
        indoor_filtered = False
        if forecast is not None and forecast.is_bad_weather and trip_setup.get("avoid_outdoor_in_bad_weather"):
            indoor_only = [r for r in available if is_indoor(r.attraction)]
            # Only commit to the hard filter if it still leaves enough to
            # fill the day -- otherwise fall back to the full pool (and the
            # unfiltered weather hint) rather than handing back a thin day.
            if len(indoor_only) >= stops_lo:
                day_available = indoor_only
                indoor_filtered = True

        prompt = _build_day_prompt(
            trip_setup, date, day_available, forecast, route_lines, day_feedback, effective_hi, indoor_filtered
        )

        structured_llm = (llm or _default_llm()).with_structured_output(_LLMDayPlan)
        try:
            result: _LLMDayPlan = structured_llm.invoke(prompt)
        except openai.OpenAIError as exc:
            logger.warning("itinerary LLM call failed for %s: %s", date, exc)
            errors.append(f"itinerary: {exc}")
            days.append(ItineraryDay(date=date, stops=[], weather_note=None, dropped_stops=[]))
            continue

        # The LLM only picks place_ids + dwell; scheduling.schedule_day turns
        # them into a timed, ordered day (and drops what won't fit).
        picks: list[ScheduleInput] = []
        seen: set[str] = set()
        closed_today: list[str] = []
        for stop in result.stops:
            if stop.place_id not in attractions_by_id or stop.place_id in used_place_ids or stop.place_id in seen:
                continue  # hallucinated place_id, or a repeat despite the filtered candidate list
            seen.add(stop.place_id)
            attraction = attractions_by_id[stop.place_id]
            if not is_open_on(attraction, date):
                # The prompt lists this as closed today, but the LLM picked it
                # anyway -- ScheduleInput has no "closed all day" concept (only
                # a within-day open_min/close_min, absent==unknown-not-closed),
                # so it must never reach the scheduler at all.
                closed_today.append(attraction.name)
                continue
            hours = (attraction.opening_hours or {}).get(date.weekday())
            open_min, close_min = hours if hours else (None, None)
            picks.append(
                ScheduleInput(
                    place_id=stop.place_id,
                    name=attraction.name,
                    dwell_minutes=clamped_dwell_minutes(attraction, stop.dwell_minutes),
                    open_min=open_min,
                    close_min=close_min,
                    rank=rank_by_id.get(stop.place_id, len(ranked)),
                )
            )

        scheduled = schedule_day_with_lunch(
            picks,
            day_start_min=day_start_min,
            day_end_min=day_end_min,
            travel_minutes=travel_minutes,
            lunch_dwell_minutes=LUNCH_DWELL_MINUTES[trip_setup["daily_pace"]],
            lunch_label=meal_lunch_label(lang),
        )
        stops: list[ItineraryStop] = [
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
        used_place_ids.update(s["place_id"] for s in stops if s["kind"] == "attraction")

        days.append(
            ItineraryDay(
                date=date,
                stops=stops,
                weather_note=_weather_note(date, stops, forecasts, attractions_by_id, lang),
                dropped_stops=[*closed_today, *scheduled.dropped],
            )
        )

    return {"itinerary": days, "refine_iterations": refine_iterations, "errors": errors}
