"""LangGraph state schema for the trip-planning agent.

Kept as plain TypedDicts (no LangGraph-specific base class needed) so the
tool/recsys modules stay importable and testable without depending on
LangGraph at all.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, TypedDict

from app.recsys.rank import RankedAttraction
from app.tools.places_client import Attraction
from app.tools.route_client import DistanceMatrix
from app.tools.weather_client import DailyForecast

DailyPace = Literal["relaxed", "moderate", "packed"]
BudgetLevel = Literal["budget", "moderate", "luxury"]
Lang = Literal["en", "zh"]

MAX_REFINE_ITERATIONS = 2


class TripSetup(TypedDict):
    destination: str
    start_date: dt.date
    end_date: dt.date
    interests: list[str]
    daily_pace: DailyPace
    budget_level: BudgetLevel
    day_start_time: dt.time
    day_end_time: dt.time
    lang: Lang


class ItineraryStop(TypedDict):
    attraction_name: str
    place_id: str
    # "attraction" or "meal" -- a meal is a break the scheduler leaves time
    # for (no place recommended), see app/graph/scheduling.py.
    kind: str
    arrival_offset_min: int
    # arrival_offset_min + dwell_minutes -- the visit's end. Stored (not just
    # derived in the UI) so the frontend can show it and so check_constraints
    # reads exactly what the scheduler produced.
    depart_offset_min: int
    dwell_minutes: int
    # Travel time from the previous stop in visit order, minutes; 0 for the
    # first stop of the day. Comes straight from app/graph/scheduling.py.
    travel_from_prev_min: int
    # Minutes waiting at the door for this place to open (arrived early). 0
    # unless opening hours pushed the arrival later.
    wait_for_opening_min: int


class ItineraryDay(TypedDict):
    date: dt.date
    stops: list[ItineraryStop]
    weather_note: str | None
    # Names of attractions the LLM picked for this day that didn't make it
    # in -- either the scheduler couldn't fit them (opening hours + travel +
    # day window) or itinerary_node filtered them out before the scheduler
    # ever saw them (closed all day that weekday). Surfaced to the user by
    # finalize.py. Empty on a normal day.
    dropped_stops: list[str]


class AgentState(TypedDict, total=False):
    trip_setup: TripSetup

    candidate_attractions: list[Attraction]
    ranked_attractions: list[RankedAttraction]
    weather_forecast: list[DailyForecast]
    # Pairwise driving matrix between the top-ranked candidates (OSRM table
    # service). None when routing failed or there were <2 candidates -- the
    # scheduler then falls back to straight-line estimates.
    distance_matrix: DistanceMatrix | None

    # ISO alpha-2 code of the destination country (from the Places address
    # components), and whether the default providers cover it poorly enough
    # to warn the user -- see app/tools/providers.py.
    destination_country: str | None
    low_coverage_destination: bool

    itinerary: list[ItineraryDay]
    summary: str
    # Advisory lines shown to the user alongside the itinerary (partial
    # forecast, low-coverage destination, unresolved constraints). Already
    # localized; finalize.py also folds them into `summary`.
    notes: list[str]

    constraint_violations: list[str]
    # Same violations as constraint_violations, bucketed by day offset --
    # see constraints.py's docstring. Lets itinerary_node re-plan only the
    # days that actually need it on a refine pass.
    violations_by_day: dict[int, list[str]]
    refine_iterations: int

    errors: list[str]
