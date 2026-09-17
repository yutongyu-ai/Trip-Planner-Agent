"""Synthetic chat scenarios for the conversational edit agent (see
run_eval.py). Each is built so the correct response genuinely requires more
than one tool call, with the second call's choice depending on what the
first one revealed -- not just phrasing an equivalent single-call request
differently. That's the property this eval exists to demonstrate: the model
is choosing tools and reacting to what it sees, not following a fixed script.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app.graph.state import AgentState, ItineraryDay, ItineraryStop
from app.recsys.rank import RankedAttraction
from app.tools.places_client import Attraction
from app.tools.weather_client import DailyForecast

START = dt.date.today() + dt.timedelta(days=10)  # beyond the weather horizon; scenarios mock it directly


def _attraction(
    place_id: str,
    name: str,
    categories: list[str],
    *,
    price_level: str | None = None,
    rating: float = 4.3,
) -> Attraction:
    return Attraction(
        place_id=place_id,
        name=name,
        address="",
        lat=0.0,
        lng=0.0,
        rating=rating,
        rating_count=200,
        price_level=price_level,
        categories=categories,
        description=None,
    )


def _ranked(attraction: Attraction) -> RankedAttraction:
    return RankedAttraction(attraction=attraction, score=1.0, similarity=1.0, normalized_rating=0.8)


def _stop(attraction: Attraction, start_min: int, dwell: int = 90) -> ItineraryStop:
    return ItineraryStop(
        attraction_name=attraction.name,
        place_id=attraction.place_id,
        kind="attraction",
        arrival_offset_min=start_min,
        depart_offset_min=start_min + dwell,
        dwell_minutes=dwell,
        travel_from_prev_min=0,
        wait_for_opening_min=0,
    )


def _trip_setup(lang: str = "en") -> dict:
    return {
        "destination": "Porto",
        "start_date": START,
        "end_date": START,
        "interests": ["museums"],
        "daily_pace": "moderate",
        "budget_level": "budget",
        "day_start_time": dt.time(9, 0),
        "day_end_time": dt.time(18, 0),
        "lang": lang,
    }


@dataclass(frozen=True)
class ChatScenario:
    name: str
    state: AgentState
    user_message: str
    min_expected_tool_calls: int
    # A tool-result substring that must appear somewhere in the trajectory
    # for the run to count as "on the right track" (not just any 2 calls).
    required_trajectory_marker: str
    # Task success: place_id that must be gone from / present in the final itinerary.
    place_expected_removed: str | None = None
    place_expected_present: str | None = None
    weather_forecast: list[DailyForecast] | None = None  # None = no weather mocking needed


def _budget_swap_scenario() -> ChatScenario:
    expensive = _attraction("museum-lux", "Grand National Museum", ["museum"], price_level="PRICE_LEVEL_VERY_EXPENSIVE")
    cheap_museum = _attraction("museum-cheap", "Community Art Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    park = _attraction("park1", "Riverside Park", ["park"], price_level="PRICE_LEVEL_FREE")
    ranked = [_ranked(expensive), _ranked(cheap_museum), _ranked(park)]
    itinerary = [ItineraryDay(date=START, stops=[_stop(expensive, 0)], weather_note=None, dropped_stops=[])]
    state: AgentState = {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="budget_swap_finds_cheaper_alternative",
        state=state,
        user_message=(
            "The museum on day 0 (place_id museum-lux) is way over my budget. Find something cheaper "
            "in the same category and swap it in."
        ),
        min_expected_tool_calls=2,  # find_alternative, then swap_stop
        required_trajectory_marker="museum-cheap",
        place_expected_removed="museum-lux",
        place_expected_present="museum-cheap",
    )


def _weather_triggered_swap_scenario() -> ChatScenario:
    outdoor = _attraction("park1", "Old Town Gardens", ["park"], price_level="PRICE_LEVEL_FREE")
    indoor = _attraction("museum1", "City History Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    ranked = [_ranked(outdoor), _ranked(indoor)]
    itinerary = [ItineraryDay(date=START, stops=[_stop(outdoor, 0)], weather_note=None, dropped_stops=[])]
    state: AgentState = {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="weather_triggered_indoor_swap",
        state=state,
        user_message=(
            f"Check the weather for {START.isoformat()} (day 0). If it looks bad, swap the outdoor stop "
            "for an indoor alternative."
        ),
        min_expected_tool_calls=3,  # check_weather, find_alternative, swap_stop
        required_trajectory_marker="museum1",
        place_expected_removed="park1",
        place_expected_present="museum1",
        weather_forecast=[
            DailyForecast(date=START, avg_temp_c=11.0, condition="Rain", description="heavy rain", is_bad_weather=True)
        ],
    )


SCENARIOS: list[ChatScenario] = [_budget_swap_scenario(), _weather_triggered_swap_scenario()]
