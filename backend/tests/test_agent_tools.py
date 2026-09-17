"""Unit tests for the chat agent's tools (app/agent/tools.py) -- no LLM
involved, calling each tool directly the same way the ReAct agent would.
"""

from __future__ import annotations

import datetime as dt

from app.agent.tools import build_tools
from app.graph.state import AgentState, ItineraryDay, ItineraryStop
from app.recsys.rank import RankedAttraction
from app.tools.places_client import Attraction
from app.tools.weather_client import DailyForecast

START = dt.date(2026, 9, 1)


def _ranked(
    place_id: str,
    categories: list[str],
    *,
    price_level: str | None = None,
    opening_hours: dict[int, tuple[int, int]] | None = None,
) -> RankedAttraction:
    attraction = Attraction(
        place_id=place_id,
        name=f"Name-{place_id}",
        address="",
        lat=0.0,
        lng=0.0,
        rating=4.0,
        rating_count=10,
        price_level=price_level,
        categories=categories,
        description=None,
        open_weekdays=frozenset(opening_hours) if opening_hours else None,
        opening_hours=opening_hours,
    )
    return RankedAttraction(attraction=attraction, score=1.0, similarity=1.0, normalized_rating=0.8)


def _trip_setup() -> dict:
    return {
        "destination": "Lisbon",
        "start_date": START,
        "end_date": START,
        "interests": ["museums"],
        "daily_pace": "moderate",
        "budget_level": "moderate",
        "day_start_time": dt.time(9, 0),
        "day_end_time": dt.time(18, 0),
        "lang": "en",
    }


def _state(ranked: list[RankedAttraction], scheduled_place_ids: list[str]) -> AgentState:
    attractions_by_id = {r.attraction.place_id: r.attraction for r in ranked}
    stops = []
    clock = 0
    for pid in scheduled_place_ids:
        stops.append(
            ItineraryStop(
                attraction_name=attractions_by_id[pid].name,
                place_id=pid,
                kind="attraction",
                arrival_offset_min=clock,
                depart_offset_min=clock + 60,
                dwell_minutes=60,
                travel_from_prev_min=0,
                wait_for_opening_min=0,
            )
        )
        clock += 90
    itinerary = [ItineraryDay(date=START, stops=stops, weather_note=None, dropped_stops=[])]
    return {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }


def _tool(state: AgentState, name: str):
    return next(t for t in build_tools(state) if t.name == name)


def test_list_itinerary_shows_scheduled_stops():
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["park"])]
    state = _state(ranked, ["p1"])

    out = _tool(state, "list_itinerary").invoke({})

    assert "Name-p1" in out
    assert "09:00-10:00" in out


def test_find_alternative_excludes_used_and_filters_by_category():
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["park"]), _ranked("p3", ["museum"])]
    state = _state(ranked, ["p1"])

    out = _tool(state, "find_alternative").invoke({"day_offset": 0, "category": "museum"})

    assert "p1" not in out  # already used
    assert "place_id=p3" in out
    assert "p2" not in out  # filtered out by category


def test_find_alternative_rejects_bad_day_offset():
    ranked = [_ranked("p1", ["museum"])]
    state = _state(ranked, [])

    out = _tool(state, "find_alternative").invoke({"day_offset": 5})

    assert "out of range" in out


def test_swap_stop_replaces_and_reschedules_the_day():
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["park"])]
    state = _state(ranked, ["p1"])

    out = _tool(state, "swap_stop").invoke({"day_offset": 0, "remove_place_id": "p1", "add_place_id": "p2"})

    day = state["itinerary"][0]
    scheduled_ids = [s["place_id"] for s in day["stops"] if s["kind"] == "attraction"]
    assert scheduled_ids == ["p2"]
    assert "Name-p2" in out


def test_swap_stop_rejects_place_not_currently_on_that_day():
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["park"])]
    state = _state(ranked, ["p1"])

    out = _tool(state, "swap_stop").invoke({"day_offset": 0, "remove_place_id": "p2", "add_place_id": "p1"})

    assert "not currently scheduled" in out
    # state must be untouched by a rejected edit
    assert [s["place_id"] for s in state["itinerary"][0]["stops"]] == ["p1"]


def test_swap_stop_rejects_hallucinated_place_id():
    ranked = [_ranked("p1", ["museum"])]
    state = _state(ranked, ["p1"])

    out = _tool(state, "swap_stop").invoke({"day_offset": 0, "remove_place_id": "p1", "add_place_id": "does-not-exist"})

    assert "not a known candidate" in out


def test_add_stop_rejects_place_already_used_elsewhere():
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["park"])]
    state = _state(ranked, ["p1", "p2"])

    out = _tool(state, "add_stop").invoke({"day_offset": 0, "place_id": "p1"})

    assert "already scheduled" in out


def test_remove_stop_drops_and_reschedules():
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["park"])]
    state = _state(ranked, ["p1", "p2"])

    _tool(state, "remove_stop").invoke({"day_offset": 0, "place_id": "p1"})

    scheduled_ids = [s["place_id"] for s in state["itinerary"][0]["stops"] if s["kind"] == "attraction"]
    assert scheduled_ids == ["p2"]


def test_add_stop_rejects_place_closed_that_day():
    # opening_hours has no entry for START's weekday -- open_weekdays excludes it entirely.
    other_weekday = (START.weekday() + 1) % 7
    ranked = [
        _ranked("p1", ["museum"]),
        _ranked("p2", ["park"], opening_hours={other_weekday: (9 * 60, 17 * 60)}),
    ]
    state = _state(ranked, ["p1"])

    out = _tool(state, "add_stop").invoke({"day_offset": 0, "place_id": "p2"})

    assert "closed on" in out
    assert [s["place_id"] for s in state["itinerary"][0]["stops"]] == ["p1"]  # untouched


def test_check_weather_reports_forecast(monkeypatch):
    ranked = [_ranked("p1", ["museum"])]
    state = _state(ranked, ["p1"])

    class _FakeWeatherClient:
        def get_daily_forecast(self, lat, lng, lang="en"):
            return [DailyForecast(date=START, avg_temp_c=12.0, condition="Rain", description="light rain", is_bad_weather=True)]

    monkeypatch.setattr("app.agent.tools.get_weather_provider", lambda country_code=None: _FakeWeatherClient())

    out = _tool(state, "check_weather").invoke({"date": START.isoformat()})

    assert "light rain" in out
    assert "bad weather" in out


def test_check_weather_rejects_malformed_date():
    ranked = [_ranked("p1", ["museum"])]
    state = _state(ranked, ["p1"])

    out = _tool(state, "check_weather").invoke({"date": "not-a-date"})

    assert "valid date" in out
