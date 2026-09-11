"""End-to-end tests of the compiled graph: real node wiring, external APIs
mocked via respx (same convention as the client tests), the LLM swapped
for a fake via app.graph.nodes.itinerary._default_llm.
"""

import datetime as dt

import httpx
import numpy as np
import respx

from app.graph import build_graph
from app.graph.nodes import itinerary as itinerary_module
from app.graph.nodes.itinerary import _LLMDayPlan, _LLMStop
from app.recsys import rank as rank_module
from app.tools.places_client import PLACES_SEARCH_URL
from app.tools.weather_client import FORECAST_HORIZON_DAYS, FORECAST_URL

OSRM_BASE_URL = "https://router.project-osrm.org"


def _place(place_id: str) -> dict:
    return {
        "id": place_id,
        "displayName": {"text": f"Name-{place_id}"},
        "formattedAddress": "123 Main St",
        "location": {"latitude": 1.0, "longitude": 1.0},
        "rating": 4.5,
        "userRatingCount": 100,
        "priceLevel": "PRICE_LEVEL_INEXPENSIVE",
        "types": ["museum"],
        "editorialSummary": {"text": "A fine museum."},
    }


def _trip_setup(start_date: dt.date, daily_pace: str = "moderate") -> dict:
    return {
        "destination": "Lisbon",
        "start_date": start_date,
        "end_date": start_date,
        "interests": ["museums"],
        "daily_pace": daily_pace,
        "budget_level": "moderate",
        "day_start_time": dt.time(9, 0),
        "day_end_time": dt.time(18, 0),
    }


class _FakeStructuredLLM:
    """Returns `results` in order, one per `.invoke()` call (itinerary_node
    calls this once per day it needs to (re)generate)."""

    def __init__(self, results: list[_LLMDayPlan]):
        self._results = results
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> _LLMDayPlan:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self._results) - 1)
        return self._results[index]


class _FakeLLM:
    def __init__(self, results: list[_LLMDayPlan]):
        self.structured = _FakeStructuredLLM(results)

    def with_structured_output(self, schema) -> _FakeStructuredLLM:
        return self.structured


def _mock_external_apis(place_ids: list[str], forecast_date: dt.date):
    respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"places": [_place(p) for p in place_ids]})
    )
    timestamp = int(dt.datetime.combine(forecast_date, dt.time(12), tzinfo=dt.timezone.utc).timestamp())
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200, json={"list": [{"dt": timestamp, "main": {"temp": 20.0}, "weather": [{"main": "Clear"}]}]}
        )
    )
    respx.get(url__startswith=f"{OSRM_BASE_URL}/table/v1/driving/").mock(
        return_value=httpx.Response(200, json=_table_response(len(place_ids)))
    )


def _table_response(n: int) -> dict:
    # Uniform 5-min / 500-m spacing between every pair, 0 on the diagonal.
    durations = [[0.0 if i == j else 300.0 for j in range(n)] for i in range(n)]
    distances = [[0.0 if i == j else 500.0 for j in range(n)] for i in range(n)]
    return {"code": "Ok", "durations": durations, "distances": distances}


@respx.mock
def test_graph_happy_path_within_weather_horizon(monkeypatch):
    monkeypatch.setattr(rank_module, "embed_texts", lambda texts: np.ones((len(texts), 2)))
    start_date = dt.date.today() + dt.timedelta(days=2)
    _mock_external_apis(["p1", "p2"], forecast_date=start_date)

    llm = _FakeLLM(
        [
            _LLMDayPlan(
                stops=[
                    _LLMStop(place_id="p1", dwell_minutes=60),
                    _LLMStop(place_id="p2", dwell_minutes=60),
                ]
            )
        ]
    )
    monkeypatch.setattr(itinerary_module, "_default_llm", lambda: llm)

    graph = build_graph()
    # relaxed pace: 2 stops is within range and doesn't trip "ends too early".
    result = graph.invoke({"trip_setup": _trip_setup(start_date, daily_pace="relaxed")})

    assert result["refine_iterations"] == 0
    assert result["constraint_violations"] == []
    assert len(result["weather_forecast"]) == 1  # weather node ran
    assert result["itinerary"][0]["weather_note"] is not None
    assert "Lisbon" in result["summary"]


@respx.mock
def test_graph_skips_weather_beyond_horizon(monkeypatch):
    monkeypatch.setattr(rank_module, "embed_texts", lambda texts: np.ones((len(texts), 2)))
    start_date = dt.date.today() + dt.timedelta(days=FORECAST_HORIZON_DAYS + 10)
    _mock_external_apis(["p1"], forecast_date=start_date)

    llm = _FakeLLM([_LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60)])])
    monkeypatch.setattr(itinerary_module, "_default_llm", lambda: llm)

    graph = build_graph()
    result = graph.invoke({"trip_setup": _trip_setup(start_date, daily_pace="relaxed")})

    assert result.get("weather_forecast", []) == []
    assert result["itinerary"][0]["weather_note"] is None
    assert any("5-day horizon" in note for note in result["notes"])


@respx.mock
def test_graph_refine_loop_recovers_from_a_pace_violation(monkeypatch):
    monkeypatch.setattr(rank_module, "embed_texts", lambda texts: np.ones((len(texts), 2)))
    start_date = dt.date.today() + dt.timedelta(days=2)
    ids = [f"p{i}" for i in range(6)]
    _mock_external_apis(ids, forecast_date=start_date)

    too_packed = _LLMDayPlan(stops=[_LLMStop(place_id=i, dwell_minutes=45) for i in ids])  # 6 > moderate's 5
    fixed = _LLMDayPlan(stops=[_LLMStop(place_id=i, dwell_minutes=45) for i in ids[:4]])
    llm = _FakeLLM([too_packed, fixed])
    monkeypatch.setattr(itinerary_module, "_default_llm", lambda: llm)

    graph = build_graph()
    result = graph.invoke({"trip_setup": _trip_setup(start_date, daily_pace="moderate")})

    assert result["refine_iterations"] == 1
    assert result["constraint_violations"] == []
    attractions = [s for s in result["itinerary"][0]["stops"] if s["kind"] == "attraction"]
    assert len(attractions) == 4
    assert len(llm.structured.prompts) == 2
    assert "had these problems" in llm.structured.prompts[1]
