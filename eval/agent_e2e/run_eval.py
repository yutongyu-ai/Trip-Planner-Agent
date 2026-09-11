"""End-to-end evaluation of the LangGraph trip-planning agent, as an
ablation over the three pieces that make an itinerary feasible.

`backend/tests/test_graph.py` already checks the graph is wired up
correctly with a scripted fake LLM. What this script measures is how good
the itinerary node's *real* gpt-4o-mini decisions are, and how much each of
these earns its complexity:

  1. the bounded refine loop         (app/graph/nodes/constraints.py)
  2. the deterministic day scheduler (app/graph/scheduling.py)
  3. the real OSRM distance matrix   (app/graph/nodes/route.py)

Each is toggled independently, giving four cumulative variants from
"LLM assigns its own times, no retry" up to the full system. Places /
Weather / Route are mocked via respx from `scenarios.SCENARIOS` so every run
is reproducible; the OpenAI call inside itinerary_node is real
(gpt-4o-mini, temperature=0), so this needs a real OPENAI_API_KEY in the
repo-root .env and makes a modest number of paid API calls (one per
scenario per variant).

Run: python eval/agent_e2e/run_eval.py
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import httpx
import numpy as np
import respx
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import get_settings  # noqa: E402
from app.graph.nodes import itinerary as itinerary_module  # noqa: E402
from app.graph.nodes import route as route_module  # noqa: E402
from app.graph.nodes._shared import attraction_lookup, is_indoor  # noqa: E402
from app.graph.nodes.attractions import attractions_node  # noqa: E402
from app.graph.nodes.constraints import check_constraints_node, should_refine  # noqa: E402
from app.graph.nodes.finalize import finalize_node  # noqa: E402
from app.graph.nodes.itinerary import _LLMDayPlan, itinerary_node  # noqa: E402
from app.graph.nodes.route import route_node  # noqa: E402
from app.graph.nodes.weather import should_check_weather, weather_node  # noqa: E402
from app.graph.scheduling import ScheduledDay, ScheduledStop  # noqa: E402
from app.graph.state import AgentState  # noqa: E402
from app.recsys import rank as rank_module  # noqa: E402
from app.tools.places_client import PLACES_SEARCH_URL  # noqa: E402
from app.tools.route_client import RouteClientError  # noqa: E402
from app.tools.weather_client import DailyForecast, FORECAST_URL  # noqa: E402
from scenarios import SCENARIOS, Scenario  # noqa: E402

OSRM_TABLE_PREFIX = "https://router.project-osrm.org/table/v1/driving/"
_ASSUMED_CITY_SPEED_KMH = 25.0


# ---------------------------------------------------------------------------
# Variants: each flag cumulatively enables one of the three feasibility pieces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    name: str
    refine: bool
    scheduler: bool
    matrix: bool


VARIANTS = [
    Variant("llm-times, no retry", refine=False, scheduler=False, matrix=False),
    Variant("+ refine loop", refine=True, scheduler=False, matrix=False),
    Variant("+ scheduler", refine=True, scheduler=True, matrix=False),
    Variant("+ osrm matrix (full)", refine=True, scheduler=True, matrix=True),
]


def build_graph_variant(refine: bool):
    graph = StateGraph(AgentState)
    graph.add_node("attractions", attractions_node)
    graph.add_node("weather", weather_node)
    graph.add_node("route", route_node)
    graph.add_node("itinerary", itinerary_node)
    graph.add_node("check_constraints", check_constraints_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "attractions")
    graph.add_conditional_edges("attractions", should_check_weather, {"weather": "weather", "route": "route"})
    graph.add_edge("weather", "route")
    graph.add_edge("route", "itinerary")
    graph.add_edge("itinerary", "check_constraints")
    if refine:
        graph.add_conditional_edges(
            "check_constraints", should_refine, {"itinerary": "itinerary", "finalize": "finalize"}
        )
    else:
        graph.add_edge("check_constraints", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Ablation stand-ins
# ---------------------------------------------------------------------------


def _naive_schedule_day(
    picks, *, day_start_min, day_end_min, travel_minutes, lunch_dwell_minutes=None, lunch_label=None
):
    """Stand-in for a run with the deterministic scheduler (and its lunch
    placement) disabled entirely: keep the LLM's pick order, pack visits
    back-to-back from the day start, ignore opening hours and travel time,
    and skip lunch -- this is what an LLM left to assign its own arrival
    times tends to do. Travel time is still recorded for the travel metric
    -- it just isn't respected in the timing. Accepts (and ignores) the
    lunch kwargs so it's a drop-in replacement for schedule_day_with_lunch."""
    stops: list[ScheduledStop] = []
    clock = 0
    prev_id: str | None = None
    for pick in picks:
        travel = 0 if prev_id is None else max(0, round(travel_minutes(prev_id, pick.place_id)))
        stops.append(
            ScheduledStop(
                place_id=pick.place_id,
                name=pick.name,
                arrival_offset_min=clock,
                depart_offset_min=clock + pick.dwell_minutes,
                travel_from_prev_min=travel,
                wait_for_opening_min=0,
            )
        )
        clock += pick.dwell_minutes
        prev_id = pick.place_id
    return ScheduledDay(stops=stops, dropped=[])


class _FailingRouteProvider:
    """Stand-in for a run with the OSRM matrix disabled -- route_node then
    stores no matrix and the scheduler falls back to straight-line
    estimates."""

    def get_distance_matrix(self, place_ids, stops):
        raise RouteClientError("ablation: matrix disabled")


# ---------------------------------------------------------------------------
# Pure metric functions (see test_metrics.py)
# ---------------------------------------------------------------------------


def constraints_satisfied(violations: list[str]) -> bool:
    return not violations


def _attractions(day: dict) -> list[dict]:
    return [s for s in day["stops"] if s.get("kind", "attraction") == "attraction"]


def empty_day_rate(itinerary: list[dict]) -> float:
    if not itinerary:
        return 0.0
    return sum(1 for day in itinerary if not _attractions(day)) / len(itinerary)


def avg_travel_minutes_per_day(itinerary: list[dict]) -> float | None:
    """Mean, over non-empty days, of the total between-stop travel time the
    plan implies. Lower is better -- a well-ordered day spends less of itself
    in transit. None if there were no non-empty days."""
    totals = [
        sum(stop.get("travel_from_prev_min", 0) for stop in day["stops"]) for day in itinerary if day["stops"]
    ]
    return float(np.mean(totals)) if totals else None


def dropped_stop_rate(itinerary: list[dict]) -> float:
    """Fraction of the attractions that reached the scheduler which it had to
    drop to make the day fit. 0.0 when the scheduler is disabled."""
    scheduled = sum(len(_attractions(day)) for day in itinerary)
    dropped = sum(len(day.get("dropped_stops", [])) for day in itinerary)
    total = scheduled + dropped
    return dropped / total if total else 0.0


def hallucinated_place_id_rate(raw_day_plans: list[_LLMDayPlan], valid_ids: set[str]) -> float:
    total = 0
    hallucinated = 0
    for day_plan in raw_day_plans:
        for stop in day_plan.stops:
            total += 1
            if stop.place_id not in valid_ids:
                hallucinated += 1
    return hallucinated / total if total else 0.0


def bad_weather_indoor_rate(
    itinerary: list[dict],
    forecasts_by_date: dict[dt.date, DailyForecast],
    attractions_by_id: dict,
) -> float | None:
    total = 0
    indoor = 0
    for day in itinerary:
        forecast = forecasts_by_date.get(day["date"])
        if forecast is None or not forecast.is_bad_weather:
            continue
        for stop in day["stops"]:
            attraction = attractions_by_id.get(stop["place_id"])
            if attraction is None:
                continue
            total += 1
            if is_indoor(attraction):
                indoor += 1
    return indoor / total if total else None


# ---------------------------------------------------------------------------
# Real-LLM recording wrapper (same monkeypatch point test_graph.py uses)
# ---------------------------------------------------------------------------


class _RecordingStructuredLLM:
    def __init__(self, structured, records: list[_LLMDayPlan]):
        self._structured = structured
        self._records = records

    def invoke(self, prompt: str) -> _LLMDayPlan:
        result = self._structured.invoke(prompt)
        self._records.append(result)
        return result


class _RecordingLLM:
    def __init__(self, real_llm, records: list[_LLMDayPlan]):
        self._real_llm = real_llm
        self._records = records

    def with_structured_output(self, schema):
        return _RecordingStructuredLLM(self._real_llm.with_structured_output(schema), self._records)


def _make_real_llm() -> ChatOpenAI:
    return ChatOpenAI(model=itinerary_module.MODEL_NAME, api_key=get_settings().openai_api_key, temperature=0)


# ---------------------------------------------------------------------------
# respx mocking of Places / Weather / Route from a Scenario
# ---------------------------------------------------------------------------


def _place_json(spec) -> dict:
    place = {
        "id": spec.place_id,
        "displayName": {"text": spec.place_id.replace("-", " ").title()},
        "formattedAddress": "123 Main St",
        "location": {"latitude": spec.lat, "longitude": spec.lng},
        "rating": spec.rating,
        "userRatingCount": 250,
        "priceLevel": spec.price_level,
        "types": spec.categories,
        "editorialSummary": {"text": f"A well-known {spec.categories[0].replace('_', ' ')}."},
    }
    if spec.open_min is not None and spec.close_min is not None:
        # Places API Period.open.day is 0=Sunday..6=Saturday; give every day
        # the same window.
        place["regularOpeningHours"] = {
            "periods": [
                {
                    "open": {"day": d, "hour": spec.open_min // 60, "minute": spec.open_min % 60},
                    "close": {"day": d, "hour": spec.close_min // 60, "minute": spec.close_min % 60},
                }
                for d in range(7)
            ]
        }
    return place


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r1, g1, r2, g2 = (math.radians(v) for v in (lat1, lng1, lat2, lng2))
    dlat, dlng = r2 - r1, g2 - g1
    h = math.sin(dlat / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlng / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _table_response(request: httpx.Request) -> httpx.Response:
    """Build an OSRM-shaped duration/distance matrix from the coordinates in
    the request path, so distances track each scenario's real geometry."""
    coord_str = request.url.path.rsplit("/", 1)[-1]
    points = [(float(lat), float(lng)) for lng, lat in (c.split(",") for c in coord_str.split(";"))]
    n = len(points)
    durations = [[0.0] * n for _ in range(n)]
    distances = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            km = _haversine_km(*points[i], *points[j]) * 1.3
            distances[i][j] = km * 1000
            durations[i][j] = km / _ASSUMED_CITY_SPEED_KMH * 3600
    return httpx.Response(200, json={"code": "Ok", "durations": durations, "distances": distances})


def _mock_apis(scenario: Scenario) -> None:
    respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"places": [_place_json(p) for p in scenario.places]})
    )

    start_date = scenario.trip_setup()["start_date"]
    forecast_entries = []
    for offset in range(scenario.forecast_days):
        date = start_date + dt.timedelta(days=offset)
        timestamp = int(dt.datetime.combine(date, dt.time(12), tzinfo=dt.timezone.utc).timestamp())
        forecast_entries.append(
            {"dt": timestamp, "main": {"temp": 18.0}, "weather": [{"main": scenario.forecast_condition}]}
        )
    respx.get(FORECAST_URL).mock(return_value=httpx.Response(200, json={"list": forecast_entries}))

    respx.get(url__startswith=OSRM_TABLE_PREFIX).mock(side_effect=_table_response)


def _fake_embeddings(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 2))


# ---------------------------------------------------------------------------
# Per-scenario run
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    final_state: dict
    raw_llm_day_plans: list[_LLMDayPlan]


def _run(graph, scenario: Scenario, variant: Variant) -> RunOutcome:
    records: list[_LLMDayPlan] = []
    recording_llm = _RecordingLLM(_make_real_llm(), records)

    patches = [
        mock.patch.object(itinerary_module, "_default_llm", lambda: recording_llm),
        mock.patch.object(rank_module, "embed_texts", _fake_embeddings),
    ]
    if not variant.scheduler:
        patches.append(mock.patch.object(itinerary_module, "schedule_day_with_lunch", _naive_schedule_day))
    if not variant.matrix:
        patches.append(
            mock.patch.object(route_module, "get_route_provider", lambda country_code=None: _FailingRouteProvider())
        )

    with respx.mock:
        _mock_apis(scenario)
        for patch in patches:
            patch.start()
        try:
            final_state = graph.invoke({"trip_setup": scenario.trip_setup()})
        finally:
            for patch in reversed(patches):
                patch.stop()

    return RunOutcome(final_state=final_state, raw_llm_day_plans=records)


def _forecasts_by_date(state: dict) -> dict[dt.date, DailyForecast]:
    return {f.date: f for f in state.get("weather_forecast", [])}


def evaluate(variant: Variant, scenarios: list[Scenario]) -> tuple[dict[str, float], list[dict]]:
    graph = build_graph_variant(variant.refine)
    metrics: dict[str, list[float]] = {
        "constraint_satisfaction_rate": [],
        "refine_iterations": [],
        "empty_day_rate": [],
        "avg_travel_min_per_day": [],
        "dropped_stop_rate": [],
        "hallucinated_place_id_rate": [],
    }
    bad_weather_indoor_rates: list[float] = []
    rows: list[dict] = []

    for scenario in scenarios:
        outcome = _run(graph, scenario, variant)
        state = outcome.final_state
        itinerary = state.get("itinerary", [])
        violations = state.get("constraint_violations", [])
        valid_ids = {p.place_id for p in scenario.places}
        attractions_by_id = attraction_lookup(state.get("ranked_attractions", []))

        metrics["constraint_satisfaction_rate"].append(1.0 if constraints_satisfied(violations) else 0.0)
        metrics["refine_iterations"].append(float(state.get("refine_iterations", 0)))
        metrics["empty_day_rate"].append(empty_day_rate(itinerary))
        travel = avg_travel_minutes_per_day(itinerary)
        if travel is not None:
            metrics["avg_travel_min_per_day"].append(travel)
        metrics["dropped_stop_rate"].append(dropped_stop_rate(itinerary))
        metrics["hallucinated_place_id_rate"].append(hallucinated_place_id_rate(outcome.raw_llm_day_plans, valid_ids))

        indoor_rate = bad_weather_indoor_rate(itinerary, _forecasts_by_date(state), attractions_by_id)
        if indoor_rate is not None:
            bad_weather_indoor_rates.append(indoor_rate)

        rows.append(
            {
                "scenario": scenario.name,
                "satisfied": not violations,
                "violations": list(violations),
                "refine_iterations": state.get("refine_iterations", 0),
                "dropped": [name for day in itinerary for name in day.get("dropped_stops", [])],
            }
        )

    aggregated = {name: (float(np.mean(v)) if v else float("nan")) for name, v in metrics.items()}
    aggregated["bad_weather_indoor_rate"] = (
        float(np.mean(bad_weather_indoor_rates)) if bad_weather_indoor_rates else float("nan")
    )
    return aggregated, rows


def main() -> None:
    if not get_settings().openai_api_key:
        sys.exit(
            "OPENAI_API_KEY is not set (checked the repo-root .env via app.config.get_settings()). "
            "This script makes real gpt-4o-mini calls and needs a real key."
        )

    import logging

    logging.getLogger("app.graph.nodes.route").setLevel(logging.ERROR)  # the matrix-off ablation logs a warning per run

    print(f"Running {len(SCENARIOS)} synthetic scenarios through {len(VARIANTS)} variants...\n")

    aggregated: dict[str, dict[str, float]] = {}
    per_scenario: dict[str, list[dict]] = {}
    for variant in VARIANTS:
        aggregated[variant.name], per_scenario[variant.name] = evaluate(variant, SCENARIOS)

    metric_names = list(next(iter(aggregated.values())).keys())
    header = f"{'metric':<30}" + "".join(f"{v.name:<22}" for v in VARIANTS)
    print(header)
    print("-" * len(header))
    for name in metric_names:
        row = f"{name:<30}" + "".join(f"{aggregated[v.name][name]:<22.3f}" for v in VARIANTS)
        print(row)

    print("\nper-scenario constraint satisfaction (Y = 0 violations):")
    print(f"{'scenario':<34}" + "".join(f"{v.name[:20]:<22}" for v in VARIANTS))
    for i, scenario in enumerate(SCENARIOS):
        marks = "".join(f"{'Y' if per_scenario[v.name][i]['satisfied'] else 'n':<22}" for v in VARIANTS)
        print(f"{scenario.name:<34}{marks}")

    print("\nremaining violations in the full variant:")
    for row in per_scenario[VARIANTS[-1].name]:
        if row["violations"]:
            print(f"  {row['scenario']} (refine x{row['refine_iterations']}):")
            for v in row["violations"]:
                print(f"    - {v}")
        if row["dropped"]:
            print(f"  {row['scenario']} dropped: {', '.join(row['dropped'])}")


if __name__ == "__main__":
    main()
