"""Tests for run_eval.py: the pure metric math, plus one smoke test that the
ablation harness itself is wired up (with a scripted LLM, no OpenAI call).
The full evaluation needs a real OPENAI_API_KEY -- see run_eval.py's
docstring.
"""

import datetime as dt
import re

import run_eval  # noqa: E402  (adds backend/ to sys.path as an import side effect)

from app.graph.nodes.itinerary import _LLMDayPlan, _LLMStop
from app.tools.places_client import Attraction
from app.tools.weather_client import DailyForecast


def _attraction(place_id: str, categories: list[str]) -> Attraction:
    return Attraction(
        place_id=place_id,
        name=place_id,
        address="",
        lat=0.0,
        lng=0.0,
        rating=4.5,
        rating_count=100,
        price_level="PRICE_LEVEL_MODERATE",
        categories=categories,
        description=None,
    )


def _day(date: dt.date, place_ids: list[str], *, travel: int = 0, dropped: list[str] | None = None) -> dict:
    return {
        "date": date,
        "stops": [
            {
                "attraction_name": pid,
                "place_id": pid,
                "arrival_offset_min": 0,
                "depart_offset_min": 60,
                "dwell_minutes": 60,
                "travel_from_prev_min": 0 if i == 0 else travel,
            }
            for i, pid in enumerate(place_ids)
        ],
        "weather_note": None,
        "dropped_stops": dropped or [],
    }


def test_constraints_satisfied_true_when_no_violations():
    assert run_eval.constraints_satisfied([]) is True


def test_constraints_satisfied_false_when_violations_present():
    assert run_eval.constraints_satisfied(["day 1: too many stops"]) is False


def test_empty_day_rate_counts_days_with_no_stops():
    date = dt.date(2026, 1, 1)
    itinerary = [_day(date, ["p1"]), _day(date + dt.timedelta(days=1), []), _day(date + dt.timedelta(days=2), [])]
    assert run_eval.empty_day_rate(itinerary) == 2 / 3


def test_empty_day_rate_is_zero_for_empty_itinerary():
    assert run_eval.empty_day_rate([]) == 0.0


def test_hallucinated_place_id_rate_counts_ids_outside_the_valid_set():
    raw_day_plans = [
        _LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60)]),
        _LLMDayPlan(stops=[_LLMStop(place_id="ghost", dwell_minutes=60)]),
    ]
    assert run_eval.hallucinated_place_id_rate(raw_day_plans, valid_ids={"p1"}) == 1 / 2


def test_avg_travel_minutes_per_day_averages_over_non_empty_days():
    date = dt.date(2026, 1, 1)
    itinerary = [
        _day(date, ["a", "b", "c"], travel=10),  # 0 + 10 + 10 = 20
        _day(date + dt.timedelta(days=1), ["d", "e"], travel=30),  # 0 + 30 = 30
        _day(date + dt.timedelta(days=2), []),  # skipped
    ]
    assert run_eval.avg_travel_minutes_per_day(itinerary) == 25.0


def test_avg_travel_minutes_per_day_is_none_without_stops():
    assert run_eval.avg_travel_minutes_per_day([]) is None


def test_dropped_stop_rate_is_dropped_over_total():
    date = dt.date(2026, 1, 1)
    itinerary = [_day(date, ["a", "b", "c"], dropped=["d"])]
    assert run_eval.dropped_stop_rate(itinerary) == 1 / 4


def test_dropped_stop_rate_is_zero_when_nothing_dropped():
    assert run_eval.dropped_stop_rate([_day(dt.date(2026, 1, 1), ["a", "b"])]) == 0.0


def test_hallucinated_place_id_rate_is_zero_with_no_stops():
    assert run_eval.hallucinated_place_id_rate([], valid_ids={"p1"}) == 0.0


def test_bad_weather_indoor_rate_ignores_days_without_bad_weather():
    date = dt.date(2026, 1, 1)
    itinerary = [_day(date, ["outdoor1"])]
    forecasts = {
        date: DailyForecast(date=date, avg_temp_c=20.0, condition="Clear", description="clear sky", is_bad_weather=False)
    }
    attractions_by_id = {"outdoor1": _attraction("outdoor1", ["park"])}
    assert run_eval.bad_weather_indoor_rate(itinerary, forecasts, attractions_by_id) is None


def test_bad_weather_indoor_rate_measures_indoor_share_on_bad_weather_days():
    date = dt.date(2026, 1, 1)
    itinerary = [_day(date, ["museum1", "park1"])]
    forecasts = {
        date: DailyForecast(date=date, avg_temp_c=10.0, condition="Rain", description="light rain", is_bad_weather=True)
    }
    attractions_by_id = {
        "museum1": _attraction("museum1", ["museum"]),
        "park1": _attraction("park1", ["park"]),
    }
    assert run_eval.bad_weather_indoor_rate(itinerary, forecasts, attractions_by_id) == 0.5


class _ScriptedStructuredLLM:
    """Picks the first few `place_id=` values it sees in the prompt, so the
    harness smoke test doesn't need a real model."""

    def invoke(self, prompt: str) -> _LLMDayPlan:
        ids = re.findall(r"place_id=(\S+)", prompt)
        hi = int(re.search(r"Return \d+-(\d+) stops", prompt).group(1))
        return _LLMDayPlan(stops=[_LLMStop(place_id=pid, dwell_minutes=120) for pid in ids[:hi]])


class _ScriptedLLM:
    def with_structured_output(self, schema):
        return _ScriptedStructuredLLM()


def test_every_variant_runs_end_to_end(monkeypatch):
    monkeypatch.setattr(run_eval, "_make_real_llm", _ScriptedLLM)
    scenario = next(s for s in run_eval.SCENARIOS if s.name == "packed_late_opening_landmarks")

    for variant in run_eval.VARIANTS:
        aggregated, rows = run_eval.evaluate(variant, [scenario])
        assert 0.0 <= aggregated["constraint_satisfaction_rate"] <= 1.0
        assert len(rows) == 1

    # With the scheduler on, the late-opening landmarks scenario should be
    # feasible; with it off (LLM assigns 09:00 to an 11:00 attraction) it
    # should not be.
    with_scheduler = run_eval.evaluate(run_eval.VARIANTS[-1], [scenario])[0]
    without_scheduler = run_eval.evaluate(run_eval.VARIANTS[0], [scenario])[0]
    assert with_scheduler["constraint_satisfaction_rate"] >= without_scheduler["constraint_satisfaction_rate"]
