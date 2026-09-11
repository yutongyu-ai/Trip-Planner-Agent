import datetime as dt

import httpx
import openai

from app.graph.nodes.itinerary import _LLMDayPlan, _LLMStop, itinerary_node
from app.recsys.rank import RankedAttraction
from app.tools.places_client import Attraction
from app.tools.route_client import DistanceMatrix
from app.tools.weather_client import DailyForecast

START = dt.date(2026, 9, 1)


class _FakeStructuredLLM:
    """Returns `results` in order, one per `.invoke()` call (itinerary_node
    calls this once per day it needs to (re)generate) -- reuses the last
    result once the list is exhausted, so a single-element list is enough
    for tests that only care about one day.
    """

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


class _FailingStructuredLLM:
    def invoke(self, prompt: str) -> _LLMDayPlan:
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))


class _FailingLLM:
    def with_structured_output(self, schema) -> _FailingStructuredLLM:
        return _FailingStructuredLLM()


def _ranked(
    place_id: str,
    categories: list[str],
    *,
    lat: float = 0.0,
    lng: float = 0.0,
    opening_hours: dict[int, tuple[int, int]] | None = None,
) -> RankedAttraction:
    attraction = Attraction(
        place_id=place_id,
        name=f"Name-{place_id}",
        address="",
        lat=lat,
        lng=lng,
        rating=4.0,
        rating_count=10,
        price_level=None,
        categories=categories,
        description=None,
        open_weekdays=frozenset(opening_hours) if opening_hours else None,
        opening_hours=opening_hours,
    )
    return RankedAttraction(attraction=attraction, score=1.0, similarity=1.0, normalized_rating=0.8)


def _trip_setup(days: int) -> dict:
    return {
        "destination": "Lisbon",
        "start_date": START,
        "end_date": START + dt.timedelta(days=days - 1),
        "interests": ["museums"],
        "daily_pace": "moderate",
        "budget_level": "moderate",
        "day_start_time": dt.time(9, 0),
        "day_end_time": dt.time(18, 0),
    }


def test_itinerary_node_resolves_llm_picks_into_stops():
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["park"])]
    llm = _FakeLLM([_LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60)])])

    result = itinerary_node({"trip_setup": _trip_setup(1), "ranked_attractions": ranked}, llm=llm)

    day = result["itinerary"][0]
    assert day["date"] == START
    # Filtered to real attraction stops -- the day spans lunchtime (9-18), so
    # a "meal" stop is also present; that's schedule_day_with_lunch's job,
    # covered separately in test_scheduling.py.
    attraction_stops = [s for s in day["stops"] if s["kind"] == "attraction"]
    assert attraction_stops == [
        {
            "attraction_name": "Name-p1",
            "place_id": "p1",
            "kind": "attraction",
            "arrival_offset_min": 0,
            "depart_offset_min": 60,
            "dwell_minutes": 60,
            "travel_from_prev_min": 0,
            "wait_for_opening_min": 0,
        }
    ]
    assert day["dropped_stops"] == []


def test_itinerary_node_drops_hallucinated_place_id():
    ranked = [_ranked("p1", ["museum"])]
    llm = _FakeLLM([_LLMDayPlan(stops=[_LLMStop(place_id="does-not-exist", dwell_minutes=60)])])

    result = itinerary_node({"trip_setup": _trip_setup(1), "ranked_attractions": ranked}, llm=llm)

    assert result["itinerary"][0]["stops"] == []


def test_itinerary_node_skips_llm_when_no_candidates():
    result = itinerary_node({"trip_setup": _trip_setup(2), "ranked_attractions": []}, llm=None)

    assert len(result["itinerary"]) == 2
    assert all(day["stops"] == [] for day in result["itinerary"])
    assert any("no ranked attractions" in e for e in result["errors"])


def test_itinerary_node_degrades_gracefully_when_llm_call_fails():
    ranked = [_ranked("p1", ["museum"])]

    result = itinerary_node({"trip_setup": _trip_setup(2), "ranked_attractions": ranked}, llm=_FailingLLM())

    assert len(result["itinerary"]) == 2
    assert all(day["stops"] == [] for day in result["itinerary"])
    assert any(e.startswith("itinerary: ") for e in result["errors"])


def test_itinerary_node_leaves_refine_iterations_unchanged_on_first_pass():
    ranked = [_ranked("p1", ["museum"])]
    llm = _FakeLLM([_LLMDayPlan(stops=[])])

    result = itinerary_node({"trip_setup": _trip_setup(1), "ranked_attractions": ranked}, llm=llm)

    assert result["refine_iterations"] == 0


def test_itinerary_node_increments_refine_iterations_when_retrying():
    ranked = [_ranked("p1", ["museum"])]
    llm = _FakeLLM([_LLMDayPlan(stops=[])])
    state = {
        "trip_setup": _trip_setup(1),
        "ranked_attractions": ranked,
        "constraint_violations": ["too packed"],
        "violations_by_day": {0: ["too packed"]},
        "refine_iterations": 1,
    }

    result = itinerary_node(state, llm=llm)

    assert result["refine_iterations"] == 2


def test_itinerary_node_calls_llm_once_per_day_on_first_pass():
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["museum"])]
    llm = _FakeLLM(
        [
            _LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60)]),
            _LLMDayPlan(stops=[_LLMStop(place_id="p2", dwell_minutes=60)]),
        ]
    )

    result = itinerary_node({"trip_setup": _trip_setup(2), "ranked_attractions": ranked}, llm=llm)

    assert len(llm.structured.prompts) == 2
    assert result["itinerary"][0]["stops"][0]["place_id"] == "p1"
    assert result["itinerary"][1]["stops"][0]["place_id"] == "p2"


def test_itinerary_node_excludes_place_ids_used_on_an_earlier_day_from_later_candidates():
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["museum"])]
    llm = _FakeLLM(
        [
            _LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60)]),
            _LLMDayPlan(stops=[_LLMStop(place_id="p2", dwell_minutes=60)]),
        ]
    )

    itinerary_node({"trip_setup": _trip_setup(2), "ranked_attractions": ranked}, llm=llm)

    day2_prompt = llm.structured.prompts[1]
    assert "place_id=p1" not in day2_prompt
    assert "place_id=p2" in day2_prompt


def test_itinerary_node_only_regenerates_days_that_have_a_violation_on_refine():
    # day 0 came out clean last attempt; day 1 (offset 1) is the one flagged.
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["museum"])]
    previous_day0 = {
        "date": START,
        "stops": [{"attraction_name": "Name-p1", "place_id": "p1", "arrival_offset_min": 0, "dwell_minutes": 60}],
        "weather_note": None,
    }
    previous_day1 = {"date": START + dt.timedelta(days=1), "stops": [], "weather_note": None}
    llm = _FakeLLM([_LLMDayPlan(stops=[_LLMStop(place_id="p2", dwell_minutes=60)])])
    state = {
        "trip_setup": _trip_setup(2),
        "ranked_attractions": ranked,
        "itinerary": [previous_day0, previous_day1],
        "constraint_violations": ["2026-09-02: 0 stops, outside the moderate range of 2-4"],
        "violations_by_day": {1: ["2026-09-02: 0 stops, outside the moderate range of 2-4"]},
        "refine_iterations": 0,
    }

    result = itinerary_node(state, llm=llm)

    assert result["itinerary"][0] == previous_day0  # untouched -- no violation, no LLM call
    attraction_stops = [s for s in result["itinerary"][1]["stops"] if s["kind"] == "attraction"]
    assert attraction_stops == [
        {
            "attraction_name": "Name-p2",
            "place_id": "p2",
            "kind": "attraction",
            "arrival_offset_min": 0,
            "depart_offset_min": 60,
            "dwell_minutes": 60,
            "travel_from_prev_min": 0,
            "wait_for_opening_min": 0,
        }
    ]
    assert len(llm.structured.prompts) == 1  # day 0 never called the LLM at all
    assert "0 stops, outside the moderate range" in llm.structured.prompts[0]


def test_itinerary_node_excludes_place_ids_from_a_later_unchanged_day_when_regenerating_an_earlier_one():
    # day 0 (offset 0) is the one flagged this time; day 1 came out clean
    # last attempt and already claimed p2. Regenerating day 0 must not be
    # able to re-pick p2, even though day 1 hasn't been reached by the loop
    # yet -- otherwise the same place ends up scheduled on both days.
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["museum"])]
    previous_day0 = {"date": START, "stops": [], "weather_note": None}
    previous_day1 = {
        "date": START + dt.timedelta(days=1),
        "stops": [{"attraction_name": "Name-p2", "place_id": "p2", "arrival_offset_min": 0, "dwell_minutes": 60}],
        "weather_note": None,
    }
    llm = _FakeLLM([_LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60)])])
    state = {
        "trip_setup": _trip_setup(2),
        "ranked_attractions": ranked,
        "itinerary": [previous_day0, previous_day1],
        "constraint_violations": ["2026-09-01: 0 stops, outside the moderate range of 2-4"],
        "violations_by_day": {0: ["2026-09-01: 0 stops, outside the moderate range of 2-4"]},
        "refine_iterations": 0,
    }

    result = itinerary_node(state, llm=llm)

    day0_prompt = llm.structured.prompts[0]
    assert "place_id=p2" not in day0_prompt  # already claimed by day 1, must not be offered
    assert result["itinerary"][1] == previous_day1  # untouched
    day0_ids = {s["place_id"] for s in result["itinerary"][0]["stops"]}
    day1_ids = {s["place_id"] for s in result["itinerary"][1]["stops"]}
    assert not (day0_ids & day1_ids)


def test_itinerary_node_sorts_stops_by_arrival_time():
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["museum"])]
    llm = _FakeLLM(
        [
            _LLMDayPlan(
                stops=[
                    _LLMStop(place_id="p2", dwell_minutes=60),
                    _LLMStop(place_id="p1", dwell_minutes=60),
                ]
            )
        ]
    )

    result = itinerary_node({"trip_setup": _trip_setup(1), "ranked_attractions": ranked}, llm=llm)

    attraction_ids = [s["place_id"] for s in result["itinerary"][0]["stops"] if s["kind"] == "attraction"]
    assert attraction_ids == ["p1", "p2"]


def test_weather_note_for_bad_weather_all_indoor_stops():
    ranked = [_ranked("p1", ["museum"])]
    llm = _FakeLLM([_LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60)])])
    forecast = [
        DailyForecast(date=START, avg_temp_c=12.0, condition="Rain", description="light rain", is_bad_weather=True)
    ]

    result = itinerary_node(
        {"trip_setup": _trip_setup(1), "ranked_attractions": ranked, "weather_forecast": forecast}, llm=llm
    )

    assert "adjusted to indoor stops" in result["itinerary"][0]["weather_note"]


def test_weather_note_for_bad_weather_not_all_indoor_stops():
    ranked = [_ranked("p1", ["park"])]
    llm = _FakeLLM([_LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60)])])
    forecast = [
        DailyForecast(date=START, avg_temp_c=12.0, condition="Rain", description="light rain", is_bad_weather=True)
    ]

    result = itinerary_node(
        {"trip_setup": _trip_setup(1), "ranked_attractions": ranked, "weather_forecast": forecast}, llm=llm
    )

    assert "consider indoor alternatives" in result["itinerary"][0]["weather_note"]


def test_weather_note_in_chinese():
    ranked = [_ranked("p1", ["museum"])]
    llm = _FakeLLM([_LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60)])])
    forecast = [
        DailyForecast(date=START, avg_temp_c=20.0, condition="Clear", description="晴", is_bad_weather=False)
    ]
    trip_setup = _trip_setup(1)
    trip_setup["lang"] = "zh"

    result = itinerary_node(
        {"trip_setup": trip_setup, "ranked_attractions": ranked, "weather_forecast": forecast}, llm=llm
    )

    assert result["itinerary"][0]["weather_note"] == "晴，约 20°C"


def test_weather_note_is_none_without_forecast():
    ranked = [_ranked("p1", ["museum"])]
    llm = _FakeLLM([_LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60)])])

    result = itinerary_node({"trip_setup": _trip_setup(1), "ranked_attractions": ranked}, llm=llm)

    assert result["itinerary"][0]["weather_note"] is None


def test_itinerary_node_schedules_a_pick_around_its_late_opening_hours():
    # START (2026-09-01) is a Tuesday, weekday() == 1. The LLM picks a place
    # that doesn't open until 11:00 as its first stop; the scheduler must not
    # leave it starting at the 09:00 day start (the real London-Eye bug).
    ranked = [
        _ranked("late", ["tourist_attraction"], opening_hours={1: (11 * 60, 18 * 60)}),
        _ranked("open", ["park"]),
    ]
    llm = _FakeLLM(
        [_LLMDayPlan(stops=[_LLMStop(place_id="late", dwell_minutes=45), _LLMStop(place_id="open", dwell_minutes=90)])]
    )

    result = itinerary_node({"trip_setup": _trip_setup(1), "ranked_attractions": ranked}, llm=llm)

    stops = {s["place_id"]: s for s in result["itinerary"][0]["stops"] if s["kind"] == "attraction"}
    assert len(stops) == 2
    # 09:00 day start + offset must be >= 11:00 for the late-opening stop.
    assert 9 * 60 + stops["late"]["arrival_offset_min"] >= 11 * 60
    assert result["itinerary"][0]["dropped_stops"] == []


def test_itinerary_node_drops_a_pick_closed_on_the_visit_weekday():
    # START (2026-09-01) is a Tuesday, weekday() == 1. "closed" only lists
    # opening hours for Monday (0), so it's shut all day Tuesday. A missing
    # weekday in opening_hours means "closed that day", not "unknown hours" --
    # ScheduleInput has no "closed all day" field (only a within-day
    # open_min/close_min, where None means "open all day"), so this pick must
    # be filtered out before it ever reaches the scheduler.
    ranked = [
        _ranked("closed", ["museum"], opening_hours={0: (9 * 60, 18 * 60)}),
        _ranked("open", ["park"]),
    ]
    llm = _FakeLLM(
        [
            _LLMDayPlan(
                stops=[_LLMStop(place_id="closed", dwell_minutes=60), _LLMStop(place_id="open", dwell_minutes=60)]
            )
        ]
    )

    result = itinerary_node({"trip_setup": _trip_setup(1), "ranked_attractions": ranked}, llm=llm)

    day = result["itinerary"][0]
    assert [s["place_id"] for s in day["stops"] if s["kind"] == "attraction"] == ["open"]
    assert day["dropped_stops"] == ["Name-closed"]


def test_itinerary_node_caps_early_days_so_late_days_keep_candidates():
    # 5 candidates, 3 relaxed days (2-3 stops each). Day 0 taking its full 3
    # would leave the last two days short, so it's held to 2.
    ranked = [_ranked(f"p{i}", ["museum"]) for i in range(5)]
    trip = {
        "destination": "Lisbon",
        "start_date": START,
        "end_date": START + dt.timedelta(days=2),
        "interests": ["museums"],
        "daily_pace": "relaxed",
        "budget_level": "moderate",
        "day_start_time": dt.time(9, 0),
        "day_end_time": dt.time(18, 0),
    }
    llm = _FakeLLM(
        [
            _LLMDayPlan(stops=[_LLMStop(place_id="p0", dwell_minutes=60), _LLMStop(place_id="p1", dwell_minutes=60)]),
            _LLMDayPlan(stops=[_LLMStop(place_id="p2", dwell_minutes=60), _LLMStop(place_id="p3", dwell_minutes=60)]),
            _LLMDayPlan(stops=[_LLMStop(place_id="p4", dwell_minutes=60)]),
        ]
    )

    result = itinerary_node({"trip_setup": trip, "ranked_attractions": ranked}, llm=llm)

    assert "about 2-2 stops" in llm.structured.prompts[0]  # capped: 5 candidates, 2 days still to plan
    attractions_per_day = [
        sum(1 for s in day["stops"] if s["kind"] == "attraction") for day in result["itinerary"]
    ]
    assert attractions_per_day == [2, 2, 1]  # no day left empty


def test_itinerary_node_uses_the_osrm_matrix_for_travel_time():
    # Matrix says p1->p2 is 40 raw driving min; realistic_transit_minutes
    # turns that into 40 * 1.6 + 8 = 72 door-to-door. A straight-line
    # estimate for two identical coords would instead be ~0.
    ranked = [_ranked("p1", ["museum"]), _ranked("p2", ["museum"])]
    matrix = DistanceMatrix(
        place_ids=["p1", "p2"],
        duration_s=[[0.0, 2400.0], [2400.0, 0.0]],
        distance_m=[[0.0, 20000.0], [20000.0, 0.0]],
    )
    llm = _FakeLLM(
        [_LLMDayPlan(stops=[_LLMStop(place_id="p1", dwell_minutes=60), _LLMStop(place_id="p2", dwell_minutes=60)])]
    )

    result = itinerary_node(
        {"trip_setup": _trip_setup(1), "ranked_attractions": ranked, "distance_matrix": matrix}, llm=llm
    )

    stops = [s for s in result["itinerary"][0]["stops"] if s["kind"] == "attraction"]
    assert [s["place_id"] for s in stops] == ["p1", "p2"]
    assert stops[1]["travel_from_prev_min"] == 72


def test_itinerary_node_drops_a_pick_that_cannot_fit_the_day():
    # Four 150-minute stops don't fit a 540-minute day (600 min needed); the
    # lowest-ranked is dropped and reported.
    ranked = [_ranked(f"p{i}", ["amusement_park"]) for i in range(4)]
    llm = _FakeLLM([_LLMDayPlan(stops=[_LLMStop(place_id=f"p{i}", dwell_minutes=150) for i in range(4)])])

    result = itinerary_node({"trip_setup": _trip_setup(1), "ranked_attractions": ranked}, llm=llm)

    day = result["itinerary"][0]
    assert day["dropped_stops"] == ["Name-p3"]
    assert {s["place_id"] for s in day["stops"] if s["kind"] == "attraction"} == {"p0", "p1", "p2"}
