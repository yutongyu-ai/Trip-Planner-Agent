import datetime as dt

from app.graph.nodes._shared import category_dwell_minutes
from app.graph.nodes.constraints import check_constraints_node, should_refine
from app.recsys.rank import RankedAttraction
from app.tools.places_client import Attraction

DAY = dt.date(2026, 9, 1)  # a Tuesday


def _ranked(
    place_id: str,
    price_level: str | None,
    open_weekdays: frozenset[int] | None = None,
    opening_hours: dict[int, tuple[int, int]] | None = None,
    lat: float = 0.0,
    lng: float = 0.0,
) -> RankedAttraction:
    attraction = Attraction(
        place_id=place_id,
        name=place_id,
        address="",
        lat=lat,
        lng=lng,
        rating=4.0,
        rating_count=10,
        price_level=price_level,
        categories=["museum"],
        description=None,
        open_weekdays=open_weekdays,
        opening_hours=opening_hours,
    )
    return RankedAttraction(attraction=attraction, score=1.0, similarity=1.0, normalized_rating=0.8)


def _state(daily_pace: str, budget_level: str, stops: list[str], ranked: list[RankedAttraction]) -> dict:
    attractions_by_id = {r.attraction.place_id: r.attraction for r in ranked}
    return {
        "trip_setup": {
            "daily_pace": daily_pace,
            "budget_level": budget_level,
            "day_start_time": dt.time(9, 0),
            "day_end_time": dt.time(18, 0),
        },
        "ranked_attractions": ranked,
        "itinerary": [
            {
                "date": DAY,
                # 90 min apart, same coordinates -- spaced out enough that the
                # timing checks (dwell + travel gap, end-of-window) never fire
                # unless a test is specifically exercising them.
                "stops": [
                    {
                        "place_id": p,
                        "attraction_name": p,
                        "arrival_offset_min": i * 90,
                        "dwell_minutes": category_dwell_minutes(attractions_by_id[p]),
                    }
                    for i, p in enumerate(stops)
                ],
                "weather_note": None,
            }
        ],
    }


def test_no_violations_within_pace_and_budget():
    ranked = [_ranked("p1", "PRICE_LEVEL_INEXPENSIVE"), _ranked("p2", "PRICE_LEVEL_INEXPENSIVE")]
    state = {
        "trip_setup": {
            "daily_pace": "moderate",
            "budget_level": "moderate",
            "day_start_time": dt.time(9, 0),
            "day_end_time": dt.time(18, 0),
        },
        "ranked_attractions": ranked,
        "itinerary": [
            {
                "date": DAY,
                # 270 min apart -- comfortably clears the dwell+travel gap
                # minimum.
                "stops": [
                    {"place_id": "p1", "attraction_name": "p1", "arrival_offset_min": 0, "dwell_minutes": 60},
                    {"place_id": "p2", "attraction_name": "p2", "arrival_offset_min": 270, "dwell_minutes": 60},
                ],
                "weather_note": None,
            }
        ],
    }

    result = check_constraints_node(state)

    assert result["constraint_violations"] == []


def test_flags_too_many_stops_for_relaxed_pace():
    ranked = [_ranked(f"p{i}", None) for i in range(5)]
    state = _state("relaxed", "luxury", [f"p{i}" for i in range(5)], ranked)

    result = check_constraints_node(state)

    assert any("stops" in v for v in result["constraint_violations"])


def test_does_not_flag_a_thin_day_when_the_candidate_pool_is_exhausted():
    # 2 ranked places, both scheduled; day 2 has 0 stops -- not the LLM's
    # fault, there's nothing left to add, so no "too few stops" violation.
    ranked = [_ranked("p1", None), _ranked("p2", None)]
    state = {
        "trip_setup": {
            "daily_pace": "moderate",  # min 2 stops/day
            "budget_level": "luxury",
            "day_start_time": dt.time(9, 0),
            "day_end_time": dt.time(18, 0),
        },
        "ranked_attractions": ranked,
        "itinerary": [
            {
                "date": DAY,
                "stops": [
                    {"place_id": "p1", "attraction_name": "p1", "arrival_offset_min": 0, "dwell_minutes": 60},
                    {"place_id": "p2", "attraction_name": "p2", "arrival_offset_min": 270, "dwell_minutes": 60},
                ],
                "weather_note": None,
            },
            {"date": DAY + dt.timedelta(days=1), "stops": [], "weather_note": None},
        ],
    }

    result = check_constraints_node(state)

    assert not any("stops" in v for v in result["constraint_violations"])


def test_still_flags_too_many_stops_even_when_pool_is_exhausted():
    ranked = [_ranked(f"p{i}", None) for i in range(5)]
    state = _state("relaxed", "luxury", [f"p{i}" for i in range(5)], ranked)

    result = check_constraints_node(state)

    assert any("stops" in v for v in result["constraint_violations"])


def test_flags_over_budget_price_level_when_cheaper_options_existed():
    # 3 cheap candidates available (>= the relaxed minimum of 2), so an
    # over-budget day is the LLM's fault, not the city's.
    ranked = [_ranked(f"cheap{i}", "PRICE_LEVEL_INEXPENSIVE") for i in range(3)] + [
        _ranked("exp", "PRICE_LEVEL_VERY_EXPENSIVE")
    ]
    state = _state("relaxed", "budget", ["cheap0", "exp"], ranked)

    result = check_constraints_node(state)

    assert any("budget cap" in v for v in result["constraint_violations"])


def test_does_not_flag_over_budget_when_the_city_has_too_few_affordable_options():
    # Only 1 cheap candidate but the relaxed minimum is 2 -- an over-budget
    # day is unavoidable, so no hard violation (finalize.py adds a note).
    ranked = [_ranked("cheap", "PRICE_LEVEL_INEXPENSIVE")] + [
        _ranked(f"exp{i}", "PRICE_LEVEL_VERY_EXPENSIVE") for i in range(4)
    ]
    state = _state("relaxed", "budget", ["cheap", "exp0"], ranked)

    result = check_constraints_node(state)

    assert not any("budget cap" in v for v in result["constraint_violations"])


def test_flags_stop_count_violation_in_chinese():
    ranked = [_ranked(f"p{i}", None) for i in range(5)]
    state = _state("relaxed", "luxury", [f"p{i}" for i in range(5)], ranked)
    state["trip_setup"]["lang"] = "zh"

    result = check_constraints_node(state)

    assert any("超出了" in v for v in result["constraint_violations"])


def test_missing_price_level_is_not_penalized():
    ranked = [_ranked("p1", None)]
    state = _state("relaxed", "budget", ["p1"], ranked)

    result = check_constraints_node(state)

    assert result["constraint_violations"] == []


def test_flags_stop_closed_on_the_scheduled_weekday():
    # DAY is a Tuesday (weekday()==1); this place is only open Saturdays (5).
    ranked = [_ranked("p1", None, open_weekdays=frozenset({5}))]
    state = _state("relaxed", "moderate", ["p1"], ranked)

    result = check_constraints_node(state)

    assert any("closed" in v for v in result["constraint_violations"])


def test_open_weekdays_none_is_not_penalized():
    ranked = [_ranked("p1", None, open_weekdays=None)]
    state = _state("relaxed", "moderate", ["p1"], ranked)

    result = check_constraints_node(state)

    assert result["constraint_violations"] == []


def test_flags_stop_scheduled_outside_opening_hours():
    # DAY is a Tuesday (weekday()==1); open 09:00-11:00 (540-660min) that day.
    ranked = [_ranked("p1", None, opening_hours={1: (540, 660)})]
    state = {
        "trip_setup": {
            "daily_pace": "relaxed",
            "budget_level": "luxury",
            "day_start_time": dt.time(9, 0),
            "day_end_time": dt.time(18, 0),
        },
        "ranked_attractions": ranked,
        "itinerary": [
            {
                "date": DAY,
                # Day starts 09:00; offset 180 = 12:00, an hour past the 11:00 close.
                "stops": [
                    {"place_id": "p1", "attraction_name": "p1", "arrival_offset_min": 180, "dwell_minutes": 60}
                ],
                "weather_note": None,
            }
        ],
    }

    result = check_constraints_node(state)

    assert any("11:00" in v for v in result["constraint_violations"])


def test_does_not_flag_a_stop_scheduled_within_opening_hours():
    # DAY is a Tuesday (weekday()==1); open 09:00-18:00 that day, the same
    # as the traveller's own day window -- a legally-timed stop always fits.
    ranked = [_ranked("p1", None, opening_hours={1: (540, 1080)})]
    state = _state("relaxed", "luxury", ["p1"], ranked)

    result = check_constraints_node(state)

    assert result["constraint_violations"] == []


def test_opening_hours_none_is_not_penalized():
    ranked = [_ranked("p1", None, opening_hours=None)]
    state = _state("relaxed", "luxury", ["p1"], ranked)

    result = check_constraints_node(state)

    assert result["constraint_violations"] == []


def test_flags_place_repeated_across_days():
    ranked = [_ranked("p1", None), _ranked("p2", None)]
    state = {
        "trip_setup": {
            "daily_pace": "relaxed",
            "budget_level": "luxury",
            "day_start_time": dt.time(9, 0),
            "day_end_time": dt.time(18, 0),
        },
        "ranked_attractions": ranked,
        "itinerary": [
            {
                "date": DAY,
                "stops": [{"place_id": "p1", "attraction_name": "p1", "arrival_offset_min": 0, "dwell_minutes": 60}],
                "weather_note": None,
            },
            {
                "date": DAY + dt.timedelta(days=1),
                # p1 again, on the next day -- should be flagged even though
                # each day individually is within the relaxed 1-2 stop range.
                "stops": [{"place_id": "p1", "attraction_name": "p1", "arrival_offset_min": 0, "dwell_minutes": 60}],
                "weather_note": None,
            },
        ],
    }

    result = check_constraints_node(state)

    assert any("already scheduled" in v for v in result["constraint_violations"])


def test_flags_gap_too_short_for_travel_between_distant_stops():
    # ~55km apart (0.5 degrees of latitude), but only 30 min between them.
    ranked = [_ranked("p1", None, lat=51.5, lng=0.0), _ranked("p2", None, lat=51.0, lng=0.0)]
    state = {
        "trip_setup": {
            "daily_pace": "packed",
            "budget_level": "luxury",
            "day_start_time": dt.time(9, 0),
            "day_end_time": dt.time(18, 0),
        },
        "ranked_attractions": ranked,
        "itinerary": [
            {
                "date": DAY,
                "stops": [
                    {"place_id": "p1", "attraction_name": "p1", "arrival_offset_min": 0, "dwell_minutes": 60},
                    {"place_id": "p2", "attraction_name": "p2", "arrival_offset_min": 30, "dwell_minutes": 60},
                ],
                "weather_note": None,
            }
        ],
    }

    result = check_constraints_node(state)

    assert any("min between" in v for v in result["constraint_violations"])


def test_flags_itinerary_running_past_the_day_end_time():
    ranked = [_ranked("p1", None)]
    state = {
        "trip_setup": {
            "daily_pace": "relaxed",
            "budget_level": "luxury",
            "day_start_time": dt.time(9, 0),
            "day_end_time": dt.time(10, 0),  # only a 1-hour window
        },
        "ranked_attractions": ranked,
        "itinerary": [
            {
                "date": DAY,
                # Starts at the very end of the window -- the 60-minute dwell
                # time alone pushes past day_end_time.
                "stops": [{"place_id": "p1", "attraction_name": "p1", "arrival_offset_min": 30, "dwell_minutes": 60}],
                "weather_note": None,
            }
        ],
    }

    result = check_constraints_node(state)

    assert any("beyond the" in v for v in result["constraint_violations"])


def test_flags_a_day_that_wraps_up_far_too_early_with_room_to_add():
    # moderate day, 3 short stops finishing by ~11:30 in a 540-min window,
    # 2 unused candidates -> "add more", not a gap-padding violation.
    ranked = [_ranked(f"p{i}", None) for i in range(5)]
    state = {
        "trip_setup": {
            "daily_pace": "moderate",
            "budget_level": "luxury",
            "day_start_time": dt.time(9, 0),
            "day_end_time": dt.time(18, 0),
        },
        "ranked_attractions": ranked,
        "itinerary": [
            {
                "date": DAY,
                "stops": [
                    {"place_id": "p0", "attraction_name": "p0", "arrival_offset_min": 0, "dwell_minutes": 50},
                    {"place_id": "p1", "attraction_name": "p1", "arrival_offset_min": 60, "dwell_minutes": 50},
                    {"place_id": "p2", "attraction_name": "p2", "arrival_offset_min": 120, "dwell_minutes": 50},
                ],
                "weather_note": None,
            }
        ],
    }

    result = check_constraints_node(state)

    assert any("half the window empty" in v or "add" in v for v in result["constraint_violations"])


def test_does_not_flag_an_underfilled_day_when_candidates_are_exhausted():
    ranked = [_ranked("p1", None), _ranked("p2", None), _ranked("p3", None)]
    state = {
        "trip_setup": {
            "daily_pace": "moderate",
            "budget_level": "luxury",
            "day_start_time": dt.time(9, 0),
            "day_end_time": dt.time(18, 0),
        },
        "ranked_attractions": ranked,
        "itinerary": [
            {
                "date": DAY,
                "stops": [
                    {"place_id": "p1", "attraction_name": "p1", "arrival_offset_min": 0, "dwell_minutes": 50},
                    {"place_id": "p2", "attraction_name": "p2", "arrival_offset_min": 60, "dwell_minutes": 50},
                    {"place_id": "p3", "attraction_name": "p3", "arrival_offset_min": 120, "dwell_minutes": 50},
                ],
                "weather_note": None,
            }
        ],
    }

    assert check_constraints_node(state)["constraint_violations"] == []


def test_ignores_a_meal_break_stop_in_the_checks():
    ranked = [_ranked("p1", None), _ranked("p2", None), _ranked("p3", None)]
    state = {
        "trip_setup": {
            "daily_pace": "moderate",
            "budget_level": "luxury",
            "day_start_time": dt.time(9, 0),
            "day_end_time": dt.time(18, 0),
        },
        "ranked_attractions": ranked,
        "itinerary": [
            {
                "date": DAY,
                "stops": [
                    {"place_id": "p1", "attraction_name": "p1", "kind": "attraction", "arrival_offset_min": 0, "dwell_minutes": 180},
                    {"place_id": "__meal_lunch__", "attraction_name": "Lunch", "kind": "meal", "arrival_offset_min": 180, "dwell_minutes": 60},
                    {"place_id": "p2", "attraction_name": "p2", "kind": "attraction", "arrival_offset_min": 240, "dwell_minutes": 180},
                    {"place_id": "p3", "attraction_name": "p3", "kind": "attraction", "arrival_offset_min": 430, "dwell_minutes": 90},
                ],
                "weather_note": None,
            }
        ],
    }

    # 3 real stops (in moderate's 3-5), the meal isn't counted or flagged as a duplicate.
    assert check_constraints_node(state)["constraint_violations"] == []


def test_should_refine_when_violations_and_under_limit():
    state = {"constraint_violations": ["oops"], "refine_iterations": 1}
    assert should_refine(state) == "itinerary"


def test_should_refine_finalizes_when_no_violations():
    state = {"constraint_violations": [], "refine_iterations": 0}
    assert should_refine(state) == "finalize"


def test_should_refine_finalizes_after_hitting_the_cap():
    state = {"constraint_violations": ["still bad"], "refine_iterations": 2}
    assert should_refine(state) == "finalize"
