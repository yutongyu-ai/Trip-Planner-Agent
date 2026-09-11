import datetime as dt

from app.graph.nodes.finalize import finalize_node

DAY = dt.date(2026, 9, 1)


def test_finalize_summarizes_stop_count():
    state = {
        "trip_setup": {"destination": "Lisbon"},
        "itinerary": [{"date": DAY, "stops": [{"place_id": "p1"}], "weather_note": "Clear, around 20C"}],
    }

    result = finalize_node(state)

    assert "Lisbon" in result["summary"]
    assert "1 day(s), 1 stop(s)" in result["summary"]
    assert result["notes"] == []


def test_finalize_notes_missing_forecast_days():
    state = {
        "trip_setup": {"destination": "Lisbon"},
        "itinerary": [
            {"date": DAY, "stops": [], "weather_note": "Clear, around 20C"},
            {"date": DAY + dt.timedelta(days=6), "stops": [], "weather_note": None},
        ],
    }

    result = finalize_node(state)

    assert any("5-day horizon" in note for note in result["notes"])


def test_finalize_notes_low_coverage_destination():
    state = {
        "trip_setup": {"destination": "Shanghai"},
        "itinerary": [{"date": DAY, "stops": [], "weather_note": "Clear, around 20C"}],
        "low_coverage_destination": True,
    }

    result = finalize_node(state)

    assert any("mainland-China" in note for note in result["notes"])


def test_finalize_reports_unresolved_violations():
    state = {
        "trip_setup": {"destination": "Lisbon"},
        "itinerary": [{"date": DAY, "stops": [], "weather_note": None}],
        "constraint_violations": ["too packed"],
        "refine_iterations": 2,
    }

    result = finalize_node(state)

    joined = " ".join(result["notes"])
    assert "Reached the refine limit" in joined
    assert "too packed" in joined


def test_finalize_notes_stops_the_scheduler_dropped():
    state = {
        "trip_setup": {"destination": "London"},
        "itinerary": [
            {"date": DAY, "stops": [{"place_id": "p1"}], "weather_note": None, "dropped_stops": ["London Zoo"]},
        ],
    }

    result = finalize_node(state)

    joined = " ".join(result["notes"])
    assert "London Zoo" in joined
    assert "couldn't be fitted" in joined


def test_finalize_summarizes_stop_count_in_chinese():
    state = {
        "trip_setup": {"destination": "Wuhan", "lang": "zh"},
        "itinerary": [{"date": DAY, "stops": [{"place_id": "p1"}], "weather_note": "晴天，约 20°C"}],
    }

    result = finalize_node(state)

    assert "Wuhan" in result["summary"]
    assert "共 1 天" in result["summary"]
    assert "1 个地点" in result["summary"]


def test_finalize_reports_weather_lookup_failure_in_chinese():
    state = {
        "trip_setup": {"destination": "Wuhan", "lang": "zh"},
        "itinerary": [{"date": DAY, "stops": [], "weather_note": None}],
        "errors": ["weather: Weather API request failed: 404"],
    }

    result = finalize_node(state)

    joined = " ".join(result["notes"])
    assert "天气查询失败" in joined
    assert "404" in joined


def test_finalize_notes_missing_forecast_days_in_chinese():
    state = {
        "trip_setup": {"destination": "Wuhan", "lang": "zh"},
        "itinerary": [
            {"date": DAY, "stops": [], "weather_note": "晴天，约 20°C"},
            {"date": DAY + dt.timedelta(days=6), "stops": [], "weather_note": None},
        ],
    }

    result = finalize_node(state)

    assert any("5 天" in note for note in result["notes"])


def test_finalize_reports_unresolved_violations_in_chinese():
    state = {
        "trip_setup": {"destination": "Wuhan", "lang": "zh"},
        "itinerary": [{"date": DAY, "stops": [], "weather_note": None}],
        "constraint_violations": ["行程太紧凑"],
        "refine_iterations": 2,
    }

    result = finalize_node(state)

    joined = " ".join(result["notes"])
    assert "已达到重排上限" in joined
    assert "行程太紧凑" in joined
