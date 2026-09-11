"""Direct unit tests for app/i18n.py's bilingual templates.

Several of these are already exercised indirectly through a handful of
node-level tests (test_graph_constraints.py, test_graph_finalize.py,
test_graph_itinerary.py) via a real caller with lang="zh". This file is the
systematic sweep: every public function in app/i18n.py, both languages,
so a broken/missing zh branch can't slip in unnoticed the way it did before
this file existed.
"""

from app.i18n import (
    localize_error,
    openweather_lang,
    places_language_code,
    refine_limit_reached,
    trip_planned,
    violation_beyond_window,
    violation_budget,
    violation_closed,
    violation_duplicate_place,
    violation_gap_too_short,
    violation_outside_hours,
    violation_stop_count,
    weather_beyond_horizon,
    weather_lookup_failed,
    weather_note_bad_adjusted,
    weather_note_bad_unadjusted,
    weather_note_ok,
    weekday_name,
)


def test_places_language_code():
    assert places_language_code("en") == "en"
    assert places_language_code("zh") == "zh-CN"


def test_openweather_lang():
    assert openweather_lang("en") == "en"
    assert openweather_lang("zh") == "zh_cn"


def test_weekday_name():
    assert weekday_name("en", 0) == "Monday"
    assert weekday_name("en", 6) == "Sunday"
    assert weekday_name("zh", 0) == "周一"
    assert weekday_name("zh", 6) == "周日"


def test_trip_planned():
    assert trip_planned("en", "Lisbon", 3, 5) == "Lisbon trip, 3 day(s), 5 stop(s) planned."
    msg = trip_planned("zh", "Lisbon", 3, 5)
    assert "Lisbon" in msg and "3 天" in msg and "5 个地点" in msg


def test_weather_lookup_failed():
    for lang in ("en", "zh"):
        assert "timeout" in weather_lookup_failed(lang, "timeout")


def test_weather_beyond_horizon():
    for lang in ("en", "zh"):
        assert "2" in weather_beyond_horizon(lang, 2)


def test_refine_limit_reached():
    for lang in ("en", "zh"):
        msg = refine_limit_reached(lang, 2, ["v1", "v2"])
        assert "v1" in msg and "v2" in msg


def test_violation_stop_count():
    for lang in ("en", "zh"):
        msg = violation_stop_count(lang, "2026-09-01", 5, "relaxed", 1, 2)
        assert "2026-09-01" in msg


def test_violation_budget():
    for lang in ("en", "zh"):
        msg = violation_budget(lang, "2026-09-01", 2.5, "budget", 1)
        assert "2026-09-01" in msg


def test_violation_closed():
    assert "Saturday" in violation_closed("en", "2026-09-01", "Museum X", 5)
    msg_zh = violation_closed("zh", "2026-09-01", "Museum X", 5)
    assert "Museum X" in msg_zh and "周六" in msg_zh


def test_violation_outside_hours():
    for lang in ("en", "zh"):
        msg = violation_outside_hours(lang, "2026-09-01", "Museum X", "16:30", "17:45", "09:00", "17:00")
        assert "Museum X" in msg and "16:30" in msg and "17:45" in msg and "09:00" in msg and "17:00" in msg


def test_violation_duplicate_place():
    for lang in ("en", "zh"):
        msg = violation_duplicate_place(lang, "2026-09-02", "Museum X", "2026-09-01")
        assert "Museum X" in msg and "2026-09-01" in msg and "2026-09-02" in msg


def test_violation_gap_too_short():
    for lang in ("en", "zh"):
        msg = violation_gap_too_short(lang, "2026-09-01", 10, "A", "B", 45.0)
        assert "A" in msg and "B" in msg


def test_violation_beyond_window():
    for lang in ("en", "zh"):
        msg = violation_beyond_window(lang, "2026-09-01", "A", 600, 540, "09:00", "18:00")
        assert "A" in msg


def test_weather_note_ok():
    assert weather_note_ok("en", "clear sky", 20.0) == "clear sky, around 20C"
    assert weather_note_ok("zh", "晴", 20.0) == "晴，约 20°C"


def test_weather_note_bad_adjusted():
    for lang in ("en", "zh"):
        assert "light rain" in weather_note_bad_adjusted(lang, "light rain")


def test_weather_note_bad_unadjusted():
    for lang in ("en", "zh"):
        assert "light rain" in weather_note_bad_unadjusted(lang, "light rain")


def test_localize_error_replaces_the_tag_and_keeps_the_raw_detail():
    assert localize_error("en", "weather: 404 Not Found") == "Weather lookup failed: 404 Not Found"
    assert localize_error("zh", "attractions: 429 Too Many Requests") == "景点搜索失败: 429 Too Many Requests"


def test_localize_error_covers_every_tag_a_node_can_emit():
    # One tag per node that appends to state["errors"]: attractions.py,
    # weather.py, route.py, itinerary.py.
    for tag in ("attractions", "weather", "route", "itinerary"):
        for lang in ("en", "zh"):
            result = localize_error(lang, f"{tag}: boom")
            assert result != f"{tag}: boom"
            assert "boom" in result


def test_localize_error_passes_through_an_unrecognized_tag_unchanged():
    assert localize_error("en", "totally-unknown: boom") == "totally-unknown: boom"


def test_localize_error_handles_a_tag_with_no_detail():
    assert localize_error("en", "weather") == "Weather lookup failed"
