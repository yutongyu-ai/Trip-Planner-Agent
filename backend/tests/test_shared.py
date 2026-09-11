import datetime as dt

from app.graph.nodes._shared import (
    DEFAULT_DWELL_MINUTES,
    category_dwell_minutes,
    clamped_dwell_minutes,
    dwell_bounds,
    estimate_travel_minutes,
    is_open_during,
    minutes_to_clock,
    realistic_transit_minutes,
)
from app.tools.places_client import Attraction

TUESDAY = dt.date(2026, 9, 1)  # weekday() == 1


def _attraction(categories: list[str], opening_hours: dict[int, tuple[int, int]] | None = None) -> Attraction:
    return Attraction(
        place_id="p1",
        name="Some Place",
        address="",
        lat=0.0,
        lng=0.0,
        rating=4.0,
        rating_count=10,
        price_level=None,
        categories=categories,
        description=None,
        opening_hours=opening_hours,
    )


def test_category_dwell_minutes_uses_first_matching_category():
    # "tourist_attraction" is a much weaker/generic signal than "museum" --
    # Google typically lists the specific type first, so the first match
    # should win rather than, say, the last or an average.
    assert category_dwell_minutes(_attraction(["museum", "tourist_attraction"])) == 90  # museum typical
    assert category_dwell_minutes(_attraction(["amusement_park"])) == 300


def test_category_dwell_minutes_falls_back_to_default_for_unknown_category():
    assert category_dwell_minutes(_attraction(["some_unrecognized_type"])) == DEFAULT_DWELL_MINUTES


def test_category_dwell_minutes_falls_back_to_default_for_no_categories():
    assert category_dwell_minutes(_attraction([])) == DEFAULT_DWELL_MINUTES


def test_dwell_bounds_are_the_category_min_and_max():
    assert dwell_bounds(_attraction(["museum"])) == (50, 240)
    assert dwell_bounds(_attraction(["landmark"])) == (10, 60)
    assert dwell_bounds(_attraction(["some_unrecognized_type"])) == (30, 180)


def test_clamped_dwell_minutes_passes_through_an_estimate_inside_the_bracket():
    # museum bracket is [50, 240]; a real "British Museum, 3 hours" survives.
    assert clamped_dwell_minutes(_attraction(["museum"]), 180) == 180


def test_clamped_dwell_minutes_clamps_a_wildly_high_llm_estimate_to_the_category_max():
    assert clamped_dwell_minutes(_attraction(["museum"]), 600) == 240
    assert clamped_dwell_minutes(_attraction(["landmark"]), 600) == 60


def test_clamped_dwell_minutes_clamps_a_wildly_low_llm_estimate_to_the_category_min():
    assert clamped_dwell_minutes(_attraction(["museum"]), 1) == 50  # "if you only have 50 min, skip it"


def test_realistic_transit_minutes_scales_and_adds_access_cost():
    # 9 raw driving min (roughly London Zoo -> British Museum on OSRM) should
    # come out around 20 door-to-door, not 9.
    assert realistic_transit_minutes(9) == 9 * 1.6 + 8
    assert 20 <= realistic_transit_minutes(9) <= 25


def test_realistic_transit_minutes_leaves_co_located_pairs_alone():
    assert realistic_transit_minutes(0) == 0
    assert realistic_transit_minutes(1.5) == 1.5


def test_estimate_travel_minutes_returns_raw_free_flow_driving_time():
    from app.tools.places_client import Attraction

    def _at(lat, lng):
        return Attraction("p", "P", "", lat, lng, 4.0, 10, None, [], None)

    # ~1 degree of latitude ~= 111 km; * 1.3 circuity / 28 kmh -> a big number.
    raw = estimate_travel_minutes(_at(51.5, 0.0), _at(52.5, 0.0))
    assert 250 < raw < 400  # raw driving, not yet door-to-door


def test_minutes_to_clock_formats_a_normal_time():
    assert minutes_to_clock(9 * 60) == "09:00"
    assert minutes_to_clock(17 * 60 + 30) == "17:30"


def test_minutes_to_clock_wraps_past_midnight():
    # 26:00 (an overnight close_minute) wraps to 02:00.
    assert minutes_to_clock(26 * 60) == "02:00"


def test_is_open_during_true_when_no_hours_data_at_all():
    assert is_open_during(_attraction(["museum"], opening_hours=None), TUESDAY, 540, 600) is True


def test_is_open_during_true_when_weekday_missing_from_hours():
    # Data exists for other days, just not this one -- "unknown", not "closed".
    assert is_open_during(_attraction(["museum"], opening_hours={0: (540, 1080)}), TUESDAY, 540, 600) is True


def test_is_open_during_true_when_fully_within_hours():
    assert is_open_during(_attraction(["museum"], opening_hours={1: (540, 1080)}), TUESDAY, 600, 660) is True


def test_is_open_during_false_when_arrival_before_opening():
    assert is_open_during(_attraction(["museum"], opening_hours={1: (600, 1080)}), TUESDAY, 540, 660) is False


def test_is_open_during_false_when_departure_after_closing():
    assert is_open_during(_attraction(["museum"], opening_hours={1: (540, 660)}), TUESDAY, 600, 720) is False
