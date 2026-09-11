"""Unit tests for the deterministic day scheduler.

The headline case (`test_london_...`) is the real failure this module
exists to fix: an LLM-produced London day that, after two refine passes,
still scheduled the London Eye at 09:00 (opens 11:00), London Zoo at 09:52
(opens 10:00), and left impossibly short gaps between far-apart stops.
"""

from __future__ import annotations

from app.graph.scheduling import LUNCH_PLACE_ID, ScheduleInput, schedule_day, schedule_day_with_lunch

NINE_AM = 9 * 60
SIX_PM = 18 * 60


def _travel(distances: dict[tuple[str, str], int]) -> "callable":
    def fn(a: str, b: str) -> float:
        if a == b:
            return 0
        return distances.get((a, b), distances.get((b, a), 0))

    return fn


def _feasible(day, inputs: list[ScheduleInput], *, day_start=NINE_AM, day_end=SIX_PM) -> None:
    """Assert every scheduled stop lands inside its opening window and the
    day fits [day_start, day_end]."""
    by_id = {s.place_id: s for s in inputs}
    for stop in day.stops:
        src = by_id[stop.place_id]
        arrival_clock = day_start + stop.arrival_offset_min
        depart_clock = day_start + stop.depart_offset_min
        assert depart_clock - arrival_clock == src.dwell_minutes
        if src.open_min is not None:
            assert arrival_clock >= src.open_min, f"{stop.name} arrives before it opens"
        if src.close_min is not None:
            assert depart_clock <= src.close_min, f"{stop.name} leaves after it closes"
        assert day_start <= arrival_clock
        assert depart_clock <= day_end
    # chronological, and each arrival leaves room for the recorded travel
    for prev, nxt in zip(day.stops, day.stops[1:]):
        assert nxt.arrival_offset_min >= prev.depart_offset_min + nxt.travel_from_prev_min


def test_london_scenario_respects_opening_hours_and_fits_all_four():
    stops = [
        ScheduleInput("kg", "Kensington Gardens", dwell_minutes=90, open_min=None, close_min=None, rank=0),
        ScheduleInput("eye", "London Eye", dwell_minutes=45, open_min=11 * 60, close_min=SIX_PM, rank=1),
        ScheduleInput("sea", "SEA LIFE London Aquarium", dwell_minutes=90, open_min=10 * 60, close_min=SIX_PM, rank=2),
        ScheduleInput("zoo", "London Zoo", dwell_minutes=120, open_min=10 * 60, close_min=17 * 60, rank=3),
    ]
    travel = _travel(
        {
            ("kg", "eye"): 18,
            ("kg", "sea"): 18,
            ("kg", "zoo"): 22,
            ("eye", "sea"): 2,  # same building (County Hall)
            ("eye", "zoo"): 20,
            ("sea", "zoo"): 20,
        }
    )

    day = schedule_day(stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=travel)

    assert day.dropped == []
    assert len(day.stops) == 4
    _feasible(day, stops)
    # The specific bugs from the real output must not recur.
    eye = next(s for s in day.stops if s.place_id == "eye")
    assert NINE_AM + eye.arrival_offset_min >= 11 * 60


def test_first_stop_starts_at_day_start_when_already_open():
    stops = [
        ScheduleInput("a", "A", dwell_minutes=60, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=60, open_min=None, close_min=None, rank=1),
    ]
    day = schedule_day(stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({("a", "b"): 15}))
    assert day.stops[0].arrival_offset_min == 0
    assert day.stops[1].travel_from_prev_min == 15


def test_reorders_to_minimise_travel_against_rank_order():
    # Rank order a,b,c would zig-zag; b,a,c (or c,a,b) is shorter.
    stops = [
        ScheduleInput("a", "A", dwell_minutes=30, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=30, open_min=None, close_min=None, rank=1),
        ScheduleInput("c", "C", dwell_minutes=30, open_min=None, close_min=None, rank=2),
    ]
    travel = _travel({("a", "b"): 5, ("a", "c"): 5, ("b", "c"): 60})
    day = schedule_day(stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=travel)
    visit_order = [s.place_id for s in day.stops]
    assert visit_order in (["b", "a", "c"], ["c", "a", "b"])
    assert sum(s.travel_from_prev_min for s in day.stops) == 10


def test_same_building_stops_scheduled_back_to_back():
    stops = [
        ScheduleInput("eye", "London Eye", dwell_minutes=45, open_min=None, close_min=None, rank=0),
        ScheduleInput("sea", "SEA LIFE", dwell_minutes=90, open_min=None, close_min=None, rank=1),
    ]
    day = schedule_day(stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({("eye", "sea"): 0}))
    assert day.stops[1].arrival_offset_min == day.stops[0].depart_offset_min
    assert day.stops[1].travel_from_prev_min == 0


def test_drops_lowest_ranked_stop_when_day_is_overbooked():
    # Four 180-min stops can't fit a 9-hour day; the rank-3 one is dropped.
    stops = [
        ScheduleInput(f"p{i}", f"P{i}", dwell_minutes=180, open_min=None, close_min=None, rank=i) for i in range(4)
    ]
    travel = _travel({})  # zero travel, so only dwell time constrains
    day = schedule_day(stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=travel)
    assert day.dropped == ["P3"]
    assert {s.place_id for s in day.stops} == {"p0", "p1", "p2"}
    _feasible(day, stops)


def test_drops_multiple_stops_when_needed():
    # 250-min stops: two fit a 540-min window, a third doesn't.
    stops = [
        ScheduleInput(f"p{i}", f"P{i}", dwell_minutes=250, open_min=None, close_min=None, rank=i) for i in range(4)
    ]
    day = schedule_day(stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({}))
    assert day.dropped == ["P3", "P2"]
    assert len(day.stops) == 2


def test_tight_closing_time_forces_a_drop_even_though_total_time_would_fit():
    # p_early closes at noon; combined with p_big (needs the whole afternoon)
    # and travel between them, no ordering is feasible -- something gives.
    stops = [
        ScheduleInput("early", "Early", dwell_minutes=120, open_min=NINE_AM, close_min=12 * 60, rank=0),
        ScheduleInput("big", "Big", dwell_minutes=300, open_min=NINE_AM, close_min=SIX_PM, rank=1),
        ScheduleInput("mid", "Mid", dwell_minutes=120, open_min=NINE_AM, close_min=SIX_PM, rank=2),
    ]
    travel = _travel({("early", "big"): 60, ("early", "mid"): 60, ("big", "mid"): 60})
    day = schedule_day(stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=travel)
    assert day.dropped  # at least one dropped
    _feasible(day, stops)


def test_drops_only_the_one_stop_that_can_never_fit_not_everyone_else():
    # rank0 closes 1 minute after day start -- no order can ever place it.
    # rank1-3 are otherwise perfectly schedulable together. A naive
    # "always evict the lowest-priority stop" rule would strip rank3, rank2,
    # then rank1 (all needlessly) before ever reaching the true culprit,
    # leaving an empty day. Only rank0 should go.
    stops = [
        ScheduleInput("rank0", "R0", dwell_minutes=60, open_min=None, close_min=NINE_AM + 1, rank=0),
        ScheduleInput("rank1", "R1", dwell_minutes=60, open_min=None, close_min=None, rank=1),
        ScheduleInput("rank2", "R2", dwell_minutes=60, open_min=None, close_min=None, rank=2),
        ScheduleInput("rank3", "R3", dwell_minutes=60, open_min=None, close_min=None, rank=3),
    ]
    day = schedule_day(stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({}))
    assert day.dropped == ["R0"]
    assert {s.place_id for s in day.stops} == {"rank1", "rank2", "rank3"}


def test_unknown_hours_are_treated_as_open_all_day():
    stops = [
        ScheduleInput("a", "A", dwell_minutes=120, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=120, open_min=None, close_min=None, rank=1),
    ]
    day = schedule_day(stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({("a", "b"): 30}))
    assert len(day.stops) == 2
    assert day.dropped == []


def test_schedules_tight_and_does_not_pad_the_day():
    # Two 60-min stops + 30 travel: the day should end at 11:30 (150 min in),
    # not be stretched toward any fill target.
    stops = [
        ScheduleInput("a", "A", dwell_minutes=60, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=60, open_min=None, close_min=None, rank=1),
    ]
    day = schedule_day(
        stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({("a", "b"): 30})
    )
    assert day.stops[0].arrival_offset_min == 0
    assert day.stops[1].arrival_offset_min == 90  # 60 dwell + 30 travel, no buffer
    assert day.stops[-1].depart_offset_min == 150
    assert all(s.wait_for_opening_min == 0 for s in day.stops)


def test_reports_wait_time_when_arriving_before_a_stop_opens():
    # A finishes at 10:00 (60 min in), B is 15 min away but doesn't open
    # until 11:00 -> 45 min waiting at the door.
    stops = [
        ScheduleInput("a", "A", dwell_minutes=60, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=60, open_min=11 * 60, close_min=SIX_PM, rank=1),
    ]
    day = schedule_day(
        stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({("a", "b"): 15})
    )
    b = next(s for s in day.stops if s.place_id == "b")
    assert NINE_AM + b.arrival_offset_min == 11 * 60
    assert b.travel_from_prev_min == 15
    assert b.wait_for_opening_min == 45  # arrived 10:15, opened 11:00


def test_empty_input_returns_empty_day():
    day = schedule_day([], day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({}))
    assert day.stops == []
    assert day.dropped == []


def test_single_stop():
    stops = [ScheduleInput("a", "A", dwell_minutes=90, open_min=10 * 60, close_min=SIX_PM, rank=0)]
    day = schedule_day(stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({}))
    assert len(day.stops) == 1
    assert day.stops[0].arrival_offset_min == 60  # waits for the 10:00 open
    assert day.stops[0].travel_from_prev_min == 0


def test_schedule_day_with_lunch_inserts_a_meal_stop_where_the_preceding_stop_ends():
    # A ends exactly at noon (the only gap there is) -- lunch starts there,
    # right after A finishes (you eat where you are), not at some
    # independently-chosen clock time.
    stops = [
        ScheduleInput("a", "A", dwell_minutes=180, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=120, open_min=None, close_min=None, rank=1),
    ]
    day = schedule_day_with_lunch(
        stops,
        day_start_min=NINE_AM,
        day_end_min=SIX_PM,
        travel_minutes=_travel({("a", "b"): 10}),
        lunch_dwell_minutes=60,
        lunch_label="Lunch",
    )
    kinds = [s.kind for s in day.stops]
    assert kinds == ["attraction", "meal", "attraction"]
    meal = next(s for s in day.stops if s.kind == "meal")
    assert meal.place_id == LUNCH_PLACE_ID and meal.name == "Lunch"
    assert NINE_AM + meal.arrival_offset_min == 12 * 60
    assert meal.depart_offset_min - meal.arrival_offset_min == 60  # the caller-supplied dwell, unshrunk


def test_schedule_day_with_lunch_preserves_the_travel_minimising_order():
    # Two geographic clusters: hyde+kensington are neighbours (5 min apart),
    # tower+eye are neighbours (5 min apart), but the two clusters are 40 min
    # apart. The best whole-day route visits each cluster together, not
    # interleaved -- lunch must not be allowed to break that up. (A version
    # of this module that scheduled the day as two independent halves around
    # a fixed lunch slot did exactly that on a real run: it put one park in
    # the morning half and its immediate neighbour in the afternoon half,
    # with a cross-town round trip sandwiched between them -- see the module
    # docstring.)
    stops = [
        ScheduleInput("hyde", "Hyde Park", dwell_minutes=60, open_min=None, close_min=None, rank=0),
        ScheduleInput("tower", "Tower Bridge", dwell_minutes=60, open_min=None, close_min=None, rank=1),
        ScheduleInput("eye", "London Eye", dwell_minutes=60, open_min=None, close_min=None, rank=2),
        ScheduleInput("kensington", "Kensington Gardens", dwell_minutes=60, open_min=None, close_min=None, rank=3),
    ]
    travel = _travel(
        {
            ("hyde", "kensington"): 5,
            ("tower", "eye"): 5,
            ("hyde", "tower"): 40,
            ("hyde", "eye"): 40,
            ("kensington", "tower"): 40,
            ("kensington", "eye"): 40,
        }
    )
    day = schedule_day_with_lunch(
        stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=travel, lunch_dwell_minutes=60, lunch_label="Lunch"
    )
    attraction_order = [s.place_id for s in day.stops if s.kind == "attraction"]
    assert abs(attraction_order.index("hyde") - attraction_order.index("kensington")) == 1, attraction_order
    assert abs(attraction_order.index("tower") - attraction_order.index("eye")) == 1, attraction_order
    # 5 (within one cluster) + 40 (crossing town once) + 5 (within the other
    # cluster) -- not 90, which is what crossing town twice (a zig-zag) costs.
    assert sum(s.travel_from_prev_min for s in day.stops) == 50


def test_schedule_day_with_lunch_tries_the_next_gap_when_the_nearest_one_is_infeasible():
    # A runs 9:00-13:00, so the gap right after it lands exactly on the
    # lunch target -- but B, next in line, closes at 13:35, and even a
    # floor-length lunch there would push B's departure past that. The gap
    # after B is farther from the target but doesn't touch B at all, since
    # B is already finished by the time lunch would land there.
    stops = [
        ScheduleInput("a", "A", dwell_minutes=240, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=30, open_min=None, close_min=13 * 60 + 35, rank=1),
        ScheduleInput("c", "C", dwell_minutes=60, open_min=None, close_min=None, rank=2),
    ]
    day = schedule_day_with_lunch(
        stops, day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({}), lunch_dwell_minutes=60, lunch_label="Lunch"
    )
    kinds = [s.kind for s in day.stops]
    assert kinds == ["attraction", "attraction", "meal", "attraction"]  # lunch lands after B, not after A
    meal = next(s for s in day.stops if s.kind == "meal")
    assert meal.depart_offset_min - meal.arrival_offset_min == 60  # the nearer gap failing didn't force a shrink here
    b = next(s for s in day.stops if s.place_id == "b")
    assert NINE_AM + b.depart_offset_min == 13 * 60 + 30  # B's own timing is untouched


def test_schedule_day_with_lunch_extends_the_previous_attraction_to_absorb_a_wait():
    # A (45 min) finishes well before B opens; instead of standing at B's
    # door, the traveller lingers at A instead -- same total elapsed time,
    # shown as a longer visit to A rather than a wait outside B. (This day
    # ends exactly at noon, so lunch never enters the picture here -- see
    # the next test for the same idea folding into lunch instead.)
    stops = [
        ScheduleInput("a", "A", dwell_minutes=45, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=60, open_min=11 * 60, close_min=SIX_PM, rank=1),
    ]
    day = schedule_day_with_lunch(
        stops,
        day_start_min=NINE_AM,
        day_end_min=SIX_PM,
        travel_minutes=_travel({("a", "b"): 35}),
        lunch_dwell_minutes=60,
        lunch_label="Lunch",
    )
    a = next(s for s in day.stops if s.place_id == "a")
    b = next(s for s in day.stops if s.place_id == "b")
    assert all(s.kind == "attraction" for s in day.stops)  # confirms lunch didn't get involved
    assert NINE_AM + a.depart_offset_min == 10 * 60 + 25  # extended from 9:45 to 10:25
    assert NINE_AM + b.arrival_offset_min == 11 * 60  # still arrives right at opening
    assert b.wait_for_opening_min == 0
    assert b.travel_from_prev_min == 35  # travel itself is unchanged


def test_schedule_day_with_lunch_keeps_the_wait_when_extending_would_close_the_previous_stop_late():
    # A must leave by 9:50 (a tight close time) -- extending it to cover B's
    # 40-min wait would run A well past that, so the wait stays a wait.
    stops = [
        ScheduleInput("a", "A", dwell_minutes=45, open_min=None, close_min=9 * 60 + 50, rank=0),
        ScheduleInput("b", "B", dwell_minutes=60, open_min=11 * 60, close_min=SIX_PM, rank=1),
    ]
    day = schedule_day_with_lunch(
        stops,
        day_start_min=NINE_AM,
        day_end_min=SIX_PM,
        travel_minutes=_travel({("a", "b"): 35}),
        lunch_dwell_minutes=60,
        lunch_label="Lunch",
    )
    a = next(s for s in day.stops if s.place_id == "a")
    b = next(s for s in day.stops if s.place_id == "b")
    assert NINE_AM + a.depart_offset_min == 9 * 60 + 45  # unchanged -- extending would blow its own close time
    assert b.wait_for_opening_min == 40


def test_schedule_day_with_lunch_extends_to_absorb_a_residual_opening_wait():
    # A ends 12:00, B is 30 min away but doesn't open until 14:00 -- a 90-min
    # gap. A full 60-min lunch only closes 60 of those 90 minutes; the
    # remaining 30 would show up as B's own wait_for_opening_min right after
    # insertion, but the final absorb-waits pass folds that residual into a
    # longer lunch instead of leaving the traveller standing at B's door.
    stops = [
        ScheduleInput("a", "A", dwell_minutes=180, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=120, open_min=14 * 60, close_min=None, rank=1),
    ]
    day = schedule_day_with_lunch(
        stops,
        day_start_min=NINE_AM,
        day_end_min=SIX_PM,
        travel_minutes=_travel({("a", "b"): 30}),
        lunch_dwell_minutes=60,
        lunch_label="Lunch",
    )
    kinds = [s.kind for s in day.stops]
    assert kinds == ["attraction", "meal", "attraction"]
    meal = next(s for s in day.stops if s.kind == "meal")
    b = next(s for s in day.stops if s.place_id == "b")
    assert meal.depart_offset_min - meal.arrival_offset_min == 90  # 60 base + the 30-min residual folded in
    assert NINE_AM + b.arrival_offset_min == 14 * 60  # B still opens right at 14:00
    assert b.wait_for_opening_min == 0  # no idle waiting at the door
    assert NINE_AM + b.depart_offset_min == 16 * 60  # unchanged: 120-min dwell from 14:00


def test_schedule_day_with_lunch_shrinks_toward_the_floor_when_a_close_time_forces_it():
    # A ends 12:00; B is 20 min away with no opening-hours wait, but its
    # 14:10 close time only leaves room for a floor-length lunch, not a full
    # 60 minutes.
    stops = [
        ScheduleInput("a", "A", dwell_minutes=180, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=90, open_min=None, close_min=14 * 60 + 10, rank=1),
    ]
    day = schedule_day_with_lunch(
        stops,
        day_start_min=NINE_AM,
        day_end_min=SIX_PM,
        travel_minutes=_travel({("a", "b"): 20}),
        lunch_dwell_minutes=60,
        lunch_label="Lunch",
    )
    kinds = [s.kind for s in day.stops]
    assert kinds == ["attraction", "meal", "attraction"]
    meal = next(s for s in day.stops if s.kind == "meal")
    b = next(s for s in day.stops if s.place_id == "b")
    assert meal.depart_offset_min - meal.arrival_offset_min == 20  # shrunk to the floor
    assert NINE_AM + b.depart_offset_min == 14 * 60 + 10  # exactly at its close time


def test_schedule_day_with_lunch_is_a_noop_when_the_day_does_not_span_lunchtime():
    stops = [
        ScheduleInput("a", "A", dwell_minutes=60, open_min=None, close_min=None, rank=0),
        ScheduleInput("b", "B", dwell_minutes=60, open_min=None, close_min=None, rank=1),
    ]
    day = schedule_day_with_lunch(
        stops,
        day_start_min=15 * 60,
        day_end_min=20 * 60,
        travel_minutes=_travel({("a", "b"): 10}),
        lunch_dwell_minutes=60,
        lunch_label="Lunch",
    )
    assert all(s.kind == "attraction" for s in day.stops)


def test_schedule_day_with_lunch_noop_with_fewer_than_two_stops():
    single = schedule_day_with_lunch(
        [ScheduleInput("a", "A", dwell_minutes=400, open_min=None, close_min=None, rank=0)],
        day_start_min=NINE_AM,
        day_end_min=SIX_PM,
        travel_minutes=_travel({}),
        lunch_dwell_minutes=60,
        lunch_label="Lunch",
    )
    assert all(s.kind == "attraction" for s in single.stops)

    empty = schedule_day_with_lunch(
        [], day_start_min=NINE_AM, day_end_min=SIX_PM, travel_minutes=_travel({}), lunch_dwell_minutes=60, lunch_label="Lunch"
    )
    assert empty.stops == [] and empty.dropped == []
