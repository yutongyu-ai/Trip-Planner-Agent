"""Synthetic chat scenarios for the conversational edit agent (see
run_eval.py). Each is built so the correct response genuinely requires more
than one tool call, with the second call's choice depending on what the
first one revealed -- not just phrasing an equivalent single-call request
differently. That's the property this eval exists to demonstrate: the model
is choosing tools and reacting to what it sees, not following a fixed script.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from app.graph.state import AgentState, ItineraryDay, ItineraryStop
from app.recsys.rank import RankedAttraction
from app.tools.places_client import Attraction
from app.tools.weather_client import DailyForecast

START = dt.date.today() + dt.timedelta(days=10)  # beyond the weather horizon; scenarios mock it directly


def _attraction(
    place_id: str,
    name: str,
    categories: list[str],
    *,
    price_level: str | None = None,
    rating: float = 4.3,
    open_weekdays: frozenset[int] | None = None,
) -> Attraction:
    return Attraction(
        place_id=place_id,
        name=name,
        address="",
        lat=0.0,
        lng=0.0,
        rating=rating,
        rating_count=200,
        price_level=price_level,
        categories=categories,
        description=None,
        open_weekdays=open_weekdays,
    )


def _ranked(attraction: Attraction) -> RankedAttraction:
    return RankedAttraction(attraction=attraction, score=1.0, similarity=1.0, normalized_rating=0.8)


def _stop(attraction: Attraction, start_min: int, dwell: int = 90) -> ItineraryStop:
    return ItineraryStop(
        attraction_name=attraction.name,
        place_id=attraction.place_id,
        kind="attraction",
        arrival_offset_min=start_min,
        depart_offset_min=start_min + dwell,
        dwell_minutes=dwell,
        travel_from_prev_min=0,
        wait_for_opening_min=0,
    )


def _trip_setup(lang: str = "en", num_days: int = 1) -> dict:
    return {
        "destination": "Porto",
        "start_date": START,
        "end_date": START + dt.timedelta(days=num_days - 1),
        "interests": ["museums"],
        "daily_pace": "moderate",
        "budget_level": "budget",
        "day_start_time": dt.time(9, 0),
        "day_end_time": dt.time(18, 0),
        "lang": lang,
    }


@dataclass(frozen=True)
class ChatScenario:
    name: str
    state: AgentState
    user_message: str
    # Multi-turn scenarios set this instead: each string is invoked in order against
    # the *same* thread_id/checkpointer, so the model sees its own prior replies and
    # the state mutations earlier turns made -- see run_eval._run. None = single turn
    # (user_message alone).
    user_messages: list[str] | None = None
    # Task success: place_id that must be gone from / present in the final itinerary.
    place_expected_removed: str | None = None
    place_expected_present: str | None = None
    weather_forecast: list[DailyForecast] | None = None  # None = no weather mocking needed
    # Tool Selection Accuracy: the tool names (not args) that must appear, in
    # this relative order, somewhere in the trajectory (extra calls, e.g. an
    # extra list_itinerary, don't fail it -- see run_eval.is_subsequence) --
    # and the tool names that must never appear at all, for scenarios whose
    # whole point is a tool the model should have known *not* to reach for.
    expected_tool_sequence: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)


def _budget_swap_scenario() -> ChatScenario:
    # Day 0 starts at 3 stops -- moderate pace's own floor (STOPS_PER_DAY_RANGE
    # in _shared.py) -- so the fixture is a plausible *complete* day rather
    # than the bare single-stop minimum earlier versions used; a swap doesn't
    # change the stop count either way, but a realistic starting day exercises
    # find_alternative's "already used" filter against more than one id.
    expensive = _attraction("museum-lux", "Grand National Museum", ["museum"], price_level="PRICE_LEVEL_VERY_EXPENSIVE")
    cheap_museum = _attraction("museum-cheap", "Community Art Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    park = _attraction("park1", "Riverside Park", ["park"], price_level="PRICE_LEVEL_FREE")
    cathedral = _attraction("cathedral1", "Old Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    ranked = [_ranked(expensive), _ranked(cheap_museum), _ranked(park), _ranked(cathedral)]
    itinerary = [
        ItineraryDay(
            date=START,
            stops=[_stop(expensive, 0), _stop(park, 120), _stop(cathedral, 240)],
            weather_note=None,
            dropped_stops=[],
        )
    ]
    state: AgentState = {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="budget_swap_finds_cheaper_alternative",
        state=state,
        user_message=(
            "The museum on day 0 (place_id museum-lux) is way over my budget. Find something cheaper "
            "in the same category and swap it in."
        ),
        place_expected_removed="museum-lux",
        place_expected_present="museum-cheap",
        expected_tool_sequence=["find_alternative", "swap_stop"],
    )


def _weather_triggered_swap_scenario() -> ChatScenario:
    outdoor = _attraction("park1", "Old Town Gardens", ["park"], price_level="PRICE_LEVEL_FREE")
    indoor = _attraction("museum1", "City History Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    cathedral = _attraction("cathedral1", "Old Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    gallery = _attraction("gallery1", "Modern Art Gallery", ["art_gallery"], price_level="PRICE_LEVEL_MODERATE")
    ranked = [_ranked(outdoor), _ranked(indoor), _ranked(cathedral), _ranked(gallery)]
    itinerary = [
        ItineraryDay(
            date=START,
            stops=[_stop(outdoor, 0), _stop(cathedral, 120), _stop(gallery, 240)],
            weather_note=None,
            dropped_stops=[],
        )
    ]
    state: AgentState = {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="weather_triggered_indoor_swap",
        state=state,
        user_message=(
            f"Check the weather for {START.isoformat()} (day 0). If it looks bad, swap the outdoor stop "
            "for an indoor alternative."
        ),
        place_expected_removed="park1",
        place_expected_present="museum1",
        weather_forecast=[
            DailyForecast(date=START, avg_temp_c=11.0, condition="Rain", description="heavy rain", is_bad_weather=True)
        ],
        expected_tool_sequence=["check_weather", "find_alternative", "swap_stop"],
    )


def _clear_weather_no_swap_scenario() -> ChatScenario:
    """Mirror of weather_triggered_indoor_swap with a clear forecast instead
    of a bad one -- the correct behaviour is to check the weather and then
    do *nothing else*. A model that finds-and-swaps regardless of the
    forecast would still pass every existing metric (valid tool calls, and
    task_success has nothing to check since no edit was requested) -- this
    is the scenario `disallowed_tools` exists for."""
    outdoor = _attraction("park1", "Old Town Gardens", ["park"], price_level="PRICE_LEVEL_FREE")
    indoor = _attraction("museum1", "City History Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    cathedral = _attraction("cathedral1", "Old Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    gallery = _attraction("gallery1", "Modern Art Gallery", ["art_gallery"], price_level="PRICE_LEVEL_MODERATE")
    ranked = [_ranked(outdoor), _ranked(indoor), _ranked(cathedral), _ranked(gallery)]
    itinerary = [
        ItineraryDay(
            date=START,
            stops=[_stop(outdoor, 0), _stop(cathedral, 120), _stop(gallery, 240)],
            weather_note=None,
            dropped_stops=[],
        )
    ]
    state: AgentState = {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="clear_weather_no_swap_needed",
        state=state,
        user_message=(
            f"Check the weather for {START.isoformat()} (day 0). If it looks bad, swap the outdoor stop "
            "for an indoor alternative."
        ),
        place_expected_present="park1",  # must NOT have been swapped out
        weather_forecast=[
            DailyForecast(date=START, avg_temp_c=22.0, condition="Clear", description="clear sky, sunny", is_bad_weather=False)
        ],
        expected_tool_sequence=["check_weather"],
        disallowed_tools=["find_alternative", "swap_stop"],
    )


def _direct_remove_scenario() -> ChatScenario:
    """The traveller names the stop to drop and explicitly says no
    replacement is wanted -- find_alternative has nothing to contribute
    here, unlike the swap scenarios where it's the only way to discover a
    valid alternative place_id."""
    # 4 stops before the edit, one above moderate pace's floor of 3 -- so
    # removing park1 leaves exactly 3, still a complete day by that floor,
    # keeping this scenario's scope purely about "did it remove the right
    # one" rather than also raising a pace question.
    museum = _attraction("museum1", "City Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    park = _attraction("park1", "Riverside Park", ["park"], price_level="PRICE_LEVEL_FREE")
    cathedral = _attraction("cathedral1", "Old Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    viewpoint = _attraction("viewpoint1", "Hilltop Viewpoint", ["tourist_attraction"], price_level="PRICE_LEVEL_FREE")
    ranked = [_ranked(museum), _ranked(park), _ranked(cathedral), _ranked(viewpoint)]
    itinerary = [
        ItineraryDay(
            date=START,
            stops=[_stop(museum, 0), _stop(park, 120), _stop(cathedral, 240), _stop(viewpoint, 360)],
            weather_note=None,
            dropped_stops=[],
        )
    ]
    state: AgentState = {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="direct_remove_no_search_needed",
        state=state,
        user_message=(
            "Please remove the park stop (place_id park1) from day 0 -- I don't want to visit it, "
            "no replacement needed."
        ),
        place_expected_removed="park1",
        expected_tool_sequence=["remove_stop"],
        disallowed_tools=["find_alternative"],
    )


def _add_stop_scenario() -> ChatScenario:
    """add_stop is the one tool the other scenarios never exercise -- this
    covers it with the simplest case that genuinely needs find_alternative
    first: the traveller doesn't name a place_id, so the only way to get a
    valid one is to search the candidate pool."""
    museum = _attraction("museum1", "City History Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    park = _attraction("park1", "Riverside Park", ["park"], price_level="PRICE_LEVEL_FREE")
    cathedral = _attraction("cathedral1", "Old Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    extra_museum = _attraction("museum-extra", "Design Museum", ["museum"], price_level="PRICE_LEVEL_MODERATE")
    ranked = [_ranked(museum), _ranked(park), _ranked(cathedral), _ranked(extra_museum)]
    itinerary = [
        ItineraryDay(
            date=START,
            stops=[_stop(museum, 0), _stop(park, 120), _stop(cathedral, 240)],
            weather_note=None,
            dropped_stops=[],
        )
    ]
    state: AgentState = {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="add_stop_fills_free_time",
        state=state,
        user_message=(
            "Day 0 has some free time left. Can you add another museum that matches my interests?"
        ),
        place_expected_present="museum-extra",
        expected_tool_sequence=["find_alternative", "add_stop"],
    )


def _multi_day_edit_scenario() -> ChatScenario:
    """Every other scenario edits day 0 -- this is a 2-day trip where the
    edit targets day 1, so a model that defaults to "day 0" out of habit
    (rather than actually reading the day_offset the traveller specified)
    shows up as a guardrail rejection (swap_stop on day 0 would find
    museum-lux-d1 isn't scheduled there) or a wrong final place_id, not as a
    silent pass."""
    day0_park = _attraction("park-d0", "North Park", ["park"], price_level="PRICE_LEVEL_FREE")
    day0_cathedral = _attraction("cathedral-d0", "North Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    day0_viewpoint = _attraction("viewpoint-d0", "North Viewpoint", ["tourist_attraction"], price_level="PRICE_LEVEL_FREE")
    expensive = _attraction("museum-lux-d1", "Grand National Museum", ["museum"], price_level="PRICE_LEVEL_VERY_EXPENSIVE")
    day1_park = _attraction("park-d1", "South Park", ["park"], price_level="PRICE_LEVEL_FREE")
    day1_cathedral = _attraction("cathedral-d1", "South Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    cheap_museum = _attraction("museum-cheap-d1", "Community Art Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    ranked = [
        _ranked(day0_park), _ranked(day0_cathedral), _ranked(day0_viewpoint),
        _ranked(expensive), _ranked(day1_park), _ranked(day1_cathedral), _ranked(cheap_museum),
    ]
    itinerary = [
        ItineraryDay(
            date=START,
            stops=[_stop(day0_park, 0), _stop(day0_cathedral, 120), _stop(day0_viewpoint, 240)],
            weather_note=None,
            dropped_stops=[],
        ),
        ItineraryDay(
            date=START + dt.timedelta(days=1),
            stops=[_stop(expensive, 0), _stop(day1_park, 120), _stop(day1_cathedral, 240)],
            weather_note=None,
            dropped_stops=[],
        ),
    ]
    state: AgentState = {
        "trip_setup": _trip_setup(num_days=2),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="multi_day_edit_targets_correct_day",
        state=state,
        user_message=(
            "The museum on day 1 (not day 0, place_id museum-lux-d1) is too expensive. Find something "
            "cheaper in the same category and swap it in on day 1."
        ),
        place_expected_removed="museum-lux-d1",
        place_expected_present="museum-cheap-d1",
        expected_tool_sequence=["find_alternative", "swap_stop"],
    )


def _info_query_no_edit_scenario() -> ChatScenario:
    """None of the other scenarios ever test a message that isn't an edit
    request at all -- every disallowed_tools case so far still asks for a
    conditional *edit*. A plain question ("what's on day 0?") should produce
    exactly one read-only call and zero itinerary mutations; any edit tool
    firing here would be the model treating curiosity as a change request."""
    museum = _attraction("museum1", "City Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    park = _attraction("park1", "Riverside Park", ["park"], price_level="PRICE_LEVEL_FREE")
    cathedral = _attraction("cathedral1", "Old Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    viewpoint = _attraction("viewpoint1", "Hilltop Viewpoint", ["tourist_attraction"], price_level="PRICE_LEVEL_FREE")
    ranked = [_ranked(museum), _ranked(park), _ranked(cathedral), _ranked(viewpoint)]
    itinerary = [
        ItineraryDay(
            date=START,
            stops=[_stop(museum, 0), _stop(park, 120), _stop(cathedral, 240), _stop(viewpoint, 360)],
            weather_note=None,
            dropped_stops=[],
        )
    ]
    state: AgentState = {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="info_query_no_edit_needed",
        state=state,
        user_message="What do I have scheduled on day 0? Just checking, don't change anything yet.",
        place_expected_present="museum1",
        expected_tool_sequence=["list_itinerary"],
        disallowed_tools=["find_alternative", "swap_stop", "remove_stop", "add_stop"],
    )


def _wrong_day_offset_recovers_scenario() -> ChatScenario:
    """The traveller misremembers which day a stop is on -- a believable
    real mistake, not a hallucinated place_id. The first remove_stop call
    must fail validate_place_used_on_day's check (the place_id is real, just
    not on the day named), which is exactly the system prompt's "read why
    and try a different valid option" case. Distinct from every other
    scenario here: this is the only one where a guardrail rejection on the
    *first* attempt is the expected, correct trajectory, not a model error.
    Day 0 deliberately has no viewpoint-category stop at all (unlike every
    other category pair in this file, which reuses the same category across
    days) -- so there's no plausible *other* stop on day 0 the model could
    defensibly match "the viewpoint stop" against instead of recognising the
    day_offset itself needs correcting."""
    day0_park = _attraction("park-d0", "North Park", ["park"], price_level="PRICE_LEVEL_FREE")
    day0_cathedral = _attraction("cathedral-d0", "North Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    day0_garden = _attraction("garden-d0", "North Botanical Garden", ["garden"], price_level="PRICE_LEVEL_FREE")
    day1_museum = _attraction("museum-d1", "South Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    day1_park = _attraction("park-d1", "South Park", ["park"], price_level="PRICE_LEVEL_FREE")
    day1_viewpoint = _attraction("viewpoint-d1", "South Viewpoint", ["tourist_attraction"], price_level="PRICE_LEVEL_FREE")
    ranked = [
        _ranked(day0_park), _ranked(day0_cathedral), _ranked(day0_garden),
        _ranked(day1_museum), _ranked(day1_park), _ranked(day1_viewpoint),
    ]
    itinerary = [
        ItineraryDay(
            date=START,
            stops=[_stop(day0_park, 0), _stop(day0_cathedral, 120), _stop(day0_garden, 240)],
            weather_note=None,
            dropped_stops=[],
        ),
        ItineraryDay(
            date=START + dt.timedelta(days=1),
            stops=[_stop(day1_museum, 0), _stop(day1_park, 120), _stop(day1_viewpoint, 240)],
            weather_note=None,
            dropped_stops=[],
        ),
    ]
    state: AgentState = {
        "trip_setup": _trip_setup(num_days=2),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="wrong_day_offset_recovers_after_guardrail_error",
        state=state,
        user_message=(
            "Remove the viewpoint stop (place_id viewpoint-d1) from day 0 -- I don't want to visit it, "
            "no replacement needed."
        ),
        place_expected_removed="viewpoint-d1",
        place_expected_present="park-d0",  # day 0's own stops must survive the model's wrong first guess
        expected_tool_sequence=["remove_stop"],
        disallowed_tools=["find_alternative"],
    )


def _add_stop_declines_when_closed_scenario() -> ChatScenario:
    """Mirror of add_stop_fills_free_time where the only unused candidate
    matching the traveller's ask happens to be closed on the trip's actual
    date. find_alternative's own output flags this (see _availability_note
    in app/agent/tools.py), and if the model tries add_stop anyway,
    _reschedule_day rejects it before writing any state -- either way the
    closed place must never end up in the final itinerary, unlike
    add_stop_fills_free_time where success is adding it."""
    museum = _attraction("museum1", "City History Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    park = _attraction("park1", "Riverside Park", ["park"], price_level="PRICE_LEVEL_FREE")
    cathedral = _attraction("cathedral1", "Old Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    closed_museum = _attraction(
        "museum-closed", "Design Museum", ["museum"], price_level="PRICE_LEVEL_MODERATE",
        open_weekdays=frozenset(d for d in range(7) if d != START.weekday()),
    )
    ranked = [_ranked(museum), _ranked(park), _ranked(cathedral), _ranked(closed_museum)]
    itinerary = [
        ItineraryDay(
            date=START,
            stops=[_stop(museum, 0), _stop(park, 120), _stop(cathedral, 240)],
            weather_note=None,
            dropped_stops=[],
        )
    ]
    state: AgentState = {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="add_stop_declines_when_closed_that_day",
        state=state,
        user_message="Day 0 has some free time left. Can you add another museum that matches my interests?",
        place_expected_removed="museum-closed",  # must never make it into the itinerary
        expected_tool_sequence=["find_alternative"],
    )


def _multi_turn_followup_edit_scenario() -> ChatScenario:
    """Every other scenario is a single turn. This one checks the model
    uses conversation memory *and* the mutated state from its own previous
    turn, not just the original fixture: turn 1 swaps the expensive museum
    for a cheaper one (like budget_swap_finds_cheaper_alternative), then
    turn 2 says "swap it again, I'd rather have a park" without repeating
    the place_id -- the only way to get this right is remembering that the
    slot now holds museum-cheap (turn 1's result), not the original
    museum-lux, which is no longer scheduled anywhere by turn 2."""
    expensive = _attraction("museum-lux", "Grand National Museum", ["museum"], price_level="PRICE_LEVEL_VERY_EXPENSIVE")
    cheap_museum = _attraction("museum-cheap", "Community Art Museum", ["museum"], price_level="PRICE_LEVEL_INEXPENSIVE")
    park = _attraction("park1", "Riverside Park", ["park"], price_level="PRICE_LEVEL_FREE")
    cathedral = _attraction("cathedral1", "Old Cathedral", ["church"], price_level="PRICE_LEVEL_FREE")
    park_alt = _attraction("park-alt", "Seaside Promenade", ["park"], price_level="PRICE_LEVEL_FREE")
    ranked = [_ranked(expensive), _ranked(cheap_museum), _ranked(park), _ranked(cathedral), _ranked(park_alt)]
    itinerary = [
        ItineraryDay(
            date=START,
            stops=[_stop(expensive, 0), _stop(park, 120), _stop(cathedral, 240)],
            weather_note=None,
            dropped_stops=[],
        )
    ]
    state: AgentState = {
        "trip_setup": _trip_setup(),
        "ranked_attractions": ranked,
        "itinerary": itinerary,
        "distance_matrix": None,
    }
    return ChatScenario(
        name="multi_turn_followup_edit_uses_conversation_memory",
        state=state,
        user_message="",  # unused -- see user_messages
        user_messages=[
            (
                "The museum on day 0 (place_id museum-lux) is way over my budget. Find something cheaper "
                "in the same category and swap it in."
            ),
            "Actually, I'd rather have a park there instead of a museum. Swap it again.",
        ],
        place_expected_removed="museum-cheap",  # only gone if turn 2 correctly builds on turn 1's result
        place_expected_present="park-alt",
        expected_tool_sequence=["find_alternative", "swap_stop", "find_alternative", "swap_stop"],
    )


SCENARIOS: list[ChatScenario] = [
    _budget_swap_scenario(),
    _weather_triggered_swap_scenario(),
    _clear_weather_no_swap_scenario(),
    _direct_remove_scenario(),
    _add_stop_scenario(),
    _multi_day_edit_scenario(),
    _info_query_no_edit_scenario(),
    _wrong_day_offset_recovers_scenario(),
    _add_stop_declines_when_closed_scenario(),
    _multi_turn_followup_edit_scenario(),
]
