"""Input validation for the chat agent's tools (app.agent.tools).

Every tool that touches the itinerary validates its arguments against the
*actual* session state before doing anything, the same discipline
itinerary_node already applies to the generation-time LLM (see its
docstring on hallucinated place_ids): a bad argument returns a plain string
the model can read and react to, never an exception, and never a silent,
partially-applied edit to `state["itinerary"]`.

This is also the project's prompt-injection boundary: tool results (place
descriptions, opening hours) are DATA about the trip, not instructions --
nothing returned by a tool here is ever re-interpreted as a command, and
these validators are what stop a model (confused or otherwise) from acting
on a place_id / day_offset it wasn't actually given by a tool call.
"""

from __future__ import annotations

from app.graph.nodes._shared import attraction_lookup
from app.graph.state import AgentState


def validate_day_offset(state: AgentState, day_offset: int) -> str | None:
    itinerary = state.get("itinerary", [])
    if not (0 <= day_offset < len(itinerary)):
        return f"day_offset {day_offset} is out of range -- this trip has days 0..{len(itinerary) - 1}."
    return None


def validate_known_place_id(state: AgentState, place_id: str) -> str | None:
    """place_id must be one this trip's candidate pool actually contains --
    catches a hallucinated id before it reaches the scheduler."""
    attractions_by_id = attraction_lookup(state.get("ranked_attractions", []))
    if place_id not in attractions_by_id:
        return f"place_id '{place_id}' is not a known candidate for this trip. Call find_alternative first."
    return None


def validate_place_used_on_day(state: AgentState, day_offset: int, place_id: str) -> str | None:
    day = state["itinerary"][day_offset]
    stop_ids = {s["place_id"] for s in day["stops"] if s["kind"] == "attraction"}
    if place_id not in stop_ids:
        return f"'{place_id}' is not currently scheduled on day {day_offset}."
    return None


def validate_place_not_used_anywhere(state: AgentState, place_id: str) -> str | None:
    for day in state.get("itinerary", []):
        for stop in day["stops"]:
            if stop["kind"] == "attraction" and stop["place_id"] == place_id:
                return f"'{place_id}' is already scheduled on {day['date']} -- pick a different candidate."
    return None
