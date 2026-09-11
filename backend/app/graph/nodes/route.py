"""Route node: a real pairwise driving-time matrix between the top-ranked
candidates, from OSRM's table service. The itinerary node's scheduler
(app/graph/scheduling.py) reads it to place stops in time and order; the
itinerary LLM sees a trimmed version as a proximity hint when picking which
places go together on a day. Falls back to a straight-line estimate
everywhere if the call fails (see _shared.estimate_travel_minutes).
"""

from __future__ import annotations

import logging

from app.graph.state import AgentState
from app.tools.providers import get_route_provider
from app.tools.route_client import Coordinate, RouteClientError

logger = logging.getLogger(__name__)

MAX_MATRIX_STOPS = 10


def route_node(state: AgentState) -> dict:
    ranked = state.get("ranked_attractions", [])[:MAX_MATRIX_STOPS]
    if len(ranked) < 2:
        return {"distance_matrix": None}

    client = get_route_provider(state.get("destination_country"))
    place_ids = [r.attraction.place_id for r in ranked]
    stops = [Coordinate(lat=r.attraction.lat, lng=r.attraction.lng) for r in ranked]

    try:
        matrix = client.get_distance_matrix(place_ids, stops)
    except RouteClientError as exc:
        logger.warning("route matrix lookup failed: %s", exc)
        errors = list(state.get("errors", []))
        errors.append(f"route: {exc}")
        return {"distance_matrix": None, "errors": errors}

    return {"distance_matrix": matrix}
