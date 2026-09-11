"""Attractions node: search candidates via the places provider, then rerank
them against the traveller's stated preferences (see app/recsys/rank.py).

Also records the destination's country code (from the Places address
components) so later nodes can pick a region-appropriate provider and
finalize.py can warn when the destination is in a poorly-covered region --
see app/tools/providers.py.
"""

from __future__ import annotations

import logging

from app.graph.state import AgentState
from app.i18n import places_language_code
from app.recsys.rank import rank_attractions
from app.tools.places_client import PlacesClientError
from app.tools.providers import get_places_provider, is_low_coverage_country

logger = logging.getLogger(__name__)

TOP_N_CANDIDATES = 12


def _preference_text(trip_setup) -> str:
    interests = ", ".join(trip_setup["interests"]) or "general sightseeing"
    return (
        f"Interested in {interests}. Prefers a {trip_setup['daily_pace']} pace "
        f"and a {trip_setup['budget_level']} budget."
    )


def attractions_node(state: AgentState) -> dict:
    trip_setup = state["trip_setup"]
    lang = trip_setup.get("lang", "en")
    client = get_places_provider()

    try:
        candidates = client.search_attractions(
            trip_setup["destination"],
            trip_setup["interests"],
            language_code=places_language_code(lang),
        )
    except PlacesClientError as exc:
        logger.warning("attractions lookup failed for %s: %s", trip_setup["destination"], exc)
        errors = list(state.get("errors", []))
        errors.append(f"attractions: {exc}")
        return {
            "candidate_attractions": [],
            "ranked_attractions": [],
            "destination_country": None,
            "low_coverage_destination": False,
            "errors": errors,
        }

    country = next((c.country_code for c in candidates if c.country_code), None)
    ranked = rank_attractions(
        candidates, _preference_text(trip_setup), top_n=TOP_N_CANDIDATES
    )
    return {
        "candidate_attractions": candidates,
        "ranked_attractions": ranked,
        "destination_country": country,
        "low_coverage_destination": is_low_coverage_country(country),
    }
