"""Attractions node: search candidates via the places provider, then rerank
them against the traveller's stated preferences (see app/recsys/rank.py).

Also records the destination's country code (from the Places address
components) so later nodes can pick a region-appropriate provider and
finalize.py can warn when the destination is in a poorly-covered region --
see app/tools/providers.py.
"""

from __future__ import annotations

import logging

from app.graph.nodes._shared import candidate_pool_target
from app.graph.state import AgentState
from app.i18n import places_language_code
from app.recsys.rank import rank_attractions, rank_attractions_by_popularity
from app.tools.places_client import PlacesClientError
from app.tools.providers import get_places_provider, is_low_coverage_country

logger = logging.getLogger(__name__)


def _preference_text(trip_setup) -> str:
    interests = ", ".join(trip_setup["interests"]) or "general sightseeing"
    text = (
        f"Interested in {interests}. Prefers a {trip_setup['daily_pace']} pace "
        f"and a {trip_setup['budget_level']} budget."
    )
    notes = trip_setup.get("preference_notes")
    if notes:
        text += f" Additional preferences: {notes}."
    return text


def attractions_node(state: AgentState) -> dict:
    trip_setup = state["trip_setup"]
    lang = trip_setup.get("lang", "en")
    client = get_places_provider()
    top_n = candidate_pool_target(trip_setup)

    try:
        candidates = client.search_attractions(
            trip_setup["destination"],
            trip_setup["interests"],
            max_results=top_n,
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

    if not candidates:
        # The Places call itself succeeded (no PlacesClientError) but matched
        # nothing at all -- almost always a misspelled/nonexistent
        # destination, not a transient API problem, so it gets its own
        # friendlier error tag rather than falling through to
        # itinerary_node's generic "empty plan" one (see i18n.py's
        # _ERROR_TAG_LABELS).
        errors = list(state.get("errors", []))
        errors.append("destination_not_found")
        return {
            "candidate_attractions": [],
            "ranked_attractions": [],
            "destination_country": None,
            "low_coverage_destination": False,
            "errors": errors,
        }

    country = next((c.country_code for c in candidates if c.country_code), None)
    # With neither stated interests nor a free-text note there's no real
    # preference to match semantically -- rank by rating/popularity instead
    # of a generic "general sightseeing" query that would just add embedding
    # noise. See rank_attractions_by_popularity's docstring. A free-text note
    # on its own (interests left as "Surprise me") still routes through
    # semantic ranking, since it's a real stated preference even without a
    # checked category.
    if trip_setup["interests"] or trip_setup.get("preference_notes"):
        ranked = rank_attractions(candidates, _preference_text(trip_setup), top_n=top_n)
    else:
        ranked = rank_attractions_by_popularity(candidates, top_n=top_n)
    return {
        "candidate_attractions": candidates,
        "ranked_attractions": ranked,
        "destination_country": country,
        "low_coverage_destination": is_low_coverage_country(country),
    }
