import httpx
import numpy as np
import pytest
import respx

from app.graph.nodes import attractions as attractions_node_module
from app.graph.nodes.attractions import attractions_node
from app.recsys import rank as rank_module
from app.tools.places_client import PLACES_SEARCH_URL, PlacesClient


def _trip_setup(**overrides) -> dict:
    base = {
        "destination": "Lisbon",
        "interests": ["museums"],
        "daily_pace": "moderate",
        "budget_level": "moderate",
    }
    base.update(overrides)
    return base


def _place(place_id: str, name: str, country: str | None = None) -> dict:
    place = {
        "id": place_id,
        "displayName": {"text": name},
        "formattedAddress": "123 Main St",
        "location": {"latitude": 1.23, "longitude": 4.56},
        "rating": 4.5,
        "userRatingCount": 100,
        "priceLevel": "PRICE_LEVEL_MODERATE",
        "types": ["museum"],
        "editorialSummary": {"text": "A fine museum."},
    }
    if country is not None:
        place["addressComponents"] = [{"types": ["country", "political"], "shortText": country}]
    return place


@pytest.fixture
def fake_embed(monkeypatch):
    def _fake_embed_texts(texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2))

    monkeypatch.setattr(rank_module, "embed_texts", _fake_embed_texts)


@pytest.fixture
def fake_provider(monkeypatch):
    monkeypatch.setattr(
        attractions_node_module, "get_places_provider", lambda country_code=None: PlacesClient(api_key="k")
    )


@respx.mock
def test_attractions_node_ranks_candidates(fake_embed, fake_provider):
    respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"places": [_place("p1", "City Museum")]})
    )

    result = attractions_node({"trip_setup": _trip_setup()})

    assert len(result["candidate_attractions"]) == 1
    assert len(result["ranked_attractions"]) == 1
    assert result["ranked_attractions"][0].attraction.place_id == "p1"


@respx.mock
def test_attractions_node_flags_low_coverage_for_mainland_china(fake_embed, fake_provider):
    respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"places": [_place("p1", "豫园", country="CN")]})
    )

    result = attractions_node({"trip_setup": _trip_setup(destination="Shanghai")})

    assert result["destination_country"] == "CN"
    assert result["low_coverage_destination"] is True


@respx.mock
def test_attractions_node_does_not_flag_low_coverage_elsewhere(fake_embed, fake_provider):
    respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"places": [_place("p1", "City Museum", country="PT")]})
    )

    result = attractions_node({"trip_setup": _trip_setup()})

    assert result["destination_country"] == "PT"
    assert result["low_coverage_destination"] is False


@respx.mock
def test_attractions_node_degrades_gracefully_on_places_error(fake_provider):
    respx.post(PLACES_SEARCH_URL).mock(return_value=httpx.Response(403))

    result = attractions_node({"trip_setup": _trip_setup()})

    assert result["candidate_attractions"] == []
    assert result["ranked_attractions"] == []
    assert result["low_coverage_destination"] is False
    assert any("attractions:" in e for e in result["errors"])
