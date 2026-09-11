import httpx
import respx

from app.graph.nodes import route as route_node_module
from app.graph.nodes.route import route_node
from app.recsys.rank import RankedAttraction
from app.tools.places_client import Attraction
from app.tools.route_client import RouteClient

BASE_URL = "https://router.project-osrm.org"


def _ranked(place_id: str, lat: float, lng: float) -> RankedAttraction:
    attraction = Attraction(
        place_id=place_id,
        name=place_id,
        address="",
        lat=lat,
        lng=lng,
        rating=4.0,
        rating_count=10,
        price_level=None,
        categories=["museum"],
        description=None,
    )
    return RankedAttraction(attraction=attraction, score=1.0, similarity=1.0, normalized_rating=0.8)


def test_route_node_returns_none_for_single_candidate():
    assert route_node({"ranked_attractions": [_ranked("p1", 1.0, 1.0)]})["distance_matrix"] is None


def test_route_node_returns_none_for_no_candidates():
    assert route_node({"ranked_attractions": []})["distance_matrix"] is None


@respx.mock
def test_route_node_returns_a_matrix(monkeypatch):
    monkeypatch.setattr(
        route_node_module, "get_route_provider", lambda country_code=None: RouteClient(base_url=BASE_URL)
    )
    respx.get(url__startswith=f"{BASE_URL}/table/v1/driving/").mock(
        return_value=httpx.Response(
            200,
            json={"code": "Ok", "durations": [[0, 300], [300, 0]], "distances": [[0, 500], [500, 0]]},
        )
    )

    result = route_node({"ranked_attractions": [_ranked("p1", 1.0, 1.0), _ranked("p2", 1.1, 1.1)]})

    assert result["distance_matrix"].minutes_between("p1", "p2") == 5.0


@respx.mock
def test_route_node_degrades_gracefully_on_error(monkeypatch):
    monkeypatch.setattr(
        route_node_module, "get_route_provider", lambda country_code=None: RouteClient(base_url=BASE_URL)
    )
    respx.get(url__startswith=f"{BASE_URL}/table/v1/driving/").mock(return_value=httpx.Response(500))

    result = route_node({"ranked_attractions": [_ranked("p1", 1.0, 1.0), _ranked("p2", 1.1, 1.1)]})

    assert result["distance_matrix"] is None
    assert any("route:" in e for e in result["errors"])
