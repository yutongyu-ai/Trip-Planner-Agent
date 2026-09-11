import httpx
import pytest
import respx

from app.tools.route_client import Coordinate, RouteClient, RouteClientError

BASE_URL = "https://router.project-osrm.org"

_STOPS = [
    Coordinate(lat=38.71, lng=-9.14),
    Coordinate(lat=38.72, lng=-9.13),
    Coordinate(lat=38.73, lng=-9.15),
]


@respx.mock
def test_get_distance_matrix_returns_pairwise_minutes():
    respx.get(url__startswith=f"{BASE_URL}/table/v1/driving/").mock(
        return_value=httpx.Response(
            200,
            json={
                "code": "Ok",
                "durations": [[0, 300, 600], [300, 0, 240], [600, 240, 0]],
                "distances": [[0, 1200, 2400], [1200, 0, 900], [2400, 900, 0]],
            },
        )
    )
    matrix = RouteClient(base_url=BASE_URL).get_distance_matrix(["a", "b", "c"], _STOPS)

    assert matrix.minutes_between("a", "b") == 5.0
    assert matrix.minutes_between("c", "b") == 4.0
    assert matrix.minutes_between("a", "missing") is None


@respx.mock
def test_get_distance_matrix_treats_an_unroutable_pair_as_none():
    # OSRM's table service returns `null` (not an omitted cell) for a pair
    # it can't find a route between at all -- e.g. split by water.
    respx.get(url__startswith=f"{BASE_URL}/table/v1/driving/").mock(
        return_value=httpx.Response(
            200,
            json={"code": "Ok", "durations": [[0, None], [None, 0]], "distances": [[0, None], [None, 0]]},
        )
    )
    matrix = RouteClient(base_url=BASE_URL).get_distance_matrix(["a", "b"], _STOPS[:2])

    assert matrix.minutes_between("a", "b") is None


@respx.mock
def test_get_distance_matrix_tolerates_missing_distance_annotation():
    respx.get(url__startswith=f"{BASE_URL}/table/v1/driving/").mock(
        return_value=httpx.Response(200, json={"code": "Ok", "durations": [[0, 120], [120, 0]]})
    )
    matrix = RouteClient(base_url=BASE_URL).get_distance_matrix(["a", "b"], _STOPS[:2])

    assert matrix.minutes_between("a", "b") == 2.0
    assert matrix.distance_m == []


def test_get_distance_matrix_returns_trivial_matrix_for_single_stop():
    matrix = RouteClient(base_url=BASE_URL).get_distance_matrix(["a"], _STOPS[:1])
    assert matrix.place_ids == ["a"]
    assert matrix.minutes_between("a", "a") == 0.0


@respx.mock
def test_get_distance_matrix_raises_on_osrm_error_code():
    respx.get(url__startswith=f"{BASE_URL}/table/v1/driving/").mock(
        return_value=httpx.Response(200, json={"code": "NoTable"})
    )
    with pytest.raises(RouteClientError):
        RouteClient(base_url=BASE_URL).get_distance_matrix(["a", "b"], _STOPS[:2])


@respx.mock
def test_get_distance_matrix_raises_on_http_error():
    respx.get(url__startswith=f"{BASE_URL}/table/v1/driving/").mock(return_value=httpx.Response(500))
    with pytest.raises(RouteClientError):
        RouteClient(base_url=BASE_URL).get_distance_matrix(["a", "b"], _STOPS[:2])


def test_get_distance_matrix_rejects_mismatched_lengths():
    with pytest.raises(RouteClientError):
        RouteClient(base_url=BASE_URL).get_distance_matrix(["a", "b"], _STOPS[:1])
