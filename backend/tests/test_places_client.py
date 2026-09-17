import json

import httpx
import pytest
import respx

from app.tools.places_client import (
    PLACES_SEARCH_URL,
    PlacesClient,
    PlacesClientError,
)


def _place(place_id: str, name: str) -> dict:
    return {
        "id": place_id,
        "displayName": {"text": name},
        "formattedAddress": "123 Main St",
        "location": {"latitude": 1.23, "longitude": 4.56},
        "rating": 4.5,
        "userRatingCount": 100,
        "priceLevel": "PRICE_LEVEL_MODERATE",
        # Includes "locality" (in addition to a realistic attraction type)
        # so the same fixture can double as a bare-name city-center lookup
        # response in tests that reuse one mock for every call -- see
        # PlacesClient._resolve_city_center's _LOCALITY_TYPES check.
        "types": ["museum", "tourist_attraction", "locality"],
        "editorialSummary": {"text": "A fine museum."},
    }


@respx.mock
def test_search_attractions_returns_parsed_results():
    respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json={"places": [_place("p1", "City Museum")]}
        )
    )
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("Lisbon", ["museums"])

    assert len(results) == 1
    assert results[0].name == "City Museum"
    assert results[0].place_id == "p1"
    assert results[0].rating == 4.5
    assert "museum" in results[0].categories


@respx.mock
def test_search_attractions_widens_radius_when_too_few_results():
    route = respx.post(PLACES_SEARCH_URL)
    route.side_effect = [
        httpx.Response(200, json={"places": [_place("anchor", "Small Town")]}),  # city-center resolution
        httpx.Response(200, json={"places": [_place("p1", "Only One")]}),
        httpx.Response(
            200,
            json={
                "places": [
                    _place(f"p{i}", f"Place {i}") for i in range(6)
                ]
            },
        ),
    ]
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("Small Town", [])

    assert len(results) == 6
    assert route.call_count == 3


@respx.mock
def test_search_attractions_returns_best_effort_after_exhausting_ladder():
    respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"places": [_place("p1", "Only One")]})
    )
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("Remote Village", [])

    assert len(results) == 1


@respx.mock
def test_search_attractions_sends_language_code():
    route = respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"places": [_place("p1", "武汉热干面")]})
    )
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("Wuhan", ["food"], language_code="zh-CN")

    assert results[0].name == "武汉热干面"
    # calls[0] is the bare-city-name location-resolution lookup; calls[1] is
    # the first actual interest-based search attempt.
    sent_body = json.loads(route.calls[1].request.content)
    assert sent_body["languageCode"] == "zh-CN"


@respx.mock
def test_search_attractions_defaults_to_english_language_code():
    route = respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"places": [_place("p1", "City Museum")]})
    )
    client = PlacesClient(api_key="test-key")

    client.search_attractions("Lisbon", ["museums"])

    sent_body = json.loads(route.calls[1].request.content)
    assert sent_body["languageCode"] == "en"


@respx.mock
def test_search_query_omits_attractions_keyword_when_interests_given():
    # "attractions" biases Google's text search toward generic landmarks,
    # diluting a themed search (e.g. food) with off-topic tourist-trap
    # results -- see app/tools/places_client.py's _build_query docstring.
    route = respx.post(PLACES_SEARCH_URL)
    route.side_effect = [
        httpx.Response(200, json={"places": [_place("anchor", "Wuhan")]}),  # city-center resolution
        *([httpx.Response(200, json={"places": []})] * 3),  # one per radius-ladder rung
    ]
    client = PlacesClient(api_key="test-key")

    client.search_attractions("Wuhan", ["food"])

    sent_body = json.loads(route.calls[1].request.content)
    assert sent_body["textQuery"] == "food in Wuhan"


@respx.mock
def test_search_query_falls_back_to_attractions_when_no_interests():
    route = respx.post(PLACES_SEARCH_URL)
    route.side_effect = [
        httpx.Response(200, json={"places": [_place("anchor", "Lisbon")]}),  # city-center resolution
        *([httpx.Response(200, json={"places": []})] * 3),  # one per radius-ladder rung
    ]
    client = PlacesClient(api_key="test-key")

    client.search_attractions("Lisbon", [])

    sent_body = json.loads(route.calls[1].request.content)
    assert sent_body["textQuery"] == "top attractions in Lisbon"


@respx.mock
def test_search_attractions_extracts_country_code():
    place = _place("p1", "Yu Garden")
    place["addressComponents"] = [
        {"types": ["locality", "political"], "shortText": "Shanghai", "longText": "Shanghai"},
        {"types": ["country", "political"], "shortText": "CN", "longText": "China"},
    ]
    respx.post(PLACES_SEARCH_URL).mock(return_value=httpx.Response(200, json={"places": [place]}))
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("Shanghai", [])

    assert results[0].country_code == "CN"


@respx.mock
def test_search_attractions_country_code_none_when_absent():
    respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"places": [_place("p1", "City Museum")]})
    )
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("Lisbon", [])

    assert results[0].country_code is None


@respx.mock
def test_search_attractions_raises_on_http_error():
    respx.post(PLACES_SEARCH_URL).mock(return_value=httpx.Response(403))
    client = PlacesClient(api_key="bad-key")

    with pytest.raises(PlacesClientError):
        client.search_attractions("Lisbon", [])


@respx.mock
def test_search_attractions_parses_open_weekdays():
    place = _place("p1", "Saturday Market")
    # Google's Period.open.day: 6 == Saturday.
    place["regularOpeningHours"] = {
        "periods": [{"open": {"day": 6, "hour": 8, "minute": 0}, "close": {"day": 6, "hour": 15, "minute": 0}}]
    }
    respx.post(PLACES_SEARCH_URL).mock(return_value=httpx.Response(200, json={"places": [place]}))
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("London", [])

    # date.weekday()'s Saturday == 5.
    assert results[0].open_weekdays == frozenset({5})


@respx.mock
def test_search_attractions_parses_opening_hours_clock_times():
    place = _place("p1", "Saturday Market")
    # Google's Period.open.day: 6 == Saturday.
    place["regularOpeningHours"] = {
        "periods": [{"open": {"day": 6, "hour": 8, "minute": 0}, "close": {"day": 6, "hour": 15, "minute": 30}}]
    }
    respx.post(PLACES_SEARCH_URL).mock(return_value=httpx.Response(200, json={"places": [place]}))
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("London", [])

    # date.weekday()'s Saturday == 5; 08:00 = 480min, 15:30 = 930min.
    assert results[0].opening_hours == {5: (480, 930)}


@respx.mock
def test_search_attractions_opening_hours_handles_overnight_period():
    place = _place("p1", "Late Bar")
    # Open Friday 20:00, closes Saturday 02:00.
    place["regularOpeningHours"] = {
        "periods": [{"open": {"day": 5, "hour": 20, "minute": 0}, "close": {"day": 6, "hour": 2, "minute": 0}}]
    }
    respx.post(PLACES_SEARCH_URL).mock(return_value=httpx.Response(200, json={"places": [place]}))
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("London", [])

    # Friday == weekday 4; 20:00 = 1200min, and 02:00 the *next* day is
    # expressed as 1200 + 6h = 1560min past Friday's own midnight.
    assert results[0].opening_hours == {4: (1200, 1560)}


@respx.mock
def test_search_attractions_opening_hours_handles_24_hour_open():
    place = _place("p1", "Always Open Store")
    place["regularOpeningHours"] = {"periods": [{"open": {"day": 1, "hour": 0, "minute": 0}}]}
    respx.post(PLACES_SEARCH_URL).mock(return_value=httpx.Response(200, json={"places": [place]}))
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("London", [])

    # Monday == weekday 0; open at minute 0, closes 24h later at minute 1440.
    assert results[0].opening_hours == {0: (0, 1440)}


@respx.mock
def test_search_attractions_open_weekdays_none_when_hours_missing():
    respx.post(PLACES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"places": [_place("p1", "City Museum")]})
    )
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("Lisbon", [])
    assert results[0].opening_hours is None
    assert results[0].open_weekdays is None


@respx.mock
def test_search_attractions_resolves_city_center_and_biases_the_search():
    # calls[0]: bare "Shanghai" lookup used only to resolve a location.
    # calls[1]: the actual interest search, which should be biased around
    # that resolved point -- this is what stops a query like "中餐 in 上海"
    # from silently falling back to wherever the server's IP is (see
    # PlacesClient.search_attractions's docstring).
    route = respx.post(PLACES_SEARCH_URL)
    anchor = dict(_place("anchor", "Shanghai"))
    anchor["location"] = {"latitude": 31.23, "longitude": 121.47}
    route.side_effect = [
        httpx.Response(200, json={"places": [anchor]}),
        httpx.Response(200, json={"places": [_place("p1", "Yu Garden")] * 5}),
    ]
    client = PlacesClient(api_key="test-key")

    client.search_attractions("Shanghai", ["中餐"])

    resolve_body = json.loads(route.calls[0].request.content)
    assert resolve_body["textQuery"] == "Shanghai"
    assert "locationBias" not in resolve_body  # nothing to bias around yet

    search_body = json.loads(route.calls[1].request.content)
    assert search_body["textQuery"] == "中餐 in Shanghai"
    bias = search_body["locationBias"]["circle"]
    assert bias["center"] == {"latitude": 31.23, "longitude": 121.47}
    assert bias["radius"] == 10_000.0  # first rung of _RADIUS_LADDER_M


@respx.mock
def test_search_attractions_returns_empty_when_city_lookup_matches_nothing():
    # The city-center lookup succeeding but matching nothing at all (e.g. a
    # garbled/nonexistent destination like "ndjed") must NOT fall through to
    # an unbiased search: with no locationBias, Google still returns
    # *something* for the "<interests> in <city>" query -- anchored to the
    # calling server's own IP location (observed for real: a nonexistent
    # city produced a fully-formed, plausible-looking Manchester itinerary)
    # -- which is worse than an honest empty result.
    route = respx.post(PLACES_SEARCH_URL)
    route.side_effect = [httpx.Response(200, json={"places": []})]
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("Nowhereville", [])

    assert results == []
    assert route.call_count == 1  # never even attempts the unbiased search


@respx.mock
def test_search_attractions_returns_empty_when_lookup_fuzzy_matches_a_non_locality_place():
    # Observed for real: the bare-name lookup for "effefev" didn't come back
    # empty -- Google's text search fuzzy-matched it to an unrelated
    # business, "EF International Language Campus" (types
    # educational_institution/point_of_interest/establishment, no locality
    # tag at all), and the graph then produced a fully-formed itinerary for
    # that business's city. A hit with no _LOCALITY_TYPES tag must be
    # treated the same as no match.
    fuzzy_match = {
        "id": "biz1",
        "displayName": {"text": "EF International Language Campus"},
        "formattedAddress": "26 Wilbraham Rd, Manchester, UK",
        "location": {"latitude": 53.44, "longitude": -2.23},
        "types": ["educational_institution", "point_of_interest", "establishment"],
    }
    route = respx.post(PLACES_SEARCH_URL)
    route.side_effect = [httpx.Response(200, json={"places": [fuzzy_match]})]
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("effefev", [])

    assert results == []
    assert route.call_count == 1  # never even attempts the unbiased search


@respx.mock
def test_search_attractions_fetches_a_second_page_when_the_first_falls_short():
    route = respx.post(PLACES_SEARCH_URL)
    page1 = [_place(f"p{i}", f"Place {i}") for i in range(20)]
    page2 = [_place(f"q{i}", f"Extra {i}") for i in range(5)]
    route.side_effect = [
        httpx.Response(200, json={"places": [_place("anchor", "London")]}),  # city-center resolution
        httpx.Response(200, json={"places": page1, "nextPageToken": "tok-1"}),
        httpx.Response(200, json={"places": page2}),  # no nextPageToken -- last page
    ]
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("London", ["museums"], max_results=25)

    assert len(results) == 25
    assert route.call_count == 3
    second_page_body = json.loads(route.calls[2].request.content)
    assert second_page_body["pageToken"] == "tok-1"


@respx.mock
def test_search_attractions_skips_the_second_page_once_max_results_is_met():
    route = respx.post(PLACES_SEARCH_URL)
    page1 = [_place(f"p{i}", f"Place {i}") for i in range(20)]
    route.side_effect = [
        httpx.Response(200, json={"places": [_place("anchor", "London")]}),  # city-center resolution
        httpx.Response(200, json={"places": page1, "nextPageToken": "tok-1"}),
    ]
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("London", ["museums"], max_results=20)

    assert len(results) == 20
    assert route.call_count == 2  # never spends a request on a page it doesn't need


@respx.mock
def test_search_attractions_stops_after_two_pages_even_with_more_available():
    route = respx.post(PLACES_SEARCH_URL)
    page1 = [_place(f"p{i}", f"Place {i}") for i in range(20)]
    page2 = [_place(f"q{i}", f"Extra {i}") for i in range(20)]
    route.side_effect = [
        httpx.Response(200, json={"places": [_place("anchor", "London")]}),  # city-center resolution
        httpx.Response(200, json={"places": page1, "nextPageToken": "tok-1"}),
        httpx.Response(200, json={"places": page2, "nextPageToken": "tok-2"}),  # a third page exists...
    ]
    client = PlacesClient(api_key="test-key")

    results = client.search_attractions("London", ["museums"], max_results=60)

    assert len(results) == 40  # ...but _MAX_PAGES_PER_SEARCH caps it at two
    assert route.call_count == 3


@respx.mock
def test_search_attractions_raises_when_city_lookup_itself_fails():
    # A real API failure during the city-center lookup (bad key, quota,
    # network) is a different situation from "this city doesn't exist" --
    # it must propagate as PlacesClientError, not be silently folded into an
    # empty "destination not found" result.
    respx.post(PLACES_SEARCH_URL).mock(return_value=httpx.Response(403))
    client = PlacesClient(api_key="bad-key")

    with pytest.raises(PlacesClientError):
        client.search_attractions("Nowhereville", [])
