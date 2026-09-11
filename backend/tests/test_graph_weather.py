import datetime as dt

import httpx
import respx

from app.graph.nodes import weather as weather_node_module
from app.graph.nodes.weather import should_check_weather, weather_node
from app.recsys.rank import RankedAttraction
from app.tools.places_client import Attraction
from app.tools.weather_client import FORECAST_HORIZON_DAYS, FORECAST_URL, WeatherClient


def _trip_setup(start_date: dt.date) -> dict:
    return {"destination": "Lisbon", "start_date": start_date}


def _ranked(lat: float, lng: float) -> RankedAttraction:
    attraction = Attraction(
        place_id="p1",
        name="Some Place",
        address="",
        lat=lat,
        lng=lng,
        rating=4.0,
        rating_count=10,
        price_level=None,
        categories=[],
        description=None,
    )
    return RankedAttraction(attraction=attraction, score=1.0, similarity=1.0, normalized_rating=0.8)


def test_should_check_weather_within_horizon():
    start_date = dt.date.today() + dt.timedelta(days=2)
    assert should_check_weather({"trip_setup": _trip_setup(start_date)}) == "weather"


def test_should_check_weather_skips_beyond_horizon():
    start_date = dt.date.today() + dt.timedelta(days=FORECAST_HORIZON_DAYS + 10)
    assert should_check_weather({"trip_setup": _trip_setup(start_date)}) == "route"


def test_should_check_weather_skips_past_start_date():
    start_date = dt.date.today() - dt.timedelta(days=1)
    assert should_check_weather({"trip_setup": _trip_setup(start_date)}) == "route"


@respx.mock
def test_weather_node_returns_forecast(monkeypatch):
    monkeypatch.setattr(
        weather_node_module, "get_weather_provider", lambda country_code=None: WeatherClient(api_key="k")
    )
    timestamp = int(dt.datetime.now(tz=dt.timezone.utc).timestamp())
    route = respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={"list": [{"dt": timestamp, "main": {"temp": 20.0}, "weather": [{"main": "Clear"}]}]},
        )
    )

    result = weather_node(
        {"trip_setup": {"destination": "Lisbon"}, "ranked_attractions": [_ranked(38.7223, -9.1393)]}
    )

    assert len(result["weather_forecast"]) == 1
    assert result["weather_forecast"][0].condition == "Clear"
    # Called with the attraction's coordinates, not the raw destination text.
    request_params = dict(httpx.QueryParams(route.calls[0].request.url.query))
    assert request_params["lat"] == "38.7223"
    assert request_params["lon"] == "-9.1393"
    assert "q" not in request_params


@respx.mock
def test_weather_node_degrades_gracefully_on_error(monkeypatch):
    monkeypatch.setattr(
        weather_node_module, "get_weather_provider", lambda country_code=None: WeatherClient(api_key="k")
    )
    respx.get(FORECAST_URL).mock(return_value=httpx.Response(500))

    result = weather_node(
        {"trip_setup": {"destination": "Lisbon"}, "ranked_attractions": [_ranked(38.7223, -9.1393)]}
    )

    assert result["weather_forecast"] == []
    assert any("weather:" in e for e in result["errors"])


def test_weather_node_skips_api_call_when_no_ranked_attractions():
    result = weather_node({"trip_setup": {"destination": "Nowhere"}, "ranked_attractions": []})

    assert result == {"weather_forecast": []}
