import datetime as dt

import httpx
import pytest
import respx

from app.tools.weather_client import FORECAST_URL, WeatherClient, WeatherClientError


def _entry(timestamp: int, temp: float, condition: str, description: str = "clear sky") -> dict:
    return {
        "dt": timestamp,
        "main": {"temp": temp},
        "weather": [{"main": condition, "description": description}],
    }


@respx.mock
def test_get_daily_forecast_buckets_by_day_and_flags_bad_weather():
    day1_noon = int(
        dt.datetime(2026, 8, 20, 12, tzinfo=dt.timezone.utc).timestamp()
    )
    day1_evening = int(
        dt.datetime(2026, 8, 20, 18, tzinfo=dt.timezone.utc).timestamp()
    )
    day2_noon = int(
        dt.datetime(2026, 8, 21, 12, tzinfo=dt.timezone.utc).timestamp()
    )

    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "list": [
                    _entry(day1_noon, 20.0, "Clear"),
                    _entry(day1_evening, 18.0, "Clear"),
                    _entry(day2_noon, 15.0, "Rain"),
                ]
            },
        )
    )
    client = WeatherClient(api_key="test-key")

    forecasts = client.get_daily_forecast(38.7223, -9.1393)

    assert len(forecasts) == 2
    assert forecasts[0].date == dt.date(2026, 8, 20)
    assert forecasts[0].avg_temp_c == pytest.approx(19.0)
    assert forecasts[0].is_bad_weather is False
    assert forecasts[1].condition == "Rain"
    assert forecasts[1].is_bad_weather is True


@respx.mock
def test_get_daily_forecast_raises_on_http_error():
    respx.get(FORECAST_URL).mock(return_value=httpx.Response(404))
    client = WeatherClient(api_key="bad-key")

    with pytest.raises(WeatherClientError):
        client.get_daily_forecast(0.0, 0.0)


@respx.mock
def test_get_daily_forecast_sends_coords_and_lang_not_city_name():
    route = respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(200, json={"list": [_entry(0, 20.0, "Clear")]})
    )
    client = WeatherClient(api_key="test-key")

    client.get_daily_forecast(30.5928, 114.3055, lang="zh_cn")

    params = route.calls[0].request.url.params
    assert params["lat"] == "30.5928"
    assert params["lon"] == "114.3055"
    assert params["lang"] == "zh_cn"
    assert "q" not in params


@respx.mock
def test_get_daily_forecast_uses_localized_description_matching_dominant_condition():
    timestamp = int(dt.datetime(2026, 8, 20, 12, tzinfo=dt.timezone.utc).timestamp())
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "list": [
                    _entry(timestamp, 20.0, "Rain", description="小雨"),
                    _entry(timestamp + 3600, 19.0, "Rain", description="小雨"),
                    _entry(timestamp + 7200, 18.0, "Clear", description="晴"),
                ]
            },
        )
    )
    client = WeatherClient(api_key="test-key")

    forecasts = client.get_daily_forecast(30.5928, 114.3055, lang="zh_cn")

    assert forecasts[0].condition == "Rain"
    assert forecasts[0].description == "小雨"
