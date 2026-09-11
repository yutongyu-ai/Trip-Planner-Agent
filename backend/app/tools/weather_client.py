"""OpenWeatherMap 5-day/3-hour forecast client.

Docs: https://openweathermap.org/forecast5

OpenWeatherMap's free tier only forecasts ~5 days out, so trips starting
further in the future have no reliable forecast. Callers (the LangGraph
planner node) should skip calling this client entirely when the trip start
date is beyond FORECAST_HORIZON_DAYS rather than getting an empty/stale
result back.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass

import httpx

FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
FORECAST_HORIZON_DAYS = 5


@dataclass(frozen=True)
class DailyForecast:
    date: dt.date
    avg_temp_c: float
    # OpenWeatherMap's stable, unlocalized classification code (e.g.
    # "Rain") -- always English, used for the is_bad_weather check below.
    # Not meant for display.
    condition: str
    # OpenWeatherMap's localized human-readable phrase for the same
    # forecast (e.g. "light rain" / "小雨", depending on the `lang` param
    # passed to get_daily_forecast) -- this is what should be shown to a
    # user.
    description: str
    is_bad_weather: bool


_BAD_WEATHER_CONDITIONS = {"Thunderstorm", "Rain", "Snow", "Drizzle"}


class WeatherClientError(RuntimeError):
    """Raised when the OpenWeatherMap API call fails or returns malformed data."""


class WeatherClient:
    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=10.0)

    def get_daily_forecast(self, lat: float, lng: float, lang: str = "en") -> list[DailyForecast]:
        """Forecast for a coordinate, not a city name.

        The `q=<city name>` form of this endpoint only matches Latin-script
        names (e.g. "Wuhan" works, "武汉" 404s) and is ambiguous across
        same-named cities worldwide. Coordinates sidestep both problems --
        callers should use a real point they already trust (e.g. an
        attraction's lat/lng from Places) rather than re-resolving the
        traveller's raw destination text a second time.

        `lang` (e.g. "zh_cn") only affects DailyForecast.description --
        OpenWeatherMap's `main` classification code stays English
        regardless, which is why `condition` (used for is_bad_weather) and
        `description` (for display) are kept as separate fields.
        """
        params = {
            "lat": lat,
            "lon": lng,
            "appid": self._api_key,
            "units": "metric",
            "lang": lang,
        }
        try:
            response = self._client.get(FORECAST_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WeatherClientError(f"Weather API request failed: {exc}") from exc

        data = response.json()
        return self._bucket_by_day(data.get("list", []))

    @staticmethod
    def _bucket_by_day(entries: list[dict]) -> list[DailyForecast]:
        temps_by_date: dict[dt.date, list[float]] = defaultdict(list)
        # Each entry is (main classification code, localized description).
        weather_by_date: dict[dt.date, list[tuple[str, str]]] = defaultdict(list)

        for entry in entries:
            timestamp = dt.datetime.fromtimestamp(entry["dt"], tz=dt.timezone.utc)
            date = timestamp.date()
            temps_by_date[date].append(entry["main"]["temp"])
            weather = entry.get("weather", [{}])[0]
            weather_by_date[date].append((weather.get("main", "Unknown"), weather.get("description", "unknown")))

        forecasts = []
        for date in sorted(temps_by_date):
            temps = temps_by_date[date]
            weathers = weather_by_date[date]
            conditions = [w[0] for w in weathers]
            dominant_condition = max(set(conditions), key=conditions.count)
            # Pick the description that actually co-occurred with the
            # dominant condition, rather than the most common description
            # overall, so the two stay consistent with each other.
            matching_descriptions = [w[1] for w in weathers if w[0] == dominant_condition]
            dominant_description = max(set(matching_descriptions), key=matching_descriptions.count)
            forecasts.append(
                DailyForecast(
                    date=date,
                    avg_temp_c=sum(temps) / len(temps),
                    condition=dominant_condition,
                    description=dominant_description,
                    is_bad_weather=dominant_condition in _BAD_WEATHER_CONDITIONS,
                )
            )
        return forecasts
