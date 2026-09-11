"""Provider selection for the three external data tools.

Each tool (places / weather / route) sits behind a Protocol so a
region-specific implementation can be swapped in per destination without
touching the graph nodes, the constraint checks, or the itinerary LLM --
those all operate on the normalized Attraction / DailyForecast /
DistanceMatrix types, not on any one provider's response shape.

Today there is exactly one implementation per slot (Google Places /
OpenWeatherMap / OSRM), returned regardless of country. The `country_code`
argument is the seam for a future mainland-China path (Amap + QWeather); see
docs/adr-001-provider-abstraction.md and docs/adr-002-china-provider.md.
"""

from __future__ import annotations

from typing import Protocol

from app.config import get_settings
from app.tools.places_client import Attraction, PlacesClient
from app.tools.route_client import Coordinate, DistanceMatrix, RouteClient
from app.tools.weather_client import DailyForecast, WeatherClient

# Countries the default (international) providers cover poorly enough that the
# user should be warned. Mainland China only -- Google/OSRM/OpenWeatherMap
# data for Hong Kong, Macau and Taiwan is fine.
LOW_COVERAGE_COUNTRIES = {"CN"}


def is_low_coverage_country(country_code: str | None) -> bool:
    return country_code in LOW_COVERAGE_COUNTRIES


class PlacesProvider(Protocol):
    def search_attractions(
        self, city: str, interests: list[str], max_results: int = 20, language_code: str = "en"
    ) -> list[Attraction]: ...


class WeatherProvider(Protocol):
    def get_daily_forecast(self, lat: float, lng: float, lang: str = "en") -> list[DailyForecast]: ...


class RouteProvider(Protocol):
    def get_distance_matrix(self, place_ids: list[str], stops: list[Coordinate]) -> DistanceMatrix: ...


def get_places_provider(country_code: str | None = None) -> PlacesProvider:
    return PlacesClient(api_key=get_settings().google_places_api_key)


def get_weather_provider(country_code: str | None = None) -> WeatherProvider:
    return WeatherClient(api_key=get_settings().openweather_api_key)


def get_route_provider(country_code: str | None = None) -> RouteProvider:
    return RouteClient(base_url=get_settings().osrm_base_url)
