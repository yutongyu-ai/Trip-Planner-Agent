"""Weather node.

The dynamic-tool-selection decision lives in `should_check_weather`: for
trips starting beyond the free-tier forecast horizon, the graph skips this
node entirely via a conditional edge rather than calling the API and
getting nothing useful back (see app/tools/weather_client.py's docstring).
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Literal

from app.graph.state import AgentState
from app.i18n import openweather_lang
from app.tools.providers import get_weather_provider
from app.tools.weather_client import FORECAST_HORIZON_DAYS, WeatherClientError

logger = logging.getLogger(__name__)


def should_check_weather(state: AgentState) -> Literal["weather", "route"]:
    days_until_start = (state["trip_setup"]["start_date"] - dt.date.today()).days
    if 0 <= days_until_start < FORECAST_HORIZON_DAYS:
        return "weather"
    return "route"


def weather_node(state: AgentState) -> dict:
    ranked = state.get("ranked_attractions", [])
    if not ranked:
        # No coordinates to anchor a forecast on (attractions lookup came up
        # empty) -- nothing useful to call the weather API with.
        return {"weather_forecast": []}

    lat = sum(r.attraction.lat for r in ranked) / len(ranked)
    lng = sum(r.attraction.lng for r in ranked) / len(ranked)

    client = get_weather_provider(state.get("destination_country"))

    try:
        forecast = client.get_daily_forecast(lat, lng, lang=openweather_lang(state["trip_setup"].get("lang", "en")))
    except WeatherClientError as exc:
        logger.warning(
            "weather lookup failed for %s at (%.4f, %.4f): %s",
            state["trip_setup"]["destination"],
            lat,
            lng,
            exc,
        )
        errors = list(state.get("errors", []))
        errors.append(f"weather: {exc}")
        return {"weather_forecast": [], "errors": errors}

    return {"weather_forecast": forecast}
