"""MCP server exposing this project's three external tool clients (Places /
Weather / Route) as MCP tools, so any MCP client -- Claude Desktop, another
agent, `scripts/mcp_client_demo.py` -- can call them directly, not just this
project's own LangGraph nodes.

The `Protocol` abstraction in app/tools/providers.py already normalizes each
provider's inputs and outputs (`Attraction` / `DailyForecast` /
`DistanceMatrix`), so this is a thin declarative wrapper, not a
reimplementation -- the real API calls and response parsing still live in
app/tools/*_client.py and are unchanged by this file existing.

Run: python -m app.mcp_server   (stdio transport, for a local MCP client)
"""

from __future__ import annotations

from dataclasses import asdict

from mcp.server.mcpserver import MCPServer

from app.tools.providers import get_places_provider, get_route_provider, get_weather_provider
from app.tools.route_client import Coordinate

mcp = MCPServer(name="trip-planner-tools")


@mcp.tool()
def search_attractions(
    city: str, interests: list[str], max_results: int = 20, language_code: str = "en"
) -> list[dict]:
    """Search for tourist attractions/venues in `city` matching `interests` (e.g. ["museums", "food"]; empty
    list searches general sightseeing). Automatically widens the search radius if too few results come back.
    Returns each match's place_id, name, address, coordinates, rating, price level, categories and opening
    hours (when Google has them)."""
    provider = get_places_provider()
    attractions = provider.search_attractions(city, interests, max_results=max_results, language_code=language_code)
    return [asdict(a) for a in attractions]


@mcp.tool()
def get_weather_forecast(lat: float, lng: float, lang: str = "en") -> list[dict]:
    """Daily weather forecast for a coordinate (use search_attractions first to get one -- this provider takes
    coordinates, not a city name). Only covers roughly the next 5 days (the provider's free-tier horizon);
    returns fewer days, or an empty list, for anything further out."""
    provider = get_weather_provider()
    return [asdict(f) for f in provider.get_daily_forecast(lat, lng, lang=lang)]


@mcp.tool()
def get_distance_matrix(place_ids: list[str], coordinates: list[list[float]]) -> dict:
    """Pairwise driving-time matrix between points, `coordinates[i] = [lat, lng]` corresponding to
    `place_ids[i]`. Returns raw free-flow driving minutes between every pair (`null` for an unroutable pair) --
    not a realistic door-to-door estimate; this project's scheduler applies that correction (congestion +
    transit-access time) separately, on the caller's side."""
    provider = get_route_provider()
    stops = [Coordinate(lat=lat, lng=lng) for lat, lng in coordinates]
    matrix = provider.get_distance_matrix(place_ids, stops)
    return {
        "place_ids": matrix.place_ids,
        "duration_minutes": [[d / 60 if d is not None else None for d in row] for row in matrix.duration_s],
    }


if __name__ == "__main__":
    mcp.run()
