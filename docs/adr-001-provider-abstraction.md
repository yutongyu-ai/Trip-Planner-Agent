# ADR-001: Provider abstraction for the external data tools

## Status

Accepted — implemented.

## Context

The agent calls three external services: Google Places (attraction search),
OpenWeatherMap (forecast), OSRM (routing). Each had its client constructed
directly inside its graph node (`PlacesClient(api_key=...)` etc.).

These three services are good for most of the world but weak for mainland
China (Google is network-blocked there, POI coverage is thin, opening hours
are often missing, the WGS-84 coordinates they return are offset ~50–500 m
from the GCJ-02 datum Chinese maps use, and there is no transit routing).
A future mainland-China path would want different providers (Amap, QWeather)
without rewriting the planning logic.

## Decision

Put each tool behind a `Protocol` and select the implementation through a
factory:

- `app/tools/providers.py` defines `PlacesProvider`, `WeatherProvider`,
  `RouteProvider` and `get_places_provider(country_code)` /
  `get_weather_provider(country_code)` / `get_route_provider(country_code)`.
- The existing `PlacesClient` / `WeatherClient` / `RouteClient` are the
  "international" implementations; their logic is unchanged.
- Graph nodes call the factory instead of constructing a client.
- `country_code` is accepted now but ignored — every country still gets the
  international implementation. It is the seam ADR-002 plugs into.

The normalized boundary types (`Attraction`, `DailyForecast`, `RouteLeg`)
are unchanged, so `check_constraints_node`, the itinerary LLM prompt, and
the recsys module never need to know which provider produced the data.

The destination's country code is discovered opportunistically: the Places
field mask now requests `addressComponents`, and `attractions_node` reads
the country from the returned candidates and stores it on the state
(`destination_country`). `low_coverage_destination` is set when that
country is one the international providers cover poorly
(`providers.LOW_COVERAGE_COUNTRIES`, currently `{"CN"}` — mainland only;
Hong Kong, Macau and Taiwan are fine).

## Consequences

- The China migration (ADR-002) is now a matter of adding three provider
  classes and a branch in the factory, not touching the graph.
- The factory functions are also what `app/mcp_server.py` calls directly to
  expose the same three tools over MCP — a concrete second beneficiary of
  this seam being real code rather than a promise: wrapping them as MCP
  tools needed no changes here, only a declarative `@mcp.tool()` layer on
  top.
- `finalize_node` can emit an honest "this destination is poorly covered"
  advisory (surfaced through the new `notes` field) instead of silently
  returning a low-quality itinerary.
- One extra field in the Places response payload (`addressComponents`).
- The factory reads settings, so node tests patch the factory function
  (`monkeypatch.setattr(node_module, "get_*_provider", ...)`) rather than
  patching `get_settings`.

## Alternatives considered

- **Leave clients constructed in the nodes, add `if country == "CN"`
  branches inline.** Rejected: spreads provider knowledge across three
  nodes and still couples the nodes to concrete response shapes.
- **A separate `resolve_destination()` call to get the country before the
  search.** Rejected: an extra Places request when the country is already
  present on every candidate the search returns.
