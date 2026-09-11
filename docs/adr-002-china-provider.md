# ADR-002: Mainland-China data providers (Amap + QWeather)

## Status

Proposed — **not implemented**. Designed now so the trade-offs are on
record; build it only if mainland-China itineraries become a real target
(e.g. applying the project to a domestic OTA / LBS context). Estimated
2–4 days.

## Context

The international providers (Google Places / OpenWeatherMap / OSRM) are the
wrong tool for mainland China:

| Problem | Effect |
|---|---|
| Google services are network-blocked in mainland China | direct calls fail from a China-hosted deployment |
| Sparse POI coverage, missing `editorialSummary`, missing `regularOpeningHours` | weak ranking input, opening-hours constraint checks mostly inactive |
| Google returns WGS-84; Chinese maps and most Chinese APIs use GCJ-02 | ~50–500 m offset if coordinates from different datums are mixed |
| OSRM public demo is driving-only, no transit, thin OSM road data in some cities | routing hint is unrealistic for how people actually move in Chinese cities |
| OpenWeatherMap is less accurate in China than domestic providers | lower-quality weather notes |

ADR-001 already put each tool behind a `Protocol` and a
`get_*_provider(country_code)` factory. `attractions_node` records
`destination_country`; `low_coverage_destination` is `True` for `CN`.

## Decision (proposed)

Add mainland-China implementations and branch the factory on
`country_code == "CN"`.

### Places → Amap (高德开放平台) Web Service API

- POI search: `/v3/place/text` and `/v3/place/around`, keyed by keyword +
  city; radius escalation maps onto the existing `_RADIUS_LADDER_M` idea.
- Detail (opening hours, rating): `/v3/place/detail`.
- Returns GCJ-02 coordinates natively — no conversion needed as long as the
  weather and route providers for CN are also GCJ-02 native (Amap is).
- Map the response into the existing `Attraction` dataclass. `price_level`
  has no clean Amap equivalent; leave `None` and let the budget constraint
  check skip stops with unknown price (it already does).

### Weather → QWeather (和风天气)

- `/v7/weather/3d` (or `7d`) by `location=<lng,lat>`.
- Easier onboarding than Amap: English docs, free "developer" tier
  (~1000 calls/day), no real-name requirement for the free tier at time of
  writing.
- Map `text` / `tempMin` / `tempMax` into `DailyForecast`. Keep a
  provider-specific bad-weather keyword set (the current one keys off
  OpenWeatherMap's English `main` codes).

### Route → Amap Direction API

- `/v3/direction/driving` for parity with the current OSRM behaviour;
  `/v3/direction/transit/integrated` is the better long-term signal for
  city trips and can be added later.
- GCJ-02 in, GCJ-02 out.

### Coordinate-system rule

Each provider owns one datum end to end. International path = WGS-84
throughout; CN path = GCJ-02 throughout. The two never mix inside one
request, because `country_code` picks a single provider set for the whole
run. If a display map is ever added, convert at that layer only.

### What does NOT change

- LangGraph structure, conditional edges, refine loop.
- `check_constraints_node` and every helper in `_shared.py`.
- The itinerary LLM prompt and structured output.
- `recsys` (already multilingual after the embedding-model swap).
- `Attraction` / `DailyForecast` / `RouteLeg` — the CN providers map into
  the same types.

## Consequences

- One codebase serves outbound (Google) and domestic (Amap) trips, switched
  by destination country.
- `tools/` roughly doubles in size (3 new provider classes + response
  mappers + tests).
- Amap individual developer registration requires real-name verification
  (Chinese ID); credit-based free quota.
- New eval scenarios needed in `eval/agent_e2e/` for CN destinations, plus
  ideally a small Chinese offline-recsys set to actually measure the
  multilingual embedding gain (today it is only argued, not measured).

## Alternatives considered

- **Migrate everything to Amap.** Rejected: Amap's non-China POI coverage
  is weak, which would break the outbound-travel use case that the
  international stack serves well.
- **Baidu Maps instead of Amap.** Comparable capability; Baidu uses its own
  BD-09 datum (one more conversion) and its API terms are stricter. Amap is
  the lighter first step; revisit if Baidu POI coverage proves better for
  the target cities.
- **Keep two forks (one per market).** Rejected: double maintenance, and
  the provider switch is a cleaner demonstration of the same capability.
