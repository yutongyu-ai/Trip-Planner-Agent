"""Google Places API (New) client for attraction/venue search.

Docs: https://developers.google.com/maps/documentation/places/web-service/text-search
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

_FIELD_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.addressComponents",
        "places.location",
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
        "places.types",
        "places.editorialSummary",
        "places.regularOpeningHours",
    ]
)

# Radius (meters) escalation ladder used when a search returns too few candidates.
_RADIUS_LADDER_M = [10_000, 25_000, 50_000]
MIN_CANDIDATES = 5


@dataclass(frozen=True)
class Attraction:
    place_id: str
    name: str
    address: str
    lat: float
    lng: float
    rating: float | None
    rating_count: int | None
    price_level: str | None
    categories: list[str]
    description: str | None
    # Weekdays the place is open, using date.weekday() convention (0=Monday
    # .. 6=Sunday). None means Google gave us no hours data -- treat as
    # "unknown, don't exclude it" rather than "closed every day". Derived
    # from (and kept consistent with) opening_hours below.
    open_weekdays: frozenset[int] | None = None
    # date.weekday() -> (open_minute, close_minute), both minutes since that
    # weekday's midnight; close_minute may exceed 1440 for hours that run
    # past midnight (e.g. 20:00-02:00 is (1200, 1560)). A weekday with no
    # entry here but present in open_weekdays shouldn't happen -- both are
    # parsed from the same `regularOpeningHours.periods` response. Only
    # covers the first period Google returns per weekday, so split hours
    # (e.g. a lunch/dinner gap) aren't represented -- a v1 simplification.
    opening_hours: dict[int, tuple[int, int]] | None = None
    # ISO 3166-1 alpha-2 code from the Places addressComponents (e.g. "CN"),
    # used to pick a region-appropriate data provider (see
    # app/tools/providers.py). None if Google returned no country component.
    country_code: str | None = None


class PlacesClientError(RuntimeError):
    """Raised when the Places API call fails or returns malformed data."""


class PlacesClient:
    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=10.0)

    def search_attractions(
        self, city: str, interests: list[str], max_results: int = 20, language_code: str = "en"
    ) -> list[Attraction]:
        """Search attractions in `city` matching `interests`, widening the
        search radius if too few results come back.

        LangGraph's attractions node retries this call at a wider radius
        (see _RADIUS_LADDER_M) instead of failing the whole graph run when a
        small/rural city yields few candidates.

        The search is geographically anchored via `locationBias`, resolved
        from a separate bare-city-name lookup (see `_resolve_city_center`)
        rather than relying on the "<interests> in <city>" query text alone
        to keep Google's text search on the right city -- observed in
        practice: certain interest phrasings (e.g. "中餐 in 上海") make
        Google drop the "in <city>" location entirely and fall back to
        biasing results around this server's own IP location instead.

        `language_code` (a BCP-47 code, e.g. "zh-CN") only changes what
        language Google returns display names/addresses/summaries in -- it
        doesn't affect which places match `query`.
        """
        query = self._build_query(city, interests)
        center = self._resolve_city_center(city, language_code)
        last_results: list[Attraction] = []
        for radius_m in _RADIUS_LADDER_M:
            last_results = self._search_text(query, radius_m, max_results, language_code, center)
            if len(last_results) >= MIN_CANDIDATES:
                return last_results
        return last_results

    def _resolve_city_center(self, city: str, language_code: str) -> tuple[float, float] | None:
        """Resolve `city` to a (lat, lng) anchor point via a bare-name Places
        lookup -- unlike "<interests> in <city>", a query that's just the
        city name on its own reliably resolves to that city's own location
        (verified against several city/interest combinations that otherwise
        misfired). Returns None if the city doesn't resolve to anything or
        the lookup itself fails, so the caller can fall back to an
        unconstrained search rather than failing the whole request over a
        location-bias nicety.
        """
        try:
            results = self._search_text(city, radius_m=None, max_results=1, language_code=language_code, center=None)
        except PlacesClientError:
            return None
        return (results[0].lat, results[0].lng) if results else None

    def _build_query(self, city: str, interests: list[str]) -> str:
        # Deliberately no "attractions" in the interest-driven case: that
        # word biases Google's text search toward generic famous landmarks
        # (see app/recsys/rank.py's docstring on why relevance filtering
        # alone doesn't reliably catch a highly-rated but off-topic result),
        # which then dilutes a themed search (e.g. "food") with unrelated
        # tourist-trap results. Only fall back to "attractions" for the
        # no-interests case, where "general sightseeing" is genuinely what
        # we want.
        if interests:
            return f"{' '.join(interests)} in {city}"
        return f"top attractions in {city}"

    def _search_text(
        self,
        query: str,
        radius_m: int | None,
        max_results: int,
        language_code: str,
        center: tuple[float, float] | None,
    ) -> list[Attraction]:
        payload = {
            "textQuery": query,
            "maxResultCount": min(max_results, 20),
            "languageCode": language_code,
        }
        if center is not None and radius_m is not None:
            lat, lng = center
            payload["locationBias"] = {
                "circle": {"center": {"latitude": lat, "longitude": lng}, "radius": float(radius_m)}
            }
        headers = {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": _FIELD_MASK,
            "Content-Type": "application/json",
        }
        try:
            response = self._client.post(
                PLACES_SEARCH_URL, json=payload, headers=headers
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise PlacesClientError(f"Places API request failed: {exc}") from exc

        data = response.json()
        return [self._parse_place(place) for place in data.get("places", [])]

    @staticmethod
    def _parse_place(place: dict) -> Attraction:
        location = place.get("location", {})
        editorial = place.get("editorialSummary", {})
        opening_hours = PlacesClient._parse_opening_hours(place)
        return Attraction(
            place_id=place["id"],
            name=place.get("displayName", {}).get("text", "Unknown"),
            address=place.get("formattedAddress", ""),
            lat=location.get("latitude", 0.0),
            lng=location.get("longitude", 0.0),
            rating=place.get("rating"),
            rating_count=place.get("userRatingCount"),
            price_level=place.get("priceLevel"),
            categories=place.get("types", []),
            description=editorial.get("text"),
            open_weekdays=frozenset(opening_hours) if opening_hours else None,
            opening_hours=opening_hours,
            country_code=PlacesClient._parse_country(place),
        )

    @staticmethod
    def _parse_country(place: dict) -> str | None:
        for component in place.get("addressComponents", []):
            if "country" in component.get("types", []):
                return component.get("shortText") or component.get("longText")
        return None

    @staticmethod
    def _parse_opening_hours(place: dict) -> dict[int, tuple[int, int]] | None:
        periods = place.get("regularOpeningHours", {}).get("periods")
        if not periods:
            return None
        # Places API's Period.open.day is 0=Sunday..6=Saturday; convert to
        # date.weekday()'s 0=Monday..6=Sunday so callers can compare directly
        # against an itinerary date.
        hours: dict[int, tuple[int, int]] = {}
        for period in periods:
            open_info = period.get("open")
            if open_info is None:
                continue
            weekday = (open_info["day"] - 1) % 7
            open_minute = open_info.get("hour", 0) * 60 + open_info.get("minute", 0)

            close_info = period.get("close")
            if close_info is None:
                close_minute = open_minute + 24 * 60  # open 24h starting this day
            else:
                close_weekday = (close_info["day"] - 1) % 7
                close_minute = close_info.get("hour", 0) * 60 + close_info.get("minute", 0)
                if close_weekday != weekday:
                    # Closes on a later day (e.g. past midnight) -- express
                    # as minutes past the *opening* weekday's midnight
                    # instead of wrapping back to a small number.
                    close_minute += ((close_weekday - weekday) % 7) * 24 * 60

            hours[weekday] = (open_minute, close_minute)  # last period wins if a day has more than one
        return hours or None
