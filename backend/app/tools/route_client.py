"""OSRM (Open Source Routing Machine) client.

Uses the public demo server by default (no API key required). The graph
only needs *pairwise* travel time between a handful of candidate
attractions -- so it asks for a full duration/distance matrix in one call
(the `table` service) rather than a single through-route:
https://project-osrm.org/docs/v5.24.0/api/#table-service
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Coordinate:
    lat: float
    lng: float


@dataclass(frozen=True)
class DistanceMatrix:
    """Pairwise driving estimates between `place_ids`, indexed the same way
    on both axes. `duration_s[i][j]` is the time from `place_ids[i]` to
    `place_ids[j]`, or `None` if OSRM found no route between them; `distance_m`
    may be empty if the OSRM server doesn't return the distance annotation."""

    place_ids: list[str]
    duration_s: list[list[float | None]]
    distance_m: list[list[float]]

    def minutes_between(self, a: str, b: str) -> float | None:
        """Driving minutes from `a` to `b`, or None if either id isn't in
        the matrix, or OSRM found no route between them at all (its `table`
        service returns `null` for an unroutable pair -- e.g. split by water
        with no bridge in range -- rather than omitting the cell). Either way
        the caller should fall back to a straight-line estimate."""
        try:
            i, j = self._index[a], self._index[b]
        except KeyError:
            return None
        duration = self.duration_s[i][j]
        return duration / 60 if duration is not None else None

    @property
    def _index(self) -> dict[str, int]:
        return {pid: i for i, pid in enumerate(self.place_ids)}


class RouteClientError(RuntimeError):
    """Raised when the OSRM API call fails or returns malformed data."""


class RouteClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=10.0)

    def get_distance_matrix(self, place_ids: list[str], stops: list[Coordinate]) -> DistanceMatrix:
        """Full pairwise driving matrix between `stops` (parallel to
        `place_ids`). One `table` request regardless of how many stops."""
        if len(stops) != len(place_ids):
            raise RouteClientError("place_ids and stops must be the same length")
        if len(stops) < 2:
            return DistanceMatrix(place_ids=list(place_ids), duration_s=[[0.0]] * len(stops), distance_m=[])

        coords = ";".join(f"{stop.lng},{stop.lat}" for stop in stops)
        url = f"{self._base_url}/table/v1/driving/{coords}"
        params = {"annotations": "duration,distance"}

        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RouteClientError(f"OSRM request failed: {exc}") from exc

        data = response.json()
        if data.get("code") != "Ok":
            raise RouteClientError(f"OSRM returned error code: {data.get('code')}")
        if "durations" not in data:
            raise RouteClientError("OSRM table response has no durations")

        return DistanceMatrix(
            place_ids=list(place_ids),
            duration_s=data["durations"],
            distance_m=data.get("distances") or [],
        )
