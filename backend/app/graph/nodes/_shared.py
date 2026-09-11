"""Small helpers shared by more than one graph node."""

from __future__ import annotations

import datetime as dt
import math

from app.recsys.rank import RankedAttraction
from app.tools.places_client import Attraction

_EARTH_RADIUS_KM = 6371.0
# Straight-line distance systematically underestimates real street distance;
# this is a rough correction, not a substitute for a real route.
_ROAD_CIRCUITY_FACTOR = 1.3
# Rough free-flow city driving speed, for the straight-line fallback when the
# OSRM matrix doesn't cover a pair. Both this and the matrix produce "raw
# free-flow driving minutes" -- realistic_transit_minutes turns that into a
# door-to-door estimate.
_FREE_FLOW_CITY_KMH = 28.0

# Raw free-flow driving time badly understates how a traveller actually gets
# between two city attractions: congestion roughly halves real driving speed,
# and most tourists take transit or walk, which carries a fixed cost (walk to
# a stop, wait, walk from it) a point-to-point car route ignores. OSRM's
# public demo server has no traffic model at all, so its times need both
# corrections.
_CONGESTION_FACTOR = 1.6
_TRANSIT_ACCESS_MIN = 8
# At or below this many raw minutes the two places are effectively the same
# spot (a museum quarter, adjacent buildings) -- just walk it, no access cost.
_COLOCATED_MAX_RAW_MIN = 2.0

# Dwell time by Places `types` category, as (min, typical, max) minutes.
# `typical` is only an anchor shown to the LLM -- the real work is done by
# the LLM's own knowledge of the specific place (the British Museum is not a
# one-room local museum, even though Google tags both `museum`). [min, max]
# is the absolute bracket clamped_dwell_minutes holds the LLM's estimate to:
# wide enough for that knowledge to move the number a lot, tight enough that
# a hallucinated "15 min at a national museum" or "4 h at a viewpoint" can't
# silently wreck the day's schedule. _DEFAULT_DWELL_BOUNDS covers a category
# not listed here.
# Category minimums are "the shortest a real visit makes sense" -- if you
# only have 30 minutes for the British Museum, don't go. Keeping them
# realistic stops the LLM from compressing every dwell to the floor to cram
# in more stops, which left days ending at 1pm.
_CATEGORY_DWELL = {
    "museum": (50, 90, 240),
    "art_gallery": (45, 75, 210),
    "aquarium": (60, 90, 150),
    "zoo": (120, 165, 240),
    "amusement_park": (150, 300, 480),
    "theme_park": (150, 300, 480),
    "national_park": (90, 180, 360),
    "park": (30, 60, 150),
    "shopping_mall": (40, 90, 180),
    "tourist_attraction": (20, 45, 150),
    "landmark": (10, 25, 60),
    "church": (15, 30, 75),
    "hindu_temple": (15, 30, 75),
    "mosque": (15, 30, 75),
    "synagogue": (15, 30, 75),
    "restaurant": (45, 75, 120),
    "cafe": (20, 40, 75),
    "bar": (30, 60, 120),
    "night_club": (60, 120, 240),
    "movie_theater": (90, 130, 180),
    "casino": (45, 120, 300),
    "book_store": (15, 30, 90),
    "library": (30, 50, 120),
    "stadium": (75, 120, 210),
}
_DEFAULT_DWELL_BOUNDS = (30, 60, 180)
DEFAULT_DWELL_MINUTES = _DEFAULT_DWELL_BOUNDS[1]

# How many stops a day should have, by pace, and roughly how many hours of
# "visits + travel" a day of that pace should add up to. Both are told to
# the LLM in itinerary.py's prompt and checked in check_constraints_node.
# This, plus each stop's own realistic dwell time, is how "the right amount
# of day" is enforced -- there is deliberately no gap-padding toward a
# window-utilisation ratio (that just makes the itinerary worse).
STOPS_PER_DAY_RANGE = {
    "relaxed": (2, 3),
    "moderate": (3, 5),
    "packed": (4, 6),
}
PACE_ACTIVITY_HOURS = {
    "relaxed": (3.5, 5.0),
    "moderate": (5.5, 7.0),
    "packed": (7.0, 8.5),
}

# Minutes reserved for lunch, by pace -- a relaxed traveller sits down for a
# proper meal, a packed one grabs something quick and keeps moving. Consumed
# by scheduling.schedule_day_with_lunch; kept here (not in scheduling.py,
# which stays agnostic to pace/business concepts -- see its docstring) so
# it lives next to the other two pace tables it's conceptually part of.
LUNCH_DWELL_MINUTES = {
    "relaxed": 90,
    "moderate": 60,
    "packed": 40,
}

# Google Places priceLevel string -> ordinal; and the highest ordinal a
# budget level tolerates.
PRICE_LEVEL_RANK = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}
MAX_PRICE_LEVEL = {"budget": 1, "moderate": 2, "luxury": 4}

_INDOOR_CATEGORY_KEYWORDS = {
    "museum",
    "art_gallery",
    "aquarium",
    "planetarium",
    "library",
    "shopping_mall",
    "movie_theater",
    "casino",
    "book_store",
}


def attraction_lookup(ranked: list[RankedAttraction]) -> dict[str, Attraction]:
    return {r.attraction.place_id: r.attraction for r in ranked}


def is_indoor(attraction: Attraction) -> bool:
    """Best-effort indoor/outdoor guess from Google Places `types` keywords."""
    categories = {c.lower() for c in attraction.categories}
    return bool(categories & _INDOOR_CATEGORY_KEYWORDS)


def time_to_minutes(t: dt.time) -> int:
    return t.hour * 60 + t.minute


def minutes_to_clock(minutes: int) -> str:
    """Inverse of time_to_minutes, tolerant of values >= 1440 (past
    midnight, as opening_hours close_minute can be) or negative -- wraps
    into a 24h clock string rather than raising, since this is only used
    for human-readable violation messages, not further arithmetic."""
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def is_open_on(attraction: Attraction, date: dt.date) -> bool:
    """True if the place is known to be open that weekday, or we have no
    hours data at all (treated as "don't exclude it", not "always open")."""
    if attraction.open_weekdays is None:
        return True
    return date.weekday() in attraction.open_weekdays


def is_open_during(attraction: Attraction, date: dt.date, arrival_min: int, departure_min: int) -> bool:
    """True if [arrival_min, departure_min) (clock minutes since `date`'s own
    midnight) falls entirely within `attraction`'s opening hours for that
    weekday, or we have no hour-level data at all for that weekday (same
    "unknown, don't exclude it" convention as is_open_on -- a weekday
    present in open_weekdays but missing from opening_hours is treated as
    open all day, not closed). Does not by itself check whether the place
    is open on `date` at all -- see is_open_on for that.
    """
    if attraction.opening_hours is None:
        return True
    hours = attraction.opening_hours.get(date.weekday())
    if hours is None:
        return True
    open_min, close_min = hours
    return open_min <= arrival_min and departure_min <= close_min


def _category_dwell_triple(attraction: Attraction) -> tuple[int, int, int]:
    """(min, typical, max) for the first of `attraction`'s Places `types`
    that's in _CATEGORY_DWELL (Google generally lists the most
    specific/primary type first); _DEFAULT_DWELL_BOUNDS otherwise."""
    for category in attraction.categories:
        triple = _CATEGORY_DWELL.get(category.lower())
        if triple is not None:
            return triple
    return _DEFAULT_DWELL_BOUNDS


def category_dwell_minutes(attraction: Attraction) -> int:
    """The typical-visit anchor for `attraction`'s category -- the starting
    point the LLM adjusts from (shown per candidate in itinerary.py's
    prompt), not a hard value."""
    return _category_dwell_triple(attraction)[1]


def dwell_bounds(attraction: Attraction) -> tuple[int, int]:
    """Absolute (min, max) minutes that clamped_dwell_minutes holds the
    LLM's per-stop estimate to for this attraction's category."""
    lo, _, hi = _category_dwell_triple(attraction)
    return lo, hi


def clamped_dwell_minutes(attraction: Attraction, llm_minutes: int) -> int:
    """The LLM's per-stop dwell_minutes estimate (see itinerary.py's
    _LLMStop), held to this attraction's category bracket (see
    _CATEGORY_DWELL) -- keeps the LLM's judgment (the British Museum needs
    hours, a photo-stop landmark doesn't) while rejecting a single
    hallucinated outlier that would break the day's schedule.
    """
    lo, hi = dwell_bounds(attraction)
    return int(max(lo, min(round(llm_minutes), hi)))


def haversine_km(a: Attraction, b: Attraction) -> float:
    lat1, lng1, lat2, lng2 = (math.radians(v) for v in (a.lat, a.lng, b.lat, b.lng))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def realistic_transit_minutes(raw_driving_min: float) -> float:
    """Turn a raw free-flow driving estimate (the OSRM matrix, or the
    straight-line fallback below) into a realistic door-to-door time --
    congestion-scaled plus a fixed transit-access cost -- except for
    effectively-co-located pairs, which are just a walk."""
    if raw_driving_min <= _COLOCATED_MAX_RAW_MIN:
        return raw_driving_min
    return raw_driving_min * _CONGESTION_FACTOR + _TRANSIT_ACCESS_MIN


def estimate_travel_minutes(a: Attraction, b: Attraction) -> float:
    """Straight-line fallback for a pair the OSRM matrix doesn't cover:
    haversine distance, street-circuity corrected, at a free-flow city speed.
    Returns raw driving minutes -- pass through realistic_transit_minutes for
    a door-to-door figure.
    """
    distance_km = haversine_km(a, b) * _ROAD_CIRCUITY_FACTOR
    return (distance_km / _FREE_FLOW_CITY_KMH) * 60
