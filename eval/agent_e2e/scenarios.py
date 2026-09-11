"""Synthetic trip scenarios for the agent end-to-end evaluation (see
run_eval.py).

Places/weather/route responses are fully synthetic and mocked via respx, so
results are reproducible and don't depend on live Places/Weather/Route API
access -- the LLM call inside itinerary_node is the only real, non-deterministic
part of a run. Each scenario is built to stress a specific piece of
`check_constraints_node`'s logic (pace, budget) or `itinerary_node`'s weather
handling, rather than being a random sample.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app.graph.state import BudgetLevel, DailyPace, TripSetup
from app.tools.weather_client import FORECAST_HORIZON_DAYS


@dataclass(frozen=True)
class PlaceSpec:
    place_id: str
    price_level: str
    categories: list[str]
    rating: float = 4.3
    # Coordinates and opening hours drive the deterministic scheduler
    # (app/graph/scheduling.py). Defaults: co-located and open all day, so
    # scenarios that don't care about geography/hours behave as before.
    lat: float = 0.0
    lng: float = 0.0
    open_min: int | None = None  # clock-minutes since midnight, every weekday
    close_min: int | None = None


_KM_IN_DEGREES = 0.009  # ~degrees latitude per km


def _places(
    prefix: str,
    n: int,
    price_level: str,
    categories: list[str],
    rating: float = 4.3,
    *,
    open_min: int | None = None,
    close_min: int | None = None,
    spread_km: float = 0.0,
    base_lat: float = 0.0,
) -> list[PlaceSpec]:
    """`n` places, all at (base_lat, 0) unless `spread_km` strings them along
    a line that many km long. `base_lat` puts a group in its own area so two
    groups can sit a fixed distance apart. `open_min`/`close_min` apply to
    all of them."""
    out = []
    for i in range(n):
        offset = (i / max(n - 1, 1)) * spread_km * _KM_IN_DEGREES
        out.append(
            PlaceSpec(
                f"{prefix}{i}",
                price_level,
                categories,
                rating,
                lat=base_lat + offset,
                lng=0.0,
                open_min=open_min,
                close_min=close_min,
            )
        )
    return out


@dataclass(frozen=True)
class Scenario:
    name: str
    destination: str
    daily_pace: DailyPace
    budget_level: BudgetLevel
    interests: list[str]
    num_days: int
    start_offset_days: int  # from today; >= FORECAST_HORIZON_DAYS skips the weather node
    forecast_condition: str  # OpenWeatherMap "main" condition applied to every forecast day
    places: list[PlaceSpec]
    day_start_time: dt.time = dt.time(9, 0)
    day_end_time: dt.time = dt.time(18, 0)

    def trip_setup(self) -> TripSetup:
        start = dt.date.today() + dt.timedelta(days=self.start_offset_days)
        end = start + dt.timedelta(days=self.num_days - 1)
        return TripSetup(
            destination=self.destination,
            start_date=start,
            end_date=end,
            interests=self.interests,
            daily_pace=self.daily_pace,
            budget_level=self.budget_level,
            day_start_time=self.day_start_time,
            day_end_time=self.day_end_time,
        )

    @property
    def forecast_days(self) -> int:
        """How many of this trip's days actually get a forecast entry."""
        if self.start_offset_days >= FORECAST_HORIZON_DAYS:
            return 0
        return min(self.num_days, FORECAST_HORIZON_DAYS - self.start_offset_days)


SCENARIOS: list[Scenario] = [
    Scenario(
        name="relaxed_budget_clear",
        destination="Porto",
        daily_pace="relaxed",
        budget_level="budget",
        interests=["museums"],
        num_days=1,
        start_offset_days=2,
        forecast_condition="Clear",
        places=_places("cheap", 6, "PRICE_LEVEL_INEXPENSIVE", ["museum", "tourist_attraction"], rating=4.5),
    ),
    Scenario(
        name="packed_luxury_rain",
        destination="Vienna",
        daily_pace="packed",
        budget_level="luxury",
        interests=["fine dining", "art"],
        num_days=1,
        start_offset_days=1,
        forecast_condition="Rain",
        places=(
            _places("gal", 4, "PRICE_LEVEL_VERY_EXPENSIVE", ["art_gallery"], rating=4.7)
            + _places("park", 4, "PRICE_LEVEL_EXPENSIVE", ["park", "tourist_attraction"], rating=4.4)
        ),
    ),
    Scenario(
        name="moderate_budget_scarce_cheap",
        destination="Zurich",
        daily_pace="moderate",
        budget_level="budget",
        interests=["sightseeing"],
        num_days=1,
        start_offset_days=3,
        forecast_condition="Clear",
        places=(
            _places("exp", 6, "PRICE_LEVEL_VERY_EXPENSIVE", ["tourist_attraction"], rating=4.6)
            + _places("cheap", 2, "PRICE_LEVEL_INEXPENSIVE", ["museum"], rating=4.0)
        ),
    ),
    Scenario(
        name="relaxed_luxury_outdoor_only",
        destination="Reykjavik",
        daily_pace="relaxed",
        budget_level="luxury",
        interests=["nature", "hiking"],
        num_days=2,
        start_offset_days=2,
        forecast_condition="Clear",
        places=_places("out", 6, "PRICE_LEVEL_VERY_EXPENSIVE", ["park", "tourist_attraction"], rating=4.8),
    ),
    Scenario(
        name="packed_moderate_3day_mixed",
        destination="Barcelona",
        daily_pace="packed",
        budget_level="moderate",
        interests=["general sightseeing"],
        num_days=3,
        start_offset_days=2,
        forecast_condition="Clear",
        places=(
            _places("mus", 4, "PRICE_LEVEL_INEXPENSIVE", ["museum"], rating=4.2)
            + _places("mod", 4, "PRICE_LEVEL_MODERATE", ["tourist_attraction"], rating=4.3)
            + _places("prk", 4, "PRICE_LEVEL_FREE", ["park"], rating=4.1)
        ),
    ),
    Scenario(
        name="moderate_budget_beyond_horizon",
        destination="Krakow",
        daily_pace="moderate",
        budget_level="budget",
        interests=["history"],
        num_days=1,
        start_offset_days=10,
        forecast_condition="Clear",  # unused: trip starts past the forecast horizon
        places=_places("cheap", 5, "PRICE_LEVEL_INEXPENSIVE", ["museum", "tourist_attraction"], rating=4.4),
    ),
    Scenario(
        name="packed_budget_rain_no_indoor_options",
        destination="Queenstown",
        daily_pace="packed",
        budget_level="budget",
        interests=["outdoors"],
        num_days=1,
        start_offset_days=1,
        forecast_condition="Rain",
        places=_places("out", 6, "PRICE_LEVEL_INEXPENSIVE", ["park", "tourist_attraction"], rating=4.3),
    ),
    Scenario(
        name="relaxed_moderate_6day_partial_forecast",
        destination="Kyoto",
        daily_pace="relaxed",
        budget_level="moderate",
        interests=["temples", "gardens"],
        num_days=6,
        start_offset_days=1,
        forecast_condition="Clouds",
        places=_places("mix", 8, "PRICE_LEVEL_MODERATE", ["museum", "tourist_attraction"], rating=4.5),
    ),
    Scenario(
        name="moderate_luxury_nightlife",
        destination="Berlin",
        daily_pace="moderate",
        budget_level="luxury",
        interests=["nightlife", "shopping"],
        num_days=1,
        start_offset_days=2,
        forecast_condition="Clear",
        places=(
            _places("bar", 4, "PRICE_LEVEL_EXPENSIVE", ["shopping_mall"], rating=4.2)
            + _places("club", 4, "PRICE_LEVEL_VERY_EXPENSIVE", ["tourist_attraction"], rating=4.5)
        ),
    ),
    Scenario(
        name="packed_luxury_2day_dense_pool",
        destination="Tokyo",
        daily_pace="packed",
        budget_level="luxury",
        interests=["food", "culture"],
        num_days=2,
        start_offset_days=3,
        forecast_condition="Clear",
        places=(
            _places("a", 3, "PRICE_LEVEL_FREE", ["park"], rating=4.0)
            + _places("b", 3, "PRICE_LEVEL_INEXPENSIVE", ["museum"], rating=4.2)
            + _places("c", 2, "PRICE_LEVEL_MODERATE", ["tourist_attraction"], rating=4.4)
            + _places("d", 2, "PRICE_LEVEL_EXPENSIVE", ["art_gallery"], rating=4.6)
        ),
    ),
    Scenario(
        name="relaxed_budget_barely_feasible",
        destination="Geneva",
        daily_pace="relaxed",
        budget_level="budget",
        interests=["sightseeing"],
        num_days=1,
        start_offset_days=2,
        forecast_condition="Clear",
        places=(
            _places("cheap", 1, "PRICE_LEVEL_FREE", ["park"], rating=4.0)
            + _places("exp", 7, "PRICE_LEVEL_VERY_EXPENSIVE", ["tourist_attraction"], rating=4.7)
        ),
    ),
    Scenario(
        name="moderate_moderate_baseline",
        destination="Lisbon",
        daily_pace="moderate",
        budget_level="moderate",
        interests=["history", "food"],
        num_days=2,
        start_offset_days=2,
        forecast_condition="Clear",
        places=(
            _places("free", 2, "PRICE_LEVEL_FREE", ["park"], rating=4.0)
            + _places("cheap", 2, "PRICE_LEVEL_INEXPENSIVE", ["museum"], rating=4.3)
            + _places("mid", 2, "PRICE_LEVEL_MODERATE", ["tourist_attraction"], rating=4.4)
            + _places("high", 2, "PRICE_LEVEL_EXPENSIVE", ["art_gallery"], rating=4.6)
        ),
    ),
    # -- scheduler stress cases (the failure this eval was extended to cover) --
    Scenario(
        name="packed_late_opening_landmarks",
        destination="London",
        daily_pace="packed",
        budget_level="moderate",
        interests=["landmarks"],
        num_days=1,
        start_offset_days=2,
        forecast_condition="Clouds",
        # Day starts 09:00; the marquee attractions don't open until 10:00-11:00.
        # An LLM assigning its own times schedules them at 09:00 anyway.
        places=(
            _places("wheel", 2, "PRICE_LEVEL_EXPENSIVE", ["tourist_attraction"], rating=4.6, open_min=11 * 60, close_min=18 * 60)
            + _places("zoo", 2, "PRICE_LEVEL_MODERATE", ["zoo"], rating=4.5, open_min=10 * 60, close_min=17 * 60)
            + _places("park", 3, "PRICE_LEVEL_FREE", ["park"], rating=4.4)
        ),
    ),
    Scenario(
        name="packed_spread_out_city",
        destination="Los Angeles",
        daily_pace="packed",
        budget_level="moderate",
        interests=["sightseeing"],
        num_days=1,
        start_offset_days=2,
        forecast_condition="Clear",
        # Candidates strung out over ~18 km -- picking a coherent subset and
        # ordering it well is the whole problem here.
        places=_places("spot", 8, "PRICE_LEVEL_MODERATE", ["tourist_attraction", "museum"], rating=4.4, spread_km=18.0),
    ),
    Scenario(
        name="packed_opening_hours_gauntlet",
        destination="Amsterdam",
        daily_pace="packed",
        budget_level="moderate",
        interests=["history"],
        num_days=1,
        start_offset_days=2,
        forecast_condition="Clouds",
        # A day where the *order* is forced by hours: the park is always open,
        # the museum opens 10:00, the tower only 12:00-15:00 (must be midday),
        # the gallery shuts at 16:00. Assign times naively and at least two
        # land outside hours.
        places=(
            _places("park", 2, "PRICE_LEVEL_FREE", ["park"], rating=4.3)
            + _places("mus", 2, "PRICE_LEVEL_INEXPENSIVE", ["museum"], rating=4.5, open_min=10 * 60, close_min=17 * 60)
            + _places("tower", 1, "PRICE_LEVEL_MODERATE", ["landmark"], rating=4.6, open_min=12 * 60, close_min=15 * 60)
            + _places("gal", 2, "PRICE_LEVEL_MODERATE", ["art_gallery"], rating=4.4, open_min=9 * 60, close_min=16 * 60)
        ),
    ),
    Scenario(
        name="moderate_two_neighborhoods",
        destination="Rome",
        daily_pace="moderate",
        budget_level="moderate",
        interests=["sightseeing"],
        num_days=1,
        start_offset_days=2,
        forecast_condition="Clear",
        # Two clusters ~13 km apart. The right plan stays in one; a plan that
        # mixes them spends the day in transit (gap-too-short violations).
        places=(
            _places("north", 4, "PRICE_LEVEL_MODERATE", ["museum", "tourist_attraction"], rating=4.5, base_lat=0.0)
            + _places("south", 4, "PRICE_LEVEL_MODERATE", ["museum", "tourist_attraction"], rating=4.4, base_lat=0.117)
        ),
    ),
    Scenario(
        name="relaxed_early_closing_highlight",
        destination="Florence",
        daily_pace="relaxed",
        budget_level="moderate",
        interests=["art"],
        num_days=1,
        start_offset_days=2,
        forecast_condition="Clear",
        # The top-rated gallery closes at 13:00, so it has to be the morning
        # stop; rank order would put it second, after 13:00.
        places=(
            _places("gallery", 1, "PRICE_LEVEL_MODERATE", ["art_gallery"], rating=4.9, open_min=9 * 60, close_min=13 * 60)
            + _places("allday", 3, "PRICE_LEVEL_INEXPENSIVE", ["museum", "church"], rating=4.4)
        ),
    ),
    Scenario(
        name="moderate_everything_opens_at_ten",
        destination="Copenhagen",
        daily_pace="moderate",
        budget_level="moderate",
        interests=["design"],
        num_days=1,
        start_offset_days=2,
        forecast_condition="Clouds",
        day_start_time=dt.time(9, 0),
        # Day starts 09:00 but nothing opens before 10:00 -- a naive first
        # stop at 09:00 is always an opening-hours violation.
        places=_places("shop", 6, "PRICE_LEVEL_MODERATE", ["shopping_mall", "museum"], rating=4.4, open_min=10 * 60, close_min=18 * 60),
    ),
    Scenario(
        name="packed_museum_quarter_plus_outliers",
        destination="Berlin",
        daily_pace="packed",
        budget_level="moderate",
        interests=["museums"],
        num_days=1,
        start_offset_days=2,
        forecast_condition="Clear",
        # Five museums on one "island" (co-located, ~0 travel between them)
        # plus three attractions 10 km out. The tight plan takes the cluster.
        places=(
            _places("isl", 5, "PRICE_LEVEL_INEXPENSIVE", ["museum"], rating=4.6, base_lat=0.0)
            + _places("out", 3, "PRICE_LEVEL_MODERATE", ["tourist_attraction"], rating=4.3, base_lat=0.09)
        ),
    ),
]
