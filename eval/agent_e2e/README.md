# Agent end-to-end evaluation

An **ablation**: how much does each of the three pieces that make an
itinerary *feasible* actually contribute?

| # | variant | refine loop | deterministic scheduler | real OSRM matrix |
|---|---|---|---|---|
| 1 | `llm-times, no retry` | – | – | – |
| 2 | `+ refine loop` | ✓ | – | – |
| 3 | `+ scheduler` | ✓ | ✓ | – |
| 4 | `+ osrm matrix (full)` | ✓ | ✓ | ✓ |

Each row adds one component to the row above, so the column-to-column delta
is that component's marginal value.

- **Scheduler off** (`_naive_schedule_day`): the LLM's pick order is kept and
  visits are packed back-to-back from the day start, ignoring opening hours
  and travel time — what an LLM left to assign its own arrival times does.
- **Matrix off**: `route_node` stores no matrix; the scheduler falls back to
  straight-line (haversine) travel estimates.

## Setup

Places / Weather / Route are mocked via `respx` from `scenarios.SCENARIOS`,
so runs are reproducible. The `gpt-4o-mini` call inside `itinerary_node` is
**real** (temperature 0) — needs `OPENAI_API_KEY` in the repo-root `.env`;
one call per scenario per variant.

```
python eval/agent_e2e/run_eval.py
```

`test_metrics.py` covers the pure metric math and smoke-tests the ablation
harness with a scripted LLM (no API key, runs in CI).

## Scenarios

19 synthetic trips. Some stress `check_constraints_node` (pace / budget /
weather) or the multi-day allocation; the rest target the scheduler
directly — opening hours that force an order, geography that forces a
choice:

- `packed_late_opening_landmarks`, `moderate_everything_opens_at_ten` — day
  starts 09:00, attractions open 10:00–11:00; a naive schedule puts them at
  09:00 anyway.
- `packed_opening_hours_gauntlet` — a park (always open), a museum (10:00),
  a tower (12:00–15:00 only), a gallery (closes 16:00): the visit order is
  forced by hours.
- `relaxed_early_closing_highlight` — the top-rated stop closes at 13:00, so
  it can't be second.
- `packed_spread_out_city`, `moderate_two_neighborhoods` — coherent-subset
  and ordering problems; a plan that hops between clusters spends the day in
  transit.
- `packed_museum_quarter_plus_outliers` — five co-located museums vs. three
  10 km out; the tight plan takes the cluster.
- `packed_moderate_3day_mixed`, `relaxed_moderate_6day_partial_forecast` —
  just enough distinct attractions; the per-day LLM must not front-load.

## Metrics

| metric | meaning | better |
|---|---|---|
| `constraint_satisfaction_rate` | fraction of runs with zero violations | higher |
| `refine_iterations` | mean refine passes used | lower |
| `empty_day_rate` | fraction of days with no stops | lower |
| `avg_travel_min_per_day` | between-stop travel time per non-empty day | lower |
| `dropped_stop_rate` | share of picked stops the scheduler couldn't fit | lower¹ |
| `hallucinated_place_id_rate` | LLM place_ids not in the candidate list | lower |
| `bad_weather_indoor_rate` | indoor share of stops on rainy days | higher |

¹ 0.0 by construction when the scheduler is off (it never drops).

## Results

19 scenarios, `gpt-4o-mini` @ temp 0 (run 2026-09-11, single run against the
current scheduler — v1/v2 don't touch the scheduler at all, so expect their
numbers to vary somewhat run to run with the model's picks; these are 19
hand-built scenarios, not a representative sample, and 1.00 means "every
constructed failure mode is covered", not "any real trip will pass"):

| metric | v1 `llm-times` | v2 `+refine` | v3 `+scheduler` | v4 `+matrix` |
|---|---|---|---|---|
| **constraint_satisfaction_rate** | 0.53 | 0.68 | **1.00** | **1.00** |
| refine_iterations (mean) | 0.00 | 0.84 | **0.00** | 0.00 |
| avg_travel_min_per_day | –¹ | –¹ | ~10 | ~10 |
| empty_day_rate | 0.02 | 0.02 | 0.02 | 0.02 |
| hallucinated_place_id_rate | 0.02 | 0.02 | 0.02 | 0.02 |

¹ v1/v2 keep the LLM's pick order unchanged, so their travel figure isn't a
controlled comparison.

**Reading it:**

- **v1 (0.53): 9 of 19 scenarios fail** with the LLM assigning its own
  times and no retry. **v2 (+refine, 0.68): 6 still fail** —
  `packed_late_opening_landmarks`, `packed_spread_out_city`,
  `moderate_two_neighborhoods`, `relaxed_early_closing_highlight`,
  `moderate_everything_opens_at_ten`, `packed_museum_quarter_plus_outliers`.
  Re-prompting the LLM fixes some cases (3 of the 9) but not these six,
  because they're arithmetic problems (an opening-hours-forced order, travel
  between neighbourhoods), not phrasing ones.
- **v2 → v3 (+scheduler): → 1.00, zero remaining violations across all 19
  scenarios** — including `relaxed_budget_barely_feasible`, a "budget
  traveller, expensive city" case that has flaked in earlier runs.
  `refine_iterations` at v3 is **0.00** — once timing, allocation and meal
  breaks are all deterministic, the refine loop is never triggered.
- **v3 → v4 (+matrix): no measurable difference here.** The matrix's real
  value is routing around obstacles a straight line misses (a river with
  few bridges, a motorway), but the eval mocks OSRM's `table` service with
  a haversine calculation, so v3 and v4 are computing nearly the same thing.
  In production the matrix feeds the ordering search real road times; the
  synthetic eval can't quantify that.

Travel times are door-to-door estimates (`realistic_transit_minutes`):
raw free-flow driving, scaled ~1.6× for congestion, plus a fixed ~8-minute
transit-access cost — because OSRM's public server has no traffic model and
a tourist crossing a city takes transit, not a point-to-point drive. The day's
attractions are scheduled as a single travel-minimising route first, then a
lunch break (length set by pace, no restaurant recommended) is slotted into
whichever gap in that route sits nearest 12:00-14:00 — trying the next-nearest
gap instead of giving up if the nearest one turns out infeasible, so lunch is
never dropped over one bad gap, and the route itself is never split into
independent halves that could separate two attractions that happen to sit
right next to each other.

Building this eval caught four problems unit tests couldn't:

1. `ends_too_early` — a check that flagged any day finishing well before a
   "fill the window to X%" target. It fought the tight scheduler on nearly
   every scenario; removed entirely, because window utilisation is a bad
   proxy for a good itinerary. "Too sparse" is now the stop-count check plus
   realistic per-stop dwell minimums plus a *hard-floored* "day underfilled"
   check whose only fix is picking more stops, never padding gaps.
2. The proximity-hint prompt block spammed "~0 min" lines for co-located
   candidates and measurably raised empty-day / hallucination rates
   (`_MIN_PROXIMITY_HINT_MINUTES`).
3. The per-day greedy LLM front-loaded tight multi-day trips, leaving the
   last days empty. Fixed structurally: `effective_hi` in `itinerary_node`
   holds back candidates for later days.
4. Raising the stop-count minimums made two budget scenarios unsatisfiable
   (a "budget" traveller in a city with only expensive attractions). Now
   `check_constraints` only hard-fails budget when there *were* enough
   affordable options; otherwise `finalize` adds an advisory note.

The numbers above are post-fix.
