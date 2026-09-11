# Trip Planner Agent

[中文说明](README.zh.md)

A single-city trip planning app. The user enters a destination and their preferences; a backend LangGraph agent dynamically calls attraction/weather/route tools, ranks candidate attractions via embedding-based retrieval, synthesizes a day-by-day itinerary, and iteratively refines it whenever the plan violates the user's stated pace or budget constraints.

The bilingual (English / 中文) UI targets the outbound-travel case — a Chinese-speaking traveller planning a trip to an international city — which the international data stack below serves well. Mainland-China destinations are detected and flagged as low-coverage; a domestic-provider path is designed in [`docs/adr-002-china-provider.md`](docs/adr-002-china-provider.md) but not built.

## Architecture

```
React (Vite + TS)
  Trip Setup ──────────┐
                        ▼
              FastAPI  /trip-plan
                        │
                  LangGraph agent
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
  Attractions Tool   Route Tool     Weather Tool
  (Google Places)     (OSRM)      (OpenWeatherMap)
        │
        ▼
  Recommendation (embedding retrieval + ranking)
```

**LangGraph earns its place here by handling three things a linear pipeline can't** (it's not just decoration):
1. **Dynamic tool selection** — decides whether to call a given tool based on the trip's dates/settings (e.g. skips the weather lookup entirely when the trip starts more than 5 days out, since OpenWeatherMap's free tier only covers a 5-day forecast horizon).
2. **Retry with graceful degradation** — automatically widens the search radius and retries when too few attraction candidates come back (see `_RADIUS_LADDER_M` in `app/tools/places_client.py`).
3. **A bounded constraint-checking loop** — after the itinerary is synthesized, checks whether it violates the user's stated daily pace/budget; if it does, loops back and re-plans, up to 2 iterations.

**Where the LLM stops and Python starts.** The itinerary LLM only *picks* which attractions go on a day and roughly how long to spend at each. It does **not** assign arrival times or visit order — that's arithmetic (line each visit up with the place's opening hours, leave room for travel between stops, order to minimise transit), which an LLM does unreliably and won't fix on a refine pass. A deterministic scheduler (`app/graph/scheduling.py`, a small travelling-salesman-with-time-windows search) does that, drops the lowest-ranked stop when a day can't fit them all, and leaves a lunch break in the middle of the day (a fixed-length window — no restaurant recommended, that's a separate problem). Travel time between stops starts from an OSRM distance matrix (straight-line fallback for uncovered pairs) and is adjusted to a realistic door-to-door figure — OSRM's public server has no traffic model, and a tourist crossing a city takes transit, not a point-to-point drive. The LLM is given a per-pace *time budget* (fill ~5.5–7 hours of a moderate day, etc.) and realistic per-category visit durations, so days don't come back half-empty.

The recommendation module is **embedding-based retrieval ranking**, not a trained model: candidate attraction descriptions and the user's stated preferences are each embedded (a multilingual sentence-transformer, so a Chinese preference string and Chinese place descriptions land in the same space), scored by cosine similarity, and blended with a normalized rating. Offline evaluation lives under `eval/offline_recsys/`, using a subset of the Yelp Open Dataset to compute NDCG@k / Precision@k against a "sort by rating" baseline.

Agent end-to-end evaluation lives under `eval/agent_e2e/` ([details + full results](eval/agent_e2e/README.md)): synthetic scenarios (Places/Weather/Route mocked, real gpt-4o-mini calls kept), run as an **ablation** over the three feasibility pieces — refine loop, deterministic scheduler, real OSRM matrix — each toggled independently. Across 19 hand-built scenarios, constraint-satisfaction rate goes from **0.53 (LLM assigns its own times, no retry) to 1.00 with the deterministic scheduler**, and mean refine passes drop to **0.00**. The cases the scheduler closes and the refine loop never can — an opening-hours-forced visit order, travel between neighbourhoods, a 09:00 start against a 10:00 opening — are arithmetic, not phrasing. (1.00 means every constructed failure mode is covered, not that any real trip passes. The matrix variant shows no separate gain because the eval mocks OSRM's `table` service with a haversine calculation — its real value, road-aware routing, needs a live server to measure.) Building the eval also surfaced four problems unit tests missed — including a window-utilisation check that turned out to be the wrong idea and was removed — see its README.

## Scope

- **Single destination city** per trip; multi-city / multi-leg trips are future work.
- **Destinations outside mainland China** are the supported case — the data stack (Google Places, OpenWeatherMap, OSRM) is strong there. Mainland-China destinations still return an itinerary but are flagged as low-coverage in the response `notes`; Hong Kong / Macau / Taiwan are not flagged. The domestic-provider design is [`docs/adr-002-china-provider.md`](docs/adr-002-china-provider.md); the tool-provider seam it plugs into is [`docs/adr-001-provider-abstraction.md`](docs/adr-001-provider-abstraction.md).

## Current Progress

- [x] Project scaffolding (backend / frontend / docker-compose)
- [x] Three external tool clients (Places / Weather / Route) + unit tests
- [x] Recommendation module (embedding-based retrieval ranking, with ratings Bayesian-shrunk by review count so "3 five-star reviews" can't outrank "thousands of 4.6-star reviews") + offline evaluation (see `eval/offline_recsys/`; alpha=0.7 improved NDCG@10 roughly 4.2x over a ratings-only baseline — measured with the earlier English-only embedding model, pending a re-run after the multilingual swap)
- [x] LangGraph agent core (`backend/app/graph/`, `POST /trip-plan`)
- [x] Agent end-to-end evaluation (see `eval/agent_e2e/`, synthetic scenarios + real LLM calls, refine loop vs. single-shot comparison)
- [x] Frontend pages (Trip Setup / Trip Plan, `frontend/src/`, talks to the backend via `POST /trip-plan`)

## Local Development

### Backend

```bash
cd backend
python3 -m venv .venv          # requires Python 3.13
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # core deps (API + tool clients)
pip install -r requirements-agent.txt    # adds LangGraph / recommendation deps (larger install)

cp ../.env.example ../.env  # fill in real API keys
pytest -q
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Docker (not yet verified end-to-end)

```bash
docker compose up --build
```

## Getting API Keys

| Service | Purpose | Card required |
|---|---|---|
| [OpenAI](https://platform.openai.com/api-keys) | LangGraph agent LLM (gpt-4o-mini) | Yes |
| [Google Places API (New)](https://console.cloud.google.com/apis/library/places-backend.googleapis.com) | Attraction search | Yes (has a free tier) |
| [OpenWeatherMap](https://openweathermap.org/api) | Weather forecasts | No (free tier) |
| OSRM public demo server | Route planning | No key needed |
