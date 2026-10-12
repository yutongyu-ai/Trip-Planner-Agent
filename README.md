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

All three of the above are *fixed* control flow — a rule decides the branch, never the model. **`POST /trip-plan/{session_id}/chat` (`app/agent/`) is the one place in this project where the model itself decides which tool to call and when to stop**: a free-text follow-up ("this museum's too expensive, find something cheaper", "check Wednesday's weather and swap the outdoor stop if it's bad") is handled by a `langgraph.prebuilt.create_react_agent` bound to six tools (`list_itinerary`, `find_alternative`, `swap_stop`, `remove_stop`, `add_stop`, `check_weather` — see `app/agent/tools.py`), looping tool-call → observation → next decision until the request is satisfied. Every tool is a thin, validated wrapper around the same deterministic scheduler the generation pipeline uses (`schedule_day_with_lunch`) — the model still never assigns a time or visit order itself, it only ever picks *which* place_id, the same division of labour as `itinerary_node`. See [`eval/agent_chat/`](eval/agent_chat/README.md) for evaluation of this agent specifically (task success, tool-selection accuracy, invalid-tool-call rate) as opposed to the generation pipeline's own eval below. Known scope limit: an edit only has to satisfy the scheduler's own feasibility (opening hours, travel time, lunch) — it is not re-checked against the traveller's stated pace/budget range the way generation is, so a chat edit can leave a day thinner than the original plan without saying so.

The main generation graph also gets a real **human-in-the-loop pause**, not just the chat agent's autonomy: when `check_constraints_node` hits a violation neither the LLM nor the bounded refine loop can fix by itself — not enough affordable candidates in the destination, or the candidate pool already fully used (`budget_unavoidable` / `pool_exhausted` on `AgentState`) — `human_review_node` (`app/graph/nodes/human_review.py`) pauses the run via LangGraph's `interrupt()` instead of silently downgrading it to an advisory note. `POST /trip-plan` returns `{"status": "needs_input", "reason", "violations", "options"}` in that case; `POST /trip-plan/{session_id}/resume` answers it ("accept" the plan as-is, or "relax_budget" to loosen one tier and give the refine loop another attempt). This needs a checkpointer (`MemorySaver`, wired in `app/main.py`) so the pause survives across what may be a separate HTTP request — the same LangGraph primitive the chat agent uses for its own conversation memory, applied to a different problem.

**Where the LLM stops and Python starts.** The itinerary LLM only *picks* which attractions go on a day and roughly how long to spend at each. It does **not** assign arrival times or visit order — that's arithmetic (line each visit up with the place's opening hours, leave room for travel between stops, order to minimise transit), which an LLM does unreliably and won't fix on a refine pass. A deterministic scheduler (`app/graph/scheduling.py`, a small travelling-salesman-with-time-windows search) does that, drops the lowest-ranked stop when a day can't fit them all, and leaves a lunch break in the middle of the day (a fixed-length window — no restaurant recommended, that's a separate problem). Travel time between stops starts from an OSRM distance matrix (straight-line fallback for uncovered pairs) and is adjusted to a realistic door-to-door figure — OSRM's public server has no traffic model, and a tourist crossing a city takes transit, not a point-to-point drive. The LLM is given a per-pace *time budget* (fill ~5.5–7 hours of a moderate day, etc.) and realistic per-category visit durations, so days don't come back half-empty.

The recommendation module is **embedding-based retrieval ranking**, not a trained model: candidate attraction descriptions and the user's stated preferences are each embedded (a multilingual sentence-transformer, so a Chinese preference string and Chinese place descriptions land in the same space), scored by cosine similarity, and blended with a normalized rating. Offline evaluation lives under `eval/offline_recsys/`, using a subset of the Yelp Open Dataset to compute NDCG@k / Precision@k against a "sort by rating" baseline.

Agent end-to-end evaluation lives under `eval/agent_e2e/` ([details + full results](eval/agent_e2e/README.md)): synthetic scenarios (Places/Weather/Route mocked, real gpt-4o-mini calls kept), run as an **ablation** over the three feasibility pieces — refine loop, deterministic scheduler, real OSRM matrix — each toggled independently. Across 19 hand-built scenarios, constraint-satisfaction rate goes from **0.42 (LLM assigns its own times, no retry) to 1.00 with the deterministic scheduler**, and mean refine passes drop to **0.00**. The cases the scheduler closes and the refine loop never can — an opening-hours-forced visit order, travel between neighbourhoods, a 09:00 start against a 10:00 opening — are arithmetic, not phrasing. (1.00 means every constructed failure mode is covered, not that any real trip passes. The matrix variant shows no separate gain because the eval mocks OSRM's `table` service with a haversine calculation — its real value, road-aware routing, needs a live server to measure.) Building the eval also surfaced four problems unit tests missed — including a window-utilisation check that turned out to be the wrong idea and was removed — see its README.

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
- [x] Conversational itinerary-edit agent (`backend/app/agent/`, `POST /trip-plan/{session_id}/chat`, ReAct + function calling; see `eval/agent_chat/`)
- [x] Human-in-the-loop pause + checkpointing on the main graph (`backend/app/graph/nodes/human_review.py`, `POST /trip-plan/{session_id}/resume`)

## Local Development

### Backend

```bash
cd backend
python3 -m venv .venv          # requires Python 3.13
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # all backend deps (API, tool clients, LangGraph, recommendation)

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

### Docker

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
