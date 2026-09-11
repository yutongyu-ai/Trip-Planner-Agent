# Improvement Plan

Status: **Parts 1–3 implemented** (see below). Part 3 = the ADR-002 design
doc; the Amap/QWeather code stays deferred. Part 4 optional, not started.

## Goals

1. Stop the silent quality failure on mainland-China destinations — turn it into an
   explicit, honest scope boundary (the same move the app already makes for the
   5-day weather horizon).
2. Make "the domestic-stack migration is understood and designed" visible in the
   repo, not just claimable verbally.
3. Keep a single codebase and keep the bilingual UI (it genuinely serves
   Chinese-speaking users planning outbound/international trips).

## Resolved decisions

| Decision | Choice |
|---|---|
| How the API flags a low-coverage destination | `notes: string[]` — a general, reusable field on `TripPlanResponse` |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, drop-in replacement) |
| Do Part 2 (provider abstraction + ADRs) | Yes |
| Normalize code comments to English | Yes (user-facing i18n stays bilingual) |

---

## Part 1 — Honest scope boundary for mainland China (~1 day)

### 1.1 Detect a mainland-China destination — DONE

- Added `places.addressComponents` to `_FIELD_MASK` in `backend/app/tools/places_client.py`.
- `Attraction` gained `country_code: str | None`; `_parse_place` fills it from
  the `addressComponents` entry whose `types` contains `"country"` (`shortText`,
  e.g. `"CN"`). Simpler than a separate `ResolvedCity` type — the country rides
  on every candidate the search already returns.
- Mainland only: `providers.LOW_COVERAGE_COUNTRIES = {"CN"}`. Hong Kong (`HK`),
  Macau (`MO`), Taiwan (`TW`) stay on the normal path.
- `attractions_node` writes `destination_country: str | None` and
  `low_coverage_destination: bool` into `AgentState`.

### 1.2 Surface the warning to the user — DONE

- New bilingual template `destination_low_coverage(lang)` in `backend/app/i18n.py`.
- `AgentState` gained `notes: list[str]`. `finalize_node` now returns
  `summary` = the trip headline only, and `notes` = the advisory lines
  (partial forecast, low-coverage destination, unresolved constraints), each
  already localized. Cleaner than folding everything into one `summary`
  string; the finalize tests now assert on `notes`.
- API: `/trip-plan` response in `backend/app/main.py` gained `notes` (passed
  through — already localized in finalize).

### 1.3 Multilingual embedding model

- `backend/app/recsys/embed.py`: `MODEL_NAME` →
  `paraphrase-multilingual-MiniLM-L12-v2`.
- Same 384 dimensions, so `rank.py`, the cosine/dot-product path, and both eval
  scripts need no changes.
- Re-run `eval/offline_recsys/run_eval.py` (Yelp data is English) to confirm
  English NDCG/Precision does not regress meaningfully; update the numbers quoted
  in `README.md` if they move.
- Known limitation, to note in the README: there is no Chinese eval set, so the
  Chinese-side improvement cannot be quantified here (that belongs to Part 3).

### 1.4 README scope paragraph

- Reposition as an **outbound / international** trip planner.
- State explicitly that it works best outside mainland China and that mainland
  support is future work; link to `docs/adr-002-china-provider.md`.

### 1.5 Frontend

- Add `notes: string[]` to `TripPlanResponse` in `frontend/src/types.ts`.
- `TripPlanView.tsx` renders `notes` as a visible banner at the top of the plan
  (not buried in the summary text).
- Add the banner-related copy to `frontend/src/i18n.tsx` as needed.

---

## Part 2 — Make the migration path visible in the repo (~1 day)

### 2.1 Provider abstraction refactor

- Define `Protocol` interfaces in `backend/app/tools/`: `PlacesProvider`,
  `WeatherProvider`, `RouteProvider`.
- The existing `PlacesClient` / `WeatherClient` / `RouteClient` become the
  "international" implementations conforming to those protocols — logic unchanged.
- New `backend/app/tools/providers.py` factory:
  `get_places_provider(country)`, `get_weather_provider(country)`,
  `get_route_provider(country)`. For now every country returns the existing
  implementation.
- Graph nodes call the factory instead of constructing clients directly.
- `Attraction` / `DailyForecast` / `RouteLeg` stay as the normalized boundary
  types (already true — this just formalizes it).
- Value: the seam Part 3 plugs into is real, tested code rather than a promise.

### 2.2 ADR documents

- `docs/adr-001-provider-abstraction.md` — records decision 2.1.
- `docs/adr-002-china-provider.md` — the full deferred design:
  - provider selection by destination country;
  - Amap (高德) Web API for places + routing (native GCJ-02, includes transit);
  - QWeather (和风天气) for weather;
  - coordinate-system isolation rule: never mix GCJ-02 and WGS-84; convert only
    at the display layer if ever needed;
  - what changes (3 new provider classes) and what does not (LangGraph nodes,
    constraint checks, itinerary LLM, recsys);
  - registration constraints (Amap real-name verification, QWeather free tier);
  - eval plan: add mainland-China scenarios to `eval/agent_e2e/`.

### 2.3 Bilingual README

- `README.md` (English, primary) + `README.zh.md` (Chinese), cross-linked at the
  top of each.

---

## Part 3 — Plan C: Amap + QWeather integration (deferred, design only)

- Full content lives in `docs/adr-002-china-provider.md`; no code now.
- Trigger to actually build it: deciding to apply to OTA / LBS companies
  (Ctrip / Amap / Meituan / Didi). Not needed for LLM-focused companies.
- Estimated 2–4 days when triggered.

---

## Part 4 — Adjacent polish (optional, low priority)

- Normalize code comments to English: `backend/requirements-agent.txt`,
  `.env.example` currently carry Chinese comments. User-facing i18n stays
  bilingual. (Done.)
- Deploy live and add a URL to the README.
- Technical write-up / blog post.

---

## Follow-up — Chinese offline-recsys eval set (deferred)

### Why

The embedding model was swapped to `paraphrase-multilingual-MiniLM-L12-v2`
to serve Chinese-language input (a zh preference string vs zh place
descriptions). `eval/offline_recsys/run_eval.py` uses the **Yelp Open
Dataset** — US businesses, English reviews — so it can confirm the English
path did not regress but **cannot measure any Chinese-side gain**. Right
now that gain is only argued (multilingual model, cross-lingual cosine
~0.97 in a smoke test), not quantified.

### What to build

A second, parallel eval — **not a replacement**. Keep the English Yelp eval
as the regression guard for the outbound/international path (the primary
supported case); add a Chinese track alongside it.

```
eval/offline_recsys/
  run_eval.py       # English, Yelp (existing, keep)
  run_eval_zh.py    # Chinese, new
  data/             # + a Chinese dataset
```

Both reuse `app/recsys/rank.py` and `app/recsys/embed.py` unchanged; each
emits its own NDCG@k / Precision@k vs the rating-only baseline. The value of
having both: a model or alpha change shows its trade-off explicitly
(e.g. "English -0.5%, Chinese +12%").

### Dataset options (cheapest first)

1. **Hand-curated small set** — ~12 Chinese personas (interest descriptions)
   + a labelled candidate pool for a few Chinese cities. Small N, but a real
   directional number.
2. **Machine-translate the Yelp eval set to Chinese** — fast; an imperfect
   proxy (tests the model on translationese rather than native Chinese).
3. **A public Chinese review dataset** (Dianping-style scrapes on
   Kaggle / academic sites) — varying quality and licensing.

### When

Not urgent — the model swap is low-risk and the English eval already
confirms no regression. Do this together with Plan C (ADR-002): once a
mainland-China provider path exists, a Chinese eval is the required
acceptance tool for it. Until then it stays a to-do, and the honest line is
"English regression checked; Chinese gain not yet quantified."

---

## Suggested sequencing

1. Part 2.1 (provider abstraction) first — it is a pure refactor with existing
   tests as the safety net, and Part 1 code then lands on the new structure.
2. Part 1 (1.1 → 1.5).
3. Part 2.2 + 2.3 (docs).
4. Part 4 items as time allows.

## Implementation status (2026-09-09)

Done — Parts 1, 2, and 3 (3 = the ADR-002 design doc; Amap/QWeather code
stays deferred):

- `backend/app/tools/providers.py` — new: Protocols + `get_*_provider`
  factories + `LOW_COVERAGE_COUNTRIES` / `is_low_coverage_country`.
- `backend/app/tools/places_client.py` — `addressComponents` in the field
  mask; `Attraction.country_code`; `_parse_country`.
- `backend/app/graph/nodes/{attractions,weather,route}.py` — call the
  factory; attractions records `destination_country` /
  `low_coverage_destination`.
- `backend/app/graph/state.py` — `destination_country`,
  `low_coverage_destination`, `notes`.
- `backend/app/graph/nodes/finalize.py` — `summary` = headline only,
  `notes` = advisory list.
- `backend/app/i18n.py` — `destination_low_coverage`.
- `backend/app/main.py` — `notes` in the response.
- `backend/app/recsys/embed.py` — model →
  `paraphrase-multilingual-MiniLM-L12-v2`.
- `frontend/src/{types.ts,components/TripPlanView.tsx,App.css}` — `notes`
  banner.
- `docs/adr-001-provider-abstraction.md`, `docs/adr-002-china-provider.md` —
  new.
- `README.md` scope section + `README.zh.md` — new.
- `.env.example`, `backend/requirements-agent.txt` — comments → English.
- Tests updated: `test_places_client`, `test_graph_attractions`,
  `test_graph_weather`, `test_graph_route`, `test_graph_finalize`,
  `test_graph`, `test_main`.

Not done:
- Part 4: deploy + URL, blog post.
- `eval/offline_recsys` NDCG re-run on the multilingual model (needs the Yelp
  data files, which are not in the repo) — update the figure quoted in
  `README.md`.
- Chinese offline-recsys eval set — see the "Follow-up" section above.

## Out of scope for this plan

Landing an internship, building a second differentiated project (multi-turn RAG
with tracing, or a Qwen + vLLM serving project), and contributing PRs to an
open-source agent/inference framework. These carry more job-search leverage than
further polishing this project, but they are not changes to this repo.
