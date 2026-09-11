import type { Lang } from './i18n'
import { translate } from './i18n'
import type { TripPlanResponse, TripSetupRequest } from './types'

// Both `npm run dev` (uvicorn on 8000) and docker-compose (backend port
// published as 8000) expose the API at this origin by default; override via
// a frontend/.env with VITE_API_BASE_URL for other setups.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// Matches the backend's own worst-case bound, derived from its client
// timeouts (not guessed):
//   - BASE covers the once-per-request calls: places_client.py's city-center
//     resolve + up to 3 radius-ladder attempts (4 x 10s httpx timeout) plus
//     weather_client.py and route_client.py (10s each) = 40 + 10 + 10 = 60s.
//   - PER_DAY covers itinerary.py: the backend plans one day per LLM call
//     (see its module docstring) and, across the graph's MAX_REFINE_ITERATIONS
//     refine rounds, a single day can be re-planned up to 3 times total (1
//     initial + 2 refine), each capped at ChatOpenAI's 30s timeout -- 3 x 30s
//     = 90s per day, worst case.
// Real runs are almost always far faster than this; the point of a timeout
// is the worst case, not the typical one -- a tighter budget than this
// would cancel a legitimately-still-working backend, not just a hung one.
const BASE_TIMEOUT_MS = 60_000
const PER_DAY_TIMEOUT_MS = 90_000

function tripDayCount(request: TripSetupRequest): number {
  const start = new Date(`${request.start_date}T00:00:00`)
  const end = new Date(`${request.end_date}T00:00:00`)
  const days = Math.round((end.getTime() - start.getTime()) / (24 * 60 * 60 * 1000)) + 1
  return Math.max(days, 1)
}

export class TripPlanError extends Error {}

export async function fetchTripPlan(request: TripSetupRequest, lang: Lang): Promise<TripPlanResponse> {
  const timeoutMs = BASE_TIMEOUT_MS + tripDayCount(request) * PER_DAY_TIMEOUT_MS
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/trip-plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'TimeoutError') {
      throw new TripPlanError(translate(lang, 'errorTimeout'))
    }
    throw new TripPlanError(translate(lang, 'errorNetwork'))
  }

  if (!response.ok) {
    const detail = await response.text()
    throw new TripPlanError(`${translate(lang, 'errorRequestFailedPrefix')} (${response.status}): ${detail || response.statusText}`)
  }

  return (await response.json()) as TripPlanResponse
}
