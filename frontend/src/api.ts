import type { Lang } from './i18n'
import { translate } from './i18n'
import type { ChatResponse, TripPlanResponse, TripSetupRequest } from './types'

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

// human_review_node (see backend/app/graph/nodes/human_review.py) can pause
// a run instead of returning a finished plan, when a constraint violation
// survived the refine loop and needs a traveller decision. There's no UI for
// that decision -- we always take "accept": finalize_node already turns the
// unresolved violation into a plain-language note (refine_limit_reached in
// app/i18n.py) shown in the results view's advisory banner, so silently
// accepting and letting the note explain "why" gets the same outcome as
// asking, without a modal.
interface TripPlanInterrupt {
  status: 'needs_input'
  session_id: string
  reason: string
  violations: string[]
  options: string[]
}

function isInterrupt(data: TripPlanResponse | TripPlanInterrupt): data is TripPlanInterrupt {
  return (data as TripPlanInterrupt).status === 'needs_input'
}

export async function fetchTripPlan(request: TripSetupRequest, lang: Lang): Promise<TripPlanResponse> {
  const timeoutMs = BASE_TIMEOUT_MS + tripDayCount(request) * PER_DAY_TIMEOUT_MS
  const data = await postTripPlan(`${API_BASE_URL}/trip-plan`, request, timeoutMs, lang)
  return isInterrupt(data) ? resumeTripPlan(data.session_id, lang) : data
}

// Resumes a paused run with a hardcoded "accept" -- see the TripPlanInterrupt
// comment above for why there's no other decision to make here. A resume
// only re-enters the graph from itinerary_node onward with human_decision
// already set to "accept", so should_resume_after_review (human_review.py)
// goes straight to finalize -- no LLM/day-planning work, so BASE_TIMEOUT_MS
// alone is generous.
async function resumeTripPlan(sessionId: string, lang: Lang): Promise<TripPlanResponse> {
  const data = await postTripPlan(
    `${API_BASE_URL}/trip-plan/${sessionId}/resume`,
    { decision: 'accept' },
    BASE_TIMEOUT_MS,
    lang,
  )
  if (isInterrupt(data)) {
    // Shouldn't happen: an "accept" decision can't route back to
    // human_review. Guard against a backend change silently looping forever
    // rather than assuming it's safe to resume again.
    throw new TripPlanError(translate(lang, 'genericError'))
  }
  return data
}

async function postTripPlan(
  url: string,
  body: unknown,
  timeoutMs: number,
  lang: Lang,
): Promise<TripPlanResponse | TripPlanInterrupt> {
  let response: Response
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
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

  return (await response.json()) as TripPlanResponse | TripPlanInterrupt
}

// A single Places lookup, not the full attraction search /trip-plan runs --
// a short timeout is enough, and keeps a hung check from leaving the field
// stuck on "Checking…" for as long as the full-plan timeout would allow.
const VALIDATE_DESTINATION_TIMEOUT_MS = 10_000

// Fast standalone destination check for the setup form's blur handler (see
// TripSetupForm.tsx) -- returns the localized "not found" message to show
// inline, or null when the destination is fine OR the check itself couldn't
// be completed (network/timeout/server error). Deliberately never throws:
// this is a nice-to-have early warning, not a hard gate -- submitting the
// full form still re-validates the destination for real either way.
export async function validateDestination(destination: string, lang: Lang): Promise<string | null> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/validate-destination`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ destination, lang }),
      signal: AbortSignal.timeout(VALIDATE_DESTINATION_TIMEOUT_MS),
    })
  } catch {
    return null
  }
  if (!response.ok) return null
  const data = (await response.json()) as { valid: boolean; message: string | null }
  return data.valid ? null : data.message
}

// The ReAct edit agent's step count (see backend/app/agent/graph.py's
// MAX_AGENT_STEPS) is data-dependent, not day-count-dependent like
// /trip-plan, so this is a flat generous bound rather than a computed one:
// worst case a handful of tool calls plus a couple of gpt-4o-mini turns,
// each bounded by the backend's own 30s LLM timeout.
const CHAT_TIMEOUT_MS = 90_000

export async function fetchChatReply(sessionId: string, message: string, lang: Lang): Promise<ChatResponse> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/trip-plan/${sessionId}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: AbortSignal.timeout(CHAT_TIMEOUT_MS),
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

  return (await response.json()) as ChatResponse
}
