import datetime as dt
import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, model_validator

from app.agent.graph import MAX_AGENT_STEPS, build_chat_agent
from app.agent.session import create_session, get_session
from app.graph import build_graph
from app.graph.state import BudgetLevel, DailyPace, Lang, TripSetup
from app.i18n import localize_error, places_language_code
from app.tools.places_client import PlacesClientError
from app.tools.providers import get_places_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Trip Planner Agent API")

# Matches the origin the frontend is served from in both `npm run dev`
# (Vite's default port) and docker-compose (frontend published on the same
# port -- see docker-compose.yml).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# A checkpointer is required for human_review_node's interrupt()/resume to
# work at all (see app/graph/nodes/human_review.py) -- same portfolio-scope
# in-memory simplification as app/agent/session.py's chat sessions: state
# lives only in this process, lost on restart or across multiple workers.
_graph = build_graph(checkpointer=MemorySaver())


# Keeps every day of the trip within reach of candidate_pool_target's
# CANDIDATE_POOL_CAP (see app/graph/nodes/_shared.py) for every pace, not
# just the gentler ones -- at MAX_TRIP_DAYS=5, even "packed" (the highest
# daily_pace, 6 stops/day) needs at most 5*6+6=36 candidates, under the cap
# of 40; a 6th day of packed sightseeing would need 42, over it. The setup
# form enforces this too (so a traveller sees it before submitting, not as a
# rejected request) -- this is the backing check for a client that skips or
# bypasses that.
MAX_TRIP_DAYS = 5


class TripSetupRequest(BaseModel):
    destination: str
    start_date: dt.date
    end_date: dt.date
    interests: list[str]
    preference_notes: str | None = None
    daily_pace: DailyPace
    budget_level: BudgetLevel
    day_start_time: dt.time
    day_end_time: dt.time
    lang: Lang = "en"
    avoid_outdoor_in_bad_weather: bool = False

    @model_validator(mode="after")
    def _check_trip_length(self) -> "TripSetupRequest":
        if (self.end_date - self.start_date).days + 1 > MAX_TRIP_DAYS:
            raise ValueError(f"Trips are limited to {MAX_TRIP_DAYS} days.")
        return self


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class ValidateDestinationRequest(BaseModel):
    destination: str
    lang: Lang = "en"


@app.post("/validate-destination")
def validate_destination(request: ValidateDestinationRequest) -> dict:
    """Fast standalone destination check for the setup form (called on the
    destination field losing focus, see frontend/src/api.ts) -- a single
    bare-name Places lookup, not the full attraction search /trip-plan runs.
    Lets the form flag an unresolvable destination before the traveller
    fills out the rest and waits through the whole planning pipeline; the
    full /trip-plan run still re-validates (via attractions_node's
    "destination_not_found") as the real safety net, since this quick check
    can be skipped (submitting without blurring the field) or itself fail.
    """
    destination = request.destination.strip()
    if not destination:
        return {"valid": True, "message": None}

    client = get_places_provider()
    try:
        exists = client.city_exists(destination, language_code=places_language_code(request.lang))
    except PlacesClientError as exc:
        # A transient/auth problem here shouldn't block someone from typing
        # -- treat it as inconclusive rather than "doesn't exist".
        logger.warning("destination validation failed for %s: %s", destination, exc)
        return {"valid": True, "message": None}

    if exists:
        return {"valid": True, "message": None}
    return {"valid": False, "message": localize_error(request.lang, "destination_not_found")}


def _interrupt_response(session_id: str, result: dict) -> dict:
    """Shape of the response when human_review_node has paused the run --
    see app/graph/nodes/human_review.py for what `payload` contains."""
    payload = result["__interrupt__"][0].value
    return {
        "status": "needs_input",
        "session_id": session_id,
        "reason": payload.get("reason"),
        "violations": payload.get("violations", []),
        "options": payload.get("options", []),
    }


@app.post("/trip-plan")
def create_trip_plan(request: TripSetupRequest) -> dict:
    trip_setup: TripSetup = request.model_dump()
    session_id = str(uuid.uuid4())
    result = _graph.invoke({"trip_setup": trip_setup}, config={"configurable": {"thread_id": session_id}})
    # `result` (the full AgentState, whether finalized or paused at
    # human_review) is kept as this session's live state -- the chat
    # agent's tools (app/agent/tools.py) read and mutate it in place as the
    # traveller edits their itinerary in later /chat turns.
    create_session(result, session_id=session_id)

    if "__interrupt__" in result:
        return _interrupt_response(session_id, result)

    errors = result.get("errors", [])
    if errors:
        logger.warning("trip-plan for %s completed with errors: %s", trip_setup["destination"], errors)
    # A stable tag (checked against the raw, unlocalized errors -- see
    # attractions_node) the frontend can act on, e.g. routing back to the
    # setup form and pointing at the destination field, rather than parsing
    # the localized text in `errors`. None for every other error, which is
    # only ever advisory alongside a real itinerary.
    error_code = "destination_not_found" if "destination_not_found" in errors else None
    return {
        "session_id": session_id,
        "itinerary": result.get("itinerary", []),
        "summary": result.get("summary", ""),
        # Already localized in finalize.py (built from trip_setup["lang"]).
        "notes": result.get("notes", []),
        "errors": [localize_error(trip_setup["lang"], e) for e in errors],
        "error_code": error_code,
    }


class ResumeRequest(BaseModel):
    decision: str


@app.post("/trip-plan/{session_id}/resume")
def resume_trip_plan(session_id: str, request: ResumeRequest) -> dict:
    """Answers a `needs_input` pause from human_review_node -- `decision` is
    one of the `options` that response listed (typically "accept" or
    "relax_budget")."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id -- call /trip-plan first.")

    result = _graph.invoke(Command(resume=request.decision), config={"configurable": {"thread_id": session_id}})
    session.state = result

    if "__interrupt__" in result:
        return _interrupt_response(session_id, result)

    errors = result.get("errors", [])
    trip_setup = result["trip_setup"]
    return {
        "session_id": session_id,
        "itinerary": result.get("itinerary", []),
        "summary": result.get("summary", ""),
        "notes": result.get("notes", []),
        "errors": [localize_error(trip_setup.get("lang", "en"), e) for e in errors],
        # Always None here: a resume only re-enters the graph from
        # itinerary_node onward (see should_resume_after_review), so
        # attractions_node's destination_not_found can't recur -- included
        # anyway to keep this response's shape consistent with /trip-plan's.
        "error_code": None,
    }


class ChatRequest(BaseModel):
    message: str


@app.post("/trip-plan/{session_id}/chat")
def chat_with_itinerary(session_id: str, request: ChatRequest) -> dict:
    """Free-text follow-up edits to an already-generated itinerary (swap a
    stop, ask about weather, drop something). Unlike /trip-plan, the model
    here decides for itself which tool(s) to call and when it's done --
    see app/agent/graph.py."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id -- call /trip-plan first.")

    agent = build_chat_agent(session.state, session.checkpointer)
    result = agent.invoke(
        {"messages": [("user", request.message)]},
        config={"configurable": {"thread_id": session_id}, "recursion_limit": MAX_AGENT_STEPS},
    )
    reply = result["messages"][-1].content
    return {"reply": reply, "itinerary": session.state.get("itinerary", [])}
