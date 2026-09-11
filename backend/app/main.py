import datetime as dt
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.graph import build_graph
from app.graph.state import BudgetLevel, DailyPace, Lang, TripSetup
from app.i18n import localize_error

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

_graph = build_graph()


class TripSetupRequest(BaseModel):
    destination: str
    start_date: dt.date
    end_date: dt.date
    interests: list[str]
    daily_pace: DailyPace
    budget_level: BudgetLevel
    day_start_time: dt.time
    day_end_time: dt.time
    lang: Lang = "en"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/trip-plan")
def create_trip_plan(request: TripSetupRequest) -> dict:
    trip_setup: TripSetup = request.model_dump()
    result = _graph.invoke({"trip_setup": trip_setup})
    errors = result.get("errors", [])
    if errors:
        logger.warning("trip-plan for %s completed with errors: %s", trip_setup["destination"], errors)
    return {
        "itinerary": result.get("itinerary", []),
        "summary": result.get("summary", ""),
        # Already localized in finalize.py (built from trip_setup["lang"]).
        "notes": result.get("notes", []),
        "errors": [localize_error(trip_setup["lang"], e) for e in errors],
    }
