"""Human-in-the-loop checkpoint between the refine loop and finalize.

check_constraints_node can find a violation the refine loop structurally
cannot fix by itself -- not enough affordable candidates in the destination,
or the candidate pool already fully used (`budget_unavoidable` /
`pool_exhausted` on AgentState). Today that just becomes a silent advisory
note in finalize.py; this node instead pauses the graph and asks the
traveller what to do, via LangGraph's dynamic `interrupt()` -- only when one
of those flags is actually set, so a normal run is never interrupted.

Requires a checkpointer at compile time (see build_graph's `checkpointer`
param): LangGraph persists state at the pause point so a resume -- quite
possibly a separate HTTP request -- can pick the run back up exactly where
it left off. See app/main.py's POST /trip-plan/{session_id}/resume.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import interrupt

from app.graph.state import AgentState

# One tier looser; luxury has nowhere further to go.
_BUDGET_TIER_UP = {"budget": "moderate", "moderate": "luxury", "luxury": "luxury"}


def human_review_node(state: AgentState) -> dict:
    # budget_unavoidable/pool_exhausted only *suppress* a specific violation
    # inside check_constraints_node -- they don't mean anything went wrong by
    # themselves (a small destination can legitimately use its whole
    # candidate pool with a perfectly satisfying itinerary). Only pause when
    # should_refine already gave up with real violations still unresolved.
    if not state.get("constraint_violations"):
        return {}
    if not (state.get("budget_unavoidable") or state.get("pool_exhausted")):
        return {}

    reason = "budget_unavoidable" if state.get("budget_unavoidable") else "pool_exhausted"
    options = ["accept", "relax_budget"] if reason == "budget_unavoidable" else ["accept"]
    decision = interrupt(
        {"reason": reason, "violations": state.get("constraint_violations", []), "options": options}
    )

    if decision == "relax_budget" and reason == "budget_unavoidable":
        trip_setup = dict(state["trip_setup"])
        trip_setup["budget_level"] = _BUDGET_TIER_UP.get(trip_setup["budget_level"], trip_setup["budget_level"])
        # refine_iterations reset: relaxing the budget is a materially
        # different situation, worth another full MAX_REFINE_ITERATIONS
        # attempt rather than inheriting whatever count the pre-pause run
        # had already spent.
        return {"trip_setup": trip_setup, "refine_iterations": 0, "human_decision": "relax_budget"}
    return {"human_decision": "accept"}


def should_resume_after_review(state: AgentState) -> Literal["itinerary", "finalize"]:
    return "itinerary" if state.get("human_decision") == "relax_budget" else "finalize"
