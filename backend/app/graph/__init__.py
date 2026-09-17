"""Assembles the trip-planning LangGraph: a linear
attractions -> [weather] -> route -> itinerary pipeline, with three
non-trivial decisions made via conditional edges rather than hardcoded
control flow:

1. Dynamic tool selection -- the weather node is skipped entirely for
   trips starting beyond OpenWeatherMap's free-tier forecast horizon
   (see `should_check_weather`).
2. A bounded refine loop -- if the generated itinerary violates the
   traveller's stated pace/budget, the graph loops back to the itinerary
   node with the violations as feedback, up to MAX_REFINE_ITERATIONS
   times (see `should_refine`). The itinerary node itself plans one day
   per LLM call and, on a refine pass, only re-calls the LLM for days
   that actually had a violation -- see itinerary.py's docstring.
3. A human-in-the-loop pause -- once the refine loop gives up on a
   violation it structurally can't fix itself (not enough affordable
   candidates, or the candidate pool already exhausted), `human_review`
   pauses the run and asks the traveller what to do instead of silently
   downgrading it to an advisory note (see nodes/human_review.py). Needs a
   `checkpointer` (see this function's parameter) to actually pause/resume;
   without one this node still runs but can never have anything to pause
   for that a checkpointer-less caller could resume anyway.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes.attractions import attractions_node
from app.graph.nodes.constraints import check_constraints_node, should_refine
from app.graph.nodes.finalize import finalize_node
from app.graph.nodes.human_review import human_review_node, should_resume_after_review
from app.graph.nodes.itinerary import itinerary_node
from app.graph.nodes.route import route_node
from app.graph.nodes.weather import should_check_weather, weather_node
from app.graph.state import AgentState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("attractions", attractions_node)
    graph.add_node("weather", weather_node)
    graph.add_node("route", route_node)
    graph.add_node("itinerary", itinerary_node)
    graph.add_node("check_constraints", check_constraints_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "attractions")
    graph.add_conditional_edges(
        "attractions", should_check_weather, {"weather": "weather", "route": "route"}
    )
    graph.add_edge("weather", "route")
    graph.add_edge("route", "itinerary")
    graph.add_edge("itinerary", "check_constraints")
    graph.add_conditional_edges(
        "check_constraints",
        should_refine,
        {"itinerary": "itinerary", "finalize": "human_review"},
    )
    graph.add_conditional_edges(
        "human_review",
        should_resume_after_review,
        {"itinerary": "itinerary", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
