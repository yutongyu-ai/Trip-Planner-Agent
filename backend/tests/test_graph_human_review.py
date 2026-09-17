"""Unit tests for human_review_node / should_resume_after_review
(app/graph/nodes/human_review.py).

The pause/resume mechanics (`interrupt()` / `Command(resume=...)`) are
exercised against a minimal standalone graph -- not the full `build_graph()`
pipeline, which would need a fake LLM run all the way to a genuinely
unresolved budget violation just to reach this node. That combination is
exercised manually (see the plan's verification notes); what matters for
unit coverage is that this node reads/writes state correctly around a real
LangGraph interrupt, which this does test.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.nodes.human_review import human_review_node, should_resume_after_review
from app.graph.state import AgentState


def test_passes_through_with_no_violations_even_if_flags_are_set():
    # A small destination can legitimately use its whole candidate pool with
    # a perfectly satisfying itinerary -- pool_exhausted alone must never
    # trigger a pause.
    state = {"constraint_violations": [], "budget_unavoidable": True, "pool_exhausted": True}
    assert human_review_node(state) == {}


def test_passes_through_when_violations_remain_but_not_budget_or_pool():
    state = {"constraint_violations": ["some other violation"], "budget_unavoidable": False, "pool_exhausted": False}
    assert human_review_node(state) == {}


def test_should_resume_after_review_defaults_to_finalize():
    assert should_resume_after_review({}) == "finalize"
    assert should_resume_after_review({"human_decision": "accept"}) == "finalize"


def test_should_resume_after_review_relax_budget_routes_to_itinerary():
    assert should_resume_after_review({"human_decision": "relax_budget"}) == "itinerary"


def _minimal_graph():
    graph = StateGraph(AgentState)
    graph.add_node("human_review", human_review_node)
    graph.add_edge(START, "human_review")
    graph.add_edge("human_review", END)
    return graph.compile(checkpointer=MemorySaver())


def test_pauses_and_resumes_with_relax_budget():
    compiled = _minimal_graph()
    state = {
        "trip_setup": {"budget_level": "budget"},
        "constraint_violations": ["day 0: over budget"],
        "budget_unavoidable": True,
        "pool_exhausted": False,
        "refine_iterations": 2,
    }
    cfg = {"configurable": {"thread_id": "relax-test"}}

    paused = compiled.invoke(state, config=cfg)
    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["reason"] == "budget_unavoidable"
    assert payload["options"] == ["accept", "relax_budget"]

    resumed = compiled.invoke(Command(resume="relax_budget"), config=cfg)
    assert resumed["trip_setup"]["budget_level"] == "moderate"
    assert resumed["refine_iterations"] == 0
    assert resumed["human_decision"] == "relax_budget"


def test_pauses_and_resumes_with_accept_leaves_budget_untouched():
    compiled = _minimal_graph()
    state = {
        "trip_setup": {"budget_level": "budget"},
        "constraint_violations": ["day 0: over budget"],
        "budget_unavoidable": True,
        "pool_exhausted": False,
        "refine_iterations": 2,
    }
    cfg = {"configurable": {"thread_id": "accept-test"}}

    compiled.invoke(state, config=cfg)
    resumed = compiled.invoke(Command(resume="accept"), config=cfg)

    assert resumed["trip_setup"]["budget_level"] == "budget"
    assert resumed["human_decision"] == "accept"


def test_pool_exhausted_reason_only_offers_accept():
    compiled = _minimal_graph()
    state = {
        "trip_setup": {"budget_level": "moderate"},
        "constraint_violations": ["day 0: too few stops"],
        "budget_unavoidable": False,
        "pool_exhausted": True,
        "refine_iterations": 2,
    }
    cfg = {"configurable": {"thread_id": "pool-test"}}

    paused = compiled.invoke(state, config=cfg)
    payload = paused["__interrupt__"][0].value
    assert payload["reason"] == "pool_exhausted"
    assert payload["options"] == ["accept"]
