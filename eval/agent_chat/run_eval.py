"""Evaluation of the conversational itinerary-edit agent (app/agent/graph.py)
-- the one place in this project where an LLM actually chooses which tool to
call and when to stop, rather than following a fixed pipeline. See
scenarios.py for why each scenario needs more than one tool call to satisfy.

Three metrics, each answering a different question invalid_tool_call_rate
alone can't: task_success (did the final state reflect the request),
tool_selection_accuracy (did it call the tools this task needed, in order,
and avoid the ones it had no reason to reach for -- see `disallowed_tools`
on clear_weather_no_swap_needed and direct_remove_no_search_needed),
invalid_tool_call_rate (were the calls it did make well-formed against the
real session state). trajectory_length is reported per scenario for context
only -- it has no universal "better" direction (the right call count is
task-specific), so it isn't trended as a metric.

Makes real gpt-4o-mini calls (temperature 0); needs a real OPENAI_API_KEY in
the repo-root .env. Weather is monkeypatched to a scripted forecast so the
weather-triggered scenarios are reproducible without a live API call.

Run: python eval/agent_chat/run_eval.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app import agent as agent_pkg  # noqa: E402,F401  (ensures app.agent.tools is importable below)
from app.agent import tools as tools_module  # noqa: E402
from app.agent.graph import MAX_AGENT_STEPS, build_chat_agent  # noqa: E402
from app.config import get_settings  # noqa: E402
from scenarios import SCENARIOS, ChatScenario  # noqa: E402

_KNOWN_ERROR_MARKERS = (
    "out of range",
    "not a known candidate",
    "not currently scheduled",
    "already scheduled",
    "closed on",
    "isn't a valid date",
    "No unused candidates",
    "No forecast available",
)


@dataclass
class RunOutcome:
    tool_calls: list[tuple[str, str]]  # (tool_name, result_content)
    final_reply: str
    final_state: dict


def _run(scenario: ChatScenario) -> RunOutcome:
    state = dict(scenario.state)  # shallow copy: itinerary list is still shared/mutated intentionally by the tools
    state["itinerary"] = [dict(d) for d in state["itinerary"]]  # avoid cross-scenario mutation of the fixture

    patches = []
    if scenario.weather_forecast is not None:
        fake_client = mock.Mock()
        fake_client.get_daily_forecast.return_value = scenario.weather_forecast
        patches.append(mock.patch.object(tools_module, "get_weather_provider", lambda country_code=None: fake_client))

    for p in patches:
        p.start()
    try:
        agent = build_chat_agent(state, MemorySaver())
        # Multi-turn scenarios (user_messages set) invoke the same thread_id
        # repeatedly, exactly as separate /chat requests for one session
        # would -- the checkpointer carries conversation history and `state`
        # itself carries prior turns' mutations forward. Only the final
        # turn's result is kept: with a checkpointer, each invoke returns
        # the full accumulated message history for the thread, not just
        # that turn's own messages.
        turns = scenario.user_messages if scenario.user_messages is not None else [scenario.user_message]
        result = None
        for turn in turns:
            result = agent.invoke(
                {"messages": [("user", turn)]},
                config={"configurable": {"thread_id": scenario.name}, "recursion_limit": MAX_AGENT_STEPS},
            )
    finally:
        for p in patches:
            p.stop()

    tool_calls = [(m.name, m.content) for m in result["messages"] if m.__class__.__name__ == "ToolMessage"]
    return RunOutcome(tool_calls=tool_calls, final_reply=result["messages"][-1].content, final_state=state)


def _used_place_ids(state: dict) -> set[str]:
    return {s["place_id"] for day in state["itinerary"] for s in day["stops"] if s["kind"] == "attraction"}


def trajectory_length(outcome: RunOutcome) -> int:
    return len(outcome.tool_calls)


def invalid_tool_call_rate(outcome: RunOutcome) -> float:
    if not outcome.tool_calls:
        return 0.0
    invalid = sum(1 for _, content in outcome.tool_calls if any(marker in content for marker in _KNOWN_ERROR_MARKERS))
    return invalid / len(outcome.tool_calls)


def is_subsequence(required: list[str], actual: list[str]) -> bool:
    """Whether `required` appears in `actual`, in order, allowing extra
    items in between (e.g. an extra list_itinerary check doesn't break it)."""
    it = iter(actual)
    return all(tool in it for tool in required)


def tool_selection_accuracy(outcome: RunOutcome, scenario: ChatScenario) -> bool:
    """Did the model call the tools this task actually needed (in the right
    relative order) and avoid the ones it had no reason to call? Orthogonal
    to invalid_tool_call_rate: every call here can be individually
    well-formed and still add up to the wrong plan (e.g. skipping
    check_weather and guessing) -- see eval/agent_chat/README.md."""
    called = [name for name, _ in outcome.tool_calls]
    required_ok = is_subsequence(scenario.expected_tool_sequence, called)
    no_violation = not any(t in called for t in scenario.disallowed_tools)
    return required_ok and no_violation


def task_success(outcome: RunOutcome, scenario: ChatScenario) -> bool:
    used = _used_place_ids(outcome.final_state)
    if scenario.place_expected_removed is not None and scenario.place_expected_removed in used:
        return False
    if scenario.place_expected_present is not None and scenario.place_expected_present not in used:
        return False
    return True


def main() -> None:
    if not get_settings().openai_api_key:
        sys.exit("OPENAI_API_KEY is not set. This script makes real gpt-4o-mini calls and needs a real key.")

    print(f"Running {len(SCENARIOS)} chat scenarios...\n")
    rows = []
    for scenario in SCENARIOS:
        outcome = _run(scenario)
        rows.append(
            {
                "scenario": scenario.name,
                "trajectory_length": trajectory_length(outcome),  # context only -- see module docstring
                "invalid_tool_call_rate": invalid_tool_call_rate(outcome),
                "tool_selection_accuracy": tool_selection_accuracy(outcome, scenario),
                "task_success": task_success(outcome, scenario),
                "tool_calls": [name for name, _ in outcome.tool_calls],
                "final_reply": outcome.final_reply,
            }
        )

    header = f"{'scenario':<38}{'steps':<8}{'invalid%':<10}{'tool_sel':<10}{'success':<8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['scenario']:<38}{row['trajectory_length']:<8}{row['invalid_tool_call_rate']:<10.2f}"
            f"{str(row['tool_selection_accuracy']):<10}{str(row['task_success']):<8}"
        )

    print("\ntool-call sequence per scenario:")
    for row in rows:
        print(f"  {row['scenario']}: {' -> '.join(row['tool_calls'])}")
        print(f"    final reply: {row['final_reply']!r}")

    n = len(rows)
    print("\naggregate:")
    print(f"  task_success_rate: {sum(r['task_success'] for r in rows) / n:.2f}")
    print(f"  tool_selection_accuracy_rate: {sum(r['tool_selection_accuracy'] for r in rows) / n:.2f}")
    print(f"  mean invalid_tool_call_rate: {sum(r['invalid_tool_call_rate'] for r in rows) / n:.2f}")


if __name__ == "__main__":
    main()
