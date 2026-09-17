"""Evaluation of the conversational itinerary-edit agent (app/agent/graph.py)
-- the one place in this project where an LLM actually chooses which tool to
call and when to stop, rather than following a fixed pipeline. See
scenarios.py for why each scenario needs more than one tool call to satisfy.

Makes real gpt-4o-mini calls (temperature 0); needs a real OPENAI_API_KEY in
the repo-root .env. Weather is monkeypatched to a scripted forecast so the
weather-triggered scenario is reproducible without a live API call.

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
        result = agent.invoke(
            {"messages": [("user", scenario.user_message)]},
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


def reached_required_marker(outcome: RunOutcome, marker: str) -> bool:
    return any(marker in content for _, content in outcome.tool_calls)


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
                "trajectory_length": trajectory_length(outcome),
                "met_min_calls": trajectory_length(outcome) >= scenario.min_expected_tool_calls,
                "reached_marker": reached_required_marker(outcome, scenario.required_trajectory_marker),
                "invalid_tool_call_rate": invalid_tool_call_rate(outcome),
                "task_success": task_success(outcome, scenario),
                "tool_calls": [name for name, _ in outcome.tool_calls],
                "final_reply": outcome.final_reply,
            }
        )

    header = f"{'scenario':<38}{'traj_len':<10}{'>=min':<8}{'marker':<8}{'invalid%':<10}{'success':<8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['scenario']:<38}{row['trajectory_length']:<10}{str(row['met_min_calls']):<8}"
            f"{str(row['reached_marker']):<8}{row['invalid_tool_call_rate']:<10.2f}{str(row['task_success']):<8}"
        )

    print("\ntool-call sequence per scenario:")
    for row in rows:
        print(f"  {row['scenario']}: {' -> '.join(row['tool_calls'])}")
        print(f"    final reply: {row['final_reply']!r}")

    n = len(rows)
    print("\naggregate:")
    print(f"  mean trajectory_length: {sum(r['trajectory_length'] for r in rows) / n:.2f}")
    print(f"  task_success_rate: {sum(r['task_success'] for r in rows) / n:.2f}")
    print(f"  mean invalid_tool_call_rate: {sum(r['invalid_tool_call_rate'] for r in rows) / n:.2f}")


if __name__ == "__main__":
    main()
