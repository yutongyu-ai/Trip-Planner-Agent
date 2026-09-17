"""Pure unit tests for the chat-agent eval's metric math (no LLM, no
network) -- mirrors eval/agent_e2e/test_metrics.py's role of keeping the
scoring logic testable (and CI-runnable) independent of the real-API run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import RunOutcome, invalid_tool_call_rate, reached_required_marker, task_success, trajectory_length  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402


def _outcome(tool_calls: list[tuple[str, str]], final_state: dict | None = None) -> RunOutcome:
    return RunOutcome(tool_calls=tool_calls, final_reply="done", final_state=final_state or {"itinerary": []})


def test_trajectory_length_counts_tool_messages():
    outcome = _outcome([("list_itinerary", "..."), ("swap_stop", "...")])
    assert trajectory_length(outcome) == 2


def test_invalid_tool_call_rate_flags_known_error_strings():
    outcome = _outcome(
        [
            ("swap_stop", "'p9' is not a known candidate for this trip. Call find_alternative first."),
            ("swap_stop", "09:00-10:00 Name-p2"),
        ]
    )
    assert invalid_tool_call_rate(outcome) == 0.5


def test_invalid_tool_call_rate_is_zero_for_no_calls():
    assert invalid_tool_call_rate(_outcome([])) == 0.0


def test_reached_required_marker():
    outcome = _outcome([("find_alternative", "place_id=museum-cheap | ...")])
    assert reached_required_marker(outcome, "museum-cheap") is True
    assert reached_required_marker(outcome, "museum-lux") is False


def test_task_success_checks_removed_and_present_place_ids():
    scenario = SCENARIOS[0]  # budget_swap: expects museum-lux removed, museum-cheap present
    final_state = {
        "itinerary": [
            {"stops": [{"place_id": "museum-cheap", "kind": "attraction"}, {"place_id": "park1", "kind": "attraction"}]}
        ]
    }
    outcome = _outcome([], final_state=final_state)
    assert task_success(outcome, scenario) is True


def test_task_success_fails_if_removed_place_still_present():
    scenario = SCENARIOS[0]
    final_state = {"itinerary": [{"stops": [{"place_id": "museum-lux", "kind": "attraction"}]}]}
    outcome = _outcome([], final_state=final_state)
    assert task_success(outcome, scenario) is False
