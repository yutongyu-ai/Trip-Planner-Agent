# Conversational itinerary-edit agent evaluation

Unlike `eval/agent_e2e` (which measures the *generation* pipeline, a fixed
workflow), this evaluates `app/agent/graph.py` -- the one part of the project
where the model actually chooses which tool to call and when to stop. The
question this eval answers isn't "did it get the right answer" alone, but
"did it actually reason across more than one tool call, reacting to what an
earlier call revealed" -- a single-call agent would fail these scenarios by
construction (see `scenarios.py`).

## Setup

`ranked_attractions` / `itinerary` fixtures are built directly in Python (no
Places/Route mocking needed, since the chat tools never call those APIs --
see `app/agent/tools.py`); `check_weather`'s provider is monkeypatched to a
scripted forecast for the weather scenario. The `gpt-4o-mini` call is real
(temperature 0) -- needs `OPENAI_API_KEY` in the repo-root `.env`.

```
python eval/agent_chat/run_eval.py
```

`test_metrics.py` unit-tests the scoring functions against hand-built
`RunOutcome`s, no LLM or network involved -- runs in CI.

## Scenarios

- `budget_swap_finds_cheaper_alternative`: traveller says a scheduled museum
  is over budget and asks for "something cheaper in the same category" --
  requires `find_alternative` (to discover what's available and its price)
  then `swap_stop` (using the place_id that call returned, not one the model
  already knew). Minimum 2 tool calls.
- `weather_triggered_indoor_swap`: traveller asks the agent to check the
  weather and *only* swap the outdoor stop for an indoor one if it looks bad
  -- requires `check_weather` first, and the decision to call
  `find_alternative` + `swap_stop` at all depends on that result (a clear-day
  mock of the same scenario should produce zero further tool calls, though
  this eval only exercises the bad-weather branch). Minimum 3 tool calls.

## Metrics

| metric | meaning | better |
|---|---|---|
| `trajectory_length` | number of tool calls made in the run | matches or exceeds the scenario's minimum |
| `invalid_tool_call_rate` | share of tool calls whose result was a validation error (see `app/agent/guardrails.py`) | lower |
| `reached_required_marker` | whether the trajectory actually surfaced the specific alternative place_id the scenario requires (not just *any* tool calls) | True |
| `task_success_rate` | whether the final itinerary actually reflects the requested edit | higher |

## Results (2026-09-12, `gpt-4o-mini` @ temperature 0)

| scenario | trajectory | task success |
|---|---|---|
| `budget_swap_finds_cheaper_alternative` | `list_itinerary -> find_alternative -> swap_stop` | ✓ |
| `weather_triggered_indoor_swap` | `check_weather -> list_itinerary -> find_alternative -> swap_stop` | ✓ |

Both runs: `invalid_tool_call_rate = 0.00`, `task_success_rate = 1.00`, mean
`trajectory_length = 3.5`.

This is the concrete evidence for "the model called tools more than once and
changed course based on what it observed": the weather scenario's tool
sequence shows the agent checking the forecast *before* deciding to look for
an alternative at all, and using the specific place_id `find_alternative`
returned (never one it invented) to make the swap.

An earlier run with `MAX_AGENT_STEPS = 6` (see `app/agent/graph.py`) cut the
weather scenario off mid-trajectory ("Sorry, need more steps to process this
request.") -- LangGraph's `recursion_limit` counts every model/tool
super-step, not tool calls, so a bound sized for "a couple of tool calls"
silently truncated a real 4-call trajectory. Raised to 20; still bounded
(a stuck model can't spin forever), just not so tight it caps off the exact
multi-step behavior this feature exists to support.
