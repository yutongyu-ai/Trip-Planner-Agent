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

Ten scenarios. Six are positive (the task genuinely requires more than one
tool call, and nothing less); two are negative (the model must recognize it
should call *fewer* tools, not just execute a fixed script); one expects a
guardrail rejection as part of the *correct* trajectory, not a model error;
one spans two conversation turns on the same thread. Day-0, single-day
fixtures start at 3 stops (the floor for `moderate` pace in
`STOPS_PER_DAY_RANGE`), so `find_alternative`'s candidate-exclusion logic
(don't re-offer an already-scheduled place) is exercised against more than
one id.

Six scenarios is too few for `task_success_rate` to mean much on its own --
with `n=6` a single flaky run swings the rate by ~17 points, and every
scenario up to that point tested some flavor of "pick the right tool and
place_id," never the agent's behavior when something goes *wrong* (a bad
user-supplied id, a closed candidate) or when state carries across turns.
The four scenarios below were added specifically to close those gaps, not
just to pad the count -- each exercises a trajectory shape none of the
original six could have caught:

- `budget_swap_finds_cheaper_alternative`: traveller says a scheduled museum
  is over budget and asks for "something cheaper in the same category" --
  requires `find_alternative` (to discover what's available and its price)
  then `swap_stop` (using the place_id that call returned, not one the model
  already knew).
- `weather_triggered_indoor_swap`: traveller asks the agent to check the
  weather and *only* swap the outdoor stop for an indoor one if it looks bad
  -- requires `check_weather` first, and the decision to call
  `find_alternative` + `swap_stop` at all depends on that result. Forecast
  mocked to bad weather, so the swap is the correct move.
- `clear_weather_no_swap_needed`: the same weather-triggered setup, but the
  forecast is mocked clear. The correct trajectory is `check_weather` and
  *nothing else* -- `find_alternative`/`swap_stop` are listed as
  `disallowed_tools`. This is the scenario that caught a real system-prompt
  bug (the model edited anyway despite saying no edit was needed) -- see
  "Fixed" below.
- `direct_remove_no_search_needed`: traveller names the stop to drop and
  explicitly says no replacement is wanted -- `find_alternative` has nothing
  to contribute here (unlike the swap scenarios, where it's the only way to
  discover a valid alternative place_id), so it's `disallowed_tools` too.
- `add_stop_fills_free_time`: traveller asks to add something matching their
  interests, no place_id named -- covers `add_stop`, the one tool none of the
  other five scenarios ever exercise.
- `multi_day_edit_targets_correct_day`: a 2-day trip where the edit explicitly
  targets day 1, not day 0 -- every other scenario here is single-day, so
  this is the only check that the model reads the stated day_offset instead
  of defaulting to the day every other scenario happens to use. A wrong day
  wouldn't pass silently either way: `swap_stop` on the wrong day would be
  rejected by `validate_place_used_on_day` (showing up as a non-zero
  `invalid_tool_call_rate`), and day 0's own stops are asserted unchanged.
- `info_query_no_edit_needed`: a plain question ("what's on day 0?") with no
  edit implied at all -- every other `disallowed_tools` scenario up to this
  point still asked for a *conditional* edit, so this is the only check that
  curiosity alone never triggers `find_alternative`/`swap_stop`/`remove_stop`/
  `add_stop`.
- `wrong_day_offset_recovers_after_guardrail_error`: the traveller names a
  real place_id but the wrong day for it (a believable mistake, not a
  hallucinated id). Day 0 deliberately has no stop in the same category as
  the one named, so there's no plausible *other* match the model could
  defensibly substitute -- the only way to succeed is noticing the mismatch
  (via `list_itinerary` or a failed `validate_place_used_on_day` guardrail)
  and acting on the day the place_id actually lives on.
- `add_stop_declines_when_closed_that_day`: mirrors `add_stop_fills_free_time`,
  but the only unused candidate matching the request is closed on the trip's
  actual date. The right outcome either way (the model notices
  `find_alternative`'s "CLOSED that day" note and says so, or tries `add_stop`
  and gets rejected by `_reschedule_day` before any state is written) is the
  same: that place must never end up in the final itinerary, and the model
  must say it couldn't do it rather than silently giving up or inventing a
  fallback.
- `multi_turn_followup_edit_uses_conversation_memory`: the only scenario that
  invokes the agent twice on the same `thread_id`. Turn 1 swaps an expensive
  museum for a cheaper one; turn 2 says "swap it again, I'd rather have a
  park" without repeating any place_id. Getting this right requires the model
  to know the slot now holds turn 1's result, not the original fixture --
  every other scenario here is a single turn, so none of them could have
  caught a regression in cross-turn state or history handling.

## Metrics

Three, each answering a different question about the trajectory:

| metric | meaning | better |
|---|---|---|
| `invalid_tool_call_rate` | share of tool calls whose result was a validation error (see `app/agent/guardrails.py`) | lower |
| `tool_selection_accuracy` | did the trajectory contain the tools the task actually needed, in the right relative order, *and* avoid the ones it had no reason to call | True |
| `task_success_rate` | whether the final itinerary actually reflects the requested edit | higher |

`invalid_tool_call_rate` and `tool_selection_accuracy` measure different
things and neither implies the other: a call can be perfectly well-formed
(valid args, guardrails have no objection) while still being a call the
model should never have made for this task -- that's what
`tool_selection_accuracy` exists to catch, and it's also why this eval needs
negative scenarios (a trajectory that's all valid calls can still have the
wrong plan behind it). See `is_subsequence` in `run_eval.py` for the exact
matching rule: an extra `list_itinerary` check doesn't fail a scenario, only
a required tool appearing out of order or a disallowed tool appearing at all
does.

`trajectory_length` (how many tool calls a run made) is reported per
scenario in Results below for context -- it isn't a metric here, because it
has no scenario-independent "better" direction: the right call count for
`direct_remove_no_search_needed` is 1, for `weather_triggered_indoor_swap`
it's 3-4, and neither number says anything about quality on its own.

## Results (2026-10-12, `gpt-4o-mini` @ temperature 0)

| scenario | steps | trajectory | tool_selection_accuracy | task success |
|---|---|---|---|---|
| `budget_swap_finds_cheaper_alternative` | 3 | `list_itinerary -> find_alternative -> swap_stop` | True | ✓ |
| `weather_triggered_indoor_swap` | 4 | `check_weather -> list_itinerary -> find_alternative -> swap_stop` | True | ✓ |
| `clear_weather_no_swap_needed` | 1 | `check_weather` | True | ✓ |
| `direct_remove_no_search_needed` | 2 | `list_itinerary -> remove_stop` | True | ✓ |
| `add_stop_fills_free_time` | 3 | `list_itinerary -> find_alternative -> add_stop` | True | ✓ |
| `multi_day_edit_targets_correct_day` | 3 | `list_itinerary -> find_alternative -> swap_stop` | True | ✓ |
| `info_query_no_edit_needed` | 1 | `list_itinerary` | True | ✓ |
| `wrong_day_offset_recovers_after_guardrail_error` | 2 | `list_itinerary -> remove_stop` | True | ✓ |
| `add_stop_declines_when_closed_that_day` | 5 | `list_itinerary -> find_alternative -> add_stop -> find_alternative -> find_alternative` | True | ✓ |
| `multi_turn_followup_edit_uses_conversation_memory` | 5 | `list_itinerary -> find_alternative -> swap_stop -> find_alternative -> swap_stop` | True | ✓ |

Aggregate: `invalid_tool_call_rate = 0.04`, `tool_selection_accuracy_rate =
1.00`, `task_success_rate = 1.00` (10/10). With `n=10` and four of the ten
scenarios testing trajectory shapes the original six never touched (see
"Scenarios" above), this rests on meaningfully broader coverage than the
earlier 5/6 result -- not just a bigger denominator.

A few results are worth calling out as evidence, not just pass/fail:

- The weather scenario's tool sequence (`check_weather` before anything
  else) shows the agent deciding whether to look for an alternative *after*
  seeing the forecast, using the specific place_id `find_alternative`
  returned -- never one it invented.
- `multi_day_edit_targets_correct_day`'s final reply says so explicitly --
  *"I replaced the Grand National Museum with the Community Art Museum
  **on day 1**"* -- and day 0's three stops came back untouched.
- `add_stop_declines_when_closed_that_day` is the first scenario where a
  non-zero `invalid_tool_call_rate` (0.40 for that run -- `add_stop`
  correctly rejected, then more `find_alternative` calls hunting for an
  open substitute) coexists with `task_success = True`: concrete proof the
  two metrics are independent axes, not two names for the same thing.
- `multi_turn_followup_edit_uses_conversation_memory`'s turn 2 ("swap it
  again, I'd rather have a park") resolved against turn 1's *result*
  (`museum-cheap`), not the original fixture (`museum-lux`), which by turn
  2 isn't scheduled anywhere any more.

### Fixed: `clear_weather_no_swap_needed` used to fail silently past `invalid_tool_call_rate`

Until this prompt fix, this scenario reliably failed: the model called
`check_weather`, got back a clear/sunny forecast, and its own final reply
said so explicitly -- *"the weather ... is clear and sunny, so no changes
were needed for outdoor activities"* -- and then, in the same turn, called
`find_alternative` and `swap_stop` anyway, swapping out the park for a
museum it was explicitly told not to touch. Every individual call was
well-formed (guardrails had nothing to object to; `invalid_tool_call_rate =
0.00` for that run, same as every passing scenario), so this was purely a
`tool_selection_accuracy` / `task_success` failure that only a dedicated
negative scenario could catch at all -- the model was treating "check the
weather" and "do the swap" as a fixed two-step script for this user message
rather than a condition it actually had to evaluate.

Root cause: `SYSTEM_PROMPT` (`app/agent/graph.py`) said nothing about *not*
acting when a stated condition isn't met -- only about grounding and not
inventing place_ids. Adding one plain sentence about checking conditions
("If the traveller's request is conditional ... check that condition ...
if it isn't met, make no edit at all") was tried first and **did not fix
it** -- the model still called `swap_stop` in the same turn, same as
before. The tool-calling decision happens before the model drafts its
final explanatory text, so a generic rule it can agree with in prose
doesn't necessarily change the tool-call path. What did work: a more
specific version naming the exact tools to withhold (`swap_stop`,
`remove_stop`, `add_stop`) *and* a concrete counter-example matching this
scenario's own phrasing almost verbatim ("if check_weather reports clear or
mild weather, the only correct response is ... calling swap_stop afterward
... is wrong, even if you explain in words that the weather was fine").
Re-running the full suite after that change confirmed no regression on
`weather_triggered_indoor_swap` (the bad-weather branch of the same
template still correctly swaps) or any other scenario -- 10/10 pass as of
the table above.

An earlier run with `MAX_AGENT_STEPS = 6` (see `app/agent/graph.py`) cut the
weather scenario off mid-trajectory ("Sorry, need more steps to process this
request.") -- LangGraph's `recursion_limit` counts every model/tool
super-step, not tool calls, so a bound sized for "a couple of tool calls"
silently truncated a real 4-call trajectory. Raised to 20; still bounded
(a stuck model can't spin forever), just not so tight it caps off the exact
multi-step behavior this feature exists to support.
