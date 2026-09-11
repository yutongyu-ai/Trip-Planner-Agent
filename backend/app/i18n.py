"""Bilingual templates for backend-authored, non-LLM user-facing text
(trip summaries, constraint-violation messages, weather notes).

The LLM never writes user-facing prose in this app -- itinerary.py's
docstring explains why (facts shouldn't be hallucinated). This module keeps
that principle for language too: every value plugged into these templates
is already computed elsewhere (a place name, a stop count, a date), and
this only picks which language wraps it. Mirrors frontend/src/i18n.tsx's
approach, just server-side for the strings the frontend can't build itself
because it doesn't have the underlying facts.

Also maps the app's own "en"/"zh" language choice to the language/region
codes the two external APIs expect (Google Places' `languageCode`,
OpenWeatherMap's `lang`).
"""

from __future__ import annotations

Lang = str  # "en" | "zh" -- kept as plain str to avoid importing state.py here

# date.weekday() convention: 0=Monday .. 6=Sunday.
_WEEKDAY_NAMES_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_WEEKDAY_NAMES_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def weekday_name(lang: Lang, weekday: int) -> str:
    return _WEEKDAY_NAMES_ZH[weekday] if lang == "zh" else _WEEKDAY_NAMES_EN[weekday]


def places_language_code(lang: Lang) -> str:
    return "zh-CN" if lang == "zh" else "en"


def openweather_lang(lang: Lang) -> str:
    return "zh_cn" if lang == "zh" else "en"


def trip_planned(lang: Lang, destination: str, num_days: int, total_stops: int) -> str:
    if lang == "zh":
        return f"{destination} 行程，共 {num_days} 天，安排了 {total_stops} 个地点。"
    return f"{destination} trip, {num_days} day(s), {total_stops} stop(s) planned."


def weather_lookup_failed(lang: Lang, reason: str) -> str:
    if lang == "zh":
        return f"天气查询失败（{reason}）；行程未做天气调整。"
    return f"Weather lookup failed ({reason}); itinerary planned without weather adjustments."


def weather_beyond_horizon(lang: Lang, missing_days: int) -> str:
    if lang == "zh":
        return f"天气预报只覆盖免费版的 5 天窗口，本次行程中有 {missing_days} 天未做天气调整。"
    return (
        "Weather forecasts only cover the free-tier 5-day horizon; "
        f"{missing_days} day(s) in this itinerary were planned without weather adjustments."
    )


def destination_low_coverage(lang: Lang) -> str:
    if lang == "zh":
        return (
            "当前数据源对中国大陆城市的地点、营业时间和路线覆盖有限，本次行程可能不够准确。"
        )
    return (
        "The current data source has limited coverage of places, opening hours and "
        "routing for mainland-China cities; this itinerary may be inaccurate."
    )


def meal_lunch_label(lang: Lang) -> str:
    return "午餐" if lang == "zh" else "Lunch"


def budget_options_limited(lang: Lang) -> str:
    if lang == "zh":
        return "该目的地符合预算的景点有限，行程里包含了一些价位偏高的地点。"
    return (
        "This destination has few budget-friendly attractions, so the plan includes some "
        "pricier stops."
    )


def stops_dropped_to_fit(lang: Lang, names: list[str]) -> str:
    if lang == "zh":
        return (
            f"有些地点当天在营业时间和通勤时间内排不下，已移除：{'、'.join(names)}。"
            "可以考虑延长每日游玩时间，或减少当天的景点数量。"
        )
    return (
        f"Some stops couldn't be fitted into their day around opening hours and travel time "
        f"and were left out: {', '.join(names)}. Consider a longer day or fewer stops."
    )


def refine_limit_reached(lang: Lang, attempts: int, violations: list[str]) -> str:
    if lang == "zh":
        return f"已达到重排上限（{attempts} 次尝试），仍有未解决的问题：{'；'.join(violations)}"
    return f"Reached the refine limit ({attempts} attempts) with unresolved constraints: {'; '.join(violations)}"


# Entries in state["errors"] (the "weather: ...", "route: ...",
# "attractions: ...", "itinerary: ..." lines) are produced with a stable,
# language-independent tag -- finalize.py string-matches on the "weather: "
# prefix to detect a weather-specific failure, so nodes must keep writing
# these in English regardless of trip_setup["lang"]. localize_error (below)
# is the one place that turns the tag into UI copy, applied at the API
# boundary in main.py *after* finalize.py has already done its prefix
# matching on the untranslated form -- the raw text after the tag (the
# underlying HTTP/library exception) stays as-is either way, since that's
# debugging content, not UI copy.

_ERROR_TAG_LABELS: dict[str, dict[Lang, str]] = {
    "attractions": {"en": "Attraction search failed", "zh": "景点搜索失败"},
    "weather": {"en": "Weather lookup failed", "zh": "天气查询失败"},
    "route": {"en": "Route lookup failed", "zh": "路线查询失败"},
    "itinerary": {"en": "Itinerary generation failed", "zh": "行程生成失败"},
}


def localize_error(lang: Lang, error: str) -> str:
    tag, _, detail = error.partition(": ")
    label = _ERROR_TAG_LABELS.get(tag, {}).get(lang)
    if label is None:
        return error  # unrecognized tag -- show the raw text rather than hide it
    return f"{label}: {detail}" if detail else label


# -- constraint violations (constraints.py) --------------------------------


def violation_stop_count(lang: Lang, date: str, count: int, pace: str, lo: int, hi: int) -> str:
    if lang == "zh":
        return f"{date}：安排了 {count} 个地点，超出了「{pace}」节奏对应的 {lo}-{hi} 个范围"
    return f"{date}: {count} stops, outside the {pace} range of {lo}-{hi}"


def violation_budget(lang: Lang, date: str, avg_price: float, budget: str, max_price: int) -> str:
    if lang == "zh":
        return f"{date}：平均价位 {avg_price:.1f} 超出了「{budget}」预算上限 {max_price}"
    return f"{date}: average price level {avg_price:.1f} exceeds the {budget} budget cap of {max_price}"


def violation_closed(lang: Lang, date: str, name: str, weekday: int) -> str:
    weekday_str = weekday_name(lang, weekday)
    if lang == "zh":
        return f"{date}：{name} 在{weekday_str}不营业"
    return f"{date}: {name} is closed on {weekday_str}s"


def violation_outside_hours(
    lang: Lang, date: str, name: str, arrival: str, departure: str, open_time: str, close_time: str
) -> str:
    if lang == "zh":
        return f"{date}：{name} 的营业时间是 {open_time}-{close_time}，但行程安排是 {arrival}-{departure}"
    return f"{date}: {name} is open {open_time}-{close_time}, but the plan has it scheduled {arrival}-{departure}"


def violation_duplicate_place(lang: Lang, date: str, name: str, first_date: str) -> str:
    if lang == "zh":
        return f"{date}：{name} 已经安排在 {first_date}，同一个地点不能出现在多天里"
    return f"{date}: {name} is already scheduled on {first_date}, the same place can't appear on multiple days"


def violation_gap_too_short(lang: Lang, date: str, gap_min: int, a: str, b: str, needed_min: float) -> str:
    if lang == "zh":
        return f"{date}：{a} 和 {b} 之间只留了 {gap_min} 分钟，至少需要 {needed_min:.0f} 分钟（游玩时间+通勤）"
    return f"{date}: only {gap_min} min between {a} and {b}, need at least {needed_min:.0f} min (visit time + travel)"


def violation_day_underfilled(lang: Lang, date: str, end_min: int, window_min: int, hi: int) -> str:
    if lang == "zh":
        return (
            f"{date}：所有安排在当天第 {end_min} 分钟就结束了，{window_min} 分钟的时间窗还剩一大半——"
            f"再加 1-2 个地点（最多 {hi} 个），别留半天空着"
        )
    return (
        f"{date}: everything wraps up {end_min} min into a {window_min}-minute day, leaving half the "
        f"window empty -- add 1-2 more stops (up to {hi}) rather than a half-empty day"
    )


def violation_beyond_window(lang: Lang, date: str, name: str, end_min: int, window_min: int, start_time: str, end_time: str) -> str:
    if lang == "zh":
        return (
            f"{date}：最后一站 {name} 结束于当天第 {end_min} 分钟，超出了 {window_min} 分钟的"
            f"可用时间窗（{start_time}-{end_time}）"
        )
    return (
        f"{date}: last stop {name} ends at {end_min} min after day start, beyond the "
        f"{window_min}-minute window ({start_time}-{end_time})"
    )


# -- weather note (itinerary.py) --------------------------------------------


def weather_note_ok(lang: Lang, description: str, avg_temp_c: float) -> str:
    if lang == "zh":
        return f"{description}，约 {avg_temp_c:.0f}°C"
    return f"{description}, around {avg_temp_c:.0f}C"


def weather_note_bad_adjusted(lang: Lang, description: str) -> str:
    if lang == "zh":
        return f"预计{description}——已调整为室内活动"
    return f"{description} expected -- adjusted to indoor stops"


def weather_note_bad_unadjusted(lang: Lang, description: str) -> str:
    if lang == "zh":
        return f"预计{description}——可以考虑室内备选"
    return f"{description} expected -- consider indoor alternatives"
