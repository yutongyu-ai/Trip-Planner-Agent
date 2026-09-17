import datetime as dt

from fastapi.testclient import TestClient

from app import main
from app.tools.places_client import PlacesClientError


def _payload(**overrides) -> dict:
    base = {
        "destination": "Lisbon",
        "start_date": "2026-09-01",
        "end_date": "2026-09-02",
        "interests": ["museums"],
        "daily_pace": "moderate",
        "budget_level": "moderate",
        "day_start_time": "09:00",
        "day_end_time": "18:00",
    }
    base.update(overrides)
    return base


def test_health():
    client = TestClient(main.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_trip_plan_invokes_graph_and_returns_its_result(monkeypatch):
    captured = {}

    class _FakeGraph:
        def invoke(self, state, config=None):
            captured["state"] = state
            return {"itinerary": [{"date": dt.date(2026, 9, 1), "stops": [], "weather_note": None}], "summary": "ok"}

    monkeypatch.setattr(main, "_graph", _FakeGraph())
    client = TestClient(main.app)

    response = client.post("/trip-plan", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "ok"
    assert body["itinerary"][0]["date"] == "2026-09-01"
    assert body["errors"] == []
    assert captured["state"]["trip_setup"]["destination"] == "Lisbon"
    assert captured["state"]["trip_setup"]["start_date"] == dt.date(2026, 9, 1)


def test_trip_plan_passes_notes_through(monkeypatch):
    class _FakeGraph:
        def invoke(self, state, config=None):
            return {"itinerary": [], "summary": "s", "notes": ["heads up"]}

    monkeypatch.setattr(main, "_graph", _FakeGraph())
    client = TestClient(main.app)

    assert client.post("/trip-plan", json=_payload()).json()["notes"] == ["heads up"]


def test_validate_destination_returns_valid_for_a_real_city(monkeypatch):
    class _FakeClient:
        def city_exists(self, city, language_code="en"):
            return True

    monkeypatch.setattr(main, "get_places_provider", lambda: _FakeClient())
    client = TestClient(main.app)

    response = client.post("/validate-destination", json={"destination": "Lisbon"})

    assert response.json() == {"valid": True, "message": None}


def test_validate_destination_returns_localized_message_when_unresolvable(monkeypatch):
    class _FakeClient:
        def city_exists(self, city, language_code="en"):
            return False

    monkeypatch.setattr(main, "get_places_provider", lambda: _FakeClient())
    client = TestClient(main.app)

    response = client.post("/validate-destination", json={"destination": "ndjed", "lang": "zh"})

    body = response.json()
    assert body["valid"] is False
    assert "找不到目的地" in body["message"]


def test_validate_destination_fails_open_on_places_error(monkeypatch):
    class _FakeClient:
        def city_exists(self, city, language_code="en"):
            raise PlacesClientError("boom")

    monkeypatch.setattr(main, "get_places_provider", lambda: _FakeClient())
    client = TestClient(main.app)

    response = client.post("/validate-destination", json={"destination": "Lisbon"})

    # A transient API problem during this quick check must never block
    # someone from typing -- only the full /trip-plan run is the real
    # safety net for a genuine failure.
    assert response.json() == {"valid": True, "message": None}


def test_validate_destination_skips_the_lookup_for_an_empty_destination(monkeypatch):
    def _unexpected_call():
        raise AssertionError("must not call the Places provider for an empty destination")

    monkeypatch.setattr(main, "get_places_provider", _unexpected_call)
    client = TestClient(main.app)

    response = client.post("/validate-destination", json={"destination": "   "})

    assert response.json() == {"valid": True, "message": None}


def test_trip_plan_surfaces_graph_errors_localized_to_english_by_default(monkeypatch):
    class _FakeGraph:
        def invoke(self, state, config=None):
            return {"itinerary": [], "summary": "degraded", "errors": ["weather: boom"]}

    monkeypatch.setattr(main, "_graph", _FakeGraph())
    client = TestClient(main.app)

    response = client.post("/trip-plan", json=_payload())

    assert response.json()["errors"] == ["Weather lookup failed: boom"]


def test_trip_plan_surfaces_graph_errors_localized_to_chinese(monkeypatch):
    class _FakeGraph:
        def invoke(self, state, config=None):
            return {"itinerary": [], "summary": "degraded", "errors": ["attractions: boom"]}

    monkeypatch.setattr(main, "_graph", _FakeGraph())
    client = TestClient(main.app)

    response = client.post("/trip-plan", json=_payload(lang="zh"))

    assert response.json()["errors"] == ["景点搜索失败: boom"]


def test_trip_plan_rejects_invalid_daily_pace(monkeypatch):
    monkeypatch.setattr(main, "_graph", None)  # should never be reached
    client = TestClient(main.app)

    response = client.post("/trip-plan", json=_payload(daily_pace="chaotic"))

    assert response.status_code == 422


def test_trip_plan_rejects_a_trip_longer_than_max_trip_days(monkeypatch):
    # See MAX_TRIP_DAYS's docstring: a trip past this many days can exceed
    # candidate_pool_target's cap for some paces, so it's rejected before
    # ever reaching the graph -- the setup form should never send one, but
    # a client that bypasses that check must still be stopped here.
    monkeypatch.setattr(main, "_graph", None)  # should never be reached
    client = TestClient(main.app)

    response = client.post(
        "/trip-plan", json=_payload(start_date="2026-09-01", end_date="2026-09-06")  # 6 days
    )

    assert response.status_code == 422


def test_trip_plan_accepts_a_trip_exactly_at_max_trip_days(monkeypatch):
    class _FakeGraph:
        def invoke(self, state, config=None):
            return {"itinerary": [], "summary": "ok"}

    monkeypatch.setattr(main, "_graph", _FakeGraph())
    client = TestClient(main.app)

    response = client.post(
        "/trip-plan", json=_payload(start_date="2026-09-01", end_date="2026-09-05")  # 5 days
    )

    assert response.status_code == 200
