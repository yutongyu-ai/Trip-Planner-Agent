import datetime as dt

from fastapi.testclient import TestClient

from app import main


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
        def invoke(self, state):
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
        def invoke(self, state):
            return {"itinerary": [], "summary": "s", "notes": ["heads up"]}

    monkeypatch.setattr(main, "_graph", _FakeGraph())
    client = TestClient(main.app)

    assert client.post("/trip-plan", json=_payload()).json()["notes"] == ["heads up"]


def test_trip_plan_surfaces_graph_errors_localized_to_english_by_default(monkeypatch):
    class _FakeGraph:
        def invoke(self, state):
            return {"itinerary": [], "summary": "degraded", "errors": ["weather: boom"]}

    monkeypatch.setattr(main, "_graph", _FakeGraph())
    client = TestClient(main.app)

    response = client.post("/trip-plan", json=_payload())

    assert response.json()["errors"] == ["Weather lookup failed: boom"]


def test_trip_plan_surfaces_graph_errors_localized_to_chinese(monkeypatch):
    class _FakeGraph:
        def invoke(self, state):
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
