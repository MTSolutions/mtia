"""Router tests with FastAPI TestClient — auth, scoping, SSE shape; no network."""
from __future__ import annotations

import json

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.plantagent import mtapi, scope
from modules.plantagent.router import router as plantagent_router
from modules.rag import llm

SECRET = "secret"


def _token(client="degasa", login="tester"):
    return jwt.encode(
        {"sub": 1746, "roles": "admin", "client": client, "login": login},
        SECRET, algorithm="HS512")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    application = FastAPI()
    application.include_router(plantagent_router)
    return application


@pytest.fixture
def client(app, monkeypatch):
    # Scope resolution stubbed -> no mtapi2 network.
    monkeypatch.setattr(scope, "validate_plant", lambda c, p, **k: {"id": p, "name": "Planta"})
    monkeypatch.setattr(scope, "named_tree", lambda c, p, **k: {
        "id": p, "name": "Planta", "type": "plant",
        "devs": [{"id": 1079, "name": "Equipo X", "type": "dev"}]})
    # devices_in is pure; no stub needed.
    # mtapi2 indicator call stubbed (resolved at call time via ctx default).
    monkeypatch.setattr(mtapi, "call", lambda fn, c, *a: 0.87)

    async def fake_chat_tools(messages, tool_specs, model=None, options=None, think=True):
        if not getattr(fake_chat_tools, "called", False):
            fake_chat_tools.called = True
            return {"tool_calls": [{"function": {
                "name": "oee", "arguments": {"devid": 1079, "period": "hoy"}}}]}
        return {"content": "ok"}

    async def fake_chat_stream(messages, model=None, options=None, think=False):
        for piece in ["El OEE ", "es 87%."]:
            yield piece

    monkeypatch.setattr(llm, "chat_tools", fake_chat_tools)
    monkeypatch.setattr(llm, "chat_stream", fake_chat_stream)
    return TestClient(app)


def _parse_sse(text: str):
    events, event, data = [], None, []
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:"):].strip())
        elif not line and event:
            events.append((event, json.loads("\n".join(data)) if data else {}))
            event, data = None, []
    return events


def test_health_ok(client):
    r = client.get("/plantagent/health")
    assert r.status_code == 200
    assert r.json()["module"] == "plantagent"


def test_unauthenticated_returns_401(client):
    r = client.post("/plantagent/chat",
                    params={"client": "degasa", "plant_id": 7, "question": "OEE?"})
    assert r.status_code == 401


def test_cross_tenant_returns_403(client):
    r = client.post("/plantagent/chat",
                    params={"client": "otrocliente", "plant_id": 7, "question": "OEE?"},
                    headers={"Authorization": f"JWT {_token(client='degasa')}"})
    assert r.status_code == 403


def test_unknown_plant_returns_404(client, monkeypatch):
    def _raise(c, p, **k):
        raise scope.PlantNotFound("nope")
    monkeypatch.setattr(scope, "validate_plant", _raise)

    r = client.post("/plantagent/chat",
                    params={"client": "degasa", "plant_id": 999, "question": "OEE?"},
                    headers={"Authorization": f"JWT {_token()}"})
    assert r.status_code == 404


def test_mtapi_unavailable_returns_503(client, monkeypatch):
    def boom(*a, **k):
        raise mtapi.MtapiError("mtapi2 unreachable")
    monkeypatch.setattr(scope, "named_tree", boom)

    r = client.post("/plantagent/chat",
                    params={"client": "degasa", "plant_id": 7, "question": "OEE?"},
                    headers={"Authorization": f"JWT {_token()}"})
    assert r.status_code == 503


def test_chat_streams_tool_token_done(client):
    r = client.post("/plantagent/chat",
                    params={"client": "degasa", "plant_id": 7, "question": "¿OEE del 1079 hoy?"},
                    headers={"Authorization": f"JWT {_token()}"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    names = [n for n, _ in events]

    assert "tool" in names
    assert names[-1] == "done"
    tool_payload = next(p for n, p in events if n == "tool")
    assert tool_payload["name"] == "oee"
    tokens = "".join(p["text"] for n, p in events if n == "token")
    assert tokens == "El OEE es 87%."
