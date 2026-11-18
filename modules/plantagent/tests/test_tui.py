"""Unit test for the TUI's SSE parser (pure)."""
from __future__ import annotations

from modules.plantagent import tui


def test_parse_sse_groups_events_in_order():
    lines = [
        "event: tool", 'data: {"name": "oee"}', "",
        "event: token", 'data: {"text": "hola"}', "",
        "event: done", "data: {}", "",
    ]
    assert list(tui.parse_sse(iter(lines))) == [
        ("tool", {"name": "oee"}),
        ("token", {"text": "hola"}),
        ("done", {}),
    ]


def test_parse_sse_handles_trailing_event_without_blank_line():
    lines = ["event: done", "data: {}"]
    assert list(tui.parse_sse(iter(lines))) == [("done", {})]


def test_mint_token_carries_the_plant_agent_claim(monkeypatch):
    # MTS-2613: /plantagent/chat answers 403 without it.
    import jwt

    monkeypatch.setenv("JWT_SECRET", "secret")
    claims = jwt.decode(tui.mint_token("degasa"), "secret", algorithms=["HS512"],
                        options={"verify_sub": False})
    assert claims["plantAgent"] is True
    assert claims["client"] == "degasa"
