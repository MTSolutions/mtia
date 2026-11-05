"""Unit tests for llm.chat_tools — Ollama tool-calling (non-streaming).

These stub httpx so they need no live Ollama (mirrors the project's
no-network unit-test style; the live smoke lives in test_ollama.py).
"""
from __future__ import annotations

from modules.rag import llm


class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Stand-in for httpx.AsyncClient used as an async context manager."""

    def __init__(self, payload: dict, recorder: dict):
        self._payload = payload
        self._recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        self._recorder["url"] = url
        self._recorder["json"] = json
        return _FakeResp(self._payload)


def _patch_client(monkeypatch, payload: dict) -> dict:
    recorder: dict = {}
    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda *a, **k: _FakeClient(payload, recorder),
    )
    return recorder


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "oee",
            "description": "OEE de un equipo en un periodo",
            "parameters": {
                "type": "object",
                "properties": {"devid": {"type": "integer"}},
                "required": ["devid"],
            },
        },
    }
]


async def test_chat_tools_parses_tool_calls(monkeypatch):
    _patch_client(monkeypatch, {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "oee", "arguments": {"devid": 1079}}}
            ],
        }
    })

    msg = await llm.chat_tools(
        [{"role": "user", "content": "¿cuál es el OEE del equipo 1079?"}],
        tools=TOOLS,
    )

    assert msg["tool_calls"][0]["function"]["name"] == "oee"
    assert msg["tool_calls"][0]["function"]["arguments"]["devid"] == 1079


async def test_chat_tools_sends_tools_non_streaming_with_thinking(monkeypatch):
    recorder = _patch_client(monkeypatch, {"message": {"role": "assistant", "content": "ok"}})

    await llm.chat_tools([{"role": "user", "content": "hola"}], tools=TOOLS)

    sent = recorder["json"]
    assert sent["tools"] == TOOLS
    assert sent["stream"] is False          # tool calls require a complete response
    assert sent["think"] is True            # thinking on by default (boosts FC accuracy)
    assert recorder["url"].endswith("/api/chat")


async def test_chat_tools_plain_message_has_no_tool_calls(monkeypatch):
    _patch_client(monkeypatch, {"message": {"role": "assistant", "content": "Hola, ¿qué tal?"}})

    msg = await llm.chat_tools([{"role": "user", "content": "saluda"}], tools=TOOLS)

    assert msg["content"] == "Hola, ¿qué tal?"
    assert "tool_calls" not in msg


async def test_chat_tools_respects_model_and_options(monkeypatch):
    recorder = _patch_client(monkeypatch, {"message": {"role": "assistant", "content": "ok"}})

    await llm.chat_tools(
        [{"role": "user", "content": "hola"}],
        tools=TOOLS,
        model="gemma4:12b",
        options={"temperature": 0},
    )

    assert recorder["json"]["model"] == "gemma4:12b"
    assert recorder["json"]["options"] == {"temperature": 0}
