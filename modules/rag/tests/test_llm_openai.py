"""Unit tests for the OpenAI-compatible (OpenRouter) backend in llm.py.

httpx is stubbed — no network. Covers env-based provider selection, message
translation in both directions (tool_calls/arguments/tool_call_id), sampling
option mapping, and SSE stream parsing.
"""
from __future__ import annotations

import json

import pytest

from modules.rag import llm


class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeStream:
    def __init__(self, lines: list[str]):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeClient:
    """Stand-in for httpx.AsyncClient capturing url/json/headers."""

    def __init__(self, payload, recorder: dict):
        self._payload = payload
        self._recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self._recorder.update(url=url, json=json, headers=headers)
        return _FakeResp(self._payload)

    def stream(self, method, url, json=None, headers=None):
        self._recorder.update(method=method, url=url, json=json, headers=headers)
        return _FakeStream(self._payload)


def _patch(monkeypatch, payload) -> dict:
    recorder: dict = {}
    monkeypatch.setattr(
        llm.httpx, "AsyncClient", lambda *a, **k: _FakeClient(payload, recorder))
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_MODEL", "google/gemma-3-27b-it")
    return recorder


TOOLS = [{
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
}]

_COMPLETION_WITH_TOOL_CALL = {
    "choices": [{"message": {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "oee", "arguments": "{\"devid\": 1079}"},
        }],
    }}],
}


async def test_chat_tools_posts_openai_endpoint_with_bearer(monkeypatch):
    recorder = _patch(monkeypatch, _COMPLETION_WITH_TOOL_CALL)

    await llm.chat_tools([{"role": "user", "content": "hola"}], tools=TOOLS)

    assert recorder["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert recorder["headers"]["Authorization"] == "Bearer sk-or-test"
    sent = recorder["json"]
    assert sent["model"] == "google/gemma-3-27b-it"
    assert sent["tools"] == TOOLS
    assert sent["stream"] is False
    assert "think" not in sent          # Ollama-only knob never leaks out


async def test_chat_tools_normalizes_tool_calls_to_internal_shape(monkeypatch):
    _patch(monkeypatch, _COMPLETION_WITH_TOOL_CALL)

    msg = await llm.chat_tools([{"role": "user", "content": "¿OEE 1079?"}], tools=TOOLS)

    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_abc123"
    assert tc["function"]["name"] == "oee"
    assert tc["function"]["arguments"] == {"devid": 1079}   # JSON string → dict
    assert msg["content"] == ""                              # None → ""


async def test_history_translated_to_openai_format(monkeypatch):
    recorder = _patch(monkeypatch, {"choices": [{"message": {"content": "ok"}}]})
    history = [
        {"role": "system", "content": "eres un agente"},
        {"role": "user", "content": "¿OEE?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1",
                         "function": {"name": "oee", "arguments": {"devid": 1079}}}]},
        {"role": "tool", "tool_name": "oee", "tool_call_id": "call_1",
         "content": "{\"oee\": 0.87}"},
    ]

    await llm.chat_tools(history, tools=TOOLS)

    sent = recorder["json"]["messages"]
    assistant = sent[2]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["type"] == "function"
    # dict args → JSON string on the wire
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"devid": 1079}
    tool = sent[3]
    assert tool == {"role": "tool", "tool_call_id": "call_1",
                    "content": "{\"oee\": 0.87}"}            # tool_name dropped


async def test_options_map_to_top_level_and_num_ctx_dropped(monkeypatch):
    recorder = _patch(monkeypatch, {"choices": [{"message": {"content": "ok"}}]})

    await llm.chat_tools([{"role": "user", "content": "hola"}], tools=TOOLS,
                         options={"temperature": 0, "num_ctx": 16384})

    sent = recorder["json"]
    assert sent["temperature"] == 0
    assert "num_ctx" not in sent
    assert "options" not in sent


async def test_llm_url_overrides_base_for_any_openai_server(monkeypatch):
    recorder = _patch(monkeypatch, {"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_URL", "http://vllm:8000/v1/")

    await llm.chat_tools([{"role": "user", "content": "hola"}], tools=TOOLS)

    assert recorder["url"] == "http://vllm:8000/v1/chat/completions"


async def test_missing_api_key_raises(monkeypatch):
    _patch(monkeypatch, {})
    monkeypatch.delenv("OPENROUTER_API_KEY")

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        await llm.chat_tools([{"role": "user", "content": "hola"}], tools=TOOLS)


async def test_chat_stream_parses_sse_and_skips_comments(monkeypatch):
    _patch(monkeypatch, [
        ": OPENROUTER PROCESSING",
        "data: " + json.dumps({"choices": [{"delta": {"content": "El OEE "}}]}),
        "",
        "data: " + json.dumps({"choices": [{"delta": {"reasoning": "pensando..."}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"content": "es 87%."}}]}),
        "data: [DONE]",
    ])

    chunks = [c async for c in llm.chat_stream([{"role": "user", "content": "?"}])]

    assert chunks == ["El OEE ", "es 87%."]   # reasoning/comments not yielded


async def test_chat_stream_raises_on_stream_error(monkeypatch):
    _patch(monkeypatch, [
        "data: " + json.dumps({"error": {"message": "rate limited"}}),
    ])

    with pytest.raises(RuntimeError, match="rate limited"):
        async for _ in llm.chat_stream([{"role": "user", "content": "?"}]):
            pass


async def test_default_provider_still_ollama(monkeypatch):
    recorder: dict = {}
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(
        llm.httpx, "AsyncClient",
        lambda *a, **k: _FakeClient({"message": {"role": "assistant", "content": "ok"}},
                                    recorder))

    await llm.chat_tools([{"role": "user", "content": "hola"}], tools=TOOLS)

    assert recorder["url"].endswith("/api/chat")
    assert recorder["headers"] is None
    assert recorder["json"]["think"] is True
