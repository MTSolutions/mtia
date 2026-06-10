"""Tool-calling loop tests — LLM and mtapi stubbed, no network."""
from __future__ import annotations

import datetime as dt

from modules.plantagent import agent
from modules.plantagent.tools import ToolContext
from modules.rag import llm

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 6, 3, 12, 0, tzinfo=UTC)


def make_ctx(device_ids=(1079,), mtapi_result=0.87):
    calls = []

    def _mtapi(fn, client, *args):
        calls.append((fn, client, args))
        return mtapi_result

    ctx = ToolContext(
        client="degasa", plant_id=7,
        devices=[{"id": d, "name": None} for d in device_ids],
        now=NOW, tz="America/Santiago", mtapi_call=_mtapi,
    )
    ctx_calls = calls  # expose for assertions
    return ctx, ctx_calls


def script_chat_tools(monkeypatch, responses):
    seq = iter(responses)

    async def _fake(messages, tool_specs, model=None, options=None, think=True):
        return next(seq)

    monkeypatch.setattr(llm, "chat_tools", _fake)


def script_chat_stream(monkeypatch, pieces):
    async def _fake(messages, model=None, options=None, think=False):
        for p in pieces:
            yield p

    monkeypatch.setattr(llm, "chat_stream", _fake)


async def _collect(question, ctx):
    return [ev async for ev in agent.run(question, ctx)]


def _oee_call(devid, period="hoy"):
    return {"tool_calls": [{"function": {"name": "oee",
                                         "arguments": {"devid": devid, "period": period}}}]}


async def test_happy_path_calls_tool_then_streams_answer(monkeypatch):
    ctx, mtapi_calls = make_ctx()
    script_chat_tools(monkeypatch, [_oee_call(1079), {"content": "listo"}])
    script_chat_stream(monkeypatch, ["El OEE ", "es 87%."])

    events = await _collect("¿OEE del equipo 1079 hoy?", ctx)
    names = [n for n, _ in events]

    assert "tool" in names
    tool_payload = next(p for n, p in events if n == "tool")
    assert tool_payload["name"] == "oee"
    assert tool_payload["period"] is not None       # resolved period in the audit event
    assert "error" not in tool_payload

    tokens = "".join(p["text"] for n, p in events if n == "token")
    assert tokens == "El OEE es 87%."
    assert names[-1] == "done"

    # The figure came from a real mtapi2 call with the JWT client, devid last.
    assert mtapi_calls[0][0] == "oee"
    assert mtapi_calls[0][1] == "degasa"
    assert mtapi_calls[0][2][-1] == 1079


async def test_out_of_scope_devid_is_rejected_without_calling_mtapi(monkeypatch):
    ctx, mtapi_calls = make_ctx(device_ids=(1079,))
    script_chat_tools(monkeypatch, [_oee_call(9999), {"content": "no disponible"}])
    script_chat_stream(monkeypatch, ["No tengo ese equipo."])

    events = await _collect("OEE del equipo 9999", ctx)

    tool_payload = next(p for n, p in events if n == "tool")
    assert "error" in tool_payload
    assert mtapi_calls == []                          # never called mtapi2
    assert [n for n, _ in events][-1] == "done"


async def test_max_tool_calls_is_bounded(monkeypatch):
    ctx, _ = make_ctx()

    async def _always_tool(messages, tool_specs, model=None, options=None, think=True):
        return _oee_call(1079)                         # never stops asking for tools

    monkeypatch.setattr(llm, "chat_tools", _always_tool)
    script_chat_stream(monkeypatch, ["fin"])

    events = await _collect("bucle", ctx)
    tool_events = [n for n, _ in events if n == "tool"]

    assert len(tool_events) == agent.MAX_TOOL_CALLS    # bounded, no infinite loop
    assert [n for n, _ in events][-1] == "done"


async def test_unexpected_tool_error_does_not_crash_stream(monkeypatch):
    ctx, _ = make_ctx()
    script_chat_tools(monkeypatch, [_oee_call(1079), {"content": "ok"}])
    script_chat_stream(monkeypatch, ["respuesta"])

    def boom(name, args, c):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(agent.tools, "dispatch", boom)

    events = await _collect("q", ctx)
    names = [n for n, _ in events]
    tool_payload = next(p for n, p in events if n == "tool")
    assert "error" in tool_payload            # surfaced, not raised
    assert names[-1] == "done"                # stream still completed


async def test_fatal_llm_error_emits_error_event(monkeypatch):
    ctx, _ = make_ctx()

    async def boom(messages, tool_specs, model=None, options=None, think=True):
        raise RuntimeError("llm down")

    monkeypatch.setattr(agent.llm, "chat_tools", boom)

    events = await _collect("q", ctx)
    assert "error" in [n for n, _ in events]   # clean error event, no crash


async def test_empty_turn_is_retried(monkeypatch):
    ctx, mtapi_calls = make_ctx()
    # first turn stalls (empty), then a real tool call, then a final turn
    script_chat_tools(monkeypatch, [{}, _oee_call(1079), {"content": "ok"}])
    script_chat_stream(monkeypatch, ["listo"])

    events = await _collect("q", ctx)
    names = [n for n, _ in events]
    assert "tool" in names              # recovered past the empty turn
    assert mtapi_calls and mtapi_calls[0][0] == "oee"
    assert names[-1] == "done"


async def test_persistent_empty_turns_stop_and_fallback(monkeypatch):
    ctx, _ = make_ctx()

    async def always_empty(messages, tool_specs, model=None, options=None, think=True):
        return {}

    monkeypatch.setattr(agent.llm, "chat_tools", always_empty)
    script_chat_stream(monkeypatch, [])   # final answer also empty

    events = await _collect("q", ctx)
    names = [n for n, _ in events]
    assert names[-1] == "done"            # bounded — no infinite loop
    assert any(n == "token" for n, _ in events)   # guard fallback message emitted


async def test_no_tool_call_just_streams(monkeypatch):
    ctx, mtapi_calls = make_ctx()
    script_chat_tools(monkeypatch, [{"content": "respuesta directa"}])
    script_chat_stream(monkeypatch, ["Hola."])

    events = await _collect("¿qué puedes hacer?", ctx)

    assert not any(n == "tool" for n, _ in events)
    assert mtapi_calls == []
    assert [n for n, _ in events] == ["token", "done"]


async def test_tool_result_carries_tool_call_id(monkeypatch):
    """OpenAI-compatible backends correlate tool results by tool_call_id."""
    ctx, _ = make_ctx()
    call = {"tool_calls": [{"id": "call_xyz",
                            "function": {"name": "oee",
                                         "arguments": {"devid": 1079, "period": "hoy"}}}]}
    script_chat_tools(monkeypatch, [call, {"content": "listo"}])

    seen = {}

    async def fake_stream(messages, model=None, options=None, think=False):
        seen["messages"] = messages
        yield "ok"

    monkeypatch.setattr(llm, "chat_stream", fake_stream)

    await _collect("¿OEE 1079 hoy?", ctx)

    tool_msgs = [m for m in seen["messages"] if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_xyz"
    # Ollama-style correlation by name is preserved alongside.
    assert tool_msgs[0]["tool_name"] == "oee"


async def test_textual_tool_call_is_dispatched_not_answered(monkeypatch):
    """A tool call emitted as JSON text (gpt-oss style) goes through dispatch."""
    ctx, mtapi_calls = make_ctx()
    script_chat_tools(monkeypatch, [
        {"content": '{"tool": "oee", "input": {"devid": 1079, "period": "hoy"}}'},
        {"content": "listo"},
    ])
    script_chat_stream(monkeypatch, ["El OEE es 87%."])

    events = await _collect("¿OEE 1079 hoy?", ctx)

    tool_events = [p for n, p in events if n == "tool"]
    assert tool_events and tool_events[0]["name"] == "oee"
    assert mtapi_calls and mtapi_calls[0][0] == "oee"


async def test_textual_unknown_tool_feeds_error_back(monkeypatch):
    """An invented tool name becomes a ToolError fed back, never answer text."""
    ctx, mtapi_calls = make_ctx()
    script_chat_tools(monkeypatch, [
        {"content": '{"tool": "list_machines", "input": {}}'},
        {"content": "no hay tal herramienta, respondo del contexto"},
    ])
    script_chat_stream(monkeypatch, ["Las máquinas visibles son…"])

    events = await _collect("¿máquinas visibles?", ctx)

    tool_events = [p for n, p in events if n == "tool"]
    assert tool_events and "error" in tool_events[0]
    assert mtapi_calls == []                      # nothing hit mtapi2
    tokens = "".join(p["text"] for n, p in events if n == "token")
    assert "list_machines" not in tokens          # JSON never leaks as answer


async def test_final_round_gets_prose_instruction(monkeypatch):
    ctx, _ = make_ctx()
    script_chat_tools(monkeypatch, [{"content": "respuesta"}])
    seen = {}

    async def fake_stream(messages, model=None, options=None, think=False):
        seen["messages"] = messages
        yield "ok"

    monkeypatch.setattr(llm, "chat_stream", fake_stream)

    await _collect("hola", ctx)

    last = seen["messages"][-1]
    assert last["role"] == "system" and "prosa" in last["content"]
