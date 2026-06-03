"""The Plant Agent tool-calling loop.

A bounded loop: ask the LLM (with the tool catalog) what to call, validate +
execute each tool call via mtapi2, feed the results back, and repeat until the
model stops requesting tools (or MAX_TOOL_CALLS is hit). Then stream the final
Spanish answer as prose. Every tool call emits one `tool` audit event so every
figure in the answer is traceable to an mtapi2 call.

`run` is an async generator of (event_name, payload) tuples; the router wraps it
in an SSE response. It is unit-testable by stubbing llm.chat_tools/chat_stream
and ToolContext.mtapi_call — no network required.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from modules.plantagent import prompts, schemas, tools
from modules.plantagent.tools import ToolContext
from modules.rag import llm

MAX_TOOL_CALLS = 6


def _coerce_args(raw) -> dict:
    """Tool-call arguments arrive as a dict, or occasionally a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _assistant_turn(msg: dict) -> dict:
    return {
        "role": "assistant",
        "content": msg.get("content", "") or "",
        "tool_calls": msg.get("tool_calls", []),
    }


async def run(question: str, ctx: ToolContext) -> AsyncIterator[tuple[str, dict]]:
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.build_user_message(question, ctx)},
    ]

    calls = 0
    while calls < MAX_TOOL_CALLS:
        msg = await llm.chat_tools(messages, tools.TOOL_SPECS)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break

        messages.append(_assistant_turn(msg))
        for tc in tool_calls:
            calls += 1
            fn = tc.get("function") or {}
            name = fn.get("name")
            args = _coerce_args(fn.get("arguments"))
            try:
                result = tools.dispatch(name, args, ctx)
                yield schemas.EVENT_TOOL, {
                    "name": name, "args": args, "period": result.get("period")}
            except tools.ToolError as e:
                result = {"error": str(e)}
                yield schemas.EVENT_TOOL, {"name": name, "args": args, "error": str(e)}
            messages.append({
                "role": "tool", "tool_name": name,
                "content": json.dumps(result, default=str)})

    # Final user-facing answer: streamed prose, no tools, no chain-of-thought.
    async for token in llm.chat_stream(messages):
        yield schemas.EVENT_TOKEN, {"text": token}
    yield schemas.EVENT_DONE, {}
