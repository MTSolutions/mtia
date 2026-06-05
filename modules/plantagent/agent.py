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
import logging
import os
import time
from typing import AsyncIterator

from modules.plantagent import prompts, schemas, tools
from modules.plantagent.tools import ToolContext
from modules.rag import llm

MAX_TOOL_CALLS = 6
# Reasoning models occasionally emit an empty turn (all in `thinking`, no
# tool_calls and no content). Retry the tool step a couple of times before
# giving up, instead of falling straight through to an empty answer.
MAX_STALL_RETRIES = 2
# Ollama's default context (4096) is too small for the tool catalog + thinking:
# the model truncates mid-reasoning and emits empty turns. Size it for the
# specs + history + reasoning budget.
NUM_CTX = int(os.environ.get("PLANTAGENT_NUM_CTX", "16384"))
_LLM_OPTS = {"num_ctx": NUM_CTX}

logger = logging.getLogger(__name__)


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

    # Latency breakdown (T9): LLM rounds vs mtapi2 tools vs answer streaming.
    t_start = time.monotonic()
    llm_s = tools_s = 0.0
    rounds = 0
    ttft_s = None

    try:
        calls = 0
        hit_cap = False
        stalls = 0
        while True:
            if calls >= MAX_TOOL_CALLS:
                hit_cap = True
                break
            t0 = time.monotonic()
            msg = await llm.chat_tools(messages, tools.TOOL_SPECS, options=_LLM_OPTS)
            llm_s += time.monotonic() - t0
            rounds += 1
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # Empty turn (no tool call, no content) -> the model stalled;
                # retry the tool step a few times before settling for an answer.
                if not (msg.get("content") or "").strip() and stalls < MAX_STALL_RETRIES:
                    stalls += 1
                    continue
                break

            messages.append(_assistant_turn(msg))
            for tc in tool_calls:
                calls += 1
                fn = tc.get("function") or {}
                name = fn.get("name")
                args = _coerce_args(fn.get("arguments"))
                t0 = time.monotonic()
                try:
                    result = tools.dispatch(name, args, ctx)
                    dt = time.monotonic() - t0
                    tools_s += dt
                    yield schemas.EVENT_TOOL, {
                        "name": name, "args": args, "period": result.get("period"),
                        "elapsed": round(dt, 2)}
                except tools.ToolError as e:
                    tools_s += time.monotonic() - t0
                    result = {"error": str(e)}
                    yield schemas.EVENT_TOOL, {"name": name, "args": args, "error": str(e)}
                except Exception:  # unexpected — never crash the stream over one tool
                    result = {"error": "error interno al ejecutar la herramienta"}
                    yield schemas.EVENT_TOOL, {
                        "name": name, "args": args, "error": "error interno"}
                messages.append({
                    "role": "tool", "tool_name": name,
                    "content": json.dumps(result, default=str)})

        if hit_cap:
            messages.append({
                "role": "system",
                "content": "Se alcanzó el límite de consultas. Responde con la "
                           "información disponible y aclara si quedó incompleta. "
                           "No inventes cifras.",
            })

        # Final user-facing answer: streamed prose, no tools, no chain-of-thought.
        streamed = False
        async for token in llm.chat_stream(messages, options=_LLM_OPTS):
            if not streamed:
                ttft_s = time.monotonic() - t_start
            streamed = True
            yield schemas.EVENT_TOKEN, {"text": token}
        if not streamed:
            # The model produced no answer (e.g. only thinking) — never return blank.
            yield schemas.EVENT_TOKEN, {"text": (
                "No pude responder con las herramientas disponibles. Reformula la "
                "pregunta indicando equipo/línea/sección y período.")}
        total_s = time.monotonic() - t_start
        yield schemas.EVENT_DONE, {"timing": {
            "total_s": round(total_s, 2),
            "llm_s": round(llm_s, 2),            # tool-selection rounds
            "tools_s": round(tools_s, 2),        # mtapi2 execution
            "answer_s": round(total_s - llm_s - tools_s, 2),
            "ttft_s": round(ttft_s, 2) if ttft_s is not None else None,
            "rounds": rounds, "tool_calls": calls}}
    except Exception:
        # Any unhandled failure (e.g. LLM transport) ends as a clean SSE error,
        # never a fabricated answer or a broken stream. Log the traceback —
        # the SSE message is generic on purpose, the log must not be.
        logger.exception("plantagent run failed (client=%s plant_id=%s)",
                         ctx.client, ctx.plant_id)
        yield schemas.EVENT_ERROR, {
            "message": "No se pudo completar la consulta. Inténtalo nuevamente."}
        yield schemas.EVENT_DONE, {}
