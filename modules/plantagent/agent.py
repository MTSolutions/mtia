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
# Greedy decoding: la selección de herramienta debe ser determinista — con la
# temperatura default de Ollama (~0.8) la misma pregunta tomaba caminos
# distintos entre corridas (58.4% vs 85.4% para el mismo OEE semanal).
TEMPERATURE = float(os.environ.get("PLANTAGENT_TEMPERATURE", "0"))
_LLM_OPTS = {"num_ctx": NUM_CTX, "temperature": TEMPERATURE}

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


def _textual_tool_call(content) -> dict | None:
    """Detect a tool call emitted as JSON text instead of structured tool_calls.

    Some models (e.g. gpt-oss via OpenRouter) write the call into `content` —
    typically when they invent a tool that isn't in the catalog, e.g.
    ``{"tool": "list_machines", "input": {}}``. Routing it through the normal
    dispatch path lets validation reject it with the list of real tools, so the
    model can self-correct instead of the raw JSON leaking into the answer.
    """
    text = (content or "").strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool") or obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("input") or obj.get("arguments") or obj.get("parameters") or {}
    return {"function": {"name": name,
                         "arguments": args if isinstance(args, dict) else {}}}


def _assistant_turn(msg: dict) -> dict:
    return {
        "role": "assistant",
        "content": msg.get("content", "") or "",
        "tool_calls": msg.get("tool_calls", []),
    }


async def run(question: str, ctx: ToolContext,
              history: list[dict] | None = None) -> AsyncIterator[tuple[str, dict]]:
    # Pista de auditoría en el log del servidor: la evidencia SSE de la UI es
    # efímera; sin esto un "¿de dónde salió esta cifra?" en QA no se puede
    # reconstruir después.
    logger.info("plantagent question client=%s plant_id=%s turns=%d q=%r",
                ctx.client, ctx.plant_id, len(history or []) // 2, question)
    # `history` carries prior prose exchanges (memory.py); only the current
    # question gets the topology framing — repeating it per turn would burn
    # NUM_CTX on identical boilerplate.
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        *(history or []),
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
                textual = _textual_tool_call(msg.get("content"))
                if textual is not None:
                    msg = {"role": "assistant", "content": "",
                           "tool_calls": [textual]}
                    tool_calls = [textual]
                # Empty turn (no tool call, no content) -> the model stalled;
                # retry the tool step a few times before settling for an answer.
                elif not (msg.get("content") or "").strip() and stalls < MAX_STALL_RETRIES:
                    stalls += 1
                    continue
                else:
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
                    logger.info("plantagent tool %s args=%s period=%s %.2fs",
                                name, args, result.get("period"), dt)
                    yield schemas.EVENT_TOOL, {
                        "name": name, "args": args, "period": result.get("period"),
                        "elapsed": round(dt, 2)}
                except tools.ToolError as e:
                    tools_s += time.monotonic() - t0
                    result = {"error": str(e)}
                    logger.warning("plantagent tool %s args=%s error=%s", name, args, e)
                    yield schemas.EVENT_TOOL, {"name": name, "args": args, "error": str(e)}
                except Exception:  # unexpected — never crash the stream over one tool
                    result = {"error": "error interno al ejecutar la herramienta"}
                    yield schemas.EVENT_TOOL, {
                        "name": name, "args": args, "error": "error interno"}
                # tool_call_id correlates the result with its call on OpenAI-
                # compatible backends (OpenRouter/vLLM); Ollama matches by
                # name/order and ignores it.
                messages.append({
                    "role": "tool", "tool_name": name,
                    "tool_call_id": tc.get("id") or f"call_{calls - 1}",
                    "content": json.dumps(result, default=str)})

        if hit_cap:
            messages.append({
                "role": "system",
                "content": "Se alcanzó el límite de consultas. Responde con la "
                           "información disponible y aclara si quedó incompleta. "
                           "No inventes cifras.",
            })

        # Final user-facing answer: streamed prose, no tools, no chain-of-thought.
        # The explicit prose instruction keeps models that lean on tool syntax
        # (gpt-oss) from emitting a JSON tool call as the visible answer.
        messages.append({
            "role": "system",
            "content": "Responde ahora al usuario en español, en prosa. No "
                       "llames herramientas ni emitas JSON.",
        })
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
        logger.info("plantagent done client=%s plant_id=%s total=%.1fs rounds=%d tools=%d",
                    ctx.client, ctx.plant_id, total_s, rounds, calls)
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
