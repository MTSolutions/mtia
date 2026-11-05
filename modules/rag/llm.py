"""LLM chat client (streaming) — Ollama native or an OpenAI-compatible backend.

The backend is selected per deployment via env (``LLM_PROVIDER``), so mtia code
does not change between dev, prod and fallback:

- ``ollama`` (default) — Ollama's native ``/api/chat``: local dev (host Metal)
  and the self-hosted GPU node. In-environment, no egress.
- ``openrouter`` — OpenAI-compatible ``/chat/completions`` with Bearer auth.
  Works against OpenRouter or any OpenAI-compatible server (vLLM, llama.cpp
  ``llama-server``, …) by overriding ``LLM_URL``. **OpenRouter is external
  egress**: opt-in only (model bake-offs, per-client fallback with sign-off);
  by construction the agent only ever sends the question, the tool schemas and
  aggregated indicator results — never raw rows.

Callers (rag, plantagent) speak one internal message shape — Ollama's, with
``tool_calls[].function.arguments`` as a dict and tool results as
``{"role": "tool", "tool_name": ..., "tool_call_id": ..., "content": ...}`` —
and the OpenAI adapter translates in both directions (arguments to/from JSON
strings, ``tool_call_id`` correlation, which frontier APIs require).
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1"


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "ollama").strip().lower()


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")


def _openai_url() -> str:
    return os.environ.get("LLM_URL", OPENROUTER_URL).rstrip("/")


def _llm_model() -> str:
    return os.environ.get("LLM_MODEL", "gemma4:e4b")


def _openai_headers() -> dict:
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "LLM_PROVIDER=%s requiere LLM_API_KEY u OPENROUTER_API_KEY" % _provider()
        )
    return {"Authorization": f"Bearer {key}"}


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Translate the internal (Ollama-shaped) history to OpenAI chat format.

    Assistant tool calls gain an ``id``/``type`` and JSON-string arguments;
    tool results are correlated by ``tool_call_id`` (OpenAI requires it; Ollama
    correlates by order/name and ignores the extra field).
    """
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for i, tc in enumerate(m["tool_calls"]):
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {}, default=str)
                calls.append({
                    "id": tc.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {"name": fn.get("name"), "arguments": args},
                })
            out.append({"role": "assistant",
                        "content": m.get("content") or "",
                        "tool_calls": calls})
        elif role == "tool":
            out.append({"role": "tool",
                        "tool_call_id": m.get("tool_call_id") or "call_0",
                        "content": m.get("content", "")})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


def _from_openai_message(data: dict) -> dict:
    """Normalize an OpenAI chat completion into the internal message shape."""
    raw = (data.get("choices") or [{}])[0].get("message") or {}
    msg = {"role": "assistant", "content": raw.get("content") or ""}
    calls = []
    for tc in raw.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({"id": tc.get("id"),
                      "function": {"name": fn.get("name"), "arguments": args or {}}})
    if calls:
        msg["tool_calls"] = calls
    return msg


def _openai_payload(messages: list[dict], model: str | None,
                    options: dict | None, stream: bool) -> dict:
    payload = {
        "model": model or _llm_model(),
        "messages": _to_openai_messages(messages),
        "stream": stream,
    }
    # Ollama `options` → OpenAI top-level sampling params. `num_ctx` has no
    # OpenAI equivalent (the server manages context) and is dropped.
    for key in ("temperature", "top_p"):
        if options and key in options:
            payload[key] = options[key]
    return payload


async def chat_tools(
    messages: list[dict],
    tools: list[dict],
    model: str | None = None,
    options: dict | None = None,
    think: bool = True,
) -> dict:
    """Call the chat backend with tools and return the assistant `message`.

    Tool calling requires a complete (non-streamed) response — Ollama only
    populates `message.tool_calls` when `stream=False`. `think` defaults to True
    because Gemma 4's docs note reasoning significantly improves function-calling
    accuracy; the caller streams the final user-facing answer with `chat_stream`.
    (`think` is Ollama-specific; OpenAI-compatible backends use each model's
    default reasoning behavior.)

    Returns the message dict in the internal shape, e.g.::

        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "oee", "arguments": {"devid": 1079}}}]}

    A response with no tool call simply omits `tool_calls`.
    """
    if _provider() != "ollama":
        payload = _openai_payload(messages, model, options, stream=False)
        payload["tools"] = tools
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{_openai_url()}/chat/completions",
                                  json=payload, headers=_openai_headers())
            r.raise_for_status()
            return _from_openai_message(r.json())

    payload = {
        "model": model or _llm_model(),
        "messages": messages,
        "tools": tools,
        "stream": False,
        "think": think,
    }
    if options:
        payload["options"] = options

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{_ollama_url()}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    return data.get("message") or {}


async def chat_stream(
    messages: list[dict],
    model: str | None = None,
    options: dict | None = None,
    think: bool = False,
) -> AsyncIterator[str]:
    """Yield content token chunks from the chat backend with stream=true.

    Reasoning-capable models (Gemma 4, GPT-OSS, DeepSeek-R1, …) emit tokens
    into a separate `thinking` field before `content` when `think` is left at
    its default. For grounded Q&A we want the answer, not the chain-of-thought,
    so we default `think=False`. (OpenAI-compatible backends already separate
    reasoning into `delta.reasoning`; only `delta.content` is yielded.)
    """
    if _provider() != "ollama":
        payload = _openai_payload(messages, model, options, stream=True)
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{_openai_url()}/chat/completions",
                                     json=payload, headers=_openai_headers()) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    # SSE frames: "data: {...}" / "data: [DONE]"; OpenRouter also
                    # interleaves ": OPENROUTER PROCESSING" keep-alive comments.
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError(
                            event["error"].get("message", "LLM stream error"))
                    delta = (event.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        yield delta["content"]
        return

    payload = {
        "model": model or _llm_model(),
        "messages": messages,
        "stream": True,
        "think": think,
    }
    if options:
        payload["options"] = options

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{_ollama_url()}/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get("done"):
                    return
                content = (event.get("message") or {}).get("content")
                if content:
                    yield content
