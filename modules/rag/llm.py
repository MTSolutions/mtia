"""Ollama chat client (streaming)."""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")


def _llm_model() -> str:
    return os.environ.get("LLM_MODEL", "gemma4:e4b")


async def chat_tools(
    messages: list[dict],
    tools: list[dict],
    model: str | None = None,
    options: dict | None = None,
    think: bool = True,
) -> dict:
    """Call Ollama /api/chat with tools and return the assistant `message`.

    Tool calling requires a complete (non-streamed) response — Ollama only
    populates `message.tool_calls` when `stream=False`. `think` defaults to True
    because Gemma 4's docs note reasoning significantly improves function-calling
    accuracy; the caller streams the final user-facing answer with `chat_stream`.

    Returns the raw `message` dict, e.g.::

        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "oee", "arguments": {"devid": 1079}}}]}

    A response with no tool call simply omits `tool_calls`.
    """
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
    """Yield content token chunks from Ollama /api/chat with stream=true.

    Reasoning-capable models (Gemma 4, GPT-OSS, DeepSeek-R1, …) emit tokens
    into a separate `thinking` field before `content` when `think` is left at
    its default. For grounded Q&A we want the answer, not the chain-of-thought,
    so we default `think=False`.
    """
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
