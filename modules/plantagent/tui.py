"""Interactive CLI to exercise the Plant Agent chat endpoint (dev/test tool).

Runs inside the mtia container — it already has httpx, pyjwt, JWT_SECRET, can
reach the service at localhost:8008, and the LLM via the configured backend.
A JWT is minted locally from the shared JWT_SECRET so no `api` login is needed.

Interactive:
    docker compose exec -it mtia python -m modules.plantagent.tui \
        --client degasa --plant-id 78

One-shot:
    docker compose exec -T mtia python -m modules.plantagent.tui \
        --client degasa --plant-id 78 --question "¿cuál es el OEE del equipo 1280 hoy?"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Iterator

import httpx
import jwt

DIM, RED, CYAN, RESET = "\033[2m", "\033[31m", "\033[36m", "\033[0m"


def mint_token(client: str, login: str = "tui") -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise SystemExit("JWT_SECRET not set in the container environment")
    return jwt.encode(
        {"sub": 1, "roles": "admin", "client": client, "login": login},
        secret, algorithm="HS512")


def parse_sse(lines: Iterator[str]):
    """Yield (event, payload) tuples from raw SSE lines."""
    event, data = None, []
    for line in lines:
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:"):].strip())
        elif line == "":
            if event is not None:
                yield event, (json.loads("\n".join(data)) if data else {})
            event, data = None, []
    if event is not None:
        yield event, (json.loads("\n".join(data)) if data else {})


def ask(base_url: str, token: str, client: str, plant_id: int, question: str,
        conversation_id: str | None = None) -> None:
    params = {"client": client, "plant_id": plant_id, "question": question}
    if conversation_id:
        params["conversation_id"] = conversation_id
    headers = {"Authorization": "JWT " + token}
    with httpx.stream("POST", base_url + "/plantagent/chat",
                      params=params, headers=headers, timeout=None) as r:
        if r.status_code != 200:
            body = r.read().decode("utf-8", "replace")
            print("{}HTTP {}: {}{}".format(RED, r.status_code, body, RESET))
            return
        for event, payload in parse_sse(r.iter_lines()):
            if event == "tool":
                if payload.get("error"):
                    print("{}  ⚙ {}({}) → error: {}{}".format(
                        DIM, payload.get("name"), payload.get("args"),
                        payload["error"], RESET))
                else:
                    print("{}  ⚙ {}({}) período={}{}".format(
                        DIM, payload.get("name"), payload.get("args"),
                        payload.get("period"), RESET))
            elif event == "token":
                sys.stdout.write(payload.get("text", ""))
                sys.stdout.flush()
            elif event == "error":
                print("{}\n[error] {}{}".format(RED, payload.get("message", ""), RESET))
            elif event == "done":
                print()
                t = payload.get("timing")
                if t:
                    print("{}  ⏱ total={}s llm={}s tools={}s answer={}s "
                          "ttft={}s rounds={} calls={}{}".format(
                              DIM, t.get("total_s"), t.get("llm_s"),
                              t.get("tools_s"), t.get("answer_s"), t.get("ttft_s"),
                              t.get("rounds"), t.get("tool_calls"), RESET))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plant Agent test client")
    ap.add_argument("--client", required=True)
    ap.add_argument("--plant-id", type=int, required=True)
    ap.add_argument("--question", help="one-shot; omit for an interactive loop")
    ap.add_argument("--base-url",
                    default=os.environ.get("PLANTAGENT_BASE_URL", "http://localhost:8008"))
    ap.add_argument("--login", default="tui")
    args = ap.parse_args()

    token = mint_token(args.client, args.login)
    if args.question:
        ask(args.base_url, token, args.client, args.plant_id, args.question)
        return

    # One conversation per interactive session: follow-ups ("¿y ayer?")
    # resolve against the previous turns. One-shot stays stateless.
    conversation_id = uuid.uuid4().hex
    print("{}Plant Agent — client={} plant={} (Ctrl-C para salir, "
          "memoria de conversación activa){}".format(
              CYAN, args.client, args.plant_id, RESET))
    try:
        while True:
            q = input("{}\n> {}".format(CYAN, RESET)).strip()
            if q:
                ask(args.base_url, token, args.client, args.plant_id, q,
                    conversation_id=conversation_id)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
