"""FastAPI router for /plantagent/* — JWT-scoped, SSE chat over plant indicators."""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from modules.plantagent import agent, memory, mtapi, schemas, scope
from modules.plantagent.tools import ToolContext
from modules.rag.auth import JwtClaims, require_client_match, verify_jwt


router = APIRouter(prefix="/plantagent", tags=["plantagent"])

# Per-plant timezone has no clean mtapi2 getter yet (it lives on
# client.time_zone_str); adding one is an ask-first change to the Py2.7 service.
# Until then the plant tz defaults from env. TODO(MTS-1285): resolve per plant.
DEFAULT_TZ = os.environ.get("PLANTAGENT_DEFAULT_TZ", "America/Santiago")


@router.get("/health")
def health():
    return {"status": "ok", "module": "plantagent"}


@router.post("/chat")
async def chat(
    client: str,
    plant_id: int,
    question: str,
    conversation_id: str | None = None,
    claims: JwtClaims = Depends(verify_jwt),
):
    """Stream a grounded answer about a plant's official indicators as SSE.

    `conversation_id` (caller-chosen, e.g. a UUID) opts into multi-turn
    memory: prior exchanges under the same id are replayed so follow-ups
    ("¿y ayer?") resolve. Without it each request stays stateless.
    """
    require_client_match(client, claims)
    try:
        plant = scope.validate_plant(client, plant_id)   # enforces plant ∈ client
        tree = scope.named_tree(client, plant_id)         # name-annotated config
        devices = scope.devices_in(tree)
    except scope.PlantNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except mtapi.MtapiError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "servicio de indicadores (mtapi2) no disponible")

    ctx = ToolContext(
        client=client,
        plant_id=plant_id,
        devices=devices,
        now=dt.datetime.now(dt.timezone.utc),
        tz=DEFAULT_TZ,
        plant_name=tree.get("name") or plant.get("name"),
        tree=tree,
    )

    # Memory key carries the JWT identity: a guessed conversation_id from
    # another client/login resolves to a different (empty) conversation.
    mem_key = ((claims.client, claims.login, plant_id, conversation_id)
               if conversation_id else None)
    history = memory.store.history(mem_key) if mem_key else []

    async def event_stream() -> AsyncIterator[dict]:
        answer_parts: list[str] = []
        failed = False
        async for event, data in agent.run(question, ctx, history=history):
            if event == schemas.EVENT_TOKEN:
                answer_parts.append(data.get("text", ""))
            elif event == schemas.EVENT_ERROR:
                failed = True
            yield {"event": event, "data": json.dumps(data, default=str)}
        # A failed turn is not remembered — replaying the apology as context
        # would only confuse the next round.
        if mem_key and not failed and answer_parts:
            memory.store.append(mem_key, question, "".join(answer_parts))

    return EventSourceResponse(event_stream())
