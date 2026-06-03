"""FastAPI router for /plantagent/* — JWT-scoped, SSE chat over plant indicators."""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from modules.plantagent import agent, mtapi, scope
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
    claims: JwtClaims = Depends(verify_jwt),
):
    """Stream a grounded answer about a plant's official indicators as SSE."""
    require_client_match(client, claims)
    try:
        scope.validate_plant(client, plant_id)
        now = dt.datetime.now(dt.timezone.utc)
        naive = now.replace(tzinfo=None)
        device_ids = scope.device_ids(client, "plant", plant_id, naive, naive)
    except scope.PlantNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except mtapi.MtapiError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "servicio de indicadores (mtapi2) no disponible")

    ctx = ToolContext(
        client=client,
        plant_id=plant_id,
        device_ids=device_ids,
        now=now,
        tz=DEFAULT_TZ,
    )

    async def event_stream() -> AsyncIterator[dict]:
        async for event, data in agent.run(question, ctx):
            yield {"event": event, "data": json.dumps(data, default=str)}

    return EventSourceResponse(event_stream())
