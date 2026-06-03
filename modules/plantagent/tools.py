"""Tool catalog — thin, validated wrappers over mtapi2 indicator functions.

Each tool advertises an OpenAI/Ollama-style JSON schema to the LLM, then on
dispatch it: (1) validates the LLM's arguments against the request's scope
(device must belong to the plant, period must be resolvable), (2) calls the
corresponding mtapi2 function via the injected client, and (3) returns a
JSON-able result the agent feeds back to the model.

The LLM never computes figures and never supplies the client — that comes from
the JWT via ToolContext.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable

from modules.plantagent import mtapi, periods


class ToolError(ValueError):
    """A tool call was invalid or could not be executed."""


@dataclass
class ToolContext:
    """Per-request scope passed to every tool. Built by the router from the JWT."""
    client: str
    plant_id: int
    device_ids: list[int]
    now: dt.datetime          # timezone-aware reference instant
    tz: str                   # IANA timezone of the plant
    # Resolved to mtapi.call at call time when None, so it stays patchable and
    # tests can inject a stub.
    mtapi_call: Callable | None = None


# Advertised to the LLM. One entry per tool; MVP skeleton ships only `oee`.
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "oee",
            "description": (
                "OEE (eficiencia general de equipos) OFICIAL de un equipo en un "
                "período. Devuelve un número entre 0 y 1."),
            "parameters": {
                "type": "object",
                "properties": {
                    "devid": {
                        "type": "integer",
                        "description": "ID del equipo; debe pertenecer a la planta.",
                    },
                    "period": {
                        "type": "string",
                        "description": "Período relativo: 'hoy', 'ayer', "
                                       "'últimos 3 días', 'esta semana', 'este mes'.",
                    },
                },
                "required": ["devid", "period"],
            },
        },
    },
]


def _resolve_devid(args: dict, ctx: ToolContext) -> int:
    raw = args.get("devid")
    try:
        devid = int(raw)
    except (TypeError, ValueError):
        raise ToolError("devid inválido: {!r}".format(raw))
    if devid not in ctx.device_ids:
        raise ToolError(
            "el equipo {} no pertenece a la planta {}".format(devid, ctx.plant_id))
    return devid


def _resolve_period(args: dict, ctx: ToolContext) -> tuple[dt.datetime, dt.datetime]:
    phrase = args.get("period") or "hoy"
    try:
        return periods.resolve(phrase, ctx.now, ctx.tz)
    except periods.PeriodError as e:
        raise ToolError(str(e))


def _tool_oee(args: dict, ctx: ToolContext) -> dict:
    devid = _resolve_devid(args, ctx)
    start, end = _resolve_period(args, ctx)
    call = ctx.mtapi_call or mtapi.call
    try:
        value = call("oee", ctx.client, start, end, devid)
    except mtapi.MtapiError as e:
        raise ToolError(str(e))
    return {
        "devid": devid,
        "oee": value,
        "period": [start.isoformat(), end.isoformat()],
    }


_DISPATCH: dict[str, Callable[[dict, ToolContext], dict]] = {
    "oee": _tool_oee,
}


def dispatch(name: str, args: dict, ctx: ToolContext) -> dict:
    """Validate and execute a tool call; return a JSON-able result dict.

    Raises:
        ToolError: unknown tool, invalid arguments, or mtapi2 failure.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        raise ToolError("herramienta desconocida: {!r}".format(name))
    return fn(args, ctx)
