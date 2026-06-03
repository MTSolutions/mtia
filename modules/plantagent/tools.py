"""Tool catalog — thin, validated wrappers over mtapi2 indicator functions.

Each tool advertises an OpenAI/Ollama-style JSON schema to the LLM, then on
dispatch it: (1) validates the LLM's arguments against the request's scope
(device must belong to the plant, period must be resolvable), (2) calls the
corresponding mtapi2 function via the injected client, and (3) returns a
JSON-able result the agent feeds back to the model.

The LLM never computes figures and never supplies the client — that comes from
the JWT via ToolContext. Comparisons ("which equipment is worst") are computed
in code from official per-device figures, not by the model.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Iterator

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


# Per-device indicators: name -> human description. Each maps 1:1 to an mtapi2
# function taking (client, start, end, devid) and returning a 0..1 ratio.
_INDICATORS = {
    "oee": "OEE (eficiencia general de equipos)",
    "disponibilidad": "Disponibilidad",
    "rendimiento": "Rendimiento (desempeño)",
    "calidad": "Calidad",
    "cumplimiento": "Cumplimiento (puede no estar disponible para todos los clientes)",
}

_PERIOD_PROP = {
    "type": "string",
    "description": "Período relativo: 'hoy', 'ayer', 'últimos 3 días', "
                   "'esta semana', 'este mes'.",
}


def _mtapi(ctx: ToolContext) -> Callable:
    return ctx.mtapi_call or mtapi.call


def _call(ctx: ToolContext, fn: str, *args):
    try:
        return _mtapi(ctx)(fn, ctx.client, *args)
    except mtapi.MtapiError as e:
        # Includes MtapiUnavailable ("indicator no disponible para esta planta").
        raise ToolError(str(e))


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


def _make_indicator_tool(fn_name: str) -> Callable[[dict, ToolContext], dict]:
    def _tool(args: dict, ctx: ToolContext) -> dict:
        devid = _resolve_devid(args, ctx)
        start, end = _resolve_period(args, ctx)
        value = _call(ctx, fn_name, start, end, devid)
        return {
            "indicator": fn_name,
            "devid": devid,
            "value": value,
            "period": [start.isoformat(), end.isoformat()],
        }
    return _tool


def _iter_dev_nodes(node: dict) -> Iterator[dict]:
    """Depth-first yield of every ``type == 'dev'`` node in a (nested) devtree."""
    if node.get("type") == "dev":
        yield node
    for child_key in ("plants", "lines", "sections", "devs"):
        for child in node.get(child_key, []):
            yield from _iter_dev_nodes(child)


def _tool_rank_oee(args: dict, ctx: ToolContext) -> dict:
    """Rank the plant's devices by OEE (worst first) from a single devtree call.

    devtree computes the official per-device OEE server-side; we only sort —
    the model never does arithmetic.
    """
    start, end = _resolve_period(args, ctx)
    tree = _call(ctx, "devtree", start, end, "plant", ctx.plant_id, ["oee"], False)
    in_scope = set(ctx.device_ids)
    devs = [
        n for n in _iter_dev_nodes(tree)
        if n.get("oee") is not None and n.get("id") in in_scope
    ]
    devs.sort(key=lambda n: n["oee"])  # ascending: worst OEE first
    return {
        "indicator": "oee",
        "ranking": "worst_first",
        "devices": [
            {"devid": n["id"], "name": n.get("name"), "oee": n["oee"]}
            for n in devs[:10]
        ],
        "period": [start.isoformat(), end.isoformat()],
    }


def _indicator_spec(name: str, desc: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "{} OFICIAL de un equipo en un período. "
                           "Devuelve un número entre 0 y 1.".format(desc),
            "parameters": {
                "type": "object",
                "properties": {
                    "devid": {
                        "type": "integer",
                        "description": "ID del equipo; debe pertenecer a la planta.",
                    },
                    "period": _PERIOD_PROP,
                },
                "required": ["devid", "period"],
            },
        },
    }


_RANK_OEE_SPEC = {
    "type": "function",
    "function": {
        "name": "rank_oee",
        "description": "Clasifica los equipos de la planta por OEE (peor primero) "
                       "en un período. Úsalo para preguntas comparativas como "
                       "'¿qué equipo afecta más el OEE?' o 'el peor equipo'.",
        "parameters": {
            "type": "object",
            "properties": {"period": _PERIOD_PROP},
            "required": ["period"],
        },
    },
}

# Advertised to the LLM.
TOOL_SPECS = [_indicator_spec(n, d) for n, d in _INDICATORS.items()] + [_RANK_OEE_SPEC]

_DISPATCH: dict[str, Callable[[dict, ToolContext], dict]] = {
    name: _make_indicator_tool(name) for name in _INDICATORS
}
_DISPATCH["rank_oee"] = _tool_rank_oee


def dispatch(name: str, args: dict, ctx: ToolContext) -> dict:
    """Validate and execute a tool call; return a JSON-able result dict.

    Raises:
        ToolError: unknown tool, invalid arguments, or mtapi2 failure.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        raise ToolError("herramienta desconocida: {!r}".format(name))
    return fn(args, ctx)
