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
    """Per-request scope passed to every tool. Built by the router from the JWT.

    ``devices`` is the name-annotated source of truth (``[{id, name}]``); the
    agent presents and resolves equipment by name, not raw id.
    """
    client: str
    plant_id: int
    devices: list[dict]       # [{"id": int, "name": str}]
    now: dt.datetime          # timezone-aware reference instant
    tz: str                   # IANA timezone of the plant
    plant_name: str | None = None
    # Resolved to mtapi.call at call time when None, so it stays patchable and
    # tests can inject a stub.
    mtapi_call: Callable | None = None

    @property
    def device_ids(self) -> list[int]:
        return [d["id"] for d in self.devices]

    def name_for(self, devid: int) -> str | None:
        for d in self.devices:
            if d["id"] == devid:
                return d.get("name")
        return None

    def resolve_device(self, ref) -> int | None:
        """Map a device reference (id, numeric string, or name) to a devid.

        Names match case-insensitively (exact first, then substring). Returns
        None if nothing in scope matches.
        """
        if isinstance(ref, bool):
            return None
        if isinstance(ref, int):
            return ref if ref in self.device_ids else None
        s = str(ref).strip()
        if s.lstrip("-").isdigit():
            devid = int(s)
            return devid if devid in self.device_ids else None
        low = s.lower()
        for d in self.devices:
            if (d.get("name") or "").strip().lower() == low:
                return d["id"]
        for d in self.devices:
            if low and low in (d.get("name") or "").strip().lower():
                return d["id"]
        return None


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


def _resolve_device(args: dict, ctx: ToolContext) -> int:
    """Resolve the 'device' (name or id) argument to a devid within the plant.

    Accepts the legacy 'devid' key too. Raises ToolError listing available
    equipment names when the reference can't be matched.
    """
    ref = args.get("device", args.get("devid"))
    devid = ctx.resolve_device(ref)
    if devid is None:
        names = [d.get("name") for d in ctx.devices][:30]
        raise ToolError(
            "no encontré el equipo {!r} en la planta. Equipos disponibles: {}".format(
                ref, names))
    return devid


def _resolve_period(args: dict, ctx: ToolContext) -> tuple[dt.datetime, dt.datetime]:
    phrase = args.get("period") or "hoy"
    try:
        return periods.resolve(phrase, ctx.now, ctx.tz)
    except periods.PeriodError as e:
        raise ToolError(str(e))


def _make_indicator_tool(fn_name: str) -> Callable[[dict, ToolContext], dict]:
    def _tool(args: dict, ctx: ToolContext) -> dict:
        devid = _resolve_device(args, ctx)
        start, end = _resolve_period(args, ctx)
        value = _call(ctx, fn_name, start, end, devid)
        return {
            "indicator": fn_name,
            "devid": devid,
            "device": ctx.name_for(devid),
            "value": value,
            "no_data": value is None,
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
    """Rank the plant's devices by OEE (worst first), via per-device oee() calls.

    We deliberately do NOT use devtree(indicators=['oee']): mtapi2 computes a
    section/plant indicator by passing a *list* of devids to oee(), which faults
    (IndexError) for the default client. Scalar per-device oee() is reliable.
    The model never does arithmetic — we only sort the official figures.
    """
    start, end = _resolve_period(args, ctx)
    scores: list[tuple[int, float]] = []
    for devid in ctx.device_ids:
        try:
            value = _call(ctx, "oee", start, end, devid)
        except ToolError:
            value = None  # skip a device whose indicator errors; don't abort the rank
        if value is not None:
            scores.append((devid, value))

    scores.sort(key=lambda t: t[1])  # ascending: worst OEE first
    return {
        "indicator": "oee",
        "ranking": "worst_first",
        "devices": [
            {"devid": d, "name": ctx.name_for(d), "oee": v} for d, v in scores[:10]
        ],
        "no_data": not scores,
        "period": [start.isoformat(), end.isoformat()],
    }


def _collect_line_nodes(node: dict) -> list[dict]:
    """Depth-first collect every ``type == 'line'`` node in a (nested) devtree."""
    out: list[dict] = []
    if node.get("type") == "line":
        out.append(node)
    for child_key in ("plants", "lines", "sections", "devs"):
        for child in node.get(child_key, []):
            out.extend(_collect_line_nodes(child))
    return out


def _match_line(line_nodes: list[dict], name: str) -> dict | None:
    target = (name or "").strip().lower()
    for ln in line_nodes:                       # exact match first
        if (ln.get("name") or "").strip().lower() == target:
            return ln
    for ln in line_nodes:                       # then contains
        if target and target in (ln.get("name") or "").strip().lower():
            return ln
    return None


def _tool_top_stops(args: dict, ctx: ToolContext) -> dict:
    """Most significant stops in a period, by total time or by occurrence count.

    Aggregation is mtapi2's official `pareto` (grouped by cod_state); we only
    choose the sort key. Optionally scoped to a named line, resolved via devtree.
    """
    start, end = _resolve_period(args, ctx)
    by = args.get("by") or "time"
    line = args.get("line")

    if line:
        tree = _call(ctx, "devtree_named", "plant", ctx.plant_id)
        line_nodes = _collect_line_nodes(tree)
        match = _match_line(line_nodes, line)
        if match is None:
            names = [n.get("name") for n in line_nodes]
            raise ToolError(
                "no encontré la línea {!r}. Líneas disponibles: {}".format(line, names))
        devids = [n["id"] for n in _iter_dev_nodes(match)]
        scope_label = match.get("name") or line
    else:
        devids = list(ctx.device_ids)
        scope_label = "planta"

    if not devids:
        raise ToolError("no hay equipos en el alcance indicado")

    data = _call(ctx, "pareto", start, end, devids) or {}
    rows = data.get("codstates", []) or []
    sort_key = "num" if by == "count" else "time_s"
    rows = sorted(rows, key=lambda r: r.get(sort_key) or 0, reverse=True)
    return {
        "scope": scope_label,
        "by": by,
        "stops": [
            {
                "desc": r.get("desc"),
                "code_f": r.get("code_f"),
                "count": r.get("num"),
                "time_h": r.get("time_s"),
            }
            for r in rows[:10]
        ],
        "no_data": not rows,
        "period": [start.isoformat(), end.isoformat()],
    }


# Production *measure* -> mtapi2 function. These are different production
# calculations, NOT units of measure: the unit (cajas, kg, …) belongs to the
# Device (Device.unit). kp is a per-product-spec multiplier, not a unit.
# 'standard' (prod_dev_kp, counter × kp) is the canonical default counter.
_PROD_MEASURE_FN = {
    "standard": "prod_dev_kp",   # counter weighted by the product-spec kp (default)
    "counter": "prod_dev_t",     # raw counter, without kp
    "tons": "total_tons",        # tonnage
}


def _tool_production(args: dict, ctx: ToolContext) -> dict:
    devid = _resolve_device(args, ctx)
    measure = args.get("measure") or "standard"
    fn = _PROD_MEASURE_FN.get(measure)
    if fn is None:
        raise ToolError(
            "medida inválida: {!r} (usa 'standard', 'counter' o 'tons')".format(measure))
    start, end = _resolve_period(args, ctx)
    produced = _call(ctx, fn, start, end, devid)
    return {
        "devid": devid,
        "device": ctx.name_for(devid),
        "measure": measure,
        "produced": produced,
        # The numeric unit of measure belongs to the device (Device.unit);
        # reporting that label is a follow-up (no mtapi2 getter yet).
        "no_data": produced is None,
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
                    "device": {
                        "type": "string",
                        "description": "Nombre del equipo (o su id) dentro de la planta.",
                    },
                    "period": _PERIOD_PROP,
                },
                "required": ["device", "period"],
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

_TOP_STOPS_SPEC = {
    "type": "function",
    "function": {
        "name": "top_stops",
        "description": "Detenciones más importantes en un período, para toda la "
                       "planta o una línea. Úsalo para '¿la detención más repetida?' "
                       "o '¿la detención más larga?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "period": _PERIOD_PROP,
                "by": {
                    "type": "string",
                    "enum": ["time", "count"],
                    "description": "Ordenar por tiempo total ('time', detención más "
                                   "larga) o por número de ocurrencias ('count', más "
                                   "repetida).",
                },
                "line": {
                    "type": "string",
                    "description": "Nombre de la línea (opcional). Si se omite, "
                                   "abarca toda la planta.",
                },
            },
            "required": ["period"],
        },
    },
}

_PRODUCTION_SPEC = {
    "type": "function",
    "function": {
        "name": "production",
        "description": "Producción de un equipo en un período, en la unidad de "
                       "medida del equipo. Para comparar contra el plan "
                       "('producción vs plan', cumplimiento), usa además la "
                       "herramienta 'cumplimiento'.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Nombre del equipo (o su id) dentro de la planta.",
                },
                "period": _PERIOD_PROP,
                "measure": {
                    "type": "string",
                    "enum": ["standard", "counter", "tons"],
                    "description": "Medida de producción: 'standard' (producción "
                                   "estándar, contador ponderado por el kp de la "
                                   "especificación del producto — es lo normal), "
                                   "'counter' (contador crudo sin kp) o 'tons' "
                                   "(toneladas). Por defecto 'standard'.",
                },
            },
            "required": ["device", "period"],
        },
    },
}

# Advertised to the LLM.
TOOL_SPECS = (
    [_indicator_spec(n, d) for n, d in _INDICATORS.items()]
    + [_RANK_OEE_SPEC, _TOP_STOPS_SPEC, _PRODUCTION_SPEC]
)

_DISPATCH: dict[str, Callable[[dict, ToolContext], dict]] = {
    name: _make_indicator_tool(name) for name in _INDICATORS
}
_DISPATCH["rank_oee"] = _tool_rank_oee
_DISPATCH["top_stops"] = _tool_top_stops
_DISPATCH["production"] = _tool_production


def dispatch(name: str, args: dict, ctx: ToolContext) -> dict:
    """Validate and execute a tool call; return a JSON-able result dict.

    Raises:
        ToolError: unknown tool, invalid arguments, or mtapi2 failure.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        raise ToolError("herramienta desconocida: {!r}".format(name))
    return fn(args, ctx)
