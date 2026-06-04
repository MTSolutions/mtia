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
import difflib
from dataclasses import dataclass, field
from typing import Callable

from modules.plantagent import mtapi, periods, scope, turns


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
    tree: dict = field(default_factory=dict)   # named config tree (devtree_named)
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
    "description": (
        "Período relativo. Usa EXACTAMENTE uno de estos valores: 'hoy', 'ayer', "
        "'anteayer', 'esta semana', 'semana pasada', 'este mes', 'mes pasado', o "
        "'últimos N días' (con N entero, p.ej. 'últimos 7 días'). "
        "Mapea la frase de la pregunta al valor correcto: "
        "'la semana pasada' → 'semana pasada'; 'el mes pasado' → 'mes pasado'; "
        "'esta semana' → 'esta semana'. No uses 'este mes' para 'la semana pasada'."
    ),
}


def _mtapi(ctx: ToolContext) -> Callable:
    return ctx.mtapi_call or mtapi.call


def _call(ctx: ToolContext, fn: str, *args):
    try:
        return _mtapi(ctx)(fn, ctx.client, *args)
    except mtapi.MtapiError as e:
        # Includes MtapiUnavailable ("indicator no disponible para esta planta").
        raise ToolError(str(e))


def _resolve_node(args: dict, ctx: ToolContext) -> tuple[str, str, list[int]]:
    """Resolve a 'node' (equipment/line/section/plant, by name or id) to its
    device-id list. Returns (label, type, dev_ids).

    Uses the named tree (any node type) when available; otherwise falls back to
    the flat device list (device-only). Raises ToolError when unresolved or the
    node has no equipment.
    """
    ref = args.get("node", args.get("device", args.get("devid")))
    node = scope.resolve_node(ctx.tree, ref) if ctx.tree else None
    if node is not None:
        dev_ids = scope.node_device_ids(node)
        label, ntype = node.get("name"), node.get("type")
    else:
        devid = ctx.resolve_device(ref)
        if devid is None:
            opts = [n.get("name") for n in (scope.nodes_in(ctx.tree) if ctx.tree
                                            else ctx.devices)][:40]
            raise ToolError(
                "no encontré '{}' en la planta. Disponibles: {}".format(ref, opts))
        dev_ids, label, ntype = [devid], ctx.name_for(devid), "dev"
    if not dev_ids:
        raise ToolError("'{}' no tiene equipos con datos".format(label))
    return label, ntype, dev_ids


def _resolve_period(args: dict, ctx: ToolContext) -> tuple[dt.datetime, dt.datetime]:
    phrase = args.get("period") or "hoy"
    try:
        return periods.resolve(phrase, ctx.now, ctx.tz)
    except periods.PeriodError as e:
        raise ToolError(str(e))


def _resolve_period_for(args: dict, ctx: ToolContext, dev_ids: list[int]
                        ) -> tuple[dt.datetime, dt.datetime]:
    """Turn-aware period resolution for a node. Turn phrases ('este turno',
    'turno noche', 'mismo turno la semana pasada') resolve via mtapi2 using a
    representative device; everything else delegates to periods.resolve.
    """
    phrase = args.get("period") or "hoy"
    if turns.is_turn_phrase(phrase) and dev_ids:
        now_naive = ctx.now.astimezone(dt.timezone.utc).replace(tzinfo=None)
        try:
            return turns.resolve_turn(phrase, ctx.client, dev_ids[0], now_naive,
                                      ctx.tz, ctx.mtapi_call or mtapi.call)
        except turns.TurnError as e:
            raise ToolError(str(e))
    return _resolve_period(args, ctx)


def _make_indicator_tool(fn_name: str) -> Callable[[dict, ToolContext], dict]:
    def _tool(args: dict, ctx: ToolContext) -> dict:
        label, ntype, dev_ids = _resolve_node(args, ctx)
        start, end = _resolve_period(args, ctx)
        # mtapi2 indicators aggregate a device list (disp/desemp/calidad iterate);
        # a single device is sent as a scalar (identical result, simpler call).
        arg = dev_ids[0] if len(dev_ids) == 1 else dev_ids
        value = _call(ctx, fn_name, start, end, arg)
        return {
            "indicator": fn_name,
            "node": label,
            "type": ntype,
            "devids": dev_ids,
            "value": value,
            "no_data": value is None,
            "period": [start.isoformat(), end.isoformat()],
        }
    return _tool


def _tool_oee_breakdown(args: dict, ctx: ToolContext) -> dict:
    """OEE of a node plus its three factors (disponibilidad, rendimiento,
    calidad), flagging the factor that drags it most (the lowest). Supports turn
    periods ('este turno'). Reuses the per-indicator calls."""
    label, ntype, dev_ids = _resolve_node(args, ctx)
    start, end = _resolve_period_for(args, ctx, dev_ids)
    arg = dev_ids[0] if len(dev_ids) == 1 else dev_ids
    vals = {}
    for ind in ("oee", "disponibilidad", "rendimiento", "calidad"):
        try:
            vals[ind] = _call(ctx, ind, start, end, arg)
        except ToolError:
            vals[ind] = None
    factors = {k: v for k, v in vals.items() if k != "oee" and v is not None}
    worst = min(factors, key=factors.get) if factors else None
    return {
        "node": label,
        "type": ntype,
        "oee": vals.get("oee"),
        "disponibilidad": vals.get("disponibilidad"),
        "rendimiento": vals.get("rendimiento"),
        "calidad": vals.get("calidad"),
        "worst_factor": worst,
        "no_data": vals.get("oee") is None,
        "period": [start.isoformat(), end.isoformat()],
    }


def _tool_compare_periods(args: dict, ctx: ToolContext) -> dict:
    """Compare a node across two periods: OEE-factor deltas and which stops
    changed. Both periods are turn-aware. Answers 'cómo se compara el turno
    actual con el mismo turno de la semana pasada y qué cambió'."""
    label, ntype, dev_ids = _resolve_node(args, ctx)
    pa = args.get("period_a") or args.get("period")
    pb = args.get("period_b")
    if not pa or not pb:
        raise ToolError("indica dos períodos: 'period_a' y 'period_b'")
    start_a, end_a = _resolve_period_for({"period": pa}, ctx, dev_ids)
    start_b, end_b = _resolve_period_for({"period": pb}, ctx, dev_ids)
    arg = dev_ids[0] if len(dev_ids) == 1 else dev_ids

    def _metrics(s, e):
        out = {}
        for ind in ("oee", "disponibilidad", "rendimiento", "calidad"):
            try:
                out[ind] = _call(ctx, ind, s, e, arg)
            except ToolError:
                out[ind] = None
        return out

    def _stops(s, e):
        data = _call(ctx, "pareto", s, e, arg) or {}
        return {r.get("desc"): (r.get("time_s") or 0) for r in data.get("codstates", [])}

    ma, mb = _metrics(start_a, end_a), _metrics(start_b, end_b)
    deltas = {
        k: (round(ma[k] - mb[k], 4) if ma[k] is not None and mb[k] is not None else None)
        for k in ma
    }
    sa, sb = _stops(start_a, end_a), _stops(start_b, end_b)
    changes = sorted(
        ({"reason": d, "delta_h": round(sa.get(d, 0) - sb.get(d, 0), 2)}
         for d in set(sa) | set(sb)),
        key=lambda c: abs(c["delta_h"]), reverse=True)[:5]
    return {
        "node": label, "type": ntype,
        "a": {"period": [start_a.isoformat(), end_a.isoformat()], **ma},
        "b": {"period": [start_b.isoformat(), end_b.isoformat()], **mb},
        "deltas": deltas,
        "stop_changes": changes,
        "period": [start_a.isoformat(), end_a.isoformat()],
    }


def _tool_production_target(args: dict, ctx: ToolContext) -> dict:
    """Target vs produced with a pace projection for a period ('hoy', 'este
    turno'). target = sum of overlapping Plan.value (mtapi2.plan_target);
    produced = prod_dev_kp per device (reused). projected = produced scaled to
    the full period at the current pace; shortfall = target - projected."""
    if args.get("node") or args.get("device"):
        label, ntype, dev_ids = _resolve_node(args, ctx)
    else:
        dev_ids = list(ctx.device_ids)
        label, ntype = ctx.plant_name or "planta", "plant"
    phrase = (args.get("period") or "hoy").strip().lower()
    start, end = _resolve_period_for(args, ctx, dev_ids)

    # Ongoing periods ("hoy", "este turno") resolve capped at now — for the
    # projection we need the FULL period end (next local midnight / scheduled
    # turn end). Past periods keep end as-is (no projection: it's complete).
    now_naive = ctx.now.astimezone(dt.timezone.utc).replace(tzinfo=None)
    eff_end = min(end, now_naive)
    full_end = end
    if abs((end - now_naive).total_seconds()) < 120:          # capped at now
        if turns.is_turn_phrase(phrase) and dev_ids:
            _name, _s, turn_end = (ctx.mtapi_call or mtapi.call)(
                "currentturn", ctx.client, dev_ids[0])
            full_end = turn_end or end
        elif phrase == "hoy":
            full_end = periods.days_in(
                start, start + dt.timedelta(hours=36), ctx.tz)[0][2]

    data = _call(ctx, "plan_target", start, full_end, dev_ids) or {}
    target = data.get("target") or 0

    produced, any_data = 0, False
    for devid in dev_ids:
        v = _call(ctx, "prod_dev_kp", start, eff_end, devid)
        if v is not None:
            produced += v
            any_data = True

    elapsed = (eff_end - start).total_seconds()
    total = (full_end - start).total_seconds()
    projected = round(produced * (total / elapsed)) if elapsed > 0 and total > 0 else None
    shortfall = (round(target - projected) if target and projected is not None else None)
    return {
        "node": label,
        "type": ntype,
        "target": target or None,
        # 'plan' (orden de producción) or 'expected_speed' (vmax threshold —
        # producto seteado sin planificación).
        "target_source": data.get("source"),
        "produced": produced if any_data else None,
        "projected_end_of_period": projected,
        "projected_shortfall": shortfall,
        "on_track": (projected >= target) if (target and projected is not None) else None,
        "elapsed_pct": round(100.0 * elapsed / total, 1) if total > 0 else None,
        "plans": (data.get("plans") or [])[:10],
        "no_data": not any_data and not target,
        "period": [start.isoformat(), end.isoformat()],
    }


def _tool_turns_oee(args: dict, ctx: ToolContext) -> dict:
    """OEE per turn of a node for a day/period, flagging best and worst turn.
    Reuses getturns (the date's turn windows) + the oee indicator. Answers
    '¿qué turno tuvo mejor OEE el 28 de mayo?'."""
    label, ntype, dev_ids = _resolve_node(args, ctx)
    start, _end = _resolve_period_for(args, ctx, dev_ids)
    data = _call(ctx, "getturns", dev_ids[0], start) or {}
    arg = dev_ids[0] if len(dev_ids) == 1 else dev_ids
    rows = []
    for name, bounds in (data.get("turns") or {}).items():
        t_start, t_end = bounds[0], bounds[1]
        try:
            value = _call(ctx, "oee", t_start, t_end, arg)
        except ToolError:
            value = None
        if value is not None:
            rows.append({"turn": name, "oee": value,
                         "start": str(t_start), "end": str(t_end)})
    rows.sort(key=lambda r: r["oee"], reverse=True)
    return {
        "node": label,
        "type": ntype,
        "turns": rows,                          # best first
        "best_turn": rows[0] if rows else None,
        "worst_turn": rows[-1] if rows else None,
        "no_data": not rows,
        "period": [start.isoformat(), _end.isoformat()],
    }


def _tool_turns_info(args: dict, ctx: ToolContext) -> dict:
    """The node's configured turns (names + local start/end times). Backed by
    mtapi2.getturns (reused) on a representative device of the node."""
    from zoneinfo import ZoneInfo
    label, ntype, dev_ids = _resolve_node(args, ctx)
    now_naive = ctx.now.astimezone(dt.timezone.utc).replace(tzinfo=None)
    data = _call(ctx, "getturns", dev_ids[0], now_naive) or {}
    zone = ZoneInfo(ctx.tz)
    out = []
    for name, bounds in (data.get("turns") or {}).items():
        start, end = bounds[0], bounds[1]
        s_local = start.replace(tzinfo=dt.timezone.utc).astimezone(zone)
        e_local = end.replace(tzinfo=dt.timezone.utc).astimezone(zone)
        out.append({
            "name": name,
            "start_local": s_local.strftime("%H:%M"),
            "end_local": e_local.strftime("%H:%M"),
        })
    out.sort(key=lambda t: t["start_local"])
    return {
        "node": label,
        "type": ntype,
        "tz": ctx.tz,
        "turns": out,
        "no_data": not out,
    }


def _tool_recent_products(args: dict, ctx: ToolContext) -> dict:
    """Latest products/SKUs produced by a node, newest first. Backed by
    mtapi2.product_intervals (prod_interval + Product name/sku, reused).
    Defaults to the last 7 days when no period is given."""
    label, ntype, dev_ids = _resolve_node(args, ctx)
    if not args.get("period"):
        args = dict(args)
        args["period"] = "últimos 7 días"
    start, end = _resolve_period_for(args, ctx, dev_ids)

    rows = _call(ctx, "product_intervals", start, end, dev_ids) or []
    items = [
        {
            "start": r.get("start"),
            "end": r.get("end"),
            "product": r.get("product"),
            "sku": r.get("sku"),
            "device": ctx.name_for(r.get("devid")),
        }
        for r in rows[:20]
    ]
    return {
        "node": label,
        "type": ntype,
        "products": items,                     # newest first
        "truncated": len(rows) > 20,
        "no_data": not items,
        "period": [start.isoformat(), end.isoformat()],
    }


def _tool_rank_devices(args: dict, ctx: ToolContext) -> dict:
    """Rank the plant's devices by an indicator, worst or best first.

    Per-device indicator calls (scalar oee()/disponibilidad()/… are reliable;
    we avoid devtree(indicators=…) which faults on empty sections). The model
    never does arithmetic — we only sort the official figures.
    order: 'worst' (ascending, e.g. "qué equipo afecta más el OEE") or 'best'
    (descending, e.g. "la máquina con más disponibilidad").
    """
    indicator = (args.get("indicator") or "oee").strip().lower()
    if indicator not in _INDICATORS:
        raise ToolError(
            "indicador inválido: {!r} (usa {})".format(
                indicator, ", ".join(_INDICATORS)))
    order = (args.get("order") or "worst").strip().lower()
    start, end = _resolve_period(args, ctx)

    scores: list[tuple[int, float]] = []
    for devid in ctx.device_ids:
        try:
            value = _call(ctx, indicator, start, end, devid)
        except ToolError:
            value = None  # skip a device whose indicator errors; don't abort the rank
        if value is not None:
            scores.append((devid, value))

    scores.sort(key=lambda t: t[1], reverse=(order == "best"))
    return {
        "indicator": indicator,
        "order": order,
        "devices": [
            {"devid": d, "name": ctx.name_for(d), "value": v} for d, v in scores[:10]
        ],
        "no_data": not scores,
        "period": [start.isoformat(), end.isoformat()],
    }


def _tool_rank_downtime(args: dict, ctx: ToolContext) -> dict:
    """Rank EQUIPMENT by total stopped time (most downtime first) over a period.

    Distinct from top_stops (which ranks stop *reasons*). Sums each device's
    operational stop time via pareto([devid]) — a per-device fan-out; for large
    nodes this is several mtapi2 calls.
    """
    if args.get("node") or args.get("device"):
        label, ntype, dev_ids = _resolve_node(args, ctx)
    else:
        dev_ids = list(ctx.device_ids)
        label, ntype = ctx.plant_name or "planta", "plant"
    start, end = _resolve_period(args, ctx)
    rows = []
    for devid in dev_ids:
        data = _call(ctx, "pareto", start, end, [devid]) or {}
        hours = sum((r.get("time_s") or 0) for r in data.get("codstates", []))
        if hours > 0:
            rows.append((devid, round(hours, 2)))
    rows.sort(key=lambda t: t[1], reverse=True)
    return {
        "node": label,
        "type": ntype,
        "ranking": "most_downtime_first",
        "devices": [
            {"devid": d, "name": ctx.name_for(d), "downtime_h": h} for d, h in rows[:10]
        ],
        "no_data": not rows,
        "period": [start.isoformat(), end.isoformat()],
    }


def _tool_sabana(args: dict, ctx: ToolContext) -> dict:
    """Sabana detail (state intervals with production) for a node + period.

    The raw sabana can be hundreds of rows, so we summarize: totals, production
    by product, and a capped sample of rows. Backed by mtapi2.sabana (the
    SabanaRows are precomputed).
    """
    if args.get("node") or args.get("device"):
        label, ntype, dev_ids = _resolve_node(args, ctx)
    else:
        dev_ids = list(ctx.device_ids)
        label, ntype = ctx.plant_name or "planta", "plant"
    start, end = _resolve_period(args, ctx)

    rows = _call(ctx, "sabana", start, end, dev_ids) or []
    by_product: dict = {}
    for r in rows:
        prod = r.get("product") or r.get("sku") or "?"
        by_product[prod] = by_product.get(prod, 0) + (r.get("production") or 0)
    products = sorted(
        ({"product": k, "production": v} for k, v in by_product.items()),
        key=lambda d: d["production"], reverse=True)[:10]
    sample = [
        {
            "start": r.get("start"), "end": r.get("end"),
            "duration_min": r.get("duration"), "production": r.get("production"),
            "product": r.get("product"), "stop": r.get("code_description") or None,
        }
        for r in rows[:25]
    ]
    return {
        "node": label,
        "type": ntype,
        "n_rows": len(rows),
        "total_production": sum((r.get("production") or 0) for r in rows),
        "total_duration_min": round(sum((r.get("duration") or 0) for r in rows), 1),
        "by_product": products,
        "rows_sample": sample,
        "truncated": len(rows) > 25,
        "no_data": not rows,
        "period": [start.isoformat(), end.isoformat()],
    }


def _tool_stops_detail(args: dict, ctx: ToolContext) -> dict:
    """Chronological list of a node's stops with START/END time, duration and
    reason (from sabana rows that carry a stop code). Optional 'reason' filters
    by reason substring. Answers '¿a qué hora fue la detención X?'.
    """
    if args.get("node") or args.get("device"):
        label, ntype, dev_ids = _resolve_node(args, ctx)
    else:
        dev_ids = list(ctx.device_ids)
        label, ntype = ctx.plant_name or "planta", "plant"
    start, end = _resolve_period(args, ctx)
    reason = (args.get("reason") or "").strip().lower()

    # S_reg-based (universal across clients), not SabanaRow.
    rows = _call(ctx, "stop_intervals", start, end, dev_ids) or []
    stops = []
    for r in rows:
        desc = (r.get("reason") or "").strip()
        if reason and reason not in desc.lower():
            continue
        stops.append({
            "start": r.get("start"),
            "end": r.get("end"),
            "duration_min": r.get("duration_min"),
            "reason": desc or None,
            "device": ctx.name_for(r.get("devid")),
        })
    stops.sort(key=lambda s: s.get("start") or "")
    return {
        "node": label,
        "type": ntype,
        "n_stops": len(stops),
        "stops": stops[:80],
        "truncated": len(stops) > 80,
        "no_data": not stops,
        "period": [start.isoformat(), end.isoformat()],
    }


def _tool_top_stops(args: dict, ctx: ToolContext) -> dict:
    """Most significant stops in a period, by total time or by occurrence count.

    Aggregation is mtapi2's official `pareto` (grouped by cod_state); we only
    choose the sort key. Optionally scoped to a named line, resolved via devtree.
    """
    start, end = _resolve_period(args, ctx)
    by = args.get("by") or "time"
    node_ref = args.get("node") or args.get("line")

    if node_ref:
        tree = ctx.tree or _call(ctx, "devtree_named", "plant", ctx.plant_id)
        node = scope.resolve_node(tree, node_ref)
        if node is None:
            names = [n.get("name") for n in scope.nodes_in(tree)]
            raise ToolError(
                "no encontré '{}'. Disponibles: {}".format(node_ref, names))
        devids = scope.node_device_ids(node)
        scope_label = node.get("name") or str(node_ref)
    else:
        devids = list(ctx.device_ids)
        scope_label = ctx.plant_name or "planta"

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
    label, ntype, dev_ids = _resolve_node(args, ctx)
    measure = args.get("measure") or "standard"
    fn = _PROD_MEASURE_FN.get(measure)
    if fn is None:
        raise ToolError(
            "medida inválida: {!r} (usa 'standard', 'counter' o 'tons')".format(measure))
    start, end = _resolve_period(args, ctx)
    # prod_* are per-device; production is additive, so sum across the node's
    # devices. Unit of measure comes from Device.unit (device_meta) — caller
    # should not mix units; for the MVP we sum.
    total, any_data = 0, False
    for devid in dev_ids:
        v = _call(ctx, fn, start, end, devid)
        if v is not None:
            total += v
            any_data = True
    return {
        "node": label,
        "type": ntype,
        "devids": dev_ids,
        "measure": measure,
        "produced": total if any_data else None,
        "no_data": not any_data,
        "period": [start.isoformat(), end.isoformat()],
    }


def _tool_daily_oee(args: dict, ctx: ToolContext) -> dict:
    """Daily OEE series for a node over a period; flags best and worst day.

    Computes one official OEE per local calendar day (no model arithmetic) and
    returns the series plus best/worst — answers '¿el mejor/peor día?'.
    """
    label, ntype, dev_ids = _resolve_node(args, ctx)
    start, end = _resolve_period(args, ctx)
    arg = dev_ids[0] if len(dev_ids) == 1 else dev_ids
    series = []
    for date_iso, day_start, day_end in periods.days_in(start, end, ctx.tz):
        value = _call(ctx, "oee", day_start, day_end, arg)
        if value is not None:
            series.append({"date": date_iso, "oee": value})
    if not series:
        return {"node": label, "type": ntype, "series": [], "no_data": True,
                "period": [start.isoformat(), end.isoformat()]}
    return {
        "node": label,
        "type": ntype,
        "series": series,
        "best_day": max(series, key=lambda d: d["oee"]),
        "worst_day": min(series, key=lambda d: d["oee"]),
        "no_data": False,
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
                    "node": {
                        "type": "string",
                        "description": "Nombre de un equipo, línea, sección o planta. "
                                       "Para líneas/secciones/plantas el indicador "
                                       "agrega sus equipos.",
                    },
                    "period": _PERIOD_PROP,
                },
                "required": ["node", "period"],
            },
        },
    }


_RANK_DEVICES_SPEC = {
    "type": "function",
    "function": {
        "name": "rank_devices",
        "description": "Clasifica los equipos de la planta por un indicador en un "
                       "período. Úsalo para comparativas: 'qué equipo afecta más el "
                       "OEE' (indicator=oee, order=worst), 'la máquina con más "
                       "disponibilidad' (indicator=disponibilidad, order=best), etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "indicator": {
                    "type": "string",
                    "enum": ["oee", "disponibilidad", "rendimiento", "calidad",
                             "cumplimiento"],
                    "description": "Indicador a clasificar (por defecto 'oee').",
                },
                "order": {
                    "type": "string",
                    "enum": ["worst", "best"],
                    "description": "'worst' = peor primero (más bajo); 'best' = "
                                   "mejor primero (más alto). Por defecto 'worst'.",
                },
                "period": _PERIOD_PROP,
            },
            "required": ["period"],
        },
    },
}

_TOP_STOPS_SPEC = {
    "type": "function",
    "function": {
        "name": "top_stops",
        "description": "Detenciones más importantes en un período, para toda la "
                       "planta o un nodo (equipo/línea/sección). Úsalo para "
                       "'¿la detención más repetida?' o '¿la más larga?'.",
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
                "node": {
                    "type": "string",
                    "description": "Nombre de equipo/línea/sección (opcional). Si se "
                                   "omite, abarca toda la planta.",
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
                "node": {
                    "type": "string",
                    "description": "Nombre de equipo/línea/sección/planta (se suma "
                                   "la producción de sus equipos).",
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
            "required": ["node", "period"],
        },
    },
}

_DAILY_OEE_SPEC = {
    "type": "function",
    "function": {
        "name": "daily_oee",
        "description": "OEE día a día de un nodo (equipo/línea/sección/planta) en "
                       "un período, con el mejor y el peor día. Úsalo para "
                       "'¿cuál fue el mejor/peor día?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "Nombre de equipo/línea/sección/planta.",
                },
                "period": _PERIOD_PROP,
            },
            "required": ["node", "period"],
        },
    },
}

_RANK_DOWNTIME_SPEC = {
    "type": "function",
    "function": {
        "name": "rank_downtime",
        "description": "Clasifica los EQUIPOS por tiempo total detenido (mayor "
                       "primero) en un período, dentro de una planta/línea/sección. "
                       "Úsalo para '¿qué máquina estuvo más tiempo detenida?'. "
                       "Es distinto de top_stops (que clasifica motivos de detención).",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "Nombre de planta/línea/sección (opcional; por "
                                   "defecto, toda la planta).",
                },
                "period": _PERIOD_PROP,
            },
            "required": ["period"],
        },
    },
}

_SABANA_SPEC = {
    "type": "function",
    "function": {
        "name": "sabana",
        "description": "Detalle de la sábana de un nodo en un período: corridas e "
                       "intervalos de estado con su producción. Devuelve totales, "
                       "producción por producto y una muestra de filas (inicio/fin, "
                       "duración, producción, detención). Úsalo para preguntas de "
                       "detalle por intervalo/corrida u órdenes.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "Nombre de equipo/línea/sección (opcional; por "
                                   "defecto toda la planta).",
                },
                "period": _PERIOD_PROP,
            },
            "required": ["period"],
        },
    },
}

_STOPS_DETAIL_SPEC = {
    "type": "function",
    "function": {
        "name": "stops_detail",
        "description": "Lista cronológica de las detenciones de un nodo en un "
                       "período, con HORA de inicio/fin, duración y motivo. Úsalo "
                       "para '¿a qué hora fue la detención X?' o el detalle temporal "
                       "de paradas. (top_stops da agregados; esto da los horarios.)",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "Nombre de equipo/línea/sección (opcional; por "
                                   "defecto toda la planta).",
                },
                "period": _PERIOD_PROP,
                "reason": {
                    "type": "string",
                    "description": "Filtro opcional por motivo (substring), p.ej. "
                                   "'PROGRAMADO', 'AVERIA', 'CAMBIO DE FORMATO'.",
                },
            },
            "required": ["period"],
        },
    },
}

_OEE_BREAKDOWN_SPEC = {
    "type": "function",
    "function": {
        "name": "oee_breakdown",
        "description": "OEE de un nodo y sus tres factores (disponibilidad, "
                       "rendimiento, calidad), indicando cuál lo afecta más (el más "
                       "bajo). Soporta períodos de turno ('este turno'). Úsalo para "
                       "'¿OEE y qué factor lo afecta más?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "Nombre de equipo/línea/sección/planta.",
                },
                "period": _PERIOD_PROP,
            },
            "required": ["node", "period"],
        },
    },
}

_TURNS_OEE_SPEC = {
    "type": "function",
    "function": {
        "name": "turns_oee",
        "description": "OEE por TURNO de un equipo/nodo en un día o período, con "
                       "el mejor y el peor turno. Úsalo para '¿qué turno tuvo "
                       "mejor/peor OEE el 28 de mayo?'. El period acepta también "
                       "fechas como '28 de mayo'.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "Nombre de equipo/línea/sección/planta.",
                },
                "period": _PERIOD_PROP,
            },
            "required": ["node", "period"],
        },
    },
}

_TURNS_INFO_SPEC = {
    "type": "function",
    "function": {
        "name": "turns_info",
        "description": "Turnos configurados de un equipo/línea/sección (nombre y "
                       "horario local de inicio/fin). Úsalo para '¿cuáles son los "
                       "turnos de X?' o '¿a qué hora empieza el turno?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "Nombre de equipo/línea/sección/planta.",
                },
            },
            "required": ["node"],
        },
    },
}

_RECENT_PRODUCTS_SPEC = {
    "type": "function",
    "function": {
        "name": "recent_products",
        "description": "Últimos productos/SKU producidos por un equipo/línea/"
                       "sección, del más reciente al más antiguo (por defecto, "
                       "últimos 7 días). Úsalo para '¿cuáles son los últimos SKU "
                       "producidos por X?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "Nombre de equipo/línea/sección/planta.",
                },
                "period": _PERIOD_PROP,
            },
            "required": ["node"],
        },
    },
}

_PRODUCTION_TARGET_SPEC = {
    "type": "function",
    "function": {
        "name": "production_target",
        "description": "Meta de producción vs producido y faltante PROYECTADO al "
                       "ritmo actual, para un período ('hoy', 'este turno'). Úsalo "
                       "para '¿vamos a cumplir la meta de hoy?' o '¿cuál es el "
                       "faltante proyectado?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "Nombre de equipo/línea/sección (opcional; por "
                                   "defecto toda la planta).",
                },
                "period": _PERIOD_PROP,
            },
            "required": ["period"],
        },
    },
}

_COMPARE_PERIODS_SPEC = {
    "type": "function",
    "function": {
        "name": "compare_periods",
        "description": "Compara un nodo entre dos períodos: deltas de OEE y sus "
                       "factores, y qué detenciones cambiaron. Úsalo para "
                       "'¿cómo se compara el turno actual con el mismo turno de la "
                       "semana pasada y qué cambió?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {"type": "string",
                         "description": "Nombre de equipo/línea/sección/planta."},
                "period_a": {"type": "string",
                             "description": "Primer período (p.ej. 'este turno')."},
                "period_b": {"type": "string",
                             "description": "Segundo período (p.ej. 'mismo turno la "
                                            "semana pasada')."},
            },
            "required": ["node", "period_a", "period_b"],
        },
    },
}

# Advertised to the LLM.
TOOL_SPECS = (
    [_indicator_spec(n, d) for n, d in _INDICATORS.items()]
    + [_OEE_BREAKDOWN_SPEC, _RANK_DEVICES_SPEC, _TOP_STOPS_SPEC, _PRODUCTION_SPEC,
       _DAILY_OEE_SPEC, _RANK_DOWNTIME_SPEC, _SABANA_SPEC, _STOPS_DETAIL_SPEC,
       _COMPARE_PERIODS_SPEC, _PRODUCTION_TARGET_SPEC, _RECENT_PRODUCTS_SPEC,
       _TURNS_INFO_SPEC, _TURNS_OEE_SPEC]
)

_DISPATCH: dict[str, Callable[[dict, ToolContext], dict]] = {
    name: _make_indicator_tool(name) for name in _INDICATORS
}
_DISPATCH["oee_breakdown"] = _tool_oee_breakdown
_DISPATCH["compare_periods"] = _tool_compare_periods
_DISPATCH["production_target"] = _tool_production_target
_DISPATCH["recent_products"] = _tool_recent_products
_DISPATCH["turns_info"] = _tool_turns_info
_DISPATCH["turns_oee"] = _tool_turns_oee
_DISPATCH["rank_devices"] = _tool_rank_devices
_DISPATCH["rank_oee"] = _tool_rank_devices   # legacy alias (defaults indicator=oee)
_DISPATCH["top_stops"] = _tool_top_stops
_DISPATCH["production"] = _tool_production
_DISPATCH["daily_oee"] = _tool_daily_oee
_DISPATCH["rank_downtime"] = _tool_rank_downtime
_DISPATCH["sabana"] = _tool_sabana
_DISPATCH["stops_detail"] = _tool_stops_detail


_RATIO_KEYS = set(_INDICATORS)  # indicator figures are 0..1 ratios


def _add_pct_fields(obj, indicator: str | None = None) -> None:
    """Recursively annotate ratio figures with a preformatted '<key>_pct' string.

    Mutates the payload in place. The model must quote figures verbatim and
    never convert them (it wrote '0.16%' for an OEE of 0.1651), so the display
    form ('16.5%') is computed here and the prompt instructs citing the *_pct
    field as-is.
    """
    if isinstance(obj, dict):
        ind = obj.get("indicator", indicator)
        for k in list(obj):
            v = obj[k]
            if isinstance(v, (dict, list)):
                _add_pct_fields(v, ind)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                if k in _RATIO_KEYS or (k == "value" and ind in _RATIO_KEYS):
                    obj[k + "_pct"] = "{:.1f}%".format(v * 100.0)
    elif isinstance(obj, list):
        for item in obj:
            _add_pct_fields(item, indicator)


def dispatch(name: str, args: dict, ctx: ToolContext) -> dict:
    """Validate and execute a tool call; return a JSON-able result dict.

    Raises:
        ToolError: unknown tool, invalid arguments, or mtapi2 failure.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        # Models occasionally garble tool names ('daney_oee' ~ 'daily_oee').
        close = difflib.get_close_matches(name or "", list(_DISPATCH), n=1, cutoff=0.75)
        if close:
            fn = _DISPATCH[close[0]]
        else:
            raise ToolError("herramienta desconocida: {!r}. Disponibles: {}".format(
                name, sorted(_DISPATCH)))
    # Models occasionally vary argument-key casing (e.g. 'Reason'); normalize.
    args = {str(k).lower(): v for k, v in (args or {}).items()}
    result = fn(args, ctx)
    _add_pct_fields(result)
    return result
