"""Unit tests for the tool catalog — dispatch + validation, mtapi stubbed."""
from __future__ import annotations

import datetime as dt

import pytest

from modules.plantagent import mtapi, tools
from modules.plantagent.tools import ToolContext

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 6, 3, 12, 0, tzinfo=UTC)


def make_ctx(device_ids=(1079, 1080, 1081), mtapi_call=None, names=None,
             devices=None, tree=None):
    if devices is None:
        names = names or {}
        devices = [{"id": d, "name": names.get(d)} for d in device_ids]
    return ToolContext(
        client="degasa", plant_id=7, devices=devices,
        now=NOW, tz="America/Santiago", mtapi_call=mtapi_call, tree=tree or {},
    )


def recording_call(result_by_fn):
    calls = []

    def _call(fn, client, *args):
        calls.append((fn, client, args))
        val = result_by_fn[fn]
        if isinstance(val, Exception):
            raise val
        return val

    _call.calls = calls  # type: ignore[attr-defined]
    return _call


@pytest.mark.parametrize("name,fn", [
    ("oee", "oee"),
    ("disponibilidad", "disponibilidad"),
    ("rendimiento", "rendimiento"),
    ("calidad", "calidad"),
    ("cumplimiento", "cumplimiento"),
])
def test_per_device_indicator_calls_right_mtapi_fn(name, fn):
    call = recording_call({fn: 0.91})
    ctx = make_ctx(mtapi_call=call)
    result = tools.dispatch(name, {"devid": 1079, "period": "hoy"}, ctx)
    assert result["value"] == 0.91
    assert result["period"] is not None
    assert call.calls[0][0] == fn
    assert call.calls[0][1] == "degasa"
    assert call.calls[0][2][-1] == 1079        # devid last


def test_all_indicators_are_advertised_in_tool_specs():
    names = {t["function"]["name"] for t in tools.TOOL_SPECS}
    assert {"oee", "disponibilidad", "rendimiento", "calidad",
            "cumplimiento", "rank_devices"} <= names


def test_out_of_scope_devid_rejected():
    call = recording_call({"oee": 0.9})
    ctx = make_ctx(device_ids=(1079,), mtapi_call=call)
    with pytest.raises(tools.ToolError):
        tools.dispatch("oee", {"devid": 9999, "period": "hoy"}, ctx)
    assert call.calls == []


def test_unavailable_indicator_becomes_toolerror():
    call = recording_call({"cumplimiento": mtapi.MtapiUnavailable("no impl")})
    ctx = make_ctx(mtapi_call=call)
    with pytest.raises(tools.ToolError):
        tools.dispatch("cumplimiento", {"devid": 1079, "period": "hoy"}, ctx)


# --- ranking via per-device oee() (devtree[indicators] faults on sections) ----

def oee_by_dev(value_map):
    """Stub: any per-device indicator returns the mapped value for that devid."""
    calls = []

    def _call(fn, client, *args):
        calls.append((fn, client, args))
        return value_map.get(args[-1])

    _call.calls = calls  # type: ignore[attr-defined]
    return _call


def test_rank_devices_worst_first_by_default():
    call = oee_by_dev({1079: 0.82, 1080: 0.55, 1081: 0.73})
    ctx = make_ctx(device_ids=(1079, 1080, 1081), names={1080: "Etiquetadora"},
                   mtapi_call=call)
    result = tools.dispatch("rank_devices", {"period": "últimos 3 días"}, ctx)

    assert result["indicator"] == "oee" and result["order"] == "worst"
    assert [c[0] for c in call.calls].count("oee") == 3   # one oee() per device
    devices = result["devices"]
    assert [d["devid"] for d in devices] == [1080, 1081, 1079]   # worst first
    assert devices[0]["value"] == 0.55
    assert devices[0]["name"] == "Etiquetadora"


def test_rank_devices_best_by_indicator():
    # "la máquina con más disponibilidad" -> indicator=disponibilidad, order=best
    call = oee_by_dev({1079: 0.82, 1080: 0.55, 1081: 0.73})
    ctx = make_ctx(device_ids=(1079, 1080, 1081), mtapi_call=call)
    result = tools.dispatch("rank_devices", {"indicator": "disponibilidad",
                                             "order": "best", "period": "mayo"}, ctx)
    assert call.calls[0][0] == "disponibilidad"
    assert [d["devid"] for d in result["devices"]] == [1079, 1081, 1080]  # best first
    assert result["devices"][0]["value"] == 0.82


def test_rank_devices_invalid_indicator_raises():
    ctx = make_ctx(mtapi_call=oee_by_dev({}))
    with pytest.raises(tools.ToolError):
        tools.dispatch("rank_devices", {"indicator": "humedad", "period": "hoy"}, ctx)


def test_rank_devices_ignores_devs_without_value():
    call = oee_by_dev({1079: 0.6, 1080: None})
    ctx = make_ctx(device_ids=(1079, 1080), mtapi_call=call)
    result = tools.dispatch("rank_devices", {"period": "hoy"}, ctx)
    assert [d["devid"] for d in result["devices"]] == [1079]


# --- detenciones (top_stops via pareto) --------------------------------------

PARETO = {"codstates": [
    {"id": 1, "desc": "Falla mecánica", "code_f": "FM", "num": 12, "time_s": 4.5},
    {"id": 2, "desc": "Cambio de formato", "code_f": "CF", "num": 3, "time_s": 8.0},
]}

PLANT_TREE_LINES = {
    "id": 7, "name": "Planta", "type": "plant",
    "lines": [
        {"id": 10, "name": "Línea 1", "type": "line", "devs": [
            {"id": 1079, "type": "dev"}]},
        {"id": 11, "name": "Línea 2", "type": "line", "devs": [
            {"id": 1080, "type": "dev"}, {"id": 1081, "type": "dev"}]},
    ],
    "sections": [], "devs": [],
}


def test_top_stops_advertised():
    names = {t["function"]["name"] for t in tools.TOOL_SPECS}
    assert "top_stops" in names


def test_top_stops_plant_wide_by_time_is_default():
    call = recording_call({"pareto": PARETO})
    ctx = make_ctx(mtapi_call=call)
    result = tools.dispatch("top_stops", {"period": "últimos 3 días"}, ctx)
    assert result["scope"] == "planta"
    # default by time -> Cambio de formato (8.0h) first
    assert result["stops"][0]["desc"] == "Cambio de formato"
    assert call.calls[0][0] == "pareto"


def test_top_stops_by_count_orders_by_occurrences():
    call = recording_call({"pareto": PARETO})
    ctx = make_ctx(mtapi_call=call)
    result = tools.dispatch("top_stops", {"period": "hoy", "by": "count"}, ctx)
    # most-repeated -> Falla mecánica (12 occurrences) first
    assert result["stops"][0]["desc"] == "Falla mecánica"
    assert result["stops"][0]["count"] == 12


def test_top_stops_line_scoped_resolves_devices_then_paretos():
    call = recording_call({"devtree_named": PLANT_TREE_LINES, "pareto": PARETO})
    ctx = make_ctx(device_ids=(1079, 1080, 1081), mtapi_call=call)
    result = tools.dispatch("top_stops", {"period": "hoy", "line": "Línea 2", "by": "count"}, ctx)
    assert result["scope"] == "Línea 2"
    fns = [c[0] for c in call.calls]
    assert fns == ["devtree_named", "pareto"]
    # pareto called with only Línea 2's devices
    assert call.calls[1][2][-1] == [1080, 1081]


def test_top_stops_unknown_line_raises_with_available_names():
    call = recording_call({"devtree_named": PLANT_TREE_LINES})
    ctx = make_ctx(mtapi_call=call)
    with pytest.raises(tools.ToolError) as ei:
        tools.dispatch("top_stops", {"period": "hoy", "line": "Línea 9"}, ctx)
    assert "Línea 2" in str(ei.value)  # lists available lines


# --- producción --------------------------------------------------------------

def test_production_advertised():
    names = {t["function"]["name"] for t in tools.TOOL_SPECS}
    assert "production" in names


@pytest.mark.parametrize("measure,fn", [
    ("standard", "prod_dev_kp"),   # kp-weighted production (counter × product-spec kp)
    ("counter", "prod_dev_t"),     # raw counter, no kp
    ("tons", "total_tons"),
])
def test_production_measure_selects_right_mtapi_fn(measure, fn):
    call = recording_call({fn: 1234})
    ctx = make_ctx(mtapi_call=call)
    result = tools.dispatch("production", {"devid": 1079, "period": "hoy", "measure": measure}, ctx)
    assert result["produced"] == 1234
    assert result["measure"] == measure
    assert call.calls[0][0] == fn
    assert call.calls[0][2][-1] == 1079


def test_production_defaults_to_standard_kp_counter():
    call = recording_call({"prod_dev_kp": 500})
    ctx = make_ctx(mtapi_call=call)
    result = tools.dispatch("production", {"devid": 1079, "period": "hoy"}, ctx)
    assert result["measure"] == "standard"
    assert call.calls[0][0] == "prod_dev_kp"     # canonical default counter


def test_production_invalid_measure_raises():
    call = recording_call({"prod_dev_t": 1})
    ctx = make_ctx(mtapi_call=call)
    with pytest.raises(tools.ToolError):
        tools.dispatch("production", {"devid": 1079, "period": "hoy", "measure": "litros"}, ctx)
    assert call.calls == []


def test_production_out_of_scope_devid_raises():
    call = recording_call({"prod_dev_kp": 1})
    ctx = make_ctx(device_ids=(1079,), mtapi_call=call)
    with pytest.raises(tools.ToolError):
        tools.dispatch("production", {"devid": 9999, "period": "hoy"}, ctx)
    assert call.calls == []


# --- oee breakdown (Q1) ------------------------------------------------------

def test_oee_breakdown_flags_worst_factor():
    vals = {"oee": 0.50, "disponibilidad": 0.90, "rendimiento": 0.60, "calidad": 0.93}
    ctx = make_ctx(devices=[{"id": 1, "name": "X"}],
                   mtapi_call=lambda fn, c, *a: vals.get(fn))
    r = tools.dispatch("oee_breakdown", {"node": "X", "period": "hoy"}, ctx)
    assert r["oee"] == 0.50
    assert r["worst_factor"] == "rendimiento"           # 0.60 is the lowest factor


def test_oee_breakdown_supports_turn_period():
    def call(fn, client, *args):
        if fn == "currentturn":
            return ("TD", dt.datetime(2026, 6, 3, 11, 0), dt.datetime(2026, 6, 3, 18, 30))
        return 0.7

    ctx = make_ctx(devices=[{"id": 1, "name": "X"}], mtapi_call=call)
    r = tools.dispatch("oee_breakdown", {"node": "X", "period": "este turno"}, ctx)
    assert r["oee"] == 0.7
    # current turn resolved via mtapi, capped at ctx.now (12:00)
    assert r["period"] == ["2026-06-03T11:00:00", "2026-06-03T12:00:00"]


def test_oee_breakdown_advertised():
    assert "oee_breakdown" in {t["function"]["name"] for t in tools.TOOL_SPECS}


# --- compare periods (Q7) ----------------------------------------------------

def test_compare_periods_returns_deltas_and_stop_changes():
    # indicator values differ per period (by start date); pareto per period.
    # ayer = 2026-06-02; "semana pasada" starts Mon 2026-05-25.
    def call(fn, client, start, end, *rest):
        day = start.date().isoformat()
        if fn in ("oee", "disponibilidad", "rendimiento", "calidad"):
            return {"2026-06-02": 0.80, "2026-05-25": 0.60}.get(day)
        if fn == "pareto":
            return {"2026-06-02": {"codstates": [{"desc": "Falla", "time_s": 5.0}]},
                    "2026-05-25": {"codstates": [{"desc": "Falla", "time_s": 2.0}]}}[day]
        raise KeyError(fn)

    ctx = make_ctx(devices=[{"id": 1, "name": "X"}], mtapi_call=call)
    r = tools.dispatch("compare_periods",
                       {"node": "X", "period_a": "ayer", "period_b": "semana pasada"}, ctx)
    # ayer = 2026-06-02 (0.80), semana pasada starts 2026-05-26 (0.60) -> delta +0.20
    assert r["deltas"]["oee"] == 0.20
    assert r["a"]["oee"] == 0.80 and r["b"]["oee"] == 0.60
    assert r["stop_changes"][0] == {"reason": "Falla", "delta_h": 3.0}


def test_production_target_projects_shortfall():
    # period "hoy": 04:00 -> 12:00 elapsed of a 24h day (NOW=12:00 UTC, SCL -4).
    def call(fn, client, start, end, *rest):
        if fn == "plan_target":
            return {"target": 1200, "source": "plan", "plans": [], "detail": []}
        if fn == "prod_dev_kp":
            return 300                       # produced so far (8h of 24h)
        raise KeyError(fn)

    ctx = make_ctx(devices=[{"id": 1, "name": "X"}], mtapi_call=call)
    r = tools.dispatch("production_target", {"node": "X", "period": "hoy"}, ctx)
    assert r["target"] == 1200 and r["target_source"] == "plan"
    assert r["produced"] == 300
    assert r["projected_end_of_period"] == 900   # 300 * 24h/8h
    assert r["projected_shortfall"] == 300       # 1200 - 900
    assert r["on_track"] is False


def test_production_target_expected_speed_fallback_source():
    def call(fn, client, start, end, *rest):
        if fn == "plan_target":
            return {"target": 800, "source": "expected_speed", "plans": [],
                    "detail": [{"devid": 1, "expected": 800}]}
        if fn == "prod_dev_kp":
            return 400
        raise KeyError(fn)

    ctx = make_ctx(devices=[{"id": 1, "name": "X"}], mtapi_call=call)
    r = tools.dispatch("production_target", {"node": "X", "period": "hoy"}, ctx)
    assert r["target_source"] == "expected_speed"
    assert r["target"] == 800


def test_production_target_advertised():
    assert "production_target" in {t["function"]["name"] for t in tools.TOOL_SPECS}


def test_turns_info_lists_turns_with_local_times():
    turns_payload = {"turns": {
        "TD": (dt.datetime(2026, 6, 3, 11, 0), dt.datetime(2026, 6, 3, 18, 30)),
        "TN": (dt.datetime(2026, 6, 4, 2, 0), dt.datetime(2026, 6, 4, 11, 0)),
    }}

    def call(fn, client, devid, d0):
        assert fn == "getturns"
        return turns_payload

    ctx = make_ctx(devices=[{"id": 1, "name": "Esc2"}], mtapi_call=call)
    r = tools.dispatch("turns_info", {"node": "Esc2"}, ctx)
    by_name = {t["name"]: t for t in r["turns"]}
    # SCL winter (-4): TD 11:00 UTC -> 07:00 local; TN 02:00 UTC -> 22:00 local.
    assert by_name["TD"]["start_local"] == "07:00"
    assert by_name["TN"]["start_local"] == "22:00"


def test_recent_products_lists_skus_newest_first_with_default_period():
    rows = [
        {"devid": 1, "start": "2026-06-02 10:00:00", "end": "2026-06-02 18:00:00",
         "product": "Escobillón rojo", "sku": "ESC-R"},
        {"devid": 1, "start": "2026-06-01 08:00:00", "end": "2026-06-02 10:00:00",
         "product": "Escobillón azul", "sku": "ESC-A"},
    ]

    def call(fn, client, start, end, dev_ids):
        assert fn == "product_intervals"
        return rows

    ctx = make_ctx(devices=[{"id": 1, "name": "Esc2"}], mtapi_call=call)
    r = tools.dispatch("recent_products", {"node": "Esc2"}, ctx)   # no period -> default
    assert [p["sku"] for p in r["products"]] == ["ESC-R", "ESC-A"]
    assert r["products"][0]["device"] == "Esc2"
    assert r["no_data"] is False


def test_compare_periods_requires_two_periods():
    ctx = make_ctx(devices=[{"id": 1, "name": "X"}], mtapi_call=lambda *a: None)
    with pytest.raises(tools.ToolError):
        tools.dispatch("compare_periods", {"node": "X", "period_a": "hoy"}, ctx)


# --- device resolution by name -----------------------------------------------

def test_indicator_accepts_device_by_name():
    call = recording_call({"oee": 0.9})
    ctx = make_ctx(devices=[{"id": 1500, "name": "Inyectoras de bases"}], mtapi_call=call)
    r = tools.dispatch("oee", {"node": "Inyectoras de bases", "period": "hoy"}, ctx)
    assert r["devids"] == [1500]
    assert r["node"] == "Inyectoras de bases"        # result labelled by name
    assert r["type"] == "dev"
    assert call.calls[0][2][-1] == 1500              # single device -> scalar


def test_device_name_match_is_case_insensitive_and_partial():
    call = recording_call({"oee": 0.9})
    ctx = make_ctx(devices=[{"id": 1500, "name": "Inyectoras de bases"}], mtapi_call=call)
    r = tools.dispatch("oee", {"node": "inyectoras", "period": "hoy"}, ctx)
    assert r["devids"] == [1500]


def test_unknown_node_raises_listing_options():
    call = recording_call({"oee": 0.9})
    ctx = make_ctx(devices=[{"id": 1500, "name": "Inyectoras de bases"}], mtapi_call=call)
    with pytest.raises(tools.ToolError) as ei:
        tools.dispatch("oee", {"node": "Robot soldador", "period": "hoy"}, ctx)
    assert "Inyectoras de bases" in str(ei.value)
    assert call.calls == []


SECTION_TREE = {
    "id": 7, "name": "Planta", "type": "plant", "lines": [], "devs": [],
    "sections": [{"id": 20, "name": "Químicos", "type": "section", "devs": [
        {"id": 1, "name": "A", "type": "dev"}, {"id": 2, "name": "B", "type": "dev"}]}],
}


def test_indicator_for_a_section_aggregates_its_devices():
    call = recording_call({"oee": 0.5})
    ctx = make_ctx(devices=[{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
                   tree=SECTION_TREE, mtapi_call=call)
    r = tools.dispatch("oee", {"node": "Químicos", "period": "hoy"}, ctx)
    assert r["type"] == "section"
    assert r["node"] == "Químicos"
    assert r["devids"] == [1, 2]
    assert call.calls[0][2][-1] == [1, 2]            # oee called with the device LIST


# --- daily series / best-worst day -------------------------------------------

def test_daily_oee_flags_best_and_worst_day():
    by_day = {"2026-06-01": 0.5, "2026-06-02": 0.8, "2026-06-03": 0.3}

    def call(fn, client, start, end, devid):
        return by_day.get(start.date().isoformat())

    ctx = make_ctx(devices=[{"id": 1, "name": "A"}], mtapi_call=call)
    r = tools.dispatch("daily_oee", {"node": "A", "period": "esta semana"}, ctx)
    assert len(r["series"]) == 3
    assert r["best_day"] == {"date": "2026-06-02", "oee": 0.8}
    assert r["worst_day"] == {"date": "2026-06-03", "oee": 0.3}


def test_daily_oee_advertised():
    assert "daily_oee" in {t["function"]["name"] for t in tools.TOOL_SPECS}


# --- rank machines by downtime -----------------------------------------------

def test_rank_downtime_orders_machines_by_total_stop_time():
    # pareto([devid]) returns codstates whose time_s sum is the device downtime.
    pareto_by_dev = {
        1079: {"codstates": [{"time_s": 2.0}, {"time_s": 1.0}]},   # 3.0 h
        1080: {"codstates": [{"time_s": 5.0}]},                    # 5.0 h
        1081: {"codstates": []},                                   # 0 -> dropped
    }

    def call(fn, client, start, end, ids):
        return pareto_by_dev[ids[0]]

    ctx = make_ctx(device_ids=(1079, 1080, 1081),
                   names={1079: "A", 1080: "B", 1081: "C"}, mtapi_call=call)
    r = tools.dispatch("rank_downtime", {"period": "ayer"}, ctx)
    assert [d["devid"] for d in r["devices"]] == [1080, 1079]      # most downtime first
    assert r["devices"][0] == {"devid": 1080, "name": "B", "downtime_h": 5.0}
    assert r["type"] == "plant"     # no node given -> whole plant


def test_rank_downtime_advertised():
    assert "rank_downtime" in {t["function"]["name"] for t in tools.TOOL_SPECS}


# --- sabana detail ------------------------------------------------------------

def test_sabana_summarizes_rows_by_product():
    sabana_rows = [
        {"start": "t0", "end": "t1", "duration": 60.0, "production": 100,
         "product": "A", "code_description": ""},
        {"start": "t1", "end": "t2", "duration": 30.0, "production": 50,
         "product": "A", "code_description": ""},
        {"start": "t2", "end": "t3", "duration": 15.0, "production": 0,
         "product": "B", "code_description": "Falla"},
    ]

    def call(fn, client, start, end, dev_ids):
        assert fn == "sabana"
        return sabana_rows

    ctx = make_ctx(devices=[{"id": 1, "name": "A"}], mtapi_call=call)
    r = tools.dispatch("sabana", {"node": "A", "period": "ayer"}, ctx)
    assert r["n_rows"] == 3
    assert r["total_production"] == 150
    assert r["total_duration_min"] == 105.0
    assert r["by_product"][0] == {"product": "A", "production": 150}   # top product
    assert len(r["rows_sample"]) == 3
    assert r["rows_sample"][2]["stop"] == "Falla"


def test_sabana_advertised():
    assert "sabana" in {t["function"]["name"] for t in tools.TOOL_SPECS}


# --- stops detail (timing) ---------------------------------------------------

# stop_intervals (S_reg) returns only stops, with 'reason' + 'devid'.
_STOP_ROWS = [
    {"devid": 1, "start": "2026-06-02 12:00:00", "end": "2026-06-02 12:30:00",
     "duration_min": 30.0, "reason": "PROGRAMADO - COLACION"},
    {"devid": 1, "start": "2026-06-02 10:00:00", "end": "2026-06-02 10:05:00",
     "duration_min": 5.0, "reason": "AVERIA - FALLA"},
]


def test_stops_detail_lists_stops_chronologically():
    def call(fn, client, start, end, dev_ids):
        assert fn == "stop_intervals"
        return _STOP_ROWS

    ctx = make_ctx(devices=[{"id": 1, "name": "Env"}], mtapi_call=call)
    r = tools.dispatch("stops_detail", {"node": "Env", "period": "ayer"}, ctx)
    assert r["n_stops"] == 2
    assert [s["start"] for s in r["stops"]] == [
        "2026-06-02 10:00:00", "2026-06-02 12:00:00"]         # sorted by start
    assert r["stops"][1]["reason"] == "PROGRAMADO - COLACION"
    assert r["stops"][1]["device"] == "Env"                  # devid -> name


def test_stops_detail_filters_by_reason():
    def call(fn, client, start, end, dev_ids):
        return _STOP_ROWS

    ctx = make_ctx(devices=[{"id": 1, "name": "Env"}], mtapi_call=call)
    r = tools.dispatch("stops_detail", {"node": "Env", "period": "ayer",
                                        "reason": "programado"}, ctx)
    assert r["n_stops"] == 1
    assert r["stops"][0]["reason"] == "PROGRAMADO - COLACION"


def test_dispatch_normalizes_arg_key_casing():
    # Models sometimes capitalize keys (e.g. 'Reason'); dispatch lowercases them.
    def call(fn, client, start, end, dev_ids):
        return _STOP_ROWS

    ctx = make_ctx(devices=[{"id": 1, "name": "Env"}], mtapi_call=call)
    r = tools.dispatch("stops_detail", {"Node": "Env", "Period": "ayer",
                                        "Reason": "programado"}, ctx)
    assert r["n_stops"] == 1
    assert r["stops"][0]["reason"] == "PROGRAMADO - COLACION"


# --- no-data flagging (T8) ----------------------------------------------------

def test_indicator_none_value_flags_no_data():
    call = recording_call({"oee": None})
    ctx = make_ctx(mtapi_call=call)
    r = tools.dispatch("oee", {"devid": 1079, "period": "hoy"}, ctx)
    assert r["value"] is None and r["no_data"] is True


def test_indicator_real_value_is_not_no_data():
    call = recording_call({"oee": 0.0})       # 0.0 is real data, not "no data"
    ctx = make_ctx(mtapi_call=call)
    r = tools.dispatch("oee", {"devid": 1079, "period": "hoy"}, ctx)
    assert r["no_data"] is False


def test_rank_devices_empty_flags_no_data():
    call = oee_by_dev({1079: None, 1080: None})       # no device has data
    ctx = make_ctx(device_ids=(1079, 1080), mtapi_call=call)
    r = tools.dispatch("rank_devices", {"period": "hoy"}, ctx)
    assert r["devices"] == [] and r["no_data"] is True


def test_top_stops_empty_flags_no_data():
    call = recording_call({"pareto": {"codstates": []}})
    ctx = make_ctx(mtapi_call=call)
    r = tools.dispatch("top_stops", {"period": "hoy"}, ctx)
    assert r["stops"] == [] and r["no_data"] is True
