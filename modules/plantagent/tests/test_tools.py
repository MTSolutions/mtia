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
            "cumplimiento", "rank_oee"} <= names


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

def oee_by_dev(oee_map, names=None):
    """Stub: oee() returns per-devid values; getdevs() returns id/name pairs."""
    calls = []

    def _call(fn, client, *args):
        calls.append((fn, client, args))
        if fn == "oee":
            return oee_map.get(args[-1])
        if fn == "getdevs":
            return [{"id": d, "name": (names or {}).get(d)} for d in oee_map]
        raise KeyError(fn)

    _call.calls = calls  # type: ignore[attr-defined]
    return _call


def test_rank_oee_returns_worst_first_from_per_device_calls():
    call = oee_by_dev({1079: 0.82, 1080: 0.55, 1081: 0.73})
    ctx = make_ctx(device_ids=(1079, 1080, 1081), names={1080: "Etiquetadora"},
                   mtapi_call=call)
    result = tools.dispatch("rank_oee", {"period": "últimos 3 días"}, ctx)

    # One oee() call per device (no faulting devtree-with-indicators).
    assert [c[0] for c in call.calls].count("oee") == 3
    devices = result["devices"]
    # worst OEE first: 0.55 (1080) < 0.73 (1081) < 0.82 (1079)
    assert [d["devid"] for d in devices] == [1080, 1081, 1079]
    assert devices[0]["oee"] == 0.55
    assert devices[0]["name"] == "Etiquetadora"


def test_rank_oee_ignores_devs_without_oee():
    call = oee_by_dev({1079: 0.6, 1080: None})
    ctx = make_ctx(device_ids=(1079, 1080), mtapi_call=call)
    result = tools.dispatch("rank_oee", {"period": "hoy"}, ctx)
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


def test_rank_oee_empty_flags_no_data():
    call = oee_by_dev({1079: None, 1080: None})       # no device has OEE data
    ctx = make_ctx(device_ids=(1079, 1080), mtapi_call=call)
    r = tools.dispatch("rank_oee", {"period": "hoy"}, ctx)
    assert r["devices"] == [] and r["no_data"] is True


def test_top_stops_empty_flags_no_data():
    call = recording_call({"pareto": {"codstates": []}})
    ctx = make_ctx(mtapi_call=call)
    r = tools.dispatch("top_stops", {"period": "hoy"}, ctx)
    assert r["stops"] == [] and r["no_data"] is True
