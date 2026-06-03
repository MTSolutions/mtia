"""Unit tests for the tool catalog — dispatch + validation, mtapi stubbed."""
from __future__ import annotations

import datetime as dt

import pytest

from modules.plantagent import mtapi, tools
from modules.plantagent.tools import ToolContext

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 6, 3, 12, 0, tzinfo=UTC)


def make_ctx(device_ids=(1079, 1080, 1081), mtapi_call=None):
    return ToolContext(
        client="degasa", plant_id=7, device_ids=list(device_ids),
        now=NOW, tz="America/Santiago", mtapi_call=mtapi_call,
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


# --- ranking via a single devtree call ---------------------------------------

PLANT_TREE_WITH_OEE = {
    "id": 7, "name": "Planta", "type": "plant",
    "lines": [
        {"id": 10, "name": "L1", "type": "line", "devs": [
            {"id": 1079, "name": "Llenadora", "type": "dev", "oee": 0.82},
            {"id": 1080, "name": "Etiquetadora", "type": "dev", "oee": 0.55},
        ]},
    ],
    "sections": [],
    "devs": [{"id": 1081, "name": "Encajonadora", "type": "dev", "oee": 0.73}],
}


def test_rank_oee_returns_worst_first_via_single_devtree_call():
    call = recording_call({"devtree": PLANT_TREE_WITH_OEE})
    ctx = make_ctx(mtapi_call=call)
    result = tools.dispatch("rank_oee", {"period": "últimos 3 días"}, ctx)

    # Exactly one mtapi call (devtree), not per-device fan-out.
    assert len(call.calls) == 1
    assert call.calls[0][0] == "devtree"
    # devtree(client, start, end, 'plant', plant_id, ['oee'], flat)
    assert call.calls[0][2][2] == "plant"
    assert call.calls[0][2][4] == ["oee"]

    devices = result["devices"]
    # worst OEE first: 0.55 (1080) < 0.73 (1081) < 0.82 (1079)
    assert [d["devid"] for d in devices] == [1080, 1081, 1079]
    assert devices[0]["oee"] == 0.55


def test_rank_oee_ignores_devs_without_oee():
    tree = {"id": 7, "type": "plant", "devs": [
        {"id": 1079, "type": "dev", "oee": 0.6},
        {"id": 1080, "type": "dev", "oee": None},
    ]}
    call = recording_call({"devtree": tree})
    ctx = make_ctx(device_ids=(1079, 1080), mtapi_call=call)
    result = tools.dispatch("rank_oee", {"period": "hoy"}, ctx)
    assert [d["devid"] for d in result["devices"]] == [1079]
