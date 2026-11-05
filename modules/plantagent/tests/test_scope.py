"""Unit tests for scope resolution — stubbed mtapi.call, no network."""
from __future__ import annotations

import datetime as dt

import pytest

from modules.plantagent import scope

START = dt.datetime(2026, 6, 1)
END = dt.datetime(2026, 6, 2)

PLANTS = [{"id": 7, "name": "Planta 1"}, {"id": 8, "name": "Planta 2"}]

# devtree('plant', 7) nested shape: lines -> devs, sections -> devs, loose devs.
PLANT7_TREE = {
    "id": 7, "name": "Planta 1", "type": "plant",
    "lines": [
        {"id": 10, "name": "L1", "type": "line", "devs": [
            {"id": 101, "name": "d1", "type": "dev"},
            {"id": 102, "name": "d2", "type": "dev"},
        ]},
    ],
    "sections": [
        {"id": 20, "name": "S1", "type": "section", "devs": [
            {"id": 201, "name": "d3", "type": "dev"},
        ]},
    ],
    "devs": [{"id": 301, "name": "d4", "type": "dev"}],
}


def make_call(responses: dict):
    calls = []

    def _call(fn, client, *args):
        calls.append((fn, client, args))
        return responses[fn]

    _call.calls = calls  # type: ignore[attr-defined]
    return _call


def test_validate_plant_returns_the_plant():
    call = make_call({"getplants": PLANTS})
    plant = scope.validate_plant("degasa", 7, mtapi_call=call)
    assert plant["id"] == 7
    # getplants is called with the JWT client.
    assert call.calls[0][:2] == ("getplants", "degasa")


def test_validate_plant_rejects_foreign_plant():
    call = make_call({"getplants": PLANTS})
    with pytest.raises(scope.PlantNotFound):
        scope.validate_plant("degasa", 999, mtapi_call=call)


def test_device_ids_collects_all_devs_under_plant():
    call = make_call({"devtree": PLANT7_TREE})
    ids = scope.device_ids("degasa", "plant", 7, START, END, mtapi_call=call)
    assert set(ids) == {101, 102, 201, 301}


def test_device_ids_passes_empty_indicators_and_client():
    call = make_call({"devtree": PLANT7_TREE})
    scope.device_ids("degasa", "plant", 7, START, END, mtapi_call=call)
    fn, client, args = call.calls[0]
    assert fn == "devtree" and client == "degasa"
    # devtree(client, start, end, _type, _id, indicators, flat)
    assert args == (START, END, "plant", 7, [], False)


def test_device_ids_for_a_single_device_returns_itself():
    dev_tree = {"id": 101, "name": "d1", "type": "dev"}
    call = make_call({"devtree": dev_tree})
    ids = scope.device_ids("degasa", "dev", 101, START, END, mtapi_call=call)
    assert ids == [101]


NAMED_TREE = {
    "id": 7, "name": "Planta 1", "type": "plant",
    "lines": [],
    "sections": [{"id": 20, "name": "Químicos", "type": "section", "devs": [
        {"id": 10, "name": "Envasadora", "type": "dev"}]}],
    "devs": [{"id": 11, "name": "Sopladora", "type": "dev"}],
}


def test_named_tree_calls_devtree_named_with_client():
    call = make_call({"devtree_named": NAMED_TREE})
    tree = scope.named_tree("degasa", 7, mtapi_call=call)
    assert tree["name"] == "Planta 1"
    assert call.calls[0][:2] == ("devtree_named", "degasa")
    assert call.calls[0][2] == ("plant", 7)


def test_devices_in_collects_named_devices():
    assert scope.devices_in(NAMED_TREE) == [
        {"id": 10, "name": "Envasadora"},
        {"id": 11, "name": "Sopladora"},
    ]


def test_resolve_node_tolerates_typos():
    tree = {"id": 7, "name": "P", "type": "plant", "lines": [], "sections": [],
            "devs": [{"id": 9, "name": "Máquina de escobillones 2", "type": "dev"}]}
    node = scope.resolve_node(tree, "Máquina de escobillenos 2")   # typo
    assert node and node["id"] == 9
