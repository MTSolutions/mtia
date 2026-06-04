"""Unit tests for turn-period resolution — stubbed mtapi turn functions."""
from __future__ import annotations

import datetime as dt

import pytest

from modules.plantagent import turns

NOW = dt.datetime(2026, 6, 3, 12, 0)            # naive UTC
SCL = "America/Santiago"                          # winter UTC-4

# virutex-like turns (naive-UTC bounds). Local: TN 22–07 (noche), TT 14:30–22
# (tarde), TD 07–14:30 (día).
TURNS = {"turns": {
    "TN": (dt.datetime(2026, 6, 4, 2, 0), dt.datetime(2026, 6, 4, 11, 0)),
    "TT": (dt.datetime(2026, 6, 3, 18, 30), dt.datetime(2026, 6, 4, 2, 0)),
    "TD": (dt.datetime(2026, 6, 3, 11, 0), dt.datetime(2026, 6, 3, 18, 30)),
}}


def make_call(responses):
    calls = []

    def _c(fn, client, *args):
        calls.append((fn,) + args)
        r = responses[fn]
        return r(*args) if callable(r) else r

    _c.calls = calls  # type: ignore[attr-defined]
    return _c


def test_is_turn_phrase():
    assert turns.is_turn_phrase("¿OEE de la Línea este turno?")
    assert turns.is_turn_phrase("turno noche")
    assert not turns.is_turn_phrase("ayer")


def test_current_turn_is_capped_at_now():
    call = make_call({"currentturn": ("TD", dt.datetime(2026, 6, 3, 11, 0),
                                      dt.datetime(2026, 6, 3, 18, 30))})
    start, end = turns.resolve_turn("este turno", "virutex", 1, NOW, SCL, call)
    assert start == dt.datetime(2026, 6, 3, 11, 0)
    assert end == NOW                              # turn end 18:30 > now -> capped


def test_same_turn_last_week_uses_current_name_seven_days_back():
    call = make_call({
        "currentturn": ("TN", dt.datetime(1, 1, 1), dt.datetime(1, 1, 1)),
        "turnbounds": lambda date, name, devid: (name, date),
    })
    turns.resolve_turn("mismo turno la semana pasada", "virutex", 1, NOW, SCL, call)
    fn, date, name, devid = call.calls[-1]
    assert fn == "turnbounds"
    assert date == NOW - dt.timedelta(days=7)
    assert name == "TN"


def test_daypart_noche_picks_night_turn_by_local_hour():
    call = make_call({"getturns": TURNS,
                      "turnbounds": lambda date, name, devid: (name, name)})
    start, _ = turns.resolve_turn("turno noche", "virutex", 1, NOW, SCL, call)
    assert start == "TN"                           # night turn chosen


def test_daypart_tarde_picks_afternoon_turn():
    call = make_call({"getturns": TURNS,
                      "turnbounds": lambda date, name, devid: (name, name)})
    start, _ = turns.resolve_turn("turno tarde", "virutex", 1, NOW, SCL, call)
    assert start == "TT"


def test_named_turn_substring():
    call = make_call({"getturns": TURNS,
                      "turnbounds": lambda date, name, devid: (name, name)})
    start, _ = turns.resolve_turn("turno TD", "virutex", 1, NOW, SCL, call)
    assert start == "TD"


def test_unrecognized_turn_raises_listing_turns():
    call = make_call({"getturns": TURNS})
    with pytest.raises(turns.TurnError) as ei:
        turns.resolve_turn("turno xyz", "virutex", 1, NOW, SCL, call)
    assert "TN" in str(ei.value)
