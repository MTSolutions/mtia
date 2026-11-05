"""Turn-aware period resolution.

Reuses (does not modify) mtapi2's currentturn/turnbounds/getturns. Turns are
per-device, so the caller passes a representative devid of the node. Resolves:

  - "este turno" / "turno actual"           -> current turn (capped at now)
  - "mismo turno (la) semana pasada"         -> current turn name, 7 days back
  - "turno <nombre>"                         -> by turn name (substring)
  - "turno noche|tarde|mañana|día"           -> by local start hour (daypart)

Returns naive-UTC (start, end), matching the indicator functions' convention.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from typing import Callable
from zoneinfo import ZoneInfo

UTC = dt.timezone.utc

# Daypart -> representative local hour (closest turn start wins).
_DAYPART_TARGET = {
    "noche": 23, "nocturno": 23,
    "tarde": 17, "vespertino": 17,
    "manana": 8, "matutino": 8,
    "dia": 11, "diurno": 11,
}


class TurnError(ValueError):
    """A turn phrase could not be resolved to a turn."""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip()


def is_turn_phrase(phrase: str) -> bool:
    return "turno" in _norm(phrase)


def _hour_dist(a: float, b: float) -> float:
    d = abs(a - b) % 24
    return min(d, 24 - d)


def _local_start_hour(start: dt.datetime, tz: str) -> float:
    local = start.replace(tzinfo=UTC).astimezone(ZoneInfo(tz))
    return local.hour + local.minute / 60.0


def resolve_turn(phrase: str, client: str, devid: int, now: dt.datetime,
                 tz: str, mtapi_call: Callable) -> tuple[dt.datetime, dt.datetime]:
    """Resolve a turn phrase to naive-UTC (start, end). Call only when
    is_turn_phrase(phrase) is true."""
    p = _norm(phrase)

    # Same turn, last week: current turn's name, 7 days back.
    if "semana pasada" in p or "semana anterior" in p:
        name, _, _ = mtapi_call("currentturn", client, devid)
        return tuple(mtapi_call("turnbounds", client, now - dt.timedelta(days=7),
                                name, devid))

    # Current turn (also the bare "turno actual"/"este turno"/"turno en curso").
    daypart_or_name = re.sub(r"^.*\bturno\b\s*", "", p).strip()
    if daypart_or_name in ("", "actual", "en curso", "de ahora", "vigente"):
        name, start, end = mtapi_call("currentturn", client, devid)
        if end and now and end > now:
            end = now                      # current turn so far
        return start, end

    # Named/daypart turn -> resolve a turn name for today.
    turns = (mtapi_call("getturns", client, devid, now) or {}).get("turns", {}) or {}
    names = list(turns.keys())

    # 1) name substring match (e.g. "turno TD")
    for name in names:
        if daypart_or_name and daypart_or_name in _norm(name):
            return tuple(mtapi_call("turnbounds", client, now, name, devid))

    # 2) daypart by local start hour (e.g. "turno noche" -> the night turn)
    target = next((h for k, h in _DAYPART_TARGET.items() if k in daypart_or_name), None)
    if target is not None and names:
        best = min(names, key=lambda n: _hour_dist(_local_start_hour(turns[n][0], tz),
                                                    target))
        return tuple(mtapi_call("turnbounds", client, now, best, devid))

    raise TurnError(
        "no identifiqué el turno en {!r}. Turnos disponibles: {}".format(phrase, names))
