"""Resolve Spanish relative-date expressions to concrete UTC datetime ranges.

Pure and deterministic: the reference ``now`` is always passed in (never read
from the clock here), so results are reproducible and unit-testable. Day
boundaries are computed in the plant's local timezone, then returned as
**naive UTC** half-open ``[start, end)`` ranges to match how mtapi2 indicator
functions expect their ``start``/``end`` arguments (see mtapi2 ``turnbounds``,
which returns naive UTC).

Turn-relative phrases ("este turno") are intentionally *not* resolved here:
they require a specific device's turn schedule from mtapi2 and are handled by
the agent where a ``devid`` is known.
"""
from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo

UTC = dt.timezone.utc

# "últimos N días" / "ultimos N dias" (optional accents, singular/plural).
_ULTIMOS_DIAS = re.compile(r"^[uú]ltimos?\s+(\d+)\s+d[ií]as?$")


class PeriodError(ValueError):
    """The relative-date expression could not be resolved."""


def _to_utc_naive(d: dt.datetime) -> dt.datetime:
    return d.astimezone(UTC).replace(tzinfo=None)


def resolve(phrase: str, now: dt.datetime, tz: str) -> tuple[dt.datetime, dt.datetime]:
    """Return naive-UTC ``(start, end)`` for a Spanish relative-date phrase.

    Args:
        phrase: e.g. "hoy", "ayer", "anteayer", "últimos 3 días",
            "esta semana", "este mes".
        now: reference instant — a timezone-aware datetime.
        tz: IANA timezone name of the plant (e.g. "America/Santiago").

    Raises:
        PeriodError: the phrase is not a supported relative-date expression.
    """
    if now.tzinfo is None:
        raise PeriodError("`now` must be timezone-aware")

    zone = ZoneInfo(tz)
    local = now.astimezone(zone)
    today = local.replace(hour=0, minute=0, second=0, microsecond=0)
    now_utc = _to_utc_naive(now)
    p = phrase.strip().lower()

    if p == "hoy":
        return _to_utc_naive(today), now_utc
    if p == "ayer":
        return _to_utc_naive(today - dt.timedelta(days=1)), _to_utc_naive(today)
    if p == "anteayer":
        return (_to_utc_naive(today - dt.timedelta(days=2)),
                _to_utc_naive(today - dt.timedelta(days=1)))

    m = _ULTIMOS_DIAS.match(p)
    if m:
        n = int(m.group(1))
        if n <= 0:
            raise PeriodError("number of days must be positive: {!r}".format(phrase))
        return _to_utc_naive(now - dt.timedelta(days=n)), now_utc

    if p in ("esta semana", "semana"):
        start = today - dt.timedelta(days=today.weekday())  # Monday
        return _to_utc_naive(start), now_utc
    if p in ("este mes", "mes"):
        return _to_utc_naive(today.replace(day=1)), now_utc

    raise PeriodError("no se reconoce el período: {!r}".format(phrase))
