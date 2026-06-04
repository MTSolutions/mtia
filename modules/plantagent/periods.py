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

_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b")


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

    if p in ("esta semana", "este semana", "semana", "la semana"):
        start = today - dt.timedelta(days=today.weekday())  # Monday
        return _to_utc_naive(start), now_utc
    if p in ("semana pasada", "la semana pasada", "ultima semana", "última semana"):
        this_monday = today - dt.timedelta(days=today.weekday())
        return (_to_utc_naive(this_monday - dt.timedelta(days=7)),
                _to_utc_naive(this_monday))
    if p in ("este mes", "esta mes", "mes", "el mes"):
        return _to_utc_naive(today.replace(day=1)), now_utc
    if p in ("mes pasado", "el mes pasado", "ultimo mes", "último mes"):
        first_this = today.replace(day=1)
        first_prev = (first_this - dt.timedelta(days=1)).replace(day=1)
        return _to_utc_naive(first_prev), _to_utc_naive(first_this)

    # Named month ("mayo", "el mes de mayo") -> its most recent past/current
    # occurrence. Current month is capped at `now`.
    mm = _MONTH_RE.search(p)
    if mm:
        month = _MONTHS[mm.group(1)]
        year = today.year if month <= today.month else today.year - 1
        start = today.replace(year=year, month=month, day=1)
        end = (start.replace(year=year + 1, month=1) if month == 12
               else start.replace(month=month + 1))
        if end > local:                       # current/ongoing month
            end = local
        return _to_utc_naive(start), _to_utc_naive(end)

    raise PeriodError("no se reconoce el período: {!r}".format(phrase))


def days_in(start: dt.datetime, end: dt.datetime, tz: str) -> list[tuple[str, dt.datetime, dt.datetime]]:
    """Split a naive-UTC ``[start, end)`` range into local calendar days.

    Returns ``[(date_iso, day_start_utc, day_end_utc)]`` — each day's bounds are
    local-midnight to next-local-midnight (the last day capped at ``end``).
    Used for daily series (e.g. best/worst day).
    """
    zone = ZoneInfo(tz)
    start_local = start.replace(tzinfo=UTC).astimezone(zone)
    end_local = end.replace(tzinfo=UTC).astimezone(zone)
    day = start_local.replace(hour=0, minute=0, second=0, microsecond=0)
    out: list[tuple[str, dt.datetime, dt.datetime]] = []
    while day < end_local:
        nxt = day + dt.timedelta(days=1)
        out.append((
            day.date().isoformat(),
            _to_utc_naive(day),
            _to_utc_naive(min(nxt, end_local)),
        ))
        day = nxt
    return out
