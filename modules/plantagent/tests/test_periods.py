"""Unit tests for period resolution — pure, deterministic, tz-aware."""
from __future__ import annotations

import datetime as dt

import pytest

from modules.plantagent import periods

UTC = dt.timezone.utc
SCL = "America/Santiago"

# Summer in Chile (CLST, UTC-3): local midnight = 03:00 UTC.
NOW_SUMMER = dt.datetime(2026, 1, 15, 18, 30, tzinfo=UTC)  # Thu 15 Jan 2026
# Winter (CLT, UTC-4): local midnight = 04:00 UTC.
NOW_WINTER = dt.datetime(2026, 7, 15, 18, 30, tzinfo=UTC)


def test_hoy_starts_at_local_midnight_ends_now():
    start, end = periods.resolve("hoy", NOW_SUMMER, SCL)
    assert start == dt.datetime(2026, 1, 15, 3, 0)   # naive UTC
    assert end == dt.datetime(2026, 1, 15, 18, 30)


def test_ayer_is_full_previous_local_day():
    start, end = periods.resolve("ayer", NOW_SUMMER, SCL)
    assert start == dt.datetime(2026, 1, 14, 3, 0)
    assert end == dt.datetime(2026, 1, 15, 3, 0)


def test_anteayer():
    start, end = periods.resolve("anteayer", NOW_SUMMER, SCL)
    assert start == dt.datetime(2026, 1, 13, 3, 0)
    assert end == dt.datetime(2026, 1, 14, 3, 0)


def test_ultimos_n_dias_is_rolling_window():
    start, end = periods.resolve("últimos 3 días", NOW_SUMMER, SCL)
    assert start == dt.datetime(2026, 1, 12, 18, 30)
    assert end == dt.datetime(2026, 1, 15, 18, 30)


def test_ultimos_accepts_no_accent_variant():
    a = periods.resolve("ultimos 7 dias", NOW_SUMMER, SCL)
    b = periods.resolve("últimos 7 días", NOW_SUMMER, SCL)
    assert a == b


def test_esta_semana_starts_monday():
    start, _ = periods.resolve("esta semana", NOW_SUMMER, SCL)  # Thu -> Mon 12 Jan
    assert start == dt.datetime(2026, 1, 12, 3, 0)


def test_este_mes_starts_first_of_month():
    start, _ = periods.resolve("este mes", NOW_SUMMER, SCL)
    assert start == dt.datetime(2026, 1, 1, 3, 0)


def test_timezone_and_dst_affect_utc_boundary():
    # Same phrase, different season -> different UTC offset proves tz/DST handling.
    s_summer, _ = periods.resolve("hoy", NOW_SUMMER, SCL)
    s_winter, _ = periods.resolve("hoy", NOW_WINTER, SCL)
    assert s_summer.hour == 3   # UTC-3
    assert s_winter.hour == 4   # UTC-4


def test_returns_naive_utc():
    start, end = periods.resolve("hoy", NOW_SUMMER, SCL)
    assert start.tzinfo is None and end.tzinfo is None


def test_now_is_injected_not_read_from_clock():
    # Pure function of its inputs: different `now` -> different result.
    a = periods.resolve("hoy", NOW_SUMMER, SCL)
    b = periods.resolve("hoy", NOW_WINTER, SCL)
    assert a != b


def test_semana_pasada_is_previous_calendar_week():
    # NOW_SUMMER = Thu 15 Jan; this Monday = 12 Jan; last week = 5–12 Jan.
    start, end = periods.resolve("semana pasada", NOW_SUMMER, SCL)
    assert start == dt.datetime(2026, 1, 5, 3, 0)
    assert end == dt.datetime(2026, 1, 12, 3, 0)


def test_mes_pasado_is_previous_calendar_month():
    start, end = periods.resolve("mes pasado", NOW_SUMMER, SCL)   # Jan -> December
    assert start == dt.datetime(2025, 12, 1, 3, 0)
    assert end == dt.datetime(2026, 1, 1, 3, 0)


def test_este_semana_typo_variant_matches_esta_semana():
    assert periods.resolve("este semana", NOW_SUMMER, SCL) == \
        periods.resolve("esta semana", NOW_SUMMER, SCL)


def test_days_in_splits_a_full_day():
    start, end = periods.resolve("ayer", NOW_SUMMER, SCL)
    days = periods.days_in(start, end, SCL)
    assert len(days) == 1
    assert days[0] == ("2026-01-14",
                       dt.datetime(2026, 1, 14, 3, 0),
                       dt.datetime(2026, 1, 15, 3, 0))


def test_named_month_past_is_full_month():
    # NOW_SUMMER = 15 Jan 2026; "el mes de diciembre" -> Dec 2025 (full).
    start, end = periods.resolve("el mes de diciembre", NOW_SUMMER, SCL)
    assert start == dt.datetime(2025, 12, 1, 3, 0)
    assert end == dt.datetime(2026, 1, 1, 3, 0)


def test_named_month_current_is_capped_at_now():
    # "enero" with now = 15 Jan -> current month, capped at now.
    start, end = periods.resolve("enero", NOW_SUMMER, SCL)
    assert start == dt.datetime(2026, 1, 1, 3, 0)
    assert end == dt.datetime(2026, 1, 15, 18, 30)            # now (naive UTC)


def test_unknown_phrase_raises():
    with pytest.raises(periods.PeriodError):
        periods.resolve("el martes pasado a las 3", NOW_SUMMER, SCL)


def test_zero_or_negative_days_raises():
    with pytest.raises(periods.PeriodError):
        periods.resolve("últimos 0 días", NOW_SUMMER, SCL)
