"""Deterministic time resolution. The LLM extracts a `TimeSpec`; calendar arithmetic is done here (small LLMs get dates wrong:
observed qwen3:8b answering "yesterday" with the wrong year)."""
from __future__ import annotations

import re
from datetime import datetime, time, timedelta

from .schema import TimeSpec

PARTS = {"morning": (6, 12), "afternoon": (12, 17), "evening": (17, 21), "night": (21, 30)}   # night = 21:00 -> 06:00 next day


class TimeError(ValueError):
    pass


def _day(s: str, now: datetime):
    t = s.strip().lower()
    if t == "today":
        return now.date()
    if t == "yesterday":
        return (now - timedelta(days=1)).date()
    if m := re.fullmatch(r"(\d{1,3}) days? ago", t):
        return (now - timedelta(days=int(m[1]))).date()
    try:
        return datetime.strptime(t, "%Y-%m-%d").date()
    except ValueError as e:
        raise TimeError(f"cannot understand the day {s!r}") from e


def _clock(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def resolve(spec: TimeSpec, now: datetime) -> tuple[float | None, float | None]:
    """-> (start_epoch, end_epoch), or (None, None) for no time constraint. Raises TimeError for unusable specs."""
    if spec.kind == "none":
        return None, None
    tz = now.tzinfo
    if spec.kind == "last":
        if not spec.amount or not spec.unit:
            raise TimeError("'last ...' needs an amount and a unit")
        return (now - timedelta(**{spec.unit: spec.amount})).timestamp(), now.timestamp()
    day = _day(spec.day or "today", now)
    at = lambda d, t: datetime.combine(d, t, tzinfo=tz)       # noqa: E731
    if spec.start_clock or spec.end_clock:
        if not (spec.start_clock and spec.end_clock):
            raise TimeError("need both a start and an end time")
        start, end = at(day, _clock(spec.start_clock)), at(day, _clock(spec.end_clock))
        if end <= start:
            end += timedelta(days=1)
    elif spec.part_of_day:
        a, b = PARTS[spec.part_of_day]
        start = at(day, time(a)); end = start + timedelta(hours=b - a)
    elif spec.kind == "between":
        raise TimeError("'between' needs start and end times")
    else:
        start = at(day, time(0)); end = start + timedelta(days=1)
    if start > now:
        raise TimeError("that time is in the future")
    return start.timestamp(), min(end, now).timestamp()
