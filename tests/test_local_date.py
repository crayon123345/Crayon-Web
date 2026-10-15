"""Freshness date must be the VISITOR's local date, not UTC (PR #3 caveat:
between 00:00-05:30 IST a UTC clock still says yesterday and mis-anchors
"today"/"latest"). Unknown or missing timezones fall back to UTC."""
import datetime
import os
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("NEON_DATABASE_URL", None)

import llm


def _fake_now(monkeypatch, utc_dt):
    class _FakeDT(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return utc_dt.astimezone(tz) if tz else utc_dt.replace(tzinfo=None)
    monkeypatch.setattr(llm.datetime, "datetime", _FakeDT)


def test_ist_early_morning_shows_next_day(monkeypatch):
    # 2026-10-16 02:00 IST = 2026-10-15 20:30 UTC: UTC says the 15th, IST is the 16th.
    _fake_now(monkeypatch, datetime.datetime(2026, 10, 15, 20, 30, tzinfo=datetime.timezone.utc))
    s = llm.to_messages([], "hi", tz="Asia/Calcutta")[0].content
    assert "2026-10-16 (Friday) (Asia/Calcutta)" in s


def test_no_tz_defaults_to_utc(monkeypatch):
    _fake_now(monkeypatch, datetime.datetime(2026, 10, 15, 20, 30, tzinfo=datetime.timezone.utc))
    s = llm.to_messages([], "hi")[0].content
    assert "2026-10-15 (Thursday) (UTC)" in s


def test_bad_tz_falls_back_to_utc(monkeypatch):
    _fake_now(monkeypatch, datetime.datetime(2026, 10, 15, 20, 30, tzinfo=datetime.timezone.utc))
    s = llm.to_messages([], "hi", tz="Not/AZone")[0].content
    assert "2026-10-15 (Thursday) (UTC)" in s


def test_zoneinfo_loads_real_zone():
    # Pins the tzdata dependency: without system tz data or the tzdata package
    # this raises and every zone silently degrades to UTC.
    from zoneinfo import ZoneInfo
    z = ZoneInfo("Asia/Calcutta")
    import datetime as _d
    dt = _d.datetime(2026, 10, 15, 12, 0, tzinfo=_d.timezone.utc).astimezone(z)
    assert dt.utcoffset() == _d.timedelta(hours=5, minutes=30)
