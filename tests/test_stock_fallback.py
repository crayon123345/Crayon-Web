"""Stock endpoint (PR #5 review v3): single VERIFIED source (yahoo). No fallback
until a source with permitting terms exists - stooq's CSV endpoint is removed /
anti-bot-walled and CNBC's terms are internal-use-only (both verified live
2026-10-15), so an outage is answered honestly with 503. The returned instrument
is bound: symbol match, stock-like type, named exchange + currency, finite
positive price, fresh timestamp; prev close is normalized before it can 500."""
import os
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("NEON_DATABASE_URL", None)
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

import time
from fastapi.testclient import TestClient
import app as appmod
import httpx


class _Resp:
    def __init__(self, payload=None, status=200):
        self._p = payload
        self.status_code = status
    def json(self):
        return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


def _yh(**meta):
    base = {"symbol": "AAPL", "shortName": "Apple Inc.", "regularMarketPrice": 232.1,
            "chartPreviousClose": 230.5, "currency": "USD", "fullExchangeName": "NasdaqGS",
            "instrumentType": "EQUITY", "regularMarketTime": int(time.time()) - 3600,
            "gmtoffset": -14400, "exchangeTimezoneName": "America/New_York"}
    base.update(meta)
    return {"chart": {"result": [{"meta": base}]}}


def _get(sym, payload=None, status=200, monkeypatch=None, raise_exc=None):
    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, url, params=None):
            if raise_exc:
                raise raise_exc
            return _Resp(payload, status)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    appmod._stcache.clear()
    appmod.hits.clear(); appmod.ip_day.clear(); appmod.day["n"] = 0  # our own rate limiter
    return TestClient(appmod.app).get(f"/api/stock?sym={sym}")


def test_ok(monkeypatch):
    r = _get("AAPL", _yh(), monkeypatch=monkeypatch)
    assert r.status_code == 200
    j = r.json()
    assert j["price"] == 232.1 and j["prev"] == 230.5 and j["delayed"] is True
    assert "America/New_York" in j["date"]


def test_http_404_is_unknown_ticker(monkeypatch):
    r = _get("ZZZZ", status=404, monkeypatch=monkeypatch)
    assert r.status_code == 404


def test_empty_result_is_unknown(monkeypatch):
    r = _get("ZZZZ", {"chart": {"result": []}}, monkeypatch=monkeypatch)
    assert r.status_code == 404


def test_symbol_mismatch_rejected(monkeypatch):
    r = _get("AAPL", _yh(symbol="MSFT"), monkeypatch=monkeypatch)
    assert r.status_code == 404


def test_non_stock_instrument_rejected(monkeypatch):
    for t in ("BOND", "CRYPTOCURRENCY", None):
        r = _get("AAPL", _yh(instrumentType=t), monkeypatch=monkeypatch)
        assert r.status_code == 404, t


def test_missing_currency_or_exchange_rejected(monkeypatch):
    r = _get("AAPL", _yh(currency=""), monkeypatch=monkeypatch)
    assert r.status_code == 404
    r = _get("AAPL", _yh(fullExchangeName="", exchangeName=""), monkeypatch=monkeypatch)
    assert r.status_code == 404


def test_invalid_price_rejected(monkeypatch):
    for p in (float("nan"), float("inf"), -1, 0, True, None, "232.1"):
        r = _get("AAPL", _yh(regularMarketPrice=p), monkeypatch=monkeypatch)
        assert r.status_code == 404, p


def test_nonfinite_prev_normalized_not_500(monkeypatch):
    for p in (float("nan"), float("inf"), -5, 0, "x", None):
        r = _get("AAPL", _yh(chartPreviousClose=p), monkeypatch=monkeypatch)
        assert r.status_code == 200, p
        assert r.json()["prev"] is None


def test_stale_timestamp_is_outage(monkeypatch):
    r = _get("AAPL", _yh(regularMarketTime=1577816000), monkeypatch=monkeypatch)  # 2020
    assert r.status_code == 503
    r = _get("AAPL", _yh(regularMarketTime=None), monkeypatch=monkeypatch)
    assert r.status_code == 503


def test_http_error_is_503(monkeypatch):
    for status in (429, 500, 503):
        r = _get("AAPL", status=status, monkeypatch=monkeypatch)
        assert r.status_code == 503, status


def test_network_error_is_503(monkeypatch):
    r = _get("AAPL", monkeypatch=monkeypatch, raise_exc=httpx.ConnectError("down"))
    assert r.status_code == 503
