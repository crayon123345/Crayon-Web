"""Stock fallback: a yahoo failure must degrade to stooq, never straight to 503
(live failure 2026-10-15: yahoo broke and /api/stock 503'd). 503 only when both
sources fail; 404 only when the ticker is genuinely unknown on both."""
import os
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("NEON_DATABASE_URL", None)
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

from fastapi.testclient import TestClient
import app as appmod
import httpx


class _Resp:
    def __init__(self, payload=None, text=None, status=200):
        self._p = payload
        self.text = text or ""
        self.status_code = status
    def json(self):
        return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


YH = {"chart": {"result": [{"meta": {"symbol": "AAPL", "shortName": "Apple Inc.",
      "regularMarketPrice": 232.1, "chartPreviousClose": 230.5, "currency": "USD",
      "fullExchangeName": "NasdaqGS", "regularMarketTime": 1760548800}}]}}
CSV_OK = "Symbol,Date,Time,Open,High,Low,Close,Volume\nAAPL.US,2026-10-15,19:59:00,231.0,233.0,230.5,232.1,123456\n"
CSV_ND = "Symbol,Date,Time,Open,High,Low,Close,Volume\nZZZZ.US,N/D,N/D,N/D,N/D,N/D,N/D,N/D\n"


def _client_factory(route):
    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, url, params=None):
            return route(url, params or {})
    return FakeAsyncClient


def _get(sym, route, monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(route))
    appmod._stcache.clear()
    return TestClient(appmod.app).get(f"/api/stock?sym={sym}")


def test_yahoo_ok_no_fallback(monkeypatch):
    def route(url, params):
        assert "yahoo" in url, "fallback must not fire when yahoo answers"
        return _Resp(YH)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "yahoo finance" and j["price"] == 232.1


def test_yahoo_http_error_falls_back_to_stooq(monkeypatch):
    def route(url, params):
        if "yahoo" in url:
            return _Resp(status=503)
        if "stooq" in url:
            assert params["s"] == "aapl.us"
            return _Resp(text=CSV_OK)
        raise AssertionError(url)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "stooq" and j["price"] == 232.1


def test_yahoo_connection_error_falls_back_to_stooq(monkeypatch):
    def route(url, params):
        if "yahoo" in url:
            raise httpx.ConnectError("down")
        return _Resp(text=CSV_OK)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 200 and r.json()["source"] == "stooq"


def test_yahoo_empty_result_tries_stooq(monkeypatch):
    def route(url, params):
        if "yahoo" in url:
            return _Resp({"chart": {"result": []}})
        return _Resp(text=CSV_OK)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 200 and r.json()["source"] == "stooq"


def test_both_down_is_503(monkeypatch):
    def route(url, params):
        if "yahoo" in url:
            return _Resp(status=503)
        if "stooq" in url:
            return _Resp(status=500)
        raise AssertionError(url)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 503


def test_unknown_on_both_is_404(monkeypatch):
    def route(url, params):
        if "yahoo" in url:
            return _Resp({"chart": {"result": []}})
        return _Resp(text=CSV_ND)
    r = _get("ZZZZ", route, monkeypatch)
    assert r.status_code == 404
