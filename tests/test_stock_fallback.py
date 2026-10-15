"""Stock fallback (PR #5 review v2): yahoo failures degrade to CNBC's free JSON
quote, never straight to 503 (live failure 2026-10-15). A yahoo 404/empty is an
UNKNOWN ticker, not an outage. CNBC fallback is restricted to plain US tickers
and binds the returned symbol to the request. Stooq was dropped: its q/l CSV
endpoint is removed and q/d/l sits behind an anti-bot check (verified live)."""
import os
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("NEON_DATABASE_URL", None)
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

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


YH = {"chart": {"result": [{"meta": {"symbol": "AAPL", "shortName": "Apple Inc.",
      "regularMarketPrice": 232.1, "chartPreviousClose": 230.5, "currency": "USD",
      "fullExchangeName": "NasdaqGS", "regularMarketTime": 1760548800}}]}}
CNBC = {"FormattedQuoteResult": {"FormattedQuote": [
    {"symbol": "AAPL", "code": 0, "name": "Apple Inc.", "last": "344.80",
     "previous_day_closing": "342.96", "currencyCode": "USD", "exchange": "NASDAQ",
     "last_time": "2026-10-15T11:05:55.163-0400"}]}}
CNBC_UNKNOWN = {"FormattedQuoteResult": {"FormattedQuote": [{"symbol": "ZZZZ", "code": 1}]}}


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


def test_yahoo_5xx_falls_back_to_cnbc(monkeypatch):
    def route(url, params):
        if "yahoo" in url:
            return _Resp(status=503)
        if "cnbc" in url:
            assert params["symbols"] == "AAPL"
            return _Resp(CNBC)
        raise AssertionError(url)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "cnbc" and j["price"] == 344.8 and j["prev"] == 342.96
    assert j["currency"] == "USD" and j["delayed"] is True


def test_yahoo_connection_error_falls_back_to_cnbc(monkeypatch):
    def route(url, params):
        if "yahoo" in url:
            raise httpx.ConnectError("down")
        return _Resp(CNBC)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 200 and r.json()["source"] == "cnbc"


def test_yahoo_404_plus_cnbc_unknown_is_404_not_503(monkeypatch):
    """MUST-1 regression: yahoo 404 means UNKNOWN ticker, not an outage."""
    def route(url, params):
        if "yahoo" in url:
            return _Resp(status=404)
        return _Resp(CNBC_UNKNOWN)
    r = _get("ZZZZ", route, monkeypatch)
    assert r.status_code == 404


def test_yahoo_404_but_cnbc_knows_it_returns_cnbc(monkeypatch):
    def route(url, params):
        if "yahoo" in url:
            return _Resp(status=404)
        return _Resp(CNBC)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 200 and r.json()["source"] == "cnbc"


def test_both_down_is_503(monkeypatch):
    def route(url, params):
        if "yahoo" in url:
            return _Resp(status=503)
        if "cnbc" in url:
            return _Resp(status=500)
        raise AssertionError(url)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 503


def test_yahoo_unknown_but_cnbc_down_is_503(monkeypatch):
    """Cannot declare unknown on one source's word when the other never answered."""
    def route(url, params):
        if "yahoo" in url:
            return _Resp(status=404)
        return _Resp(status=500)
    r = _get("ZZZZ", route, monkeypatch)
    assert r.status_code == 503


def test_suffixed_symbol_never_goes_to_cnbc(monkeypatch):
    """MUST-2: yahoo namespaces (.NS, ^NSEI) must not be guessed into CNBC."""
    def route(url, params):
        assert "cnbc" not in url, "suffixed symbol leaked into the CNBC namespace"
        return _Resp(status=503)
    r = _get("RELIANCE.NS", route, monkeypatch)
    assert r.status_code == 503


def test_cnbc_symbol_mismatch_rejected(monkeypatch):
    bad = {"FormattedQuoteResult": {"FormattedQuote": [dict(CNBC["FormattedQuoteResult"]["FormattedQuote"][0], symbol="MSFT")]}}
    def route(url, params):
        if "yahoo" in url:
            return _Resp(status=503)
        return _Resp(bad)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 503


def test_nan_price_treated_as_unknown(monkeypatch):
    nan_meta = {"chart": {"result": [{"meta": {"symbol": "AAPL", "regularMarketPrice": float("nan")}}]}}
    def route(url, params):
        if "yahoo" in url:
            return _Resp(nan_meta)
        return _Resp(CNBC)
    r = _get("AAPL", route, monkeypatch)
    assert r.status_code == 200 and r.json()["source"] == "cnbc"
