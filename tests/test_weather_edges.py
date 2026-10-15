"""Weather edge cases (PR #2 caveats): wttr.in resolves place names itself, so a
geocoding OUTAGE must not 503 the whole endpoint, and a geocoder 'no results'
must still try wttr.in before declaring a place unknown. 404 only when the
geocoder ran cleanly AND wttr.in has nothing; any outage -> 503."""
import os
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("NEON_DATABASE_URL", None)
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

from fastapi.testclient import TestClient
import app as appmod
import httpx


class _Resp:
    def __init__(self, payload):
        self._p = payload
    def json(self):
        return self._p


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


GEO = {"results": [{"name": "Kolkata", "country": "India", "latitude": 22.56, "longitude": 88.36}]}
GEO_NONE = {"results": []}
WTTR = {"current_condition": [{"temp_C": "27", "FeelsLikeC": "29", "humidity": "71",
                               "windspeedKmph": "7", "weatherDesc": [{"value": "Clear "}]}],
        "weather": [{"date": "2026-10-15", "maxtempC": "34", "mintempC": "26",
                     "hourly": [{"chanceofrain": "18"}]}]}
WTTR_EMPTY = {"current_condition": [], "weather": []}


def _get(route, monkeypatch, q="Kolkata"):
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(route))
    return TestClient(appmod.app).get(f"/api/weather?q={q}")


def test_geocoding_outage_still_answers_via_wttr(monkeypatch):
    """The gap: geocoding raised -> old code 503'd even though wttr.in never
    needed the geocoder."""
    def route(url, params):
        if "geocoding" in url:
            raise httpx.ConnectError("geocoder down")
        if "wttr.in" in url:
            return _Resp(WTTR)
        raise AssertionError(url)
    r = _get(route, monkeypatch, q="Kolkata")
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "wttr.in" and j["place"] == "Kolkata" and j["temp"] == 27.0


def test_geocoder_no_results_still_tries_wttr(monkeypatch):
    def route(url, params):
        if "geocoding" in url:
            return _Resp(GEO_NONE)
        if "wttr.in" in url:
            return _Resp(WTTR)
        raise AssertionError(url)
    r = _get(route, monkeypatch, q="Some Tiny Hamlet")
    assert r.status_code == 200
    assert r.json()["source"] == "wttr.in"


def test_unknown_on_both_is_404(monkeypatch):
    def route(url, params):
        if "geocoding" in url:
            return _Resp(GEO_NONE)
        if "wttr.in" in url:
            return _Resp(WTTR_EMPTY)
        raise AssertionError(url)
    r = _get(route, monkeypatch, q="Zzz nowhere")
    assert r.status_code == 404


def test_geocoder_outage_plus_wttr_empty_is_503_not_404(monkeypatch):
    """Cannot declare 'place not found' when the geocoder never answered."""
    def route(url, params):
        if "geocoding" in url:
            raise httpx.ConnectError("down")
        if "wttr.in" in url:
            return _Resp(WTTR_EMPTY)
        raise AssertionError(url)
    r = _get(route, monkeypatch)
    assert r.status_code == 503


def test_real_place_both_sources_down_is_503(monkeypatch):
    def route(url, params):
        if "geocoding" in url:
            return _Resp(GEO)
        if "open-meteo.com" in url:
            raise httpx.ConnectError("forecast down")
        if "wttr.in" in url:
            raise httpx.ConnectError("wttr down")
        raise AssertionError(url)
    r = _get(route, monkeypatch)
    assert r.status_code == 503
