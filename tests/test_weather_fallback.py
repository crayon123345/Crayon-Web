"""Weather fallback: an open-meteo error body must not render as a null/"Unknown"
card - the handler falls back to wttr.in, and only 503s when both sources fail."""
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
            return _Resp(route(url, params or {}))
    return FakeAsyncClient


GEO = {"results": [{"name": "Kolkata", "country": "India", "latitude": 22.56, "longitude": 88.36}]}
OM_ERR = {"error": True, "reason": "API request limit exceeded"}
OM_OK = {"current": {"temperature_2m": 27.1, "apparent_temperature": 29.0,
                     "relative_humidity_2m": 71, "weather_code": 0, "wind_speed_10m": 7.2},
         "daily": {"time": ["2026-10-15"], "temperature_2m_max": [34.0],
                   "temperature_2m_min": [26.0], "precipitation_probability_max": [18]}}
WTTR = {"current_condition": [{"temp_C": "27", "FeelsLikeC": "29", "humidity": "71",
                               "windspeedKmph": "7", "weatherDesc": [{"value": "Clear "}]}],
        "weather": [{"date": "2026-10-15", "maxtempC": "34", "mintempC": "26",
                     "hourly": [{"chanceofrain": "18"}]}]}


def test_openmeteo_error_body_falls_back_to_wttr(monkeypatch):
    """Repro of the live 2026-10-15 failure: forecast returns {"error": true, ...}
    (HTTP success, no exception) and the card used to render all-null weather."""
    def route(url, params):
        if "geocoding" in url:
            return GEO
        if "open-meteo.com" in url:
            return OM_ERR
        if "wttr.in" in url:
            return WTTR
        raise AssertionError(url)
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(route))
    r = TestClient(appmod.app).get("/api/weather?q=Kolkata")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["source"] == "wttr.in"
    assert j["temp"] == 27.0 and j["cond"] == "Clear"
    assert j["days"][0]["max"] == 34.0 and j["days"][0]["rain"] == 18


def test_openmeteo_ok_stays_primary(monkeypatch):
    """Happy path regression: a good open-meteo forecast wins and the fallback
    is never called."""
    def route(url, params):
        if "geocoding" in url:
            return GEO
        if "open-meteo.com" in url:
            return OM_OK
        raise AssertionError("fallback must not be called: " + url)
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(route))
    r = TestClient(appmod.app).get("/api/weather?q=Kolkata")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["source"] == "open-meteo.com"
    assert j["temp"] == 27.1 and j["cond"] == "Clear sky"
    assert j["days"][0]["rain"] == 18


def test_both_sources_down_is_503(monkeypatch):
    def route(url, params):
        if "geocoding" in url:
            return GEO
        if "open-meteo.com" in url:
            return OM_ERR
        raise RuntimeError("network down")
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(route))
    r = TestClient(appmod.app).get("/api/weather?q=Kolkata")
    assert r.status_code == 503, r.text
