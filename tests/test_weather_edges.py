"""Weather edge cases v2 (PR #7 review): wttr.in fallback must be BOUND -
geocoded places bind by coordinates, free-text fallback must show real
resolution evidence (nearest_area, nonzero population) and is labeled with the
area wttr actually resolved, never the raw query. Provider error bodies, empty
bodies, HTTP errors and unusable readings are OUTAGES (503), never 'place not
found'; 404 is reserved for a clean geocoder miss + wttr.in having nothing."""
import os
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("NEON_DATABASE_URL", None)
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

from fastapi.testclient import TestClient
import app as appmod
import httpx


class _Resp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status
    def json(self):
        return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


GEO = {"results": [{"name": "Kolkata", "country": "India", "latitude": 22.56, "longitude": 88.36}]}
GEO_MISS = {"generationtime_ms": 0.55}  # open-meteo's real clean-miss shape
NA_KOLKATA = [{"areaName": [{"value": "Kolkata"}], "country": [{"value": "India"}],
               "latitude": "22.57", "longitude": "88.36", "population": "5000000"}]
NA_PARIS = [{"areaName": [{"value": "Paris"}], "country": [{"value": "France"}],
             "latitude": "48.86", "longitude": "2.35"}]
NA_NULL = [{"areaName": [{"value": "Place"}], "country": [{"value": "France"}],
            "latitude": "48.25", "longitude": "-0.78", "population": "0"}]
CC = [{"temp_C": "27", "FeelsLikeC": "29", "humidity": "71", "windspeedKmph": "7",
       "weatherDesc": [{"value": "Clear"}]}]


def _wttr(na, cc=None):
    return {"nearest_area": na, "current_condition": cc if cc is not None else CC, "weather": []}


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


def _get(route, monkeypatch, q="Kolkata"):
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(route))
    return TestClient(appmod.app).get(f"/api/weather?q={q}")


def test_geocoding_outage_bound_fallback_via_wttr(monkeypatch):
    def route(url, params):
        if "geocoding" in url:
            raise httpx.ConnectError("geocoder down")
        if "wttr.in" in url:
            return _Resp(_wttr(NA_KOLKATA))
        raise AssertionError(url)
    r = _get(route, monkeypatch)
    assert r.status_code == 200
    assert r.json()["source"] == "wttr.in" and r.json()["place"] == "Kolkata, India"


def test_freetext_fallback_labeled_with_resolved_area_not_query(monkeypatch):
    """Never relabel mismatched weather: the card names what wttr resolved."""
    def route(url, params):
        if "geocoding" in url:
            return _Resp({"results": []})
        if "wttr.in" in url:
            return _Resp(_wttr(NA_PARIS))
        raise AssertionError(url)
    r = _get(route, monkeypatch, q="Some Tiny Hamlet")
    assert r.status_code == 200
    assert r.json()["place"] == "Paris, France"


def test_wttr_null_island_population_zero_is_not_a_place(monkeypatch):
    """Live repro 2026-10-15: garbage query -> wttr resolves its default
    location (population "0") - that is NOT a resolution."""
    def route(url, params):
        if "geocoding" in url:
            return _Resp(GEO_MISS)
        if "wttr.in" in url:
            return _Resp(_wttr(NA_NULL))
        raise AssertionError(url)
    r = _get(route, monkeypatch, q="ZzxxyyNoSuchPlace987654")
    assert r.status_code == 404


def test_known_place_coordinate_mismatch_is_outage(monkeypatch):
    """Kolkata geocoded + forecast down + Paris weather must NOT be served as
    Kolkata."""
    def route(url, params):
        if "geocoding" in url:
            return _Resp(GEO)
        if "open-meteo.com" in url:
            return _Resp({"error": True, "reason": "down"})
        if "wttr.in" in url:
            return _Resp(_wttr(NA_PARIS))
        raise AssertionError(url)
    r = _get(route, monkeypatch)
    assert r.status_code == 503


def test_geocoder_error_body_is_outage_not_404(monkeypatch):
    for body, status in (({"error": True, "reason": "outage"}, 200), ({}, 200), ({"reason": "rate limited"}, 429)):
        def route(url, params, body=body, status=status):
            if "geocoding" in url:
                return _Resp(body, status=status)
            if "wttr.in" in url:
                return _Resp({"current_condition": []})
            raise AssertionError(url)
        r = _get(route, monkeypatch)
        assert r.status_code == 503, body


def test_wttr_http_error_is_outage_even_with_data(monkeypatch):
    def route(url, params):
        if "geocoding" in url:
            return _Resp({"results": []})
        if "wttr.in" in url:
            return _Resp(_wttr(NA_KOLKATA), status=503)
        raise AssertionError(url)
    r = _get(route, monkeypatch)
    assert r.status_code == 503


def test_nonnumeric_temp_is_outage(monkeypatch):
    bad = [{"temp_C": "N/A", "FeelsLikeC": "N/A", "humidity": "71", "windspeedKmph": "7",
            "weatherDesc": [{"value": "Clear"}]}]
    def route(url, params):
        if "geocoding" in url:
            return _Resp({"results": []})
        if "wttr.in" in url:
            return _Resp(_wttr(NA_KOLKATA, cc=bad))
        raise AssertionError(url)
    r = _get(route, monkeypatch)
    assert r.status_code == 503


def test_clean_miss_plus_wttr_nothing_is_404(monkeypatch):
    def route(url, params):
        if "geocoding" in url:
            return _Resp({"results": []})
        if "wttr.in" in url:
            return _Resp({"current_condition": []})
        raise AssertionError(url)
    r = _get(route, monkeypatch, q="Zzz nowhere")
    assert r.status_code == 404


def test_known_place_both_sources_down_is_503(monkeypatch):
    def route(url, params):
        if "geocoding" in url:
            return _Resp(GEO)
        raise httpx.ConnectError("down")
    r = _get(route, monkeypatch)
    assert r.status_code == 503
