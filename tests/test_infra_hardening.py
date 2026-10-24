"""Infra hardening: boot-time DB degrade (dead Postgres -> ephemeral sqlite),
/health provider/db surface, and mid-stream provider failover."""

import os
os.environ.setdefault("SECRET_KEY", "test-secret")

import gmail_auth
import llm


def test_boot_db_degrades_when_neon_dead(monkeypatch):
    import app
    dead = "postgres://u:p@127.0.0.1:1/nope"  # connection refused, fast
    monkeypatch.setenv("NEON_DATABASE_URL", dead)
    monkeypatch.setattr(gmail_auth, "_DB", dead)
    app._boot_db()
    assert gmail_auth._DB == ""  # degraded to ephemeral sqlite, not a dead URL


def test_boot_db_noop_without_postgres(monkeypatch):
    import app
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.setattr(gmail_auth, "_DB", "")
    app._boot_db()
    assert gmail_auth._DB == ""


def test_health_surfaces_providers_and_db(monkeypatch):
    import app
    monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(gmail_auth, "_DB", "")
    from fastapi.testclient import TestClient
    r = TestClient(app.app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["providers"] == []
    assert body["db"] == "sqlite"


class _Chunk:
    def __init__(self, text):
        self.content = text


class FakeModel:
    """'raise_after' streams text then dies; 'fence_raise' streams only an
    unclosed fenced block (all held back) then dies; else streams text."""

    def __init__(self, behavior, text=""):
        self.behavior = behavior
        self.text = text

    async def astream(self, msgs):
        if self.behavior == "raise_after":
            for piece in self.text.split(" "):
                yield _Chunk(piece + " ")
            raise Exception("connection reset")
        if self.behavior == "fence_raise":
            yield _Chunk(self.text)
            raise Exception("connection reset")
        for piece in self.text.split(" "):
            yield _Chunk(piece + " ")


def _patch(monkeypatch, steps, models):
    import app
    monkeypatch.setattr(llm, "plan", lambda text, mail=False: steps)
    monkeypatch.setattr(llm, "build_or", lambda nm: models[nm])
    monkeypatch.setattr(llm, "build_groq", lambda nm: models[nm])
    monkeypatch.setattr(llm, "build_model", lambda streaming=True, name=None: models[name])
    app._models.clear()
    return app


def _post(app):
    from fastapi.testclient import TestClient
    return TestClient(app.app).post("/api/chat", json={"message": "hi"})


def test_midstream_death_before_output_fails_over(monkeypatch):
    # everything the first model produced was fence-holdback, so nothing reached
    # the page: the partial buffer must be dropped and the next provider tried
    steps = [("gemini", "g1", "g1"), ("groq", "q1", "q1")]
    models = {"g1": FakeModel("fence_raise", "```web-search\n{cut off"),
              "q1": FakeModel("stream", "clean fallback answer")}
    app = _patch(monkeypatch, steps, models)
    r = _post(app)
    assert r.status_code == 200
    assert "clean " in r.text and "fallback " in r.text
    assert '{"m": "q1"}' in r.text
    assert "```web-search" not in r.text


def test_midstream_death_after_output_keeps_partial_and_says_so(monkeypatch):
    steps = [("gemini", "g1", "g1"), ("groq", "q1", "q1")]
    models = {"g1": FakeModel("raise_after", "partial answer here"),
              "q1": FakeModel("stream", "should not be used")}
    app = _patch(monkeypatch, steps, models)
    r = _post(app)
    assert r.status_code == 200
    assert "partial " in r.text
    assert "cut short" in r.text
    assert "should not be used" not in r.text
