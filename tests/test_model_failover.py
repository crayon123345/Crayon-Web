"""Streaming failover loop (PR #8 review): an empty non-exception stream must
fall through to the next model, and a hard model-ID error must put that model
in cooldown. Uses FastAPI TestClient with fake model objects; no network.
"""

import os
os.environ.setdefault("SECRET_KEY", "test-secret")

import llm


class _Chunk:
    def __init__(self, text):
        self.content = text


class FakeModel:
    """behavior: 'empty' streams nothing, 'raise' raises, else streams the text."""

    def __init__(self, behavior, text=""):
        self.behavior = behavior
        self.text = text

    async def astream(self, msgs):
        if self.behavior == "raise":
            raise Exception(self.text)
        if self.behavior == "empty":
            return
            yield  # pragma: no cover
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
    client = TestClient(app.app)
    return client.post("/api/chat", json={"message": "hi"})


def test_empty_stream_falls_through(monkeypatch):
    steps = [("gemini", "g1", "g1"), ("groq", "q1", "q1")]
    models = {"g1": FakeModel("empty"), "q1": FakeModel("stream", "hello world")}
    app = _patch(monkeypatch, steps, models)
    r = _post(app)
    assert r.status_code == 200
    assert "hello " in r.text and "world " in r.text
    # the emitted model label must be the fallback's, not the empty one's
    assert '{"m": "q1"}' in r.text


def test_hard_error_marks_model_bad_and_falls_through(monkeypatch):
    llm._FAILED_MODELS.clear()
    steps = [("gemini", "g1", "g1"), ("groq", "q1", "q1")]
    models = {"g1": FakeModel("raise", "Error code: 404 - model_not_found: decommissioned"),
              "q1": FakeModel("stream", "still here")}
    app = _patch(monkeypatch, steps, models)
    r = _post(app)
    try:
        assert r.status_code == 200
        assert "still " in r.text and "here " in r.text
        assert llm.model_cooling("g1")
    finally:
        llm._FAILED_MODELS.clear()


def test_transient_error_does_not_cool_down(monkeypatch):
    llm._FAILED_MODELS.clear()
    steps = [("gemini", "g1", "g1"), ("groq", "q1", "q1")]
    models = {"g1": FakeModel("raise", "429 rate limit exceeded"),
              "q1": FakeModel("stream", "ok")}
    app = _patch(monkeypatch, steps, models)
    r = _post(app)
    assert r.status_code == 200
    assert not llm.model_cooling("g1")


def test_last_step_empty_still_gets_retry(monkeypatch):
    """Reconciliation of #4 retry and #8 fallthrough: an empty stream falls
    through while a next step exists, but the LAST step's empty stream goes
    to the silent same-model retry instead of dying as a snag."""
    class ScriptModel:
        def __init__(self, scripts):
            self.scripts = list(scripts)
            self.calls = 0
        async def astream(self, msgs):
            self.calls += 1
            for piece in self.scripts.pop(0) if self.scripts else []:
                yield _Chunk(piece)

    g1 = ScriptModel([[]])                       # empty -> must fall through
    q1 = ScriptModel([[], ["recovered"]])
    steps = [("gemini", "g1", "g1"), ("groq", "q1", "q1")]
    app = _patch(monkeypatch, steps, {"g1": g1, "q1": q1})
    r = _post(app)
    assert r.status_code == 200
    assert g1.calls == 1          # no retry on the non-last empty step
    assert q1.calls == 2          # last step empty -> silent retry once
    assert "recovered" in r.text
