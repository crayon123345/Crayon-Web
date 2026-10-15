"""Stream honesty + reliability: a truncated fenced tool block must never render
as raw markup on the page (live bug 2026-10-15: "```web-search ..." shown to the
visitor), an empty answer gets one silent retry, and a still-broken retry falls
back to the empty-answer message instead of leaking partial markup."""
import json
import os
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("NEON_DATABASE_URL", None)
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

from fastapi.testclient import TestClient
import app as appmod
import llm


class _Chunk:
    def __init__(self, text):
        self.content = text


class _FakeModel:
    """Plays back one list-of-chunks script per astream() call."""
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = 0
    async def astream(self, msgs):
        self.calls += 1
        for piece in self.scripts.pop(0) if self.scripts else []:
            yield _Chunk(piece)


def _sse_texts(resp):
    texts = []
    for line in resp.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line[6:])
        if "t" in payload:
            texts.append(payload["t"])
    return texts


def _post(fake, monkeypatch):
    monkeypatch.setattr(llm, "plan", lambda text: [("or", "fake", "default")])
    monkeypatch.setattr(llm, "build_or", lambda nm: fake)
    monkeypatch.setattr(appmod.memory, "load_context", lambda vid: [])
    monkeypatch.setattr(appmod.memory, "save_turn", lambda *a, **k: None)
    monkeypatch.setattr(appmod.memory, "save_chat_turn", lambda *a, **k: None)
    monkeypatch.setattr(appmod.memory, "event", lambda *a, **k: None)
    r = TestClient(appmod.app).post("/api/chat", json={"message": "hi", "session_id": "t-" + os.urandom(4).hex()})
    assert r.status_code == 200, r.text
    return r


def test_unclosed_block_never_streamed_and_retry_replaces(monkeypatch):
    """Live repro: the whole reply was one big unclosed ```web-search block.
    Nothing may reach the page; the silent retry's clean answer is what shows."""
    fake = _FakeModel([
        ["```web-search\n", "search('kolkata weather')\n"],          # truncated tool call
        ["Kolkata is 28C and clear right now."],                       # retry
    ])
    texts = _sse_texts(_post(fake, monkeypatch))
    assert fake.calls == 2
    assert texts and "".join(texts).strip() == "Kolkata is 28C and clear right now."
    assert not any("```" in t for t in texts)


def test_partial_answer_kept_truncated_tail_dropped(monkeypatch):
    """A good partial answer was already on screen when the stream cut off inside
    a fenced block: keep the answer, drop the tail, no retry (can't un-show text)."""
    fake = _FakeModel([
        ["Sure, here it is. ", "```web-search\nsearch("],
    ])
    texts = _sse_texts(_post(fake, monkeypatch))
    assert fake.calls == 1
    assert "".join(texts).strip() == "Sure, here it is."
    assert not any("```" in t for t in texts)


def test_empty_answer_gets_one_retry(monkeypatch):
    fake = _FakeModel([[], ["Here is the answer."]])
    texts = _sse_texts(_post(fake, monkeypatch))
    assert fake.calls == 2
    assert "".join(texts).strip() == "Here is the answer."


def test_empty_after_retry_shows_friendly_message(monkeypatch):
    fake = _FakeModel([[], []])
    texts = _sse_texts(_post(fake, monkeypatch))
    assert fake.calls == 2
    assert texts == ["I got an empty answer. Try rephrasing?"]


def test_still_broken_retry_dropped_not_leaked(monkeypatch):
    fake = _FakeModel([
        ["```web-search\n", "search("],          # truncated
        ["```python-run\n", "print("],           # retry truncated too
    ])
    texts = _sse_texts(_post(fake, monkeypatch))
    assert fake.calls == 2
    assert texts == ["I got an empty answer. Try rephrasing?"]
    assert not any("```" in t for t in texts)


def test_system_prompt_forbids_fake_future_actions():
    s = llm.to_messages([], "hi")[0].content
    assert "NO reminder" in s
    assert "NEVER promise a future or deferred action" in s
