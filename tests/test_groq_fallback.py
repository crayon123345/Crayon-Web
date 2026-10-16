"""Groq free-tier fallback leg (PR: free model fallback).

plan() must include Groq models only when GROQ_API_KEY is set, keep the
existing gemini/openrouter relative order, and build_groq must target the
Groq OpenAI-compatible endpoint. Network is never touched: a fake
langchain_openai module is injected into sys.modules (build_groq imports it
lazily inside the function).
"""

import sys
import types

import llm


def _clear(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CRAYON_GROQ_MODELS", raising=False)
    monkeypatch.delenv("CRAYON_CODE_MODELS", raising=False)
    monkeypatch.delenv("CRAYON_DEEPSEEK_MODEL", raising=False)


def test_no_groq_without_key(monkeypatch):
    _clear(monkeypatch)
    steps = llm.plan("hello there")
    assert [k for k, _, _ in steps] == ["gemini"] * len(llm.model_names())


def test_groq_before_openrouter_on_general_query(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    kinds = [k for k, _, _ in llm.plan("what is the capital of France?")]
    assert kinds[0] == "gemini"
    assert kinds[-1] == "or"
    assert "groq" in kinds
    assert kinds.index("groq") < kinds.index("or")


def test_code_query_keeps_openrouter_first_and_groq_last(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    kinds = [k for k, _, _ in llm.plan("fix this python function bug")]
    assert kinds[0] == "or"
    assert "gemini" in kinds
    assert kinds[-1] == "groq"


def test_groq_models_env_override(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("CRAYON_GROQ_MODELS", " a/one , b/two ,")
    assert llm.groq_models() == ["a/one", "b/two"]


def test_build_groq_uses_groq_endpoint(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    fake = types.ModuleType("langchain_openai")
    fake.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake)
    llm.build_groq("openai/gpt-oss-120b")
    assert captured["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["model"] == "openai/gpt-oss-120b"
    assert captured["api_key"] == "gsk_test"


def test_mail_mode_excludes_groq_leg(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    kinds = [k for k, _, _ in llm.plan("summarize my inbox", mail=True)]
    assert "groq" not in kinds
    assert kinds[0] == "gemini"
    assert "or" in kinds  # OpenRouter leg predates this PR and stays


def test_cooling_model_is_skipped(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    bad = llm.groq_models()[0]
    llm.mark_model_bad(bad)
    try:
        assert llm.model_cooling(bad)
        names = [n for _, n, _ in llm.plan("hello")]
        assert bad not in names
        # other groq model still present
        assert llm.groq_models()[1] in names
    finally:
        llm._FAILED_MODELS.clear()


def test_hard_model_error_detection():
    assert llm.hard_model_error(Exception("Error code: 404 - model_not_found: decommissioned"))
    assert llm.hard_model_error(Exception("The model `x` does not exist"))
    assert llm.hard_model_error(Exception("404: model not found"))
    assert not llm.hard_model_error(Exception("429 rate limit exceeded"))
    assert not llm.hard_model_error(Exception("500 internal server error"))
    # bare "not found" is NOT enough: a generic 404 (endpoint/route issue)
    # must not cool a healthy model for 10 minutes
    assert not llm.hard_model_error(Exception("404 page not found"))


def test_low_uptime_does_not_mark_unfailed_models_cooling(monkeypatch):
    """Fresh containers (Render restarts) have monotonic() near 0: with the old
    .get(name, 0) default every model looked cooling and plan() returned []."""
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("time.monotonic", lambda: 80.0)
    llm._FAILED_MODELS.clear()
    assert not llm.model_cooling("openai/gpt-oss-120b")
    steps = llm.plan("hello")
    assert steps, "plan() must not be empty on a low-uptime host"
    # and a genuinely failed model still cools under low uptime
    llm.mark_model_bad("openai/gpt-oss-120b")
    try:
        assert llm.model_cooling("openai/gpt-oss-120b")
    finally:
        llm._FAILED_MODELS.clear()


def test_redact_strips_all_provider_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    out = llm.redact("failed gem-secret then groq-secret then or-secret")
    assert "secret" not in out.replace("or-secret", "")
    assert "gem-secret" not in out and "groq-secret" not in out and "or-secret" not in out
    assert out.count("***") == 3
