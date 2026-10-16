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
