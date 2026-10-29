"""DeepSeek-R1-Distill-Llama-70B on the Groq free tier: chain placement and
<think> stripping. Network is never touched."""

import llm


def _clear(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CRAYON_GROQ_MODELS", raising=False)
    monkeypatch.delenv("CRAYON_GROQ_DEEPSEEK", raising=False)
    monkeypatch.delenv("CRAYON_CODE_MODELS", raising=False)
    monkeypatch.delenv("CRAYON_DEEPSEEK_MODEL", raising=False)


def test_deepseek_last_on_general_query(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    groq_names = [n for k, n, _ in llm.plan("what is the capital of France?") if k == "groq"]
    assert groq_names == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "deepseek-r1-distill-llama-70b"]


def test_deepseek_first_on_reasoning_query(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    steps = llm.plan("solve this math equation")
    groq_names = [n for k, n, _ in steps if k == "groq"]
    assert groq_names[0] == "deepseek-r1-distill-llama-70b"
    assert steps[0][0] == "gemini"  # no OpenRouter key -> Gemini still leads


def test_deepseek_excluded_in_mail_mode(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert not any(k == "groq" for k, _, _ in llm.plan("debug this python bug", mail=True))


def test_deepseek_absent_without_groq_key(monkeypatch):
    _clear(monkeypatch)
    assert all("deepseek" not in n for _, n, _ in llm.plan("solve this math equation"))


def test_deepseek_disable_env(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("CRAYON_GROQ_DEEPSEEK", "")
    assert llm.groq_models() == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def test_explicit_groq_models_kept_exact(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("CRAYON_GROQ_MODELS", " a/one , b/two ,")
    assert llm.groq_models() == ["a/one", "b/two"]  # deepseek NOT auto-added to an explicit list
    monkeypatch.setenv("CRAYON_GROQ_DEEPSEEK", "deepseek-r1-distill-llama-70b")
    assert llm.groq_models() == ["a/one", "b/two", "deepseek-r1-distill-llama-70b"]


def test_think_stripper_across_chunk_boundaries():
    s = llm.ThinkStripper()
    out = "".join(s.feed(c) for c in ["<thi", "nk>hidden reasoning...", "</th", "ink>Hello ", "world", ""])
    out += s.flush()
    assert out == "Hello world"


def test_think_stripper_plain_text_passthrough():
    s = llm.ThinkStripper()
    out = "".join(s.feed(c) for c in ["plain ", "answer"])
    out += s.flush()
    assert out == "plain answer"


def test_think_stripper_short_answer_under_tag_length():
    s = llm.ThinkStripper()
    assert s.feed("Hi") + s.flush() == "Hi"


def test_think_stripper_unterminated_think_is_dropped():
    s = llm.ThinkStripper()
    out = s.feed("<think>reasoning that never closes and keeps growing past the tail")
    out += s.flush()
    assert out == ""
