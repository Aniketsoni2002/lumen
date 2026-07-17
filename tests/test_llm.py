"""Tests for the LLM provider factory (no real models constructed)."""
from __future__ import annotations

import pytest

from lumen import config
from lumen.agent import llm


@pytest.fixture(autouse=True)
def _clear_settings():
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_ollama_provider_builds(monkeypatch):
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "ollama")
    model = llm.build_chat_model()
    assert model.__class__.__name__ == "ChatOllama"


def test_groq_provider_requires_key(monkeypatch):
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LUMEN_GROQ_API_KEY", raising=False)
    with pytest.raises(llm.LLMConfigError) as exc:
        llm.build_chat_model()
    assert "GROQ_API_KEY" in str(exc.value)


def test_groq_provider_builds_with_key(monkeypatch):
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    model = llm.build_chat_model()
    assert model.__class__.__name__ == "ChatGroq"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "openai")
    with pytest.raises(llm.LLMConfigError):
        llm.build_chat_model()


def test_active_model_name_ollama(monkeypatch):
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "qwen2.5:7b")
    assert llm.active_model_name() == "ollama:qwen2.5:7b"


def test_active_model_name_groq(monkeypatch):
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "groq")
    monkeypatch.setenv("LUMEN_GROQ_MODEL", "llama-3.3-70b-versatile")
    assert llm.active_model_name() == "groq:llama-3.3-70b-versatile"


def test_groq_api_key_reads_standard_env(monkeypatch):
    # The standard GROQ_API_KEY (not just LUMEN_-prefixed) must be picked up.
    monkeypatch.setenv("GROQ_API_KEY", "abc")
    config.get_settings.cache_clear()
    assert config.get_settings().groq_api_key == "abc"
