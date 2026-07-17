"""LLM provider factory.

Lumen is provider-agnostic: it runs on a local Ollama model (free, private) or
a hosted Groq model (free tier, fast, and — crucially — cloud-deployable, since
managed hosts like Streamlit Cloud cannot run Ollama). The rest of the codebase
never touches a provider directly; it calls :func:`build_chat_model` and gets a
LangChain chat model that supports ``.bind_tools`` either way.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from lumen.config import get_settings
from lumen.utils.logging import get_logger

logger = get_logger("llm")


class LLMConfigError(RuntimeError):
    """Raised when a provider is selected but not properly configured."""


def build_chat_model() -> BaseChatModel:
    """Return a chat model for the configured provider.

    Raises LLMConfigError with an actionable message if the provider is
    misconfigured (e.g. Groq selected but no API key), so failures are clear
    rather than a deep stack trace.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower().strip()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    if provider == "groq":
        if not settings.groq_api_key:
            raise LLMConfigError(
                "LUMEN_LLM_PROVIDER=groq but no GROQ_API_KEY is set. Get a free "
                "key at https://console.groq.com and set GROQ_API_KEY (or "
                "LUMEN_GROQ_API_KEY)."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
        )

    raise LLMConfigError(
        f"Unknown LUMEN_LLM_PROVIDER={provider!r}. Use 'ollama' or 'groq'."
    )


def active_model_name() -> str:
    """Human-readable 'provider:model' string for health checks and the UI."""
    settings = get_settings()
    provider = settings.llm_provider.lower().strip()
    model = settings.groq_model if provider == "groq" else settings.llm_model
    return f"{provider}:{model}"
