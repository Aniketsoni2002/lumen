"""Tests for the pluggable embedding backend selection.

We patch the embedding classes so no model is actually downloaded — the point
is to verify the provider routing and error handling, cheaply and offline.
"""
from __future__ import annotations

import sys
import types

import pytest

from lumen import config
from lumen.core import vectorstore


@pytest.fixture(autouse=True)
def _clear():
    config.get_settings.cache_clear()
    vectorstore.get_embeddings.cache_clear()
    yield
    config.get_settings.cache_clear()
    vectorstore.get_embeddings.cache_clear()


def _install_fake(module_path: str, class_name: str) -> dict:
    """Install a fake embeddings class at ``module_path`` and record calls."""
    seen = {}

    class _Fake:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    mod = types.ModuleType(module_path)
    setattr(mod, class_name, _Fake)
    sys.modules[module_path] = mod
    return seen


def test_fastembed_provider_selected(monkeypatch):
    monkeypatch.setenv("LUMEN_EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("LUMEN_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    seen = _install_fake(
        "langchain_community.embeddings", "FastEmbedEmbeddings"
    )
    emb = vectorstore.get_embeddings()
    assert emb.__class__.__name__ == "_Fake"
    assert seen["kwargs"]["model_name"] == "BAAI/bge-small-en-v1.5"


def test_huggingface_provider_selected(monkeypatch):
    monkeypatch.setenv("LUMEN_EMBEDDING_PROVIDER", "huggingface")
    seen = _install_fake("langchain_huggingface", "HuggingFaceEmbeddings")
    emb = vectorstore.get_embeddings()
    assert emb.__class__.__name__ == "_Fake"
    assert "model_name" in seen["kwargs"]


def test_unknown_embedding_provider_raises(monkeypatch):
    monkeypatch.setenv("LUMEN_EMBEDDING_PROVIDER", "word2vec")
    with pytest.raises(ValueError, match="Unknown"):
        vectorstore.get_embeddings()
