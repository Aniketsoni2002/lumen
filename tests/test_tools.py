"""Tests for the retrieval and web-search tools, with externals mocked."""
from __future__ import annotations

from langchain_core.documents import Document

from lumen.tools import retrieval, websearch


def test_search_documents_formats_sources(monkeypatch):
    # Mock the retrieval seam so we test formatting independent of the
    # dense/hybrid retriever choice.
    docs = [
        Document(page_content="Refunds within 30 days.",
                 metadata={"source": "policy.pdf"}),
    ]
    monkeypatch.setattr(retrieval, "_retrieve", lambda q: docs)
    out = retrieval._search_knowledge_base("refund")
    assert "policy.pdf" in out
    assert "Refunds within 30 days" in out


def test_search_documents_handles_empty(monkeypatch):
    monkeypatch.setattr(retrieval, "_retrieve", lambda q: [])
    out = retrieval._search_knowledge_base("anything")
    assert out.startswith("NO_RESULTS")


def test_retrieve_uses_hybrid_when_enabled(monkeypatch):
    """Verify the dense-vs-hybrid routing decision itself."""
    from lumen import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("LUMEN_HYBRID_RETRIEVAL", "true")

    called = {}

    class _FakeHybrid:
        def __init__(self, **kw):
            called["constructed"] = True

        def retrieve(self, query, top_k=None):
            called["retrieved"] = query
            return [Document(page_content="hybrid hit", metadata={})]

    import lumen.core.hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "HybridRetriever", _FakeHybrid)
    out = retrieval._retrieve("q")
    assert called.get("constructed") and called.get("retrieved") == "q"
    assert out[0].page_content == "hybrid hit"
    config.get_settings.cache_clear()


def test_web_search_formats_results(monkeypatch):
    fake_results = [
        {"title": "Python 3.14", "body": "New release.", "href": "https://x"},
    ]

    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            return fake_results

    # Patch the lazily-imported ddgs module.
    import sys
    import types

    fake_mod = types.ModuleType("ddgs")
    fake_mod.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_mod)

    out = websearch._run_web_search("python", max_results=1)
    assert "Python 3.14" in out
    assert "https://x" in out


def test_web_search_degrades_on_error(monkeypatch):
    class _BoomDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            raise RuntimeError("rate limited")

    import sys
    import types

    fake_mod = types.ModuleType("ddgs")
    fake_mod.DDGS = _BoomDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_mod)

    out = websearch._run_web_search("python", max_results=1)
    assert out.startswith("WEB_SEARCH_ERROR")
