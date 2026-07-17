"""Tests for the retrieval and web-search tools, with externals mocked."""
from __future__ import annotations

from langchain_core.documents import Document

from agentrag.tools import retrieval, websearch


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, _query):
        return self._docs


class _FakeStore:
    def __init__(self, docs):
        self._docs = docs

    def as_retriever(self, **_kwargs):
        return _FakeRetriever(self._docs)


def test_search_documents_formats_sources(monkeypatch):
    docs = [
        Document(page_content="Refunds within 30 days.",
                 metadata={"source": "policy.pdf"}),
    ]
    monkeypatch.setattr(retrieval, "get_vectorstore", lambda: _FakeStore(docs))
    out = retrieval._search_knowledge_base("refund")
    assert "policy.pdf" in out
    assert "Refunds within 30 days" in out


def test_search_documents_handles_empty(monkeypatch):
    monkeypatch.setattr(retrieval, "get_vectorstore", lambda: _FakeStore([]))
    out = retrieval._search_knowledge_base("anything")
    assert out.startswith("NO_RESULTS")


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
