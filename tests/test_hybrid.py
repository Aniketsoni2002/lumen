"""Tests for hybrid retrieval and RRF fusion (pure, no models/network)."""
from __future__ import annotations

from langchain_core.documents import Document

from lumen.core import hybrid
from lumen.core.hybrid import reciprocal_rank_fusion


def _doc(text, source="s", start=0):
    return Document(
        page_content=text, metadata={"source": source, "start_index": start}
    )


def test_rrf_rewards_agreement_between_retrievers():
    a, b, c = _doc("a", start=0), _doc("b", start=1), _doc("c", start=2)
    # 'a' is top of dense and 2nd of sparse -> should rank first overall.
    dense = [a, b, c]
    sparse = [b, a, c]
    fused = reciprocal_rank_fusion([(dense, 0.5), (sparse, 0.5)], top_k=3)
    assert fused[0].page_content == "a"
    assert {d.page_content for d in fused} == {"a", "b", "c"}


def test_rrf_dedupes_same_document():
    a = _doc("shared", start=0)
    # Same doc from both retrievers must not appear twice.
    fused = reciprocal_rank_fusion([([a], 0.5), ([a], 0.5)], top_k=5)
    assert len(fused) == 1


def test_rrf_respects_top_k():
    docs = [_doc(str(i), start=i) for i in range(10)]
    fused = reciprocal_rank_fusion([(docs, 1.0)], top_k=3)
    assert len(fused) == 3


def test_rrf_weight_biases_ranking():
    a, b = _doc("a", start=0), _doc("b", start=1)
    # Sparse strongly prefers 'b'; give sparse a much higher weight.
    dense = [a, b]
    sparse = [b, a]
    fused = reciprocal_rank_fusion([(dense, 0.1), (sparse, 0.9)], top_k=2)
    assert fused[0].page_content == "b"


def test_hybrid_retriever_falls_back_to_dense_when_corpus_empty(monkeypatch):
    """If BM25 has no corpus, hybrid returns the dense results unchanged."""
    dense_docs = [_doc("only-dense")]

    class _FakeRetriever:
        def invoke(self, _q):
            return dense_docs

    class _FakeStore:
        def as_retriever(self, **_):
            return _FakeRetriever()

        def get(self, **_):
            return {"documents": [], "metadatas": []}

    monkeypatch.setattr(hybrid, "get_vectorstore", lambda: _FakeStore())
    out = hybrid.HybridRetriever().retrieve("q", top_k=4)
    assert [d.page_content for d in out] == ["only-dense"]
