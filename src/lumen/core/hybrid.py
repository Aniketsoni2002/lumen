"""Hybrid retrieval: dense (semantic) + sparse (BM25 keyword), fused.

Why this exists
---------------
Pure vector search is great at *meaning* but can miss exact keyword matches
(product codes, error strings, rare proper nouns). BM25 is the opposite: great
at exact terms, blind to synonyms. Combining them beats either alone on real
corpora.

We fuse the two ranked lists with **Reciprocal Rank Fusion (RRF)** — a simple,
robust, score-normalisation-free method: a document's fused score is the sum of
``weight / (k + rank)`` across each retriever that returned it. Documents ranked
highly by *either* method bubble up; documents ranked highly by *both* win.

Implemented directly (no EnsembleRetriever dependency) so the ranking logic is
explicit and unit-testable without any model or network.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from lumen.config import get_settings
from lumen.core.vectorstore import get_vectorstore
from lumen.utils.logging import get_logger

logger = get_logger("hybrid")

# RRF constant. 60 is the value from the original Cormack et al. paper; it
# damps the influence of exact rank so the fusion is stable.
RRF_K = 60


def _doc_key(doc: Document) -> str:
    """Stable identity for de-duplicating a doc across the two retrievers."""
    src = doc.metadata.get("source", "")
    start = doc.metadata.get("start_index", "")
    # Fall back to content hash if metadata is missing.
    return f"{src}:{start}:{hash(doc.page_content)}"


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[Document], float]],
    *,
    top_k: int,
    k: int = RRF_K,
) -> list[Document]:
    """Fuse several (ranked_docs, weight) lists into one ranking via RRF.

    Pure function — no I/O — so it is trivially unit-testable.
    """
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for docs, weight in ranked_lists:
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            docs_by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)

    ranked_keys = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [docs_by_key[key] for key in ranked_keys[:top_k]]


@dataclass
class HybridRetriever:
    """Combines the vector store with an in-memory BM25 index over the same docs.

    The BM25 index is built lazily from whatever is currently in the vector
    store, so it always reflects the indexed corpus without a second copy on
    disk.
    """

    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    def _all_documents(self) -> list[Document]:
        """Pull every stored chunk out of Chroma to build the BM25 index."""
        store = get_vectorstore()
        raw = store.get(include=["documents", "metadatas"])
        docs: list[Document] = []
        for content, meta in zip(
            raw.get("documents", []) or [],
            raw.get("metadatas", []) or [],
            strict=False,
        ):
            docs.append(Document(page_content=content, metadata=meta or {}))
        return docs

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        settings = get_settings()
        k = top_k or settings.top_k

        # 1. Dense (semantic) results from Chroma.
        dense = get_vectorstore().as_retriever(
            search_kwargs={"k": k}
        ).invoke(query)

        # 2. Sparse (BM25 keyword) results over the same corpus.
        corpus = self._all_documents()
        sparse: list[Document] = []
        if corpus:
            bm25 = BM25Retriever.from_documents(corpus)
            bm25.k = k
            sparse = bm25.invoke(query)

        if not corpus:
            return dense

        fused = reciprocal_rank_fusion(
            [(dense, self.dense_weight), (sparse, self.sparse_weight)],
            top_k=k,
        )
        logger.info(
            "Hybrid retrieval: %d dense + %d sparse -> %d fused",
            len(dense), len(sparse), len(fused),
        )
        return fused
