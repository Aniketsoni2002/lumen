"""Document-retrieval tool.

Exposes the local knowledge base (ChromaDB) to the agent as a callable tool.
The agent decides *when* to use this — e.g. for questions about the user's own
uploaded documents — rather than us hard-wiring a retrieve-always pipeline.
"""
from __future__ import annotations

from langchain_core.tools import tool

from lumen.config import get_settings
from lumen.core.vectorstore import get_vectorstore
from lumen.utils.logging import get_logger

logger = get_logger("tool.retrieval")


def _retrieve(query: str):
    """Return the most relevant chunks, using hybrid retrieval if enabled."""
    settings = get_settings()
    if settings.hybrid_retrieval:
        from lumen.core.hybrid import HybridRetriever

        return HybridRetriever(
            dense_weight=settings.hybrid_dense_weight,
            sparse_weight=settings.hybrid_sparse_weight,
        ).retrieve(query, top_k=settings.top_k)
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    ).invoke(query)


def _search_knowledge_base(query: str) -> str:
    """Core retrieval logic, separated so it is easy to unit-test."""
    docs = _retrieve(query)
    if not docs:
        return "NO_RESULTS: The knowledge base returned nothing for this query."

    blocks = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source: {src})\n{doc.page_content}")
    logger.info("Knowledge base returned %d chunk(s)", len(docs))
    return "\n\n".join(blocks)


@tool("search_documents")
def search_documents(query: str) -> str:
    """Search the user's uploaded documents (the private knowledge base).

    Use this for questions that are likely answered by the user's own files,
    handbooks, papers, or notes. Returns the most relevant passages with their
    source file names so you can cite them.
    """
    return _search_knowledge_base(query)
