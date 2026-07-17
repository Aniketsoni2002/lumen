"""ChromaDB vector store + local HuggingFace embeddings."""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from lumen.config import get_settings
from lumen.utils.logging import get_logger

logger = get_logger("vectorstore")


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def add_documents(chunks: list[Document]) -> int:
    if not chunks:
        return 0
    get_vectorstore().add_documents(chunks)
    logger.info("Added %d chunk(s) to vector store", len(chunks))
    return len(chunks)


def clear_collection() -> None:
    get_vectorstore().delete_collection()
    logger.info("Cleared vector store collection")
