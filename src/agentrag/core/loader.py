"""Document loading and chunking (PDF / TXT / Markdown).

Pure and dependency-light so it is trivial to unit-test without a model or
vector store running.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentrag.config import get_settings
from agentrag.utils.logging import get_logger

logger = get_logger("loader")

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


class UnsupportedFileError(ValueError):
    """Raised for a file type we can't parse."""


def _loader_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8")
    raise UnsupportedFileError(
        f"Unsupported file type: {suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def load_documents(path: str | Path) -> list[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    docs = _loader_for(path).load()
    # Normalise "source" to the bare filename for clean citations.
    for doc in docs:
        doc.metadata["source"] = path.name
    logger.info("Loaded %d section(s) from %s", len(docs), path.name)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunk(s)", len(chunks))
    return chunks


def load_and_split(path: str | Path) -> list[Document]:
    return split_documents(load_documents(path))
