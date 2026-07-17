"""High-level ingestion: file on disk -> chunks -> vector store."""
from __future__ import annotations

from pathlib import Path

from lumen.core.loader import load_and_split
from lumen.core.vectorstore import add_documents
from lumen.utils.logging import get_logger

logger = get_logger("ingest")


def ingest_file(path: str | Path) -> int:
    chunks = load_and_split(path)
    count = add_documents(chunks)
    logger.info("Ingested %s -> %d chunks", Path(path).name, count)
    return count
