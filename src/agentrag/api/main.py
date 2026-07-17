"""FastAPI application exposing the agentic RAG pipeline.

GET  /health      -> liveness + configured model and tools
POST /ingest      -> upload a document and index it
POST /ask         -> ask the agent a question; returns answer + tools used
DELETE /documents -> clear the knowledge base
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from agentrag.agent.graph import run_agent
from agentrag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestResponse,
)
from agentrag.config import get_settings
from agentrag.core.ingest import ingest_file
from agentrag.core.loader import SUPPORTED_SUFFIXES, UnsupportedFileError
from agentrag.core.vectorstore import clear_collection
from agentrag.tools import ALL_TOOLS
from agentrag.utils.logging import get_logger

logger = get_logger("api")

app = FastAPI(
    title="AgentRAG API",
    description="Agentic Retrieval-Augmented Generation: an LLM agent that "
    "decides when to search your documents, the web, or compute.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        llm_model=settings.llm_model,
        tools=[t.name for t in ALL_TOOLS],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    settings = get_settings()
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        allowed = sorted(SUPPORTED_SUFFIXES)
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type {suffix!r}. Allowed: {allowed}",
        )

    dest = settings.upload_dir / filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        chunks = ingest_file(dest)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    return IngestResponse(filename=filename, chunks_indexed=chunks)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = run_agent(req.question)
    return AskResponse(
        answer=result.answer,
        tools_used=result.tool_calls,
        steps=result.steps,
    )


@app.delete("/documents", status_code=204)
def reset() -> None:
    clear_collection()
