"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    steps: int


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    tools: list[str]
