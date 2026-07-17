"""Central configuration for AgentRAG.

Everything is overridable via environment variables (prefix ``AGENTRAG_``) or a
``.env`` file, so the same code runs locally, in Docker and in CI. Defaults
target a fully local, free stack: Ollama for the LLM, HuggingFace for
embeddings, ChromaDB for vectors, and DuckDuckGo for web search (no API key).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (local via Ollama) -------------------------------------------
    ollama_base_url: str = Field(default="http://localhost:11434")
    # qwen2.5 is a strong local tool-calling model — reliable at the multi-step
    # tool routing this agent needs. Override with any Ollama model that
    # supports tool calling (e.g. llama3.1, mistral-nemo).
    llm_model: str = Field(default="qwen2.5:3b")
    llm_temperature: float = Field(default=0.0)

    # --- Embeddings (local HuggingFace) -----------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Vector store (local ChromaDB) ------------------------------------
    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    collection_name: str = Field(default="agentrag")

    # --- Chunking / retrieval ---------------------------------------------
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=150)
    top_k: int = Field(default=4)

    # --- Agent ------------------------------------------------------------
    # Hard cap on reasoning steps so a confused agent can never loop forever.
    max_agent_steps: int = Field(default=6)
    # How many web results the search tool returns per call.
    web_results: int = Field(default=4)

    # --- Storage ----------------------------------------------------------
    upload_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")

    def ensure_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
