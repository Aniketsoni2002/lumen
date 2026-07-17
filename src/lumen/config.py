"""Central configuration for Lumen.

Everything is overridable via environment variables (prefix ``LUMEN_``) or a
``.env`` file, so the same code runs locally, in Docker and in CI. Defaults
target a fully local, free stack: Ollama for the LLM, HuggingFace for
embeddings, ChromaDB for vectors, and DuckDuckGo for web search (no API key).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LUMEN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider -----------------------------------------------------
    # "ollama" runs a local model (free, private, needs Ollama installed).
    # "groq"  calls Groq's hosted API (free tier, very fast) — this is what
    # makes cloud deployment possible, since hosts like Streamlit Cloud can't
    # run Ollama. Set LUMEN_LLM_PROVIDER=groq and provide GROQ_API_KEY.
    llm_provider: str = Field(default="ollama")

    # --- LLM (local via Ollama) -------------------------------------------
    ollama_base_url: str = Field(default="http://localhost:11434")
    # qwen2.5:7b is a strong local tool-calling model — reliable at the
    # multi-step tool routing and self-reflection this agent needs, and it
    # stays coherent across multi-turn conversations. The 3b variant also
    # works and is lighter, but is more prone to looping on harder chains.
    # Override with any Ollama model that supports tool calling.
    llm_model: str = Field(default="qwen2.5:7b")
    llm_temperature: float = Field(default=0.0)

    # --- LLM (cloud via Groq) ---------------------------------------------
    # Used only when llm_provider == "groq". The API key is read from the
    # standard GROQ_API_KEY as well as LUMEN_GROQ_API_KEY, so it works with
    # Groq's own conventions and Streamlit Cloud secrets out of the box.
    groq_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GROQ_API_KEY", "LUMEN_GROQ_API_KEY"),
    )
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    # --- Embeddings (local HuggingFace) -----------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Vector store (local ChromaDB) ------------------------------------
    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    collection_name: str = Field(default="lumen")

    # --- Chunking / retrieval ---------------------------------------------
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=150)
    top_k: int = Field(default=4)

    # Hybrid retrieval: fuse dense (semantic) + sparse (BM25 keyword) search.
    # Weights are relative; they need not sum to 1.
    hybrid_retrieval: bool = Field(default=True)
    hybrid_dense_weight: float = Field(default=0.5)
    hybrid_sparse_weight: float = Field(default=0.5)

    # --- Agent ------------------------------------------------------------
    # Hard cap on reasoning steps so a confused agent can never loop forever.
    max_agent_steps: int = Field(default=6)
    # How many web results the search tool returns per call.
    web_results: int = Field(default=4)
    # Self-reflection: grade the answer against gathered evidence; on an
    # UNGROUNDED verdict, give the agent one chance to correct itself.
    enable_reflection: bool = Field(default=True)

    # --- Storage ----------------------------------------------------------
    upload_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")
    # SQLite file backing conversation memory. Persisting to disk means a
    # session's history survives across separate CLI invocations and API
    # restarts, not just within one process.
    memory_db: Path = Field(default=PROJECT_ROOT / "data" / "memory.sqlite")

    def ensure_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
