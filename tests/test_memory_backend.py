"""Tests for conversation-memory checkpointer backend selection."""
from __future__ import annotations

import pytest

from lumen import config
from lumen.agent import graph


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.setenv("LUMEN_MEMORY_DB", str(tmp_path / "m.sqlite"))
    graph._CHECKPOINTER = None
    yield
    graph._CHECKPOINTER = None
    config.get_settings.cache_clear()


def test_default_backend_is_in_memory(monkeypatch):
    monkeypatch.delenv("LUMEN_MEMORY_BACKEND", raising=False)
    cp = graph._get_checkpointer()
    # langgraph exposes this as MemorySaver (an alias of InMemorySaver).
    assert cp.__class__.__name__ in {"MemorySaver", "InMemorySaver"}


def test_sqlite_backend_when_requested(monkeypatch):
    monkeypatch.setenv("LUMEN_MEMORY_BACKEND", "sqlite")
    cp = graph._get_checkpointer()
    assert cp.__class__.__name__ == "SqliteSaver"


def test_checkpointer_is_cached(monkeypatch):
    monkeypatch.delenv("LUMEN_MEMORY_BACKEND", raising=False)
    first = graph._get_checkpointer()
    second = graph._get_checkpointer()
    assert first is second
