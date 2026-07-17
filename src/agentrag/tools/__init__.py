"""Agent tools. ``ALL_TOOLS`` is the toolbelt handed to the agent."""
from __future__ import annotations

from agentrag.tools.calculator import calculator
from agentrag.tools.retrieval import search_documents
from agentrag.tools.websearch import web_search

ALL_TOOLS = [search_documents, web_search, calculator]

__all__ = ["ALL_TOOLS", "search_documents", "web_search", "calculator"]
