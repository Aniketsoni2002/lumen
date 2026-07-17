"""Agent tools. ``ALL_TOOLS`` is the toolbelt handed to the agent."""
from __future__ import annotations

from lumen.tools.calculator import calculator
from lumen.tools.retrieval import search_documents
from lumen.tools.websearch import web_search

ALL_TOOLS = [search_documents, web_search, calculator]

__all__ = ["ALL_TOOLS", "search_documents", "web_search", "calculator"]
