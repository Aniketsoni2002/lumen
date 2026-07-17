"""Web-search tool (DuckDuckGo — free, no API key).

Gives the agent a fallback for questions the private knowledge base can't
answer (current events, general facts). Network access is wrapped so a failure
degrades gracefully into a message the agent can reason about, instead of
crashing the whole run.
"""
from __future__ import annotations

from langchain_core.tools import tool

from agentrag.config import get_settings
from agentrag.utils.logging import get_logger

logger = get_logger("tool.websearch")


def _run_web_search(query: str, max_results: int) -> str:
    """Perform the search. Imported lazily so tests can run offline."""
    try:
        from ddgs import DDGS
    except ImportError:  # pragma: no cover - dependency guard
        return "WEB_SEARCH_UNAVAILABLE: the 'ddgs' package is not installed."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # network / rate-limit / parsing issues
        logger.warning("Web search failed: %s", exc)
        return f"WEB_SEARCH_ERROR: {exc}"

    if not results:
        return "NO_RESULTS: The web search returned nothing for this query."

    blocks = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        blocks.append(f"[{i}] {title}\n{body}\nURL: {href}")
    logger.info("Web search returned %d result(s)", len(results))
    return "\n\n".join(blocks)


@tool("web_search")
def web_search(query: str) -> str:
    """Search the public web for current or general-knowledge information.

    Use this when the question is about recent events, external facts, or
    anything unlikely to be in the user's uploaded documents. Returns titles,
    snippets, and source URLs.
    """
    settings = get_settings()
    return _run_web_search(query, settings.web_results)
