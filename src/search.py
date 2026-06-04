"""Web-search grounding backend.

Pluggable, but ships with a Tavily implementation (a search API designed for
LLM grounding). Set TAVILY_API_KEY to enable. With no key configured,
``web_search`` returns an empty list and the debate proceeds ungrounded.
"""
from __future__ import annotations

import logging
import os

import httpx

from .state import Source

log = logging.getLogger("ensemble.search")

TAVILY_URL = "https://api.tavily.com/search"


def search_enabled() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY"))


async def web_search(query: str, max_results: int = 6, timeout: float = 30.0) -> list[Source]:
    """Return grounding sources for a query, or [] if search is unavailable."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        log.debug("web_search: no TAVILY_API_KEY set; skipping grounding")
        return []
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                TAVILY_URL,
                json={
                    "api_key": key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # grounding is best-effort; never fail the debate
        log.warning("web_search failed (%s); proceeding ungrounded", e)
        return []

    sources: list[Source] = []
    for r in data.get("results", [])[:max_results]:
        sources.append(
            Source(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=(r.get("content", "") or "")[:500],
            )
        )
    return sources
