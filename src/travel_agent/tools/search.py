"""Web search, used for destination research.

Tavily needs `TAVILY_API_KEY`. Without it the agent still runs — it can
structure trips, cost them, and use the availability tools — so a missing key
degrades research rather than failing startup.
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def build_search_tools(max_results: int = 5) -> list[BaseTool]:
    """Return the web search tool, or an empty list when unconfigured."""
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning(
            "TAVILY_API_KEY is not set — web search is disabled and the "
            "destination-researcher subagent will have no way to look things up."
        )
        return []

    from langchain_tavily import TavilySearch

    return [
        TavilySearch(
            max_results=max_results,
            topic="general",
            search_depth="advanced",
            include_answer=True,
        )
    ]
