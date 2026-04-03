"""
tools/web_search.py — Web search via ddgs package

No API key required. Returns real web results.
Install: pip install ddgs
"""

import logging

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    Returns a formatted string of titles + snippets.
    No API key required.
    """
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body  = r.get("body", "")
                href  = r.get("href", "")
                results.append(f"• {title}\n  {body}\n  Source: {href}")

        if not results:
            return f"No results found for: {query}"

        return f"Search results for '{query}':\n\n" + "\n\n".join(results)

    except ImportError:
        return "[web_search unavailable: run 'pip install ddgs']"
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return f"[Web search error: {e} — using internal knowledge only]"