"""
agents/researcher.py — Researcher sub-agent
"""
import logging
from agents.base_agent import BaseAgent
from tools.web_search import web_search

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    agent_id = "researcher"
    description = (
        "You research topics thoroughly using web search and your knowledge. "
        "You summarise findings in clear, structured prose with key facts highlighted. "
        "You always distinguish between what you found vs. what you inferred."
    )

    def run(self, task: str, context: str = "", **kwargs) -> str:
        ctx = f"\nPrior context:\n{context}" if context else ""

        # Try web search (policy-gated)
        search_results = ""
        try:
            search_results = self.use_tool("web_search", web_search, query=task[:200])
            search_context = f"\n\nWeb search results:\n{search_results}"
        except Exception as e:
            logger.warning(f"Web search unavailable: {e}")
            search_context = "\n(Web search unavailable — using internal knowledge only)"

        prompt = (
            f"Research task: {task}{ctx}{search_context}\n\n"
            "Provide a thorough, well-structured research summary. "
            "Include key facts, comparisons where relevant, and any important caveats. "
            "Format with clear sections."
        )
        result = self.think_fresh(prompt)
        self.log_action("research", task, result)
        return result
