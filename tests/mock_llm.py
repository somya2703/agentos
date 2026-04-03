"""
tests/mock_llm.py — Deterministic mock LLM for offline testing

Returns realistic-looking but hardcoded responses so the full agent
pipeline can be tested without any API key or local Ollama instance.
"""

import json
from llm_backend import LLMMessage, LLMResponse


PLAN_RESPONSE = json.dumps([
    {"step": 1, "agent": "researcher", "subtask": "Research the pros and cons of LangGraph vs AutoGen"},
    {"step": 2, "agent": "coder",      "subtask": "Write a Python comparison table for the two frameworks"},
    {"step": 3, "agent": "verifier",   "subtask": "Verify the facts in the research and code output"},
])

RESEARCH_RESPONSE = (
    "## LangGraph vs AutoGen — Research Summary\n\n"
    "**LangGraph** (LangChain Inc.) is a graph-based orchestration framework. "
    "It uses explicit state machines, making agent flows fully traceable and testable. "
    "Best suited for enterprise workflows requiring auditability.\n\n"
    "**AutoGen** (Microsoft Research) uses a conversation-based multi-agent model "
    "where agents communicate through structured message passing. "
    "Well-suited for rapid prototyping and research use cases.\n\n"
    "Key difference: LangGraph gives explicit control; AutoGen favours flexibility."
)

CODE_RESPONSE = (
    "Here is a Python comparison table:\n\n"
    "```python\n"
    "frameworks = [\n"
    "    {'name': 'LangGraph', 'paradigm': 'State machine', 'traceability': 'High', 'enterprise': 'Yes'},\n"
    "    {'name': 'AutoGen',   'paradigm': 'Conversation',  'traceability': 'Medium','enterprise': 'Partial'},\n"
    "]\n"
    "for f in frameworks:\n"
    "    print(f['name'], '|', f['paradigm'], '|', f['traceability'])\n"
    "```\n"
    "**Output:**\n```\nLangGraph | State machine | High\nAutoGen   | Conversation  | Medium\n```"
)

VERIFY_RESPONSE = (
    "Verification complete.\n\n"
    "- Claim 'LangGraph created by LangChain Inc.' — **CORRECT** (high confidence)\n"
    "- Claim 'AutoGen by Microsoft Research' — **CORRECT** (high confidence)\n"
    "- Code output matches described behavior — **CORRECT**\n"
    "- No unsupported claims detected.\n"
    "Overall assessment: outputs are factually accurate."
)

CRITIQUE_RESPONSE = json.dumps({
    "detected_failures": [],
    "severity": "none",
    "details": "No failure modes detected. Planning was appropriate (3 steps for a 2-framework comparison). Delegation was correct. Verifier performed genuine assessment.",
    "recommendation": "None required.",
})

SYNTHESISE_RESPONSE = (
    "# LangGraph vs AutoGen — Comparison Report\n\n"
    "## Summary\n"
    "Both LangGraph and AutoGen are mature open-source multi-agent frameworks, "
    "each optimised for different contexts.\n\n"
    "## Framework Comparison\n"
    "| Feature | LangGraph | AutoGen |\n"
    "|---|---|---|\n"
    "| Paradigm | Explicit state machine | Conversation-based |\n"
    "| Traceability | High | Medium |\n"
    "| Enterprise ready | Yes | Partial |\n"
    "| Creator | LangChain Inc. | Microsoft Research |\n\n"
    "## Recommendation\n"
    "For enterprise use with auditability requirements: **LangGraph**.\n"
    "For rapid research prototyping: **AutoGen**."
)


class MockLLMBackend:
    """
    Deterministic mock LLM backend.
    Returns preset responses based on keywords in the prompt.
    No API key, no network, fully offline.
    """

    backend_name = "mock"

    def invoke(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        prompt = messages[-1].content.lower() if messages else ""
        response = self._pick_response(prompt)
        return LLMResponse(response, backend_used="mock", model="mock-model")

    def invoke_text(self, prompt: str, system: str = "") -> str:
        return self.invoke([LLMMessage("user", prompt)], system=system).content

    def _pick_response(self, prompt: str) -> str:
        if "json array" in prompt or "execution plan" in prompt or "plan" in prompt[:80]:
            return PLAN_RESPONSE
        if "research" in prompt and ("langraph" in prompt or "autogen" in prompt or "framework" in prompt):
            return RESEARCH_RESPONSE
        if "comparison table" in prompt or "python" in prompt and "table" in prompt:
            return CODE_RESPONSE
        if "verify" in prompt or "fact" in prompt and "check" in prompt:
            return VERIFY_RESPONSE
        if "failure mode" in prompt or "critique" in prompt or "detected_failures" in prompt:
            return CRITIQUE_RESPONSE
        if "synthesise" in prompt or "final" in prompt and "structured" in prompt:
            return SYNTHESISE_RESPONSE
        if "hallucination" in prompt:
            if "langangraph was created by anthropic" in prompt or "2019" in prompt:
                return "HALLUCINATION"
            return "CORRECT"
        if "graceful" in prompt or "cannot" in prompt.lower() or "database" in prompt:
            return "I cannot access external databases or live systems. I don't have real-time data access."
        # Generic fallback
        return "Task completed successfully with the requested information."
