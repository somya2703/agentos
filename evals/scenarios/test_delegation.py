"""
evals/scenarios/test_delegation.py — Eval: correct delegation
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from evals.eval_base import BaseEval, EvalResult
from llm_backend import LLMMessage


DELEGATION_CASES = [
    {
        "subtask": "Search for recent papers on multi-agent systems",
        "expected_agent": "researcher",
        "wrong_agents": ["coder", "critic"],
    },
    {
        "subtask": "Write a Python function that sorts a list of agent scores",
        "expected_agent": "coder",
        "wrong_agents": ["researcher", "verifier"],
    },
    {
        "subtask": "Check whether the summary contains any factual errors",
        "expected_agent": "verifier",
        "wrong_agents": ["coder", "researcher"],
    },
    {
        "subtask": "Identify failure modes in the above execution trace",
        "expected_agent": "critic",
        "wrong_agents": ["researcher", "coder"],
    },
]

AGENT_DESCRIPTIONS = """Agent roles:
- researcher: finds information, searches the web, gathers knowledge
- coder: writes Python code, implements functions and scripts
- verifier: checks facts, validates correctness, identifies errors
- critic: identifies failure modes, evaluates quality, finds systemic problems"""


class DelegationAccuracyEval(BaseEval):
    name = "correct_delegation"
    description = "Orchestrator assigns subtasks to the correct specialist agents."
    threshold = 0.75

    def run(self, llm, policy_engine, tracer) -> EvalResult:
        checks = []
        details_lines = []

        for case in DELEGATION_CASES:
            prompt = (
                f"{AGENT_DESCRIPTIONS}\n\n"
                f"Which single agent should handle this subtask?\n"
                f"Subtask: {case['subtask']}\n\n"
                f"Reply with ONLY one word — the agent name."
            )
            response = llm.invoke([LLMMessage("user", prompt)]).content.strip().lower()

            # Extract just the agent name
            chosen = None
            for agent in ["researcher", "coder", "verifier", "critic"]:
                if agent in response:
                    chosen = agent
                    break
            chosen = chosen or response.split()[0] if response.split() else "unknown"

            correct = chosen == case["expected_agent"]
            checks.append(correct)
            icon = "✓" if correct else "✗"
            details_lines.append(
                f"  {icon} '{case['subtask'][:50]}' → got '{chosen}' (expected '{case['expected_agent']}')"
            )

        details = f"Delegation accuracy: {sum(checks)}/{len(checks)}\n" + "\n".join(details_lines)
        return self._make_result(checks, details)