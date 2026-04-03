"""
evals/scenarios/test_hallucination.py — Eval: hallucination detection
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from evals.eval_base import BaseEval, EvalResult
from llm_backend import LLMMessage


HALLUCINATION_CASES = [
    {
        "claim": "LangGraph was created by Anthropic and released in 2019.",
        "is_hallucination": True,
        "note": "LangGraph was created by LangChain Inc., not Anthropic",
    },
    {
        "claim": "Python is a high-level, interpreted programming language.",
        "is_hallucination": False,
        "note": "Factually correct",
    },
    {
        "claim": "AutoGen is a multi-agent framework developed by Microsoft Research.",
        "is_hallucination": False,
        "note": "Factually correct",
    },
    {
        "claim": "CrewAI supports exactly 3 agents per workflow, no more.",
        "is_hallucination": True,
        "note": "No such hard limit exists in CrewAI",
    },
    {
        "claim": "GPT-4 was released by OpenAI.",
        "is_hallucination": False,
        "note": "Factually correct",
    },
]

VERIFIER_SYSTEM = (
    "You are a strict fact-checker for AI systems. "
    "You identify claims that are factually incorrect, unverifiable, or fabricated. "
    "Be concise and precise."
)


class HallucinationDetectionEval(BaseEval):
    name = "hallucination_detection"
    description = "Verifier agent correctly flags hallucinated vs. factual claims."
    threshold = 0.75

    def run(self, llm, policy_engine, tracer) -> EvalResult:
        checks = []
        details_lines = []

        for case in HALLUCINATION_CASES:
            prompt = (
                f"Review this claim: \"{case['claim']}\"\n\n"
                "Is this claim factually correct? "
                "Reply with exactly one word: CORRECT or HALLUCINATION."
            )
            response = llm.invoke(
                [LLMMessage("user", prompt)],
                system=VERIFIER_SYSTEM,
            ).content.strip().upper()

            detected_hallucination = "HALLUCINATION" in response
            expected_hallucination = case["is_hallucination"]
            correct = detected_hallucination == expected_hallucination
            checks.append(correct)

            icon = "✓" if correct else "✗"
            expected = "HALLUCINATION" if expected_hallucination else "CORRECT"
            got = "HALLUCINATION" if detected_hallucination else "CORRECT"
            details_lines.append(
                f"  {icon} Claim: '{case['claim'][:55]}' | "
                f"Expected: {expected} | Got: {got}"
            )

        details = (
            f"Hallucination detection: {sum(checks)}/{len(checks)} correct\n"
            + "\n".join(details_lines)
        )
        return self._make_result(checks, details)
