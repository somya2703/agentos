"""
agents/verifier.py — Verifier sub-agent
"""
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class VerifierAgent(BaseAgent):
    agent_id = "verifier"
    description = (
        "You are a critical fact-checker. You review the outputs of other agents "
        "and identify: factual errors, logical inconsistencies, missing information, "
        "and unsupported claims. You are skeptical by default and never approve "
        "outputs without genuine scrutiny."
    )

    def run(self, input_data, **kwargs) -> str:
        # Accept both plain string (delegated mid-plan) and structured dict
        if isinstance(input_data, str):
            input_data = {"task": input_data, "results": []}
        task = input_data.get("task", "")
        results = input_data.get("results", [])
        results_text = "\n\n".join(
            f"[{r.get('agent', '?')}] Step {r.get('step', '?')}:\n{r.get('result', '')}"
            for r in results
        )
        prompt = (
            f"Original task: {task}\n\n"
            f"Agent outputs to verify:\n{results_text}\n\n"
            "Carefully verify the above outputs. For each issue found, state:\n"
            "- What the claim is\n"
            "- Why it may be incorrect or unsupported\n"
            "- Your confidence in this assessment (low/medium/high)\n\n"
            "If the outputs appear correct, state that clearly with your reasoning. "
            "Be specific — vague approval is not useful."
        )
        result = self.think_fresh(prompt)

        # Detect verification blindness (very short approval with no reasoning)
        if len(result.strip()) < 80 and any(w in result.lower() for w in ["correct", "accurate", "good", "fine"]):
            self.tracer.record_failure_mode("verification_blindness")

        self.log_action("verify", input_data, result)
        return result
