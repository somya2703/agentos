"""
agents/critic.py — Critic sub-agent (agentic failure mode detector)
"""
import json
import re
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

KNOWN_FAILURE_MODES = [
    "planning_hallucination",
    "delegation_loop",
    "tool_policy_bypass",
    "verification_blindness",
    "critic_silence",
    "context_loss",
    "over_planning",
    "under_delegation",
    "hallucinated_facts",
    "incomplete_task",
]


class CriticAgent(BaseAgent):
    agent_id = "critic"
    description = (
        "You are an agentic systems expert who identifies failure modes in multi-agent pipelines. "
        "You analyse the full execution trace and flag any signs of: hallucination, delegation problems, "
        "planning errors, security issues, or quality failures. "
        "You are methodical and output structured assessments."
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
            f"Task: {task}\n\n"
            f"Full agent execution results:\n{results_text}\n\n"
            f"Known agentic failure modes to check for: {KNOWN_FAILURE_MODES}\n\n"
            "Analyse the above execution for failure modes. "
            "Return a JSON object with these keys:\n"
            "  detected_failures: list of failure mode names found\n"
            "  severity: 'none' | 'low' | 'medium' | 'high'\n"
            "  details: string explanation of each failure found\n"
            "  recommendation: string with suggested fix\n"
            "Return ONLY the JSON, no preamble."
        )
        response = self.think_fresh(prompt)
        parsed = _parse_critique(response)

        # Auto-record detected failures into the trace
        for mode in parsed.get("detected_failures", []):
            if mode in KNOWN_FAILURE_MODES:
                self.tracer.record_failure_mode(mode)

        # Detect critic silence (critic found nothing despite long, complex execution)
        if not parsed.get("detected_failures") and len(results) > 3:
            self.tracer.record_failure_mode("critic_silence")

        result_text = json.dumps(parsed, indent=2)
        self.log_action("critique", input_data, parsed)
        return result_text


def _parse_critique(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {
        "detected_failures": [],
        "severity": "unknown",
        "details": text,
        "recommendation": "Manual review required.",
    }
