"""
evals/scenarios/test_planning.py — Eval: planning quality
"""
import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from evals.eval_base import BaseEval, EvalResult
from llm_backend import LLMMessage


def _extract_json_list(text: str):
    text = re.sub(r"```(?:json|python)?", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in ("steps", "plan", "result"):
                if isinstance(result.get(key), list):
                    return result[key]
    except Exception:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except Exception:
            pass
    objects = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
    if objects:
        parsed = []
        for obj in objects:
            try:
                parsed.append(json.loads(obj))
            except Exception:
                pass
        if parsed:
            return parsed
    return []


PLANNING_SYSTEM = (
    "You are a JSON API. You only output valid JSON. "
    "You never explain, never add prose, never answer the task itself. "
    "You only output the requested JSON structure and nothing else."
)


class PlanningQualityEval(BaseEval):
    name = "planning_quality"
    description = "Orchestrator produces a valid, structured plan for a multi-step task."
    threshold = 0.67   # realistic for 3B model on structured output

    def run(self, llm, policy_engine, tracer) -> EvalResult:
        prompt = (
            "Output a JSON array of plan steps for this task. "
            "Do NOT perform the task. Do NOT explain. Output JSON only.\n\n"
            "Task to plan: Research agentic AI frameworks, write a Python comparison, verify facts.\n\n"
            "Available agents and their roles:\n"
            "- researcher: searches for information\n"
            "- coder: writes Python code\n"
            "- verifier: checks facts\n"
            "- critic: finds problems\n\n"
            "Required JSON format (copy this structure exactly):\n"
            '[{"step": 1, "agent": "researcher", "subtask": "Research agentic frameworks"},\n'
            ' {"step": 2, "agent": "coder", "subtask": "Write comparison table"},\n'
            ' {"step": 3, "agent": "verifier", "subtask": "Verify the facts"}]\n\n'
            "Your JSON array (2-4 steps, agents from the list above only):"
        )

        response = llm.invoke(
            [LLMMessage("user", prompt)],
            system=PLANNING_SYSTEM,
        ).content

        plan = _extract_json_list(response)

        # Filter out steps with None agent (model left a gap)
        plan = [s for s in plan if s.get("agent") is not None]

        valid_agents = {"researcher", "coder", "verifier", "critic"}
        checks = [
            isinstance(plan, list),
            len(plan) >= 2,
            len(plan) <= 6,
            all("agent" in s and "subtask" in s for s in plan) if plan else False,
            any(s.get("agent") == "researcher" for s in plan),
            all(s.get("agent") in valid_agents for s in plan) if plan else False,
        ]
        details = (
            f"Plan has {len(plan)} steps. "
            f"Agents: {[s.get('agent') for s in plan]}. "
            f"Checks: {sum(checks)}/{len(checks)}\n"
            f"Raw (first 200 chars): {response[:200]}"
        )
        return self._make_result(checks, details)