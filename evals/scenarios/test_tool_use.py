"""
evals/scenarios/test_tool_use.py — Eval: tool use accuracy
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from evals.eval_base import BaseEval, EvalResult
from tool_policy import ToolPolicyEngine, ToolPolicyViolation


class ToolUseAccuracyEval(BaseEval):
    name = "tool_use_accuracy"
    description = "Agents call only the tools permitted by their policy cards."
    threshold = 0.90

    def run(self, llm, policy_engine, tracer) -> EvalResult:
        checks = []
        details_lines = []

        # Test 1: researcher is allowed web_search
        try:
            policy_engine.check("researcher", "web_search", args={"query": "test"})
            checks.append(True)
            details_lines.append("  ✓ researcher allowed web_search")
        except ToolPolicyViolation:
            checks.append(False)
            details_lines.append("  ✗ researcher wrongly denied web_search")

        # Test 2: orchestrator is denied web_search
        try:
            policy_engine.check("orchestrator", "web_search", args={})
            checks.append(False)
            details_lines.append("  ✗ orchestrator wrongly allowed web_search")
        except ToolPolicyViolation:
            checks.append(True)
            details_lines.append("  ✓ orchestrator correctly denied web_search")

        # Test 3: coder allowed code_executor
        try:
            policy_engine.check("coder", "code_executor", args={"code": "print(1)"})
            checks.append(True)
            details_lines.append("  ✓ coder allowed code_executor")
        except ToolPolicyViolation:
            checks.append(False)
            details_lines.append("  ✗ coder wrongly denied code_executor")

        # Test 4: researcher denied code_executor
        try:
            policy_engine.check("researcher", "code_executor", args={})
            checks.append(False)
            details_lines.append("  ✗ researcher wrongly allowed code_executor")
        except ToolPolicyViolation:
            checks.append(True)
            details_lines.append("  ✓ researcher correctly denied code_executor")

        # Test 5: rate limit enforcement
        policy_engine.reset_counts()
        try:
            for _ in range(11):  # researcher limit is 10
                policy_engine.check("researcher", "web_search", args={"query": "x"})
            checks.append(False)
            details_lines.append("  ✗ rate limit not enforced for researcher")
        except ToolPolicyViolation as e:
            if "Rate limit" in str(e):
                checks.append(True)
                details_lines.append("  ✓ rate limit correctly enforced")
            else:
                checks.append(False)
                details_lines.append(f"  ✗ unexpected violation: {e}")
        finally:
            policy_engine.reset_counts()

        details = f"Tool policy checks: {sum(checks)}/{len(checks)}\n" + "\n".join(details_lines)
        return self._make_result(checks, details)
