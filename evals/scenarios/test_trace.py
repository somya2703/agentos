"""
evals/scenarios/test_trace.py — Eval: trace completeness & observability
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from evals.eval_base import BaseEval, EvalResult
from trace_logger import TraceLogger


class TraceCompletenessEval(BaseEval):
    name = "trace_completeness"
    description = "Trace logger produces complete, valid JSON traces for all runs."
    threshold = 1.0

    def run(self, llm, policy_engine, tracer) -> EvalResult:
        checks = []
        details_lines = []

        # Create a fresh trace and simulate a run
        t = TraceLogger(task="eval: trace completeness test", output_dir="/tmp/agentos_eval_traces")
        t.set_backend("test", "test-model")
        t.log_step("orchestrator", "plan", input="test task", output=[{"step": 1, "agent": "researcher"}])
        t.log_step("researcher", "research", input="test task", output="some research result")
        t.log_tool_call("researcher", "web_search", args={"query": "test"}, result="result", success=True, duration_ms=42.0)
        t.record_failure_mode("test_failure_mode")
        t.record_policy_violation("orchestrator", "web_search", "not allowed")
        output_path = t.finish(final_output="test output", status="success")

        # Load and validate
        with open(output_path) as f:
            data = json.load(f)

        required_top_keys = ["run_id", "task", "started_at", "finished_at", "steps", "final_output", "status"]
        for key in required_top_keys:
            present = key in data
            checks.append(present)
            icon = "✓" if present else "✗"
            details_lines.append(f"  {icon} Top-level key: '{key}'")

        # Steps populated
        has_steps = len(data.get("steps", [])) >= 2
        checks.append(has_steps)
        details_lines.append(f"  {'✓' if has_steps else '✗'} Steps recorded: {len(data.get('steps', []))}")

        # Tool call recorded
        has_tool_call = any(
            len(s.get("tool_calls", [])) > 0 for s in data.get("steps", [])
        )
        checks.append(has_tool_call)
        details_lines.append(f"  {'✓' if has_tool_call else '✗'} Tool call in steps")

        # Failure mode recorded
        has_failure = len(data.get("failure_modes_detected", [])) > 0
        checks.append(has_failure)
        details_lines.append(f"  {'✓' if has_failure else '✗'} Failure modes recorded")

        # Policy violation recorded
        has_violation = len(data.get("policy_violations", [])) > 0
        checks.append(has_violation)
        details_lines.append(f"  {'✓' if has_violation else '✗'} Policy violations recorded")

        # Status is 'success'
        correct_status = data.get("status") == "success"
        checks.append(correct_status)
        details_lines.append(f"  {'✓' if correct_status else '✗'} Status = 'success'")

        details = f"Trace checks: {sum(checks)}/{len(checks)} passed\n" + "\n".join(details_lines)
        return self._make_result(checks, details)
