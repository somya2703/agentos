"""
tests/test_e2e.py — End-to-end integration tests for AgentOS
Run with: pytest tests/test_e2e.py -v
No API key required — uses MockLLMBackend.
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.mock_llm import MockLLMBackend
from tool_policy import ToolPolicyEngine
from trace_logger import TraceLogger
from agents.orchestrator import OrchestratorAgent
from agents.researcher import ResearcherAgent
from agents.coder import CoderAgent
from agents.verifier import VerifierAgent
from agents.critic import CriticAgent


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def llm():
    return MockLLMBackend()


@pytest.fixture
def policy():
    return ToolPolicyEngine("agent_cards")


@pytest.fixture
def tracer():
    return TraceLogger(task="e2e_test", output_dir="/tmp/agentos_e2e_traces")


@pytest.fixture
def all_agents(llm, policy, tracer):
    shared = dict(llm=llm, policy=policy, tracer=tracer)
    return {
        "researcher": ResearcherAgent(**shared),
        "coder":      CoderAgent(**shared),
        "verifier":   VerifierAgent(**shared),
        "critic":     CriticAgent(**shared),
    }


@pytest.fixture
def orchestrator(llm, policy, tracer, all_agents):
    return OrchestratorAgent(llm=llm, policy=policy, tracer=tracer, sub_agents=all_agents)


# ─── Graph structure tests ────────────────────────────────────────────────────

class TestGraphStructure:
    def test_graph_has_all_nodes(self, orchestrator):
        nodes = list(orchestrator._graph.get_graph().nodes.keys())
        for expected in ["plan", "execute", "verify", "critique", "synthesise"]:
            assert expected in nodes, f"Missing node: {expected}"

    def test_graph_has_correct_edges(self, orchestrator):
        edges = {(e.source, e.target) for e in orchestrator._graph.get_graph().edges}
        assert ("plan",       "execute")    in edges
        assert ("verify",     "critique")   in edges
        assert ("critique",   "synthesise") in edges
        assert ("synthesise", "__end__")    in edges

    def test_execute_has_conditional_edges(self, orchestrator):
        # execute should have 3 possible outgoing targets
        edge_targets = {
            e.target for e in orchestrator._graph.get_graph().edges
            if e.source == "execute"
        }
        assert len(edge_targets) >= 2, "execute node should have conditional routing"


# ─── Full pipeline test ───────────────────────────────────────────────────────

class TestFullPipeline:
    def test_run_produces_output(self, orchestrator):
        result = orchestrator.run(
            "Research LangGraph vs AutoGen and write a comparison table."
        )
        assert result.final_output, "Final output should not be empty"
        assert len(result.final_output) > 50

    def test_plan_is_populated(self, orchestrator):
        result = orchestrator.run(
            "Research LangGraph vs AutoGen and write a comparison table."
        )
        assert len(result.plan) >= 1
        assert len(result.plan) <= 6

    def test_steps_completed(self, orchestrator):
        result = orchestrator.run(
            "Research LangGraph vs AutoGen and write a comparison table."
        )
        assert result.steps_completed >= 1

    def test_status_is_success(self, orchestrator):
        result = orchestrator.run(
            "Research LangGraph vs AutoGen and write a comparison table."
        )
        assert result.status == "success"

    def test_plan_uses_valid_agents(self, orchestrator):
        result = orchestrator.run(
            "Research LangGraph vs AutoGen and write a comparison table."
        )
        valid = {"researcher", "coder", "verifier", "critic"}
        for step in result.plan:
            assert step["agent"] in valid, f"Invalid agent: {step['agent']}"


# ─── Tool policy tests ────────────────────────────────────────────────────────

class TestToolPolicy:
    def test_researcher_can_search(self, policy):
        from tool_policy import ToolPolicyViolation
        # Should not raise
        assert policy.check("researcher", "web_search", args={"query": "test"})

    def test_orchestrator_cannot_search(self, policy):
        from tool_policy import ToolPolicyViolation
        with pytest.raises(ToolPolicyViolation):
            policy.check("orchestrator", "web_search", args={})

    def test_coder_can_execute(self, policy):
        assert policy.check("coder", "code_executor", args={"code": "print(1)"})

    def test_critic_cannot_execute(self, policy):
        from tool_policy import ToolPolicyViolation
        with pytest.raises(ToolPolicyViolation):
            policy.check("critic", "code_executor", args={})

    def test_rate_limit_enforced(self, policy):
        from tool_policy import ToolPolicyViolation
        policy.reset_counts()
        try:
            for _ in range(11):  # limit is 10
                policy.check("researcher", "web_search", args={"query": "x"})
            pytest.fail("Rate limit should have been triggered")
        except ToolPolicyViolation as e:
            assert "Rate limit" in str(e)
        finally:
            policy.reset_counts()


# ─── Trace logger tests ───────────────────────────────────────────────────────

class TestTraceLogger:
    def test_trace_saves_valid_json(self, tracer):
        tracer.log_step("orchestrator", "plan", "task", [{"step": 1}])
        tracer.log_tool_call("researcher", "web_search", {"query": "x"}, "result", True, duration_ms=10)
        tracer.record_failure_mode("test_mode")
        path = tracer.finish(final_output="done", status="success")

        with open(path) as f:
            data = json.load(f)

        assert data["status"] == "success"
        assert len(data["steps"]) >= 1
        assert "test_mode" in data["failure_modes_detected"]
        assert data["final_output"] == "done"

    def test_tool_call_attached_to_step(self, tracer):
        tracer.log_step("researcher", "research", "task", "output")
        tracer.log_tool_call("researcher", "web_search", {"query": "x"}, "result", True, duration_ms=5)
        path = tracer.finish(status="success")

        with open(path) as f:
            data = json.load(f)

        tool_calls = []
        for step in data["steps"]:
            tool_calls.extend(step.get("tool_calls", []))
        assert len(tool_calls) >= 1
        assert tool_calls[0]["tool_name"] == "web_search"


# ─── Code executor sandbox tests ─────────────────────────────────────────────

class TestCodeSandbox:
    def test_safe_code_executes(self):
        from tools.code_executor import execute_python
        r = execute_python("print(2 + 2)")
        assert r["success"]
        assert "4" in r["output"]

    def test_os_import_blocked(self):
        from tools.code_executor import execute_python
        r = execute_python("import os; os.system('ls')")
        assert not r["success"]
        assert "os" in r["error"]

    def test_subprocess_blocked(self):
        from tools.code_executor import execute_python
        r = execute_python("import subprocess")
        assert not r["success"]

    def test_complex_safe_code(self):
        from tools.code_executor import execute_python
        code = """
data = [3, 1, 4, 1, 5, 9, 2, 6]
result = sorted(data)
print(result)
"""
        r = execute_python(code)
        assert r["success"]
        assert "[1, 1, 2, 3, 4, 5, 6, 9]" in r["output"]
