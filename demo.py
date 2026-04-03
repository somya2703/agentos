#!/usr/bin/env python3
"""
demo.py — Offline AgentOS demo (no API key required)

Runs the full multi-agent pipeline using the MockLLMBackend,
then prints the LangGraph execution trace and final output.

Usage:
    python demo.py
    python demo.py --task "Your custom task here"
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.mock_llm import MockLLMBackend
from tool_policy import ToolPolicyEngine
from trace_logger import TraceLogger
from agents.orchestrator import OrchestratorAgent
from agents.researcher import ResearcherAgent
from agents.coder import CoderAgent
from agents.verifier import VerifierAgent
from agents.critic import CriticAgent

DEMO_TASK = (
    "Research the top open-source agentic AI frameworks (LangGraph, AutoGen, CrewAI), "
    "write a Python comparison table, and verify the facts."
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEMO_TASK)
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              AgentOS — Offline Demo                         ║")
    print("║         (MockLLM — no API key required)                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nTask: {args.task}\n")

    # Wire up the system
    llm    = MockLLMBackend()
    policy = ToolPolicyEngine("agent_cards")
    tracer = TraceLogger(task=args.task, output_dir="traces")

    shared = dict(llm=llm, policy=policy, tracer=tracer)
    sub_agents = {
        "researcher": ResearcherAgent(**shared),
        "coder":      CoderAgent(**shared),
        "verifier":   VerifierAgent(**shared),
        "critic":     CriticAgent(**shared),
    }
    orchestrator = OrchestratorAgent(**shared, sub_agents=sub_agents)

    print("── LangGraph nodes:", list(orchestrator._graph.get_graph().nodes.keys()))
    print("── Running pipeline...\n")

    result = orchestrator.run(args.task)

    # Print execution trace
    print("─" * 64)
    print("EXECUTION TRACE")
    print("─" * 64)
    steps = tracer.trace.steps
    for i, step in enumerate(steps):
        tc_count = len(step.tool_calls)
        tc_info  = f"  [{tc_count} tool call(s)]" if tc_count else ""
        print(f"  [{i+1:02d}] {step.agent_id:<16} {step.action:<16} {tc_info}")

    print()
    print("─" * 64)
    print("PLAN")
    print("─" * 64)
    for s in result.plan:
        print(f"  Step {s['step']} → {s['agent']:<12} {s['subtask'][:60]}")

    print()
    print("─" * 64)
    print("FINAL OUTPUT")
    print("─" * 64)
    print(result.final_output)

    print()
    print("─" * 64)
    print(f"Status:        {result.status}")
    print(f"Steps done:    {result.steps_completed}")
    print(f"Failure modes: {result.failure_modes or 'none detected'}")

    # Save and show trace path
    trace_path = tracer.finish(final_output=result.final_output, status=result.status)
    print(f"Trace saved:   {trace_path}")
    print()

    # Pretty-print a snippet of the JSON trace
    with open(trace_path) as f:
        data = json.load(f)
    print("── Trace preview (top-level keys):", list(data.keys()))
    print(f"── Total steps in trace: {len(data['steps'])}")
    print()


if __name__ == "__main__":
    main()
