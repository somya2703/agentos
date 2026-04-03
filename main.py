#!/usr/bin/env python3
"""
main.py — AgentOS entry point

Usage:
    python main.py --task "Research LangGraph vs AutoGen and write a comparison."
    python main.py --task "..." --backend ollama
    python main.py --task "..." --verbose
"""

import argparse
import logging
import sys
import json
from pathlib import Path

from llm_backend import LLMBackend, LLMConfig
from tool_policy import ToolPolicyEngine
from trace_logger import TraceLogger
from agents.orchestrator import OrchestratorAgent
from agents.researcher import ResearcherAgent
from agents.coder import CoderAgent
from agents.verifier import VerifierAgent
from agents.critic import CriticAgent


def build_system(llm: LLMBackend, policy: ToolPolicyEngine, tracer: TraceLogger) -> OrchestratorAgent:
    """Wire up all agents and return the orchestrator."""
    shared = dict(llm=llm, policy=policy, tracer=tracer)
    sub_agents = {
        "researcher": ResearcherAgent(**shared),
        "coder":      CoderAgent(**shared),
        "verifier":   VerifierAgent(**shared),
        "critic":     CriticAgent(**shared),
    }
    return OrchestratorAgent(**shared, sub_agents=sub_agents)


def main():
    parser = argparse.ArgumentParser(description="AgentOS — Cloud-agnostic multi-agent system")
    parser.add_argument("--task", required=True, help="The task for the agent system to complete")
    parser.add_argument("--backend", choices=["auto", "claude", "ollama", "openai"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", help="Write final output to this file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'═'*60}")
    print(f"  AgentOS")
    print(f"{'═'*60}")
    print(f"  Task: {args.task[:80]}{'...' if len(args.task) > 80 else ''}")

    # Initialise infrastructure
    try:
        llm = LLMBackend(LLMConfig(backend=args.backend, temperature=0.2))
        print(f"  Backend: {llm.backend_name}")
    except RuntimeError as e:
        print(f"\n  ✗ LLM backend error: {e}")
        sys.exit(1)

    policy = ToolPolicyEngine("agent_cards")
    tracer = TraceLogger(task=args.task)
    print(f"  Run ID: {tracer.run_id}")
    print(f"{'═'*60}\n")

    # Build and run
    orchestrator = build_system(llm, policy, tracer)
    result = orchestrator.run(args.task)

    # Finish trace
    trace_path = tracer.finish(
        final_output=result.final_output,
        status=result.status,
    )

    # Print results
    print(f"\n{'═'*60}")
    print("  RESULT")
    print(f"{'═'*60}")
    print(result.final_output)

    print(f"\n{'─'*60}")
    print(f"  Status:        {result.status}")
    print(f"  Steps:         {result.steps_completed}")
    print(f"  Failure modes: {result.failure_modes or 'none'}")
    print(f"  Trace saved:   {trace_path}")
    print(f"{'─'*60}\n")

    # Optionally write output to file
    if args.output:
        Path(args.output).write_text(result.final_output, encoding="utf-8")
        print(f"  Output written to: {args.output}\n")

    sys.exit(0 if result.status == "success" else 1)


if __name__ == "__main__":
    main()
