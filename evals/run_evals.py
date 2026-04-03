#!/usr/bin/env python3
"""
evals/run_evals.py — AgentOS eval CLI dashboard

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --format json
    python evals/run_evals.py --scenario planning_quality
    python evals/run_evals.py --fast   # skip LLM-heavy evals

Runs all eval scenarios and prints a pass/fail report.
Saves results to traces/eval_run_<timestamp>.json
"""

import sys
import os
import json
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_backend import LLMBackend, LLMConfig
from tool_policy import ToolPolicyEngine
from trace_logger import TraceLogger

# Import all eval scenarios
from evals.scenarios.test_planning import PlanningQualityEval
from evals.scenarios.test_delegation import DelegationAccuracyEval
from evals.scenarios.test_tool_use import ToolUseAccuracyEval
from evals.scenarios.test_hallucination import HallucinationDetectionEval
from evals.scenarios.test_robustness import RobustnessEval
from evals.scenarios.test_security import SecurityEval
from evals.scenarios.test_trace import TraceCompletenessEval
from evals.scenarios.test_graceful_failure import GracefulFailureEval

logging.basicConfig(level=logging.WARNING)  # quiet during evals

ALL_EVALS = [
    PlanningQualityEval(),
    DelegationAccuracyEval(),
    ToolUseAccuracyEval(),
    HallucinationDetectionEval(),
    RobustnessEval(),
    SecurityEval(),
    TraceCompletenessEval(),
    GracefulFailureEval(),
]

# Evals that don't need LLM calls (safe for --fast / offline)
OFFLINE_EVALS = {"tool_use_accuracy", "security_tool_policy", "trace_completeness"}


def run_all(scenarios, llm, policy, tracer, fmt="human"):
    results = []
    for eval_scenario in scenarios:
        if fmt == "human":
            print(f"  Running {eval_scenario.name}...", end="\r", flush=True)
        result = eval_scenario.evaluate(llm, policy, tracer)
        results.append(result)
        if fmt == "human":
            print(f"  {result}")
    return results


def print_report(results, total_ms):
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_score = sum(r.score for r in results) / total if total else 0

    width = 56
    print()
    print("═" * width)
    print("  AgentOS Eval Suite")
    print("═" * width)
    for r in results:
        icon = "✓" if r.passed else "✗"
        bar = _score_bar(r.score)
        print(f"  {icon} {r.scenario_name:<32} {bar}  {r.score:.2f}")
    print("═" * width)
    print(f"  Score: {passed}/{total} passed  |  Avg: {avg_score:.2f}  |  {total_ms:.0f}ms")
    print("═" * width)

    if passed < total:
        print("\n  Failed scenario details:")
        for r in results:
            if not r.passed:
                print(f"\n  ── {r.scenario_name} (score={r.score:.2f}, threshold={r.threshold})")
                for line in r.details.split("\n"):
                    print(f"     {line}")
    print()


def _score_bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def save_results(results, total_ms):
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_ms": total_ms,
        "passed": sum(1 for r in results if r.passed),
        "total": len(results),
        "avg_score": sum(r.score for r in results) / len(results) if results else 0,
        "results": [
            {
                "scenario": r.scenario_name,
                "passed": r.passed,
                "score": r.score,
                "threshold": r.threshold,
                "duration_ms": r.duration_ms,
                "details": r.details,
                "failure_modes": r.failure_modes_found,
            }
            for r in results
        ],
    }
    Path("traces").mkdir(exist_ok=True)
    fname = f"traces/eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)
    return fname, output


def main():
    parser = argparse.ArgumentParser(description="AgentOS Eval Suite")
    parser.add_argument("--format", choices=["human", "json"], default="human")
    parser.add_argument("--scenario", help="Run a single scenario by name")
    parser.add_argument("--fast", action="store_true", help="Offline-only evals (no LLM calls)")
    args = parser.parse_args()

    # Initialise shared infrastructure
    try:
        llm = LLMBackend(LLMConfig(temperature=0.0))
        backend_name = llm.backend_name
    except RuntimeError as e:
        print(f"\n⚠ No LLM backend available: {e}")
        print("  Running offline evals only.\n")
        llm = None
        backend_name = "none"

    policy = ToolPolicyEngine("agent_cards")
    tracer = TraceLogger(task="eval_suite", output_dir="traces")

    # Select scenarios
    scenarios = ALL_EVALS
    if args.scenario:
        scenarios = [s for s in ALL_EVALS if s.name == args.scenario]
        if not scenarios:
            print(f"Unknown scenario '{args.scenario}'. Available: {[s.name for s in ALL_EVALS]}")
            sys.exit(1)
    if args.fast or llm is None:
        scenarios = [s for s in scenarios if s.name in OFFLINE_EVALS]

    if args.format == "human":
        print(f"\n  AgentOS Eval Suite  |  backend: {backend_name}  |  {len(scenarios)} scenarios\n")

    start = time.monotonic()
    results = run_all(scenarios, llm, policy, tracer, fmt=args.format)
    total_ms = (time.monotonic() - start) * 1000

    fname, output = save_results(results, total_ms)

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        print_report(results, total_ms)
        print(f"  Results saved to: {fname}\n")

    # Exit code: 0 if all passed, 1 if any failed
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
