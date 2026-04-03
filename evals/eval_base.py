"""
evals/eval_base.py — Base class for all AgentOS eval scenarios

Each eval scenario:
  - Has a name, description, and pass threshold
  - Implements run() which returns an EvalResult
  - Can use a mock LLM to avoid API costs during testing
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    scenario_name: str
    passed: bool
    score: float                    # 0.0 – 1.0
    threshold: float
    details: str
    duration_ms: float = 0.0
    failure_modes_found: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        icon = "✓" if self.passed else "✗"
        return (
            f"{icon} {self.scenario_name:<35} "
            f"{'PASS' if self.passed else 'FAIL'}  "
            f"({self.score:.2f})"
        )


class BaseEval(ABC):
    """
    Base class for eval scenarios.

    Subclasses implement `run(llm, policy_engine)` and return an EvalResult.
    """

    name: str = "unnamed_eval"
    description: str = ""
    threshold: float = 0.75         # minimum score to pass

    def evaluate(self, llm, policy_engine, tracer) -> EvalResult:
        start = time.monotonic()
        try:
            result = self.run(llm, policy_engine, tracer)
        except Exception as e:
            logger.error(f"Eval {self.name} crashed: {e}", exc_info=True)
            result = EvalResult(
                scenario_name=self.name,
                passed=False,
                score=0.0,
                threshold=self.threshold,
                details=f"Eval crashed: {e}",
            )
        result.duration_ms = round((time.monotonic() - start) * 1000, 2)
        return result

    @abstractmethod
    def run(self, llm, policy_engine, tracer) -> EvalResult:
        ...

    def _score(self, checks: list[bool]) -> float:
        """Helper: compute score as fraction of True checks."""
        if not checks:
            return 0.0
        return sum(checks) / len(checks)

    def _make_result(self, checks: list[bool], details: str, **metadata) -> EvalResult:
        score = self._score(checks)
        return EvalResult(
            scenario_name=self.name,
            passed=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            details=details,
            metadata=metadata,
        )
