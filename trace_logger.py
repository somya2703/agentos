"""
trace_logger.py — Structured JSON trace logger for AgentOS

Every agent run produces a complete, human-readable + machine-parseable
trace: task → plan → delegations → tool calls → outputs → final result.

This is the observability backbone of the ADLC layer.
Each trace is saved to traces/{run_id}.json
"""

import json
import uuid
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    tool_name: str
    agent_id: str
    args: dict
    result: Any
    success: bool
    duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None


@dataclass
class AgentStep:
    step_id: str
    agent_id: str
    action: str                  # "plan" | "delegate" | "tool_call" | "respond" | "verify" | "critique"
    input: Any
    output: Any
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentTrace:
    run_id: str
    task: str
    started_at: str
    finished_at: Optional[str] = None
    backend_used: str = ""
    model: str = ""
    status: str = "running"       # "running" | "success" | "failed" | "partial"
    steps: list[AgentStep] = field(default_factory=list)
    final_output: Optional[str] = None
    failure_modes_detected: list[str] = field(default_factory=list)
    policy_violations: list[dict] = field(default_factory=list)
    total_duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class TraceLogger:
    """
    Records all agent activity and saves to a JSON file on completion.

    Usage:
        logger = TraceLogger(run_id="run_001", task="Research X")
        logger.log_step("orchestrator", "plan", input=task, output=plan)
        logger.log_tool_call("researcher", "web_search", args={...}, result={...})
        logger.finish(final_output="...", status="success")
    """

    def __init__(
        self,
        task: str,
        run_id: Optional[str] = None,
        output_dir: str = "traces",
    ):
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._start_time = time.monotonic()
        self._step_start: Optional[float] = None

        self.trace = AgentTrace(
            run_id=self.run_id,
            task=task,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"TraceLogger started: {self.run_id}")

    # ─── Step logging ─────────────────────────────────────────────────────

    def begin_step(self):
        """Call at the start of a timed step."""
        self._step_start = time.monotonic()

    def log_step(
        self,
        agent_id: str,
        action: str,
        input: Any,
        output: Any,
        metadata: dict = None,
    ) -> AgentStep:
        duration = 0.0
        if self._step_start is not None:
            duration = (time.monotonic() - self._step_start) * 1000
            self._step_start = None

        step = AgentStep(
            step_id=f"step_{len(self.trace.steps) + 1:03d}",
            agent_id=agent_id,
            action=action,
            input=_safe_serialize(input),
            output=_safe_serialize(output),
            duration_ms=round(duration, 2),
            metadata=metadata or {},
        )
        self.trace.steps.append(step)
        logger.debug(f"[{self.run_id}] {agent_id}:{action} ({duration:.0f}ms)")
        return step

    # ─── Tool call logging ─────────────────────────────────────────────────

    def log_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        args: dict,
        result: Any,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> ToolCall:
        tc = ToolCall(
            tool_name=tool_name,
            agent_id=agent_id,
            args=_safe_serialize(args),
            result=_safe_serialize(result),
            success=success,
            duration_ms=round(duration_ms, 2),
            error=error,
        )
        # Attach to the most recent step for this agent, or create a bare step
        for step in reversed(self.trace.steps):
            if step.agent_id == agent_id:
                step.tool_calls.append(tc)
                break
        else:
            self.log_step(agent_id, "tool_call", input=args, output=result)
            self.trace.steps[-1].tool_calls.append(tc)

        status = "OK" if success else f"ERROR: {error}"
        logger.debug(f"[{self.run_id}] tool:{tool_name} by {agent_id} → {status}")
        return tc

    # ─── Failure mode & policy violation recording ─────────────────────────

    def record_failure_mode(self, mode: str):
        if mode not in self.trace.failure_modes_detected:
            self.trace.failure_modes_detected.append(mode)
            logger.warning(f"[{self.run_id}] Failure mode detected: {mode}")

    def record_policy_violation(self, agent_id: str, tool: str, reason: str):
        self.trace.policy_violations.append({
            "agent_id": agent_id,
            "tool": tool,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.warning(f"[{self.run_id}] Policy violation: {agent_id} → {tool}: {reason}")

    # ─── Metadata ─────────────────────────────────────────────────────────

    def set_backend(self, backend_used: str, model: str):
        self.trace.backend_used = backend_used
        self.trace.model = model

    # ─── Finish & save ────────────────────────────────────────────────────

    def finish(
        self,
        final_output: Optional[str] = None,
        status: str = "success",
    ) -> Path:
        self.trace.finished_at = datetime.now(timezone.utc).isoformat()
        self.trace.final_output = final_output
        self.trace.status = status
        self.trace.total_duration_ms = round(
            (time.monotonic() - self._start_time) * 1000, 2
        )

        output_path = self.output_dir / f"{self.run_id}.json"
        with open(output_path, "w") as f:
            json.dump(asdict(self.trace), f, indent=2, default=str)

        logger.info(
            f"Trace saved: {output_path} | "
            f"status={status} | "
            f"steps={len(self.trace.steps)} | "
            f"duration={self.trace.total_duration_ms:.0f}ms"
        )
        return output_path

    def to_dict(self) -> dict:
        return asdict(self.trace)

    def summary(self) -> str:
        steps = len(self.trace.steps)
        tools = sum(len(s.tool_calls) for s in self.trace.steps)
        failures = self.trace.failure_modes_detected
        violations = len(self.trace.policy_violations)
        return (
            f"Run {self.run_id}: {steps} steps, {tools} tool calls, "
            f"{len(failures)} failure modes, {violations} policy violations"
        )


# ─── Utility ──────────────────────────────────────────────────────────────

def _safe_serialize(obj: Any) -> Any:
    """Make any object JSON-safe without crashing."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(i) for i in obj]
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"
