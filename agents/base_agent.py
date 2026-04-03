"""
agents/base_agent.py — Base class for all AgentOS agents

All agents inherit from BaseAgent which provides:
  - Structured LLM calls with system prompt injection
  - Tool policy enforcement (pre-call check)
  - Automatic step logging to TraceLogger
  - Failure mode detection hooks
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any

from llm_backend import LLMBackend, LLMMessage
from tool_policy import ToolPolicyEngine, ToolPolicyViolation
from trace_logger import TraceLogger

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents. Subclasses implement `run()`.

    Each agent has:
      - agent_id: unique identifier matching its agent card
      - system_prompt: injected into every LLM call
      - llm: shared LLMBackend instance
      - policy: ToolPolicyEngine for permission checks
      - tracer: TraceLogger for full observability
    """

    agent_id: str = "base"
    description: str = "Base agent"

    def __init__(
        self,
        llm: LLMBackend,
        policy: ToolPolicyEngine,
        tracer: TraceLogger,
    ):
        self.llm = llm
        self.policy = policy
        self.tracer = tracer
        self._conversation: list[LLMMessage] = []

    @property
    def system_prompt(self) -> str:
        return (
            f"You are the {self.agent_id} agent in a multi-agent system called AgentOS.\n"
            f"Role: {self.description}\n"
            "Be concise, structured, and precise. "
            "Always respond in the format requested by the orchestrator."
        )

    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Any:
        """Execute this agent's primary task. Must be implemented by subclasses."""
        ...

    # ─── LLM helpers ──────────────────────────────────────────────────────

    def think(self, prompt: str, fresh: bool = False) -> str:
        """
        Send a prompt to the LLM and return the response text.
        If fresh=True, starts a new conversation (clears history).
        """
        if fresh:
            self._conversation = []

        self._conversation.append(LLMMessage("user", prompt))
        self.tracer.begin_step()

        response = self.llm.invoke(self._conversation, system=self.system_prompt)

        self._conversation.append(LLMMessage("assistant", response.content))
        self.tracer.log_step(
            agent_id=self.agent_id,
            action="llm_call",
            input=prompt,
            output=response.content,
        )
        return response.content

    def think_fresh(self, prompt: str) -> str:
        """Convenience: think with a clean conversation history."""
        return self.think(prompt, fresh=True)

    # ─── Tool call wrapper ────────────────────────────────────────────────

    def use_tool(self, tool_name: str, tool_fn, **kwargs) -> Any:
        """
        Call a tool after checking policy. Logs the call and result.

        Example:
            result = self.use_tool("web_search", search_fn, query="LangGraph vs AutoGen")
        """
        try:
            self.policy.check(self.agent_id, tool_name, args=kwargs)
        except ToolPolicyViolation as e:
            self.tracer.record_policy_violation(self.agent_id, tool_name, str(e))
            self.tracer.record_failure_mode("tool_policy_bypass_attempt")
            raise

        start = time.monotonic()
        try:
            result = tool_fn(**kwargs)
            duration_ms = (time.monotonic() - start) * 1000
            self.tracer.log_tool_call(
                agent_id=self.agent_id,
                tool_name=tool_name,
                args=kwargs,
                result=result,
                success=True,
                duration_ms=duration_ms,
            )
            return result
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            self.tracer.log_tool_call(
                agent_id=self.agent_id,
                tool_name=tool_name,
                args=kwargs,
                result=None,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise

    # ─── Failure mode helpers ─────────────────────────────────────────────

    def detect_and_record(self, condition: bool, failure_mode: str):
        """Record a failure mode if condition is True."""
        if condition:
            self.tracer.record_failure_mode(failure_mode)

    def log_action(self, action: str, input_data: Any, output_data: Any, metadata: dict = None):
        """Manually log a step (for custom actions not covered by think/use_tool)."""
        self.tracer.log_step(
            agent_id=self.agent_id,
            action=action,
            input=input_data,
            output=output_data,
            metadata=metadata or {},
        )
