"""
tool_policy.py — Per-agent tool permission engine for AgentOS

Each agent has a policy defined in its agent card (.yaml).
This module enforces those policies at runtime before any tool is called.

Policy levels:
  - ALLOW  : agent may use this tool
  - DENY   : agent may never use this tool (raises ToolPolicyViolation)
  - AUDIT  : allowed, but every call is logged with full args for review
"""

import yaml
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class PolicyLevel(str, Enum):
    ALLOW = "allow"
    DENY  = "deny"
    AUDIT = "audit"


class ToolPolicyViolation(Exception):
    """Raised when an agent attempts to use a tool it is not permitted to use."""
    def __init__(self, agent_id: str, tool_name: str, reason: str = ""):
        self.agent_id = agent_id
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(
            f"Policy violation: agent '{agent_id}' is not allowed to use tool '{tool_name}'. {reason}"
        )


@dataclass
class ToolPolicy:
    tool_name: str
    level: PolicyLevel
    reason: str = ""
    rate_limit_per_run: Optional[int] = None   # max calls per agent per run


@dataclass
class AgentPolicy:
    agent_id: str
    trust_level: str           # "low" | "medium" | "high"
    tools: dict[str, ToolPolicy] = field(default_factory=dict)
    default_policy: PolicyLevel = PolicyLevel.DENY   # deny unknown tools by default


class ToolPolicyEngine:
    """
    Loads agent policies from YAML agent cards and enforces them at runtime.

    Usage:
        engine = ToolPolicyEngine("agent_cards/")
        engine.check("researcher", "web_search", args={"query": "..."})
    """

    def __init__(self, agent_cards_dir: str = "agent_cards"):
        self.cards_dir = Path(agent_cards_dir)
        self._policies: dict[str, AgentPolicy] = {}
        self._call_counts: dict[str, dict[str, int]] = {}   # {agent_id: {tool: count}}
        self._audit_log: list[dict] = []
        self._load_all()

    def _load_all(self):
        for yaml_file in self.cards_dir.glob("*.yaml"):
            try:
                self._load_card(yaml_file)
            except Exception as e:
                logger.warning(f"Failed to load agent card {yaml_file}: {e}")

    def _load_card(self, path: Path):
        with open(path) as f:
            card = yaml.safe_load(f)

        agent_id = card["agent_id"]
        trust_level = card.get("trust_level", "medium")
        default_pol = PolicyLevel(card.get("default_tool_policy", "deny"))

        tool_policies = {}
        for tool_entry in card.get("tools", []):
            tool_name = tool_entry["name"]
            level = PolicyLevel(tool_entry.get("policy", "allow"))
            tool_policies[tool_name] = ToolPolicy(
                tool_name=tool_name,
                level=level,
                reason=tool_entry.get("reason", ""),
                rate_limit_per_run=tool_entry.get("rate_limit_per_run"),
            )

        self._policies[agent_id] = AgentPolicy(
            agent_id=agent_id,
            trust_level=trust_level,
            tools=tool_policies,
            default_policy=default_pol,
        )
        self._call_counts[agent_id] = {}
        logger.info(f"Loaded policy for agent '{agent_id}' ({trust_level} trust, {len(tool_policies)} tools)")

    def check(self, agent_id: str, tool_name: str, args: dict = None) -> bool:
        """
        Check if agent_id is allowed to call tool_name with given args.
        Returns True if allowed.
        Raises ToolPolicyViolation if denied.
        """
        args = args or {}

        if agent_id not in self._policies:
            raise ToolPolicyViolation(agent_id, tool_name, "No policy found for this agent.")

        policy = self._policies[agent_id]
        tool_pol = policy.tools.get(tool_name)
        effective_level = tool_pol.level if tool_pol else policy.default_policy

        # Rate limit check
        if tool_pol and tool_pol.rate_limit_per_run is not None:
            count = self._call_counts[agent_id].get(tool_name, 0)
            if count >= tool_pol.rate_limit_per_run:
                raise ToolPolicyViolation(
                    agent_id, tool_name,
                    f"Rate limit exceeded: {count}/{tool_pol.rate_limit_per_run} calls this run."
                )

        if effective_level == PolicyLevel.DENY:
            reason = tool_pol.reason if tool_pol else "Tool not in agent's allowed list."
            raise ToolPolicyViolation(agent_id, tool_name, reason)

        # Track call count
        self._call_counts[agent_id][tool_name] = (
            self._call_counts[agent_id].get(tool_name, 0) + 1
        )

        if effective_level == PolicyLevel.AUDIT:
            entry = {
                "agent_id": agent_id,
                "tool": tool_name,
                "args": args,
                "call_count": self._call_counts[agent_id][tool_name],
            }
            self._audit_log.append(entry)
            logger.warning(f"AUDIT: {agent_id} called {tool_name} | args={args}")

        logger.debug(f"Policy ALLOW: {agent_id} → {tool_name}")
        return True

    def reset_counts(self):
        """Call at the start of each new run."""
        for agent_id in self._call_counts:
            self._call_counts[agent_id] = {}

    def get_audit_log(self) -> list[dict]:
        return list(self._audit_log)

    def allowed_tools(self, agent_id: str) -> list[str]:
        """Return list of tools this agent is explicitly allowed to use."""
        if agent_id not in self._policies:
            return []
        return [
            name for name, pol in self._policies[agent_id].tools.items()
            if pol.level != PolicyLevel.DENY
        ]

    def summary(self) -> dict:
        return {
            agent_id: {
                "trust_level": pol.trust_level,
                "allowed_tools": self.allowed_tools(agent_id),
                "call_counts": self._call_counts.get(agent_id, {}),
            }
            for agent_id, pol in self._policies.items()
        }
