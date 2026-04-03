"""
agents/orchestrator.py — Orchestrator agent for AgentOS

Uses LangGraph StateGraph to create an explicit, traceable agent pipeline:

    [plan] → [execute] → (loop or done) → [verify] → [critique] → [synthesise] → END

Each node is a pure function: receives AgentState, returns a partial update.
LangGraph merges updates automatically. The execution loop runs until all
plan steps are consumed, then routes to verify → critique → synthesise.
"""

import json
import logging
import operator
import re
from dataclasses import dataclass, field
from typing import TypedDict, Annotated, Optional, Callable

from langgraph.graph import StateGraph, END

from agents.base_agent import BaseAgent
from llm_backend import LLMBackend
from tool_policy import ToolPolicyEngine
from trace_logger import TraceLogger

logger = logging.getLogger(__name__)

MAX_PLAN_STEPS  = 4
MAX_LOOP_GUARD  = 20


# ─── LangGraph state ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    task:              str
    plan:              list[dict]
    step_results:      Annotated[list, operator.add]   # append-only across loop
    current_step_idx:  int
    final_output:      Optional[str]
    status:            str     # planning | executing | done | failed
    error:             Optional[str]
    iteration_guard:   int


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorResult:
    final_output:    str
    plan:            list[dict]
    steps_completed: int
    failure_modes:   list[str] = field(default_factory=list)
    status:          str = "success"


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class OrchestratorAgent(BaseAgent):
    """
    LangGraph-based orchestrator.

    Graph:
        plan ──► execute ──► [route] ──► verify ──► critique ──► synthesise ──► END
                    ▲            │
                    └────────────┘  (while steps remain)
    """

    agent_id    = "orchestrator"
    description = (
        "You decompose complex tasks into clear subtasks, delegate each to the correct "
        "specialist agent (researcher, coder, verifier, critic), and aggregate their outputs. "
        "You do NOT attempt tasks that belong to sub-agents. Always plan before acting."
    )

    def __init__(
        self,
        llm:        LLMBackend,
        policy:     ToolPolicyEngine,
        tracer:     TraceLogger,
        sub_agents: dict = None,
    ):
        super().__init__(llm, policy, tracer)
        self.sub_agents = sub_agents or {}
        self._graph     = self._build_graph()

    # ── Public ────────────────────────────────────────────────────────────

    def run(self, task: str, **kwargs) -> OrchestratorResult:
        logger.info(f"Orchestrator starting: {task[:80]}")
        self.tracer.set_backend(self.llm.backend_name, "")

        initial: AgentState = {
            "task":             task,
            "plan":             [],
            "step_results":     [],
            "current_step_idx": 0,
            "final_output":     None,
            "status":           "planning",
            "error":            None,
            "iteration_guard":  0,
        }

        final = self._graph.invoke(initial)

        return OrchestratorResult(
            final_output    = final.get("final_output") or "",
            plan            = final.get("plan", []),
            steps_completed = len(final.get("step_results", [])),
            failure_modes   = self.tracer.trace.failure_modes_detected,
            status          = final.get("status", "unknown"),
        )

    # ── Graph builder ─────────────────────────────────────────────────────

    def _build_graph(self) -> Callable:
        g = StateGraph(AgentState)

        g.add_node("plan",       self._node_plan)
        g.add_node("execute",    self._node_execute)
        g.add_node("verify",     self._node_verify)
        g.add_node("critique",   self._node_critique)
        g.add_node("synthesise", self._node_synthesise)

        g.set_entry_point("plan")
        g.add_edge("plan",      "execute")
        g.add_edge("verify",    "critique")
        g.add_edge("critique",  "synthesise")
        g.add_edge("synthesise", END)

        g.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {
                "execute":    "execute",
                "verify":     "verify",
                "synthesise": "synthesise",
            },
        )

        return g.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────

    def _node_plan(self, state: AgentState) -> dict:
        task      = state["task"]
        available = list(self.sub_agents.keys())

        prompt = (
            f"Task: {task}\n\n"
            f"Available agents: {available}\n\n"
            "Return a JSON array of plan steps. Rules:\n"
            "  - Each step has: 'step' (int), 'agent' (must be from available agents), 'subtask' (string)\n"
            f"  - Maximum {MAX_PLAN_STEPS} steps total\n"
            "  - Use ONLY agents from the available list above\n"
            "  - Do NOT repeat the same subtask twice\n"
            "  - Keep the plan simple and direct — 2 to 4 steps is usually enough\n"
            "  - researcher: for research and information gathering\n"
            "  - coder: for writing and running Python code\n"
            "  - verifier: for checking facts and outputs\n"
            "  - critic: for identifying problems in the overall approach\n"
            "Return ONLY a valid JSON array, no explanation, no markdown."
        )
        plan = _parse_json_list(self.think_fresh(prompt), fallback=[
            {"step": 1, "agent": available[0] if available else "researcher", "subtask": task}
        ])

        valid = set(self.sub_agents.keys())
        clean = []
        for s in plan[:MAX_PLAN_STEPS]:
            if s.get("agent") not in valid:
                self.tracer.record_failure_mode("planning_hallucination")
                s["agent"] = available[0]
            clean.append(s)
        if len(plan) > MAX_PLAN_STEPS:
            self.tracer.record_failure_mode("over_planning")

        self.log_action("plan", task, clean)
        logger.info(f"Plan: {len(clean)} steps — {[s['agent'] for s in clean]}")
        return {"plan": clean, "status": "executing", "current_step_idx": 0}

    def _node_execute(self, state: AgentState) -> dict:
        guard = state["iteration_guard"]
        if guard >= MAX_LOOP_GUARD:
            self.tracer.record_failure_mode("max_iterations_exceeded")
            return {"status": "failed", "error": "Max iterations exceeded",
                    "iteration_guard": guard + 1}

        plan = state["plan"]
        idx  = state["current_step_idx"]

        if idx >= len(plan):
            return {"status": "done", "iteration_guard": guard + 1}

        step     = plan[idx]
        agent_id = step["agent"]
        subtask  = step["subtask"]

        # Delegation-loop guard
        seen = {r.get("subtask", "") for r in state["step_results"]}
        if subtask in seen:
            self.tracer.record_failure_mode("delegation_loop")
            logger.warning(f"Delegation loop at step {idx+1}, skipping.")
            return {
                "step_results":     [{"step": idx+1, "agent": agent_id,
                                      "subtask": subtask,
                                      "result": "[SKIPPED: delegation loop]"}],
                "current_step_idx": idx + 1,
                "iteration_guard":  guard + 1,
            }

        context = _format_context(state["step_results"])
        logger.info(f"Step {idx+1}/{len(plan)}: {agent_id} → {subtask[:60]}")
        result  = self._delegate(agent_id, subtask, context)
        self.log_action("delegate", {"agent": agent_id, "subtask": subtask},
                        result, metadata={"step": idx + 1})

        return {
            "step_results":     [{"step": idx+1, "agent": agent_id,
                                  "subtask": subtask, "result": result}],
            "current_step_idx": idx + 1,
            "iteration_guard":  guard + 1,
        }

    def _node_verify(self, state: AgentState) -> dict:
        if "verifier" not in self.sub_agents:
            return {}
        result = self.sub_agents["verifier"].run(
            {"task": state["task"], "results": state["step_results"]}
        )
        return {"step_results": [{"step": "verify", "agent": "verifier", "result": result}]}

    def _node_critique(self, state: AgentState) -> dict:
        if "critic" not in self.sub_agents:
            return {}
        result = self.sub_agents["critic"].run(
            {"task": state["task"], "results": state["step_results"]}
        )
        return {"step_results": [{"step": "critique", "agent": "critic", "result": result}]}

    def _node_synthesise(self, state: AgentState) -> dict:
        task    = state["task"]
        results = state["step_results"]
        main    = [r for r in results if r.get("agent") not in ("verifier", "critic")]
        steps_text = "\n\n".join(
            f"[{r['agent']}] Step {r['step']}:\n{r['result']}" for r in main
        )
        prompt = (
            f"Original task: {task}\n\nAgent results:\n{steps_text}\n\n"
            "Write a final, well-structured, comprehensive response to the original task."
        )
        final = self.think_fresh(prompt)
        self.log_action("synthesise", results, final)
        return {"final_output": final, "status": "success"}

    # ── Router ────────────────────────────────────────────────────────────

    def _route_after_execute(self, state: AgentState) -> str:
        if state.get("status") == "failed":
            return "synthesise"
        if state["current_step_idx"] < len(state["plan"]):
            return "execute"
        return "verify" if "verifier" in self.sub_agents else "synthesise"

    # ── Helpers ───────────────────────────────────────────────────────────

    def _delegate(self, agent_id: str, subtask: str, context: str) -> str:
        if agent_id not in self.sub_agents:
            self.tracer.record_failure_mode("unknown_agent_delegation")
            agent_id = list(self.sub_agents.keys())[0]
        try:
            return str(self.sub_agents[agent_id].run(subtask, context=context))
        except Exception as e:
            self.tracer.record_failure_mode("delegation_error")
            logger.error(f"Agent {agent_id} failed: {e}")
            return f"[ERROR: {agent_id} — {e}]"


# ─── Utilities ────────────────────────────────────────────────────────────────

def _parse_json_list(text: str, fallback: list) -> list:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return fallback


def _format_context(results: list) -> str:
    if not results:
        return ""
    return "\n".join(
        f"[{r.get('agent','?')}]: {str(r.get('result',''))[:400]}"
        for r in results[-3:]
    )
