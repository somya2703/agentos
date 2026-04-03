# Agentic Failure Mode Analysis

> Built from observations while developing and testing AgentOS.
> These are real failure patterns encountered in multi-agent LLM pipelines.

---

## 1. Planning Hallucination

**What it is**: The orchestrator's plan includes steps, subtasks, or agent types that were not requested and do not exist.

**Example**: Asked to "research and summarise topic X", the orchestrator produces a 7-step plan that includes a "database_agent" and a "reporting_agent" — neither of which exists.

**Why it happens**: LLMs generalise from training data about agentic systems and invent plausible-sounding agent names.

**Countermeasure**:
- Explicitly enumerate available agents in the planning prompt
- Post-process: filter plan steps where `agent` is not in the known agent registry
- Log `planning_hallucination` in the trace when this occurs

---

## 2. Delegation Loop

**What it is**: A sub-agent re-delegates a task back to the orchestrator or to another sub-agent, creating a cycle.

**Example**: Orchestrator → Researcher → "I need to run some code for this" → re-delegates to Coder → Coder asks Orchestrator for clarification → loop.

**Why it happens**: Sub-agents are not given hard boundaries about when to delegate vs. handle locally.

**Countermeasure**:
- Sub-agents must never re-delegate — enforce this in the system prompt
- Track `seen_subtasks` in the orchestrator; skip duplicates
- Set `MAX_ITERATIONS` on the orchestration loop

---

## 3. Tool Policy Bypass

**What it is**: An agent attempts to call a tool it is not permitted to use, either intentionally (via prompt injection) or through mis-delegation.

**Example**: The orchestrator, prompted with a malicious task, tries to call `code_executor` directly instead of routing through the coder agent.

**Why it happens**: Agent system prompts don't always enforce tool boundaries; LLMs infer "the path of least resistance."

**Countermeasure**:
- Runtime policy check via `ToolPolicyEngine.check()` before every tool call
- `ToolPolicyViolation` exceptions are non-negotiable — they cannot be caught by the agent
- All violations are logged to the trace with `record_policy_violation()`

---

## 4. Verification Blindness

**What it is**: The verifier agent approves outputs without genuine scrutiny — returning very short "looks correct" responses to avoid the work of verification.

**Example**: Given a 500-word report with a factual error, the verifier responds "The output appears accurate and complete." in 12 words.

**Why it happens**: LLMs trained to be helpful tend toward agreement. Without pressure to be critical, verification becomes rubber-stamping.

**Countermeasure**:
- Verifier system prompt: "You are skeptical by default. Vague approval is not useful."
- Detect verification blindness heuristically: flag responses < 80 chars that contain approval words
- Require structured output from verifier (claim-by-claim assessment)

---

## 5. Critic Silence

**What it is**: The critic agent finds no failure modes even when obvious problems exist — especially when the execution trace is long and the critic runs last.

**Example**: After a 6-step run where the verifier was blind and the planner hallucinated an agent, the critic returns `"detected_failures": []`.

**Why it happens**: Long context causes LLMs to miss earlier issues; being "nice" is the path of least effort.

**Countermeasure**:
- Provide the explicit list of known failure modes to check against in the critic prompt
- Auto-detect critic silence: if `detected_failures == []` after a complex run (>3 steps), flag it
- Structure the critic output as JSON — harder to "softly approve" in JSON than in prose

---

## 6. Context Loss

**What it is**: Later agents in the chain have lost the context established by earlier agents, producing outputs that contradict or ignore prior work.

**Example**: Researcher gathers facts about Framework A. Coder writes code for Framework B without acknowledging the research.

**Why it happens**: Each LLM call is stateless; context is passed manually between agents.

**Countermeasure**:
- Pass a `context` string to every sub-agent call (last N step results)
- Cap context at ~3 prior steps to avoid token bloat
- Include the original task in every agent prompt

---

## 7. Over-Planning

**What it is**: The orchestrator creates an excessive number of steps for a simple task, introducing unnecessary latency and potential for cascading errors.

**Example**: "Summarise this paragraph" becomes a 9-step plan involving all 4 agents.

**Why it happens**: LLMs trained on complex agentic examples tend to over-decompose simple tasks.

**Countermeasure**:
- Hard cap `MAX_PLAN_STEPS = 6` in the orchestrator
- Include "Maximum N steps" in the planning prompt
- Log `over_planning` when plan exceeds the cap

---

## 8. Hallucinated Facts

**What it is**: Agents produce outputs that contain specific false claims stated with confidence — especially about version numbers, release dates, and API details.

**Example**: "LangGraph 3.2 introduced native support for X in Q1 2024." — the version number and feature are fabricated.

**Why it happens**: LLMs fill in plausible-sounding specifics when uncertain.

**Countermeasure**:
- The verifier is specifically prompted to flag specific, unverifiable claims
- The critic checks for `hallucinated_facts` in its failure mode list
- Web search tool can cross-check specific claims (available to verifier)

---

## Summary Table

| Failure Mode | Severity | Detection | Mitigation |
|---|---|---|---|
| planning_hallucination | Medium | Post-plan agent registry check | Enumerate agents in prompt + filter |
| delegation_loop | High | `seen_subtasks` set | Skip duplicates, `MAX_ITERATIONS` |
| tool_policy_bypass | Critical | `ToolPolicyViolation` exception | Runtime policy enforcement (non-bypassable) |
| verification_blindness | Medium | Heuristic (length + approval words) | Structured output, skeptical system prompt |
| critic_silence | Medium | No failures after complex run | Force JSON output, explicit checklist |
| context_loss | Low | Manual review | Pass `context` string to every agent |
| over_planning | Low | Plan length > MAX | Hard cap + prompt constraint |
| hallucinated_facts | High | Verifier + critic | Web search cross-check, structured verification |
