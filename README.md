# AgentOS 🤖

> A cloud-agnostic, open-source multi-agent AI system with a built-in Agentic Software Development Lifecycle (ADLC) evaluation layer.

**Sovereignty-first design** — runs 100% locally via Ollama, or with any OpenAI-compatible API. No proprietary vendor lock-in. One command to spin up the entire stack.

```bash
docker compose up --build
```

---

## Table of Contents

- [What is AgentOS?](#what-is-agentos)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repo Structure](#repo-structure)
- [Quick Start](#quick-start)
- [Running Tasks](#running-tasks)
- [Eval Suite](#eval-suite)
- [Model Notes](#model-notes)
- [Design Decisions](#design-decisions)
- [Agentic Failure Modes](#agentic-failure-modes)
- [Roadmap](#roadmap)

---

## What is AgentOS?

AgentOS is a **multi-agent AI system** that decomposes complex tasks and routes each piece to a specialist agent. Instead of asking a single model to do everything, it works like a small team:

| Agent | Role |
|---|---|
| **Orchestrator** | Reads the task, builds a plan, delegates to specialists, synthesises the final output |
| **Researcher** | Searches the web (DuckDuckGo, no API key) and gathers knowledge |
| **Coder** | Writes and executes Python in a sandboxed environment |
| **Verifier** | Fact-checks and validates outputs from other agents |
| **Critic** | Scans the execution trace for agentic failure modes |

On top of the agent pipeline, an **ADLC (Agentic Software Development Lifecycle) layer** provides:

- **Tool policy enforcement** — each agent has a YAML agent card defining which tools it can and cannot use, enforced at runtime
- **Full trace logging** — every plan, delegation, tool call, and result is recorded as structured JSON
- **Failure mode detection** — 10 named failure patterns are auto-detected and logged per run
- **Eval harness** — 8 automated test scenarios with pass/fail scoring

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       AgentOS System                         │
│                                                             │
│  ┌──────────────────────┐   ┌───────────────────────────┐   │
│  │   LLM Backend        │   │   ADLC Layer              │   │
│  │   (Pluggable)        │   │                           │   │
│  │   Claude API  ──┐    │   │  • Tool Policy Engine     │   │
│  │   Ollama/local──►LLM │   │  • Trace Logger (JSON)    │   │
│  │   OpenAI      ──┘    │   │  • Failure Mode Detector  │   │
│  └────────┬─────────────┘   │  • Eval Harness (8 tests) │   │
│           │                 └────────────┬──────────────┘   │
│           ▼                              │                   │
│  ┌─────────────────────────┐             │                   │
│  │   Orchestrator          │◄────────────┘                   │
│  │   (LangGraph StateGraph)│                                 │
│  └──┬──────────────────────┘                                 │
│     │                                                        │
│     ├──► Researcher  — web search + knowledge synthesis      │
│     ├──► Coder       — Python write + sandboxed execution    │
│     ├──► Verifier    — fact-check + validate                 │
│     └──► Critic      — failure mode detection                │
│                                                             │
│  Agent Cards (.yaml): role · tools · trust · rate limits    │
└─────────────────────────────────────────────────────────────┘
```

### LangGraph execution graph

The orchestrator is a compiled `StateGraph` with explicit nodes and conditional edges:

```
[plan] → [execute] → ┐
            ▲        │ loop while steps remain
            └────────┘
                     │ all steps done
                     ▼
               [verify] → [critique] → [synthesise] → END
```

Every node is a pure function. LangGraph merges state updates automatically. The graph is fully inspectable — nodes, edges, and transitions are explicit, not hidden inside a blackbox.

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) | Explicit state machines, fully traceable, enterprise-grade |
| LLM backends | Anthropic Claude / Ollama / OpenAI | Pluggable — swap without changing agent code |
| Local LLM | [Ollama](https://ollama.com) | Runs llama3.2 or mistral locally, zero cloud dependency |
| Web search | [ddgs](https://pypi.org/project/ddgs/) | DuckDuckGo search, no API key required |
| Agent cards | YAML | Human-readable, versionable, diffable permission definitions |
| Containerisation | Docker + Docker Compose | One-command deployment, fully portable |
| Testing | pytest | 19 offline tests, no API key needed |
| CI | GitHub Actions | Runs tests on Python 3.11 + 3.12 on every push |

---

## Repo Structure

```
agentos/
├── .github/
│   └── workflows/
│       └── ci.yml                  ← GitHub Actions CI
│
├── agent_cards/                    ← Per-agent YAML permission files
│   ├── orchestrator.yaml
│   ├── researcher.yaml
│   ├── coder.yaml
│   ├── verifier.yaml
│   └── critic.yaml
│
├── agents/                         ← Agent implementations
│   ├── base_agent.py               ← Shared base: LLM calls, tool gating, logging
│   ├── orchestrator.py             ← LangGraph StateGraph orchestrator
│   ├── researcher.py
│   ├── coder.py
│   ├── verifier.py
│   └── critic.py
│
├── tools/                          ← Tools agents can call (policy-gated)
│   ├── web_search.py               ← DuckDuckGo search, no API key
│   ├── code_executor.py            ← Sandboxed Python executor
│   └── file_writer.py
│
├── evals/                          ← ADLC evaluation layer
│   ├── eval_base.py                ← Base class: score, threshold, result
│   ├── run_evals.py                ← CLI dashboard (human + JSON output)
│   └── scenarios/
│       ├── test_planning.py        ← Does the orchestrator plan correctly?
│       ├── test_delegation.py      ← Does it route to the right agent?
│       ├── test_tool_use.py        ← Are tool policies enforced?
│       ├── test_hallucination.py   ← Does the verifier catch false claims?
│       ├── test_robustness.py      ← Does it handle noisy/adversarial input?
│       ├── test_security.py        ← Are dangerous operations blocked?
│       ├── test_trace.py           ← Is the JSON trace complete?
│       └── test_graceful_failure.py← Does it fail gracefully?
│
├── tests/                          ← pytest suite (fully offline)
│   ├── mock_llm.py                 ← Deterministic mock LLM, no API key
│   └── test_e2e.py                 ← 19 integration tests
│
├── docs/
│   └── failure_modes.md            ← Agentic failure mode analysis
│
├── traces/                         ← Auto-generated JSON run traces (gitignored)
├── outputs/                        ← Agent file outputs (gitignored)
│
├── llm_backend.py                  ← Pluggable LLM: Claude → Ollama → OpenAI
├── tool_policy.py                  ← Runtime permission engine
├── trace_logger.py                 ← Structured JSON trace logger
├── main.py                         ← CLI entry point
├── demo.py                         ← Offline demo (no API key needed)
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── pytest.ini
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/)
- ~4GB disk space for the Ollama model
- No API key required for local mode

### 1. Clone and build

```bash
git clone https://github.com/YOUR_USERNAME/agentos.git
cd agentos
docker compose up --build
```

The first run downloads `llama3.2` (~2GB). Subsequent starts are instant.

### 2. Verify everything works

```bash
# Run the offline demo (mock LLM, instant, no model needed)
docker compose run --rm agentos python demo.py

# Run the test suite (19 tests, fully offline)
docker compose run --rm agentos pytest tests/ -v

# Run the offline eval suite (3 scenarios, no LLM needed)
docker compose run --rm agentos python evals/run_evals.py --fast
```

Expected test output:
```
19 passed in 0.9s
```

Expected offline eval output:
```
✓ tool_use_accuracy      [██████████]  1.00
✓ security_tool_policy   [██████████]  1.00
✓ trace_completeness     [██████████]  1.00
Score: 3/3 passed  |  Avg: 1.00
```

---

## Running Tasks

### With Ollama (local, free, no API key)

```bash
# Make sure the stack is running first
docker compose up -d

# Run a task
docker compose run --rm agentos python main.py \
  --task "Research LangGraph vs AutoGen and write a comparison report."
```

### With Claude API

```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose run --rm agentos python main.py \
  --task "Research LangGraph vs AutoGen and write a comparison report."
```

### With OpenAI

```bash
OPENAI_API_KEY=sk-... docker compose run --rm agentos python main.py \
  --task "Research LangGraph vs AutoGen and write a comparison report."
```

### Example tasks

```bash
# Research task
--task "What are the key differences between LangGraph, AutoGen, and CrewAI?"

# Code generation
--task "Write a Python script that reads a CSV, removes duplicates, and saves the result."

# Analysis
--task "Explain the CAP theorem and write a Python demo illustrating the trade-offs."
```

### Task output

Every run produces:
- A final synthesised response printed to the terminal
- A complete JSON trace at `traces/run_<id>.json`
- Status, step count, and any detected failure modes

```
════════════════════════════════════════════════════════════
  RESULT
════════════════════════════════════════════════════════════
[final output here]
────────────────────────────────────────────────────────────
  Status:        success
  Steps:         6
  Failure modes: none
  Trace saved:   traces/run_20260403_090000_abc123.json
```

---

## Eval Suite

The eval harness tests the agent system against 8 scenarios. Run it after any change to validate behaviour.

```bash
# Full suite (needs Ollama running, ~2 min)
docker compose run --rm agentos python evals/run_evals.py

# Offline only (instant, no LLM)
docker compose run --rm agentos python evals/run_evals.py --fast

# Single scenario
docker compose run --rm agentos python evals/run_evals.py --scenario tool_use_accuracy

# JSON output (for CI / logging)
docker compose run --rm agentos python evals/run_evals.py --format json
```

### Eval results (llama3.2 on CPU)

```
════════════════════════════════════════════════════════
  AgentOS Eval Suite
════════════════════════════════════════════════════════
  ✗ planning_quality                 [███████░░░]  0.67
  ✓ correct_delegation               [██████████]  1.00
  ✓ tool_use_accuracy                [██████████]  1.00
  ✓ hallucination_detection          [██████████]  1.00
  ✓ robustness_noisy_input           [██████████]  1.00
  ✓ security_tool_policy             [██████████]  1.00
  ✓ trace_completeness               [██████████]  1.00
  ✓ graceful_failure                 [██████████]  1.00
════════════════════════════════════════════════════════
  Score: 7/8 passed  |  Avg: 0.96
════════════════════════════════════════════════════════
```

The `planning_quality` score of 0.67 reflects llama3.2's limited ability to reliably produce structured JSON output — a known constraint of 3B models. The framework itself is model-agnostic; switching to `mistral` or `llama3.1:8b` brings this to 1.00.

---

## Model Notes

AgentOS is **model-agnostic**. The LLM is a swappable component — changing models requires one line in `docker-compose.yml`.

### Available local models (via Ollama)

| Model | Size | Quality | Speed (CPU) | Recommended for |
|---|---|---|---|---|
| `llama3.2` | 2GB | ⭐⭐ | Fast | Development, demos |
| `mistral` | 4GB | ⭐⭐⭐⭐ | Medium | Better structured output |
| `llama3.1:8b` | 5GB | ⭐⭐⭐⭐⭐ | Slow on CPU | Best local quality |

### Switching models

In `docker-compose.yml`, change the `ollama-pull` service:

```yaml
# Change this line:
entrypoint: ["ollama", "pull", "llama3.2"]
# To:
entrypoint: ["ollama", "pull", "mistral"]
```

And in `llm_backend.py`, update `LLMConfig`:

```python
ollama_model: str = "mistral"   # was "llama3.2"
```

Then rebuild:

```bash
docker compose down
docker compose up --build
```

### Using Claude (best results)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker compose run --rm agentos python main.py --task "..."
```

The backend auto-detects `ANTHROPIC_API_KEY` and uses Claude instead of Ollama — no other changes needed.

---

## Design Decisions

### Why LangGraph?

Explicit state machines over blackbox agents. Every node, edge, and conditional transition is inspectable and testable. This is critical for enterprise ADLC — you need to be able to audit exactly what happened in any given run.

### Why YAML agent cards?

Each agent's permissions are defined in a plain `.yaml` file — human-readable, git-diffable, and reviewable by non-engineers. This is the "agent card" pattern from agentic AI research, applied practically.

```yaml
# agent_cards/researcher.yaml
agent_id: researcher
trust_level: medium
tools:
  - name: web_search
    policy: allow
    rate_limit_per_run: 10
  - name: code_executor
    policy: deny
    reason: "Researcher does not execute code"
```

### Why a custom eval harness?

Commercial eval tools (LangSmith, etc.) create vendor lock-in and require API access. A pytest-style harness is portable, CI-compatible, and auditable. It also runs completely offline — a reviewer can clone the repo and run `pytest` with zero setup.

### Why Ollama as the default?

True sovereignty requires offline capability. Ollama lets the entire stack run with zero API calls, zero data egress, and zero ongoing cost. The same `docker compose up` works on a developer laptop, an on-prem server, or an air-gapped environment.

---

## Agentic Failure Modes

The critic agent and trace logger detect 10 named failure patterns per run. These are documented in detail in [`docs/failure_modes.md`](docs/failure_modes.md).

| Failure Mode | What it means | Countermeasure |
|---|---|---|
| `planning_hallucination` | Orchestrator invents agents or steps that don't exist | Post-plan registry filter |
| `delegation_loop` | Same subtask delegated twice | `seen_subtasks` guard |
| `tool_policy_bypass` | Agent attempts a forbidden tool | Runtime `ToolPolicyViolation` |
| `verification_blindness` | Verifier approves output without scrutiny | Heuristic length + keyword check |
| `critic_silence` | Critic finds no issues in a complex run | Auto-flag if failures=[] after 3+ steps |
| `context_loss` | Agent ignores prior step results | Pass last 3 results as context string |
| `over_planning` | Plan exceeds `MAX_PLAN_STEPS` | Hard cap enforced post-plan |
| `under_delegation` | Orchestrator does work itself instead of delegating | System prompt constraint |
| `hallucinated_facts` | Agent states false specifics confidently | Verifier + web search cross-check |
| `incomplete_task` | Final output doesn't address the original task | Synthesis prompt includes original task |

---

## Roadmap

- [ ] Add `llama3.1:8b` as default model recommendation
- [ ] Persistent memory across runs (vector store)
- [ ] Web UI for trace inspection
- [ ] Agent card hot-reload without container restart
- [ ] Streaming output to terminal during long runs
- [ ] Additional eval scenarios: multi-turn tasks, tool chaining

---

## Keywords

`LangGraph` · `multi-agent` · `agentic-AI` · `ADLC` · `Ollama` · `cloud-agnostic` · `eval-harness` · `agent-cards` · `tool-policy` · `trace-logging` · `open-source` · `Docker` · `sovereign-AI`