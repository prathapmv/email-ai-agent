# Email AI Agent — Autonomous Gmail Assistant

![Tests](https://github.com/YOUR_USERNAME/email-ai-agent/actions/workflows/tests.yml/badge.svg)

An autonomous AI agent that reads, triages, drafts, sends, labels, and archives
Gmail messages — built from scratch to learn agentic AI engineering end to end.

**Stack:** Python · Gmail API (OAuth2) · Ollama LLM (Anthropic Claude - Not used for now as I'm preferring local LLM Llama) · LangChain · LangGraph

## Why this project exists

This is a learning-in-public build. Every phase below maps to a real production-agent skill, not just a tutorial checkbox. 
Commit history = progress log — each commit is a working, testable increment.

## Architecture

```
Gmail Inbox
    │
    ▼
[fetch_node] ──> pulls unread threads via Gmail API
    │
    ▼
[classify_node] ──> Llama scores intent/urgency/category
    │
    ▼
[decide_node] ──> Llama picks an action: reply | draft | label | archive | escalate
    │
    ▼
[guardrail_node] ──> confidence + risk check (blocks unsafe autonomous sends)
    │
    ▼
[act_node] ──> executes via Gmail API tools
    │
    ▼
[memory_node] ──> logs thread + action to memory store for future context
```

This is a LangGraph **state machine**, not a single prompt loop — state
persists across nodes, edges are conditional (e.g. low-confidence → escalate
to a human-review queue instead of auto-sending).

## Learning roadmap (phases = commits)

**Three things in this project could only be written, not run-verified,
in the environment this was built in — see `RUNBOOK.md` for the exact
steps to verify them yourself: real Gmail OAuth, the Docker build, and
neural embedding quality.**

| Phase | Skill learned | Files |
|---|---|---|
| 1 | OAuth2 flows, Gmail API scopes, token refresh | `src/gmail_auth.py` |
| 2 | LangChain tool schemas, structured I/O | `src/gmail_tools.py` |
| 3 | LangGraph state design, conditional edges | `src/agent_state.py`, `src/agent_graph.py` |
| 4 | Prompt engineering for decision-making, structured output parsing | `src/agent_graph.py` (decide_node) |
| 5 | Memory/context management, lightweight RAG over thread history | `src/memory.py` |
| 6 | Guardrails, human-in-the-loop escalation, dry-run safety | `src/guardrails.py` |
| 7 | Mock-inbox testing without live credentials | `tests/test_agent_mock.py` |
| 8 | Vector similarity search (TF-IDF), cross-sender RAG | `src/vector_memory.py` |
| 9 | Scheduling, per-item error isolation for unattended runs | `src/scheduler.py` |
| 10 | Human-in-the-loop review UI, guardrails applying to human actions too | `src/dashboard.py` |
| 11 | Containerization, secrets-out-of-image deployment practice | `Dockerfile`, `docker-compose.yml` |
| 12 | Neural embeddings with graceful degradation to a simpler backend | `src/vector_memory.py` |
| 13 | CI -- tests that prove they keep passing, not just claims they did | `.github/workflows/tests.yml` |
| 14 | Real API pagination vs. silent single-page truncation | `src/gmail_tools.py` |
| 15 | Cost/rate controls enforced at the call site, not hoped for at the edges | `src/cost_control.py` |
| 16 | Multi-turn conversation state -- why per-email isolation breaks down | `src/memory.py` (has_already_replied) |
| 17 | Observability -- turning logs into answerable questions | `src/observability.py` |
| 18 | Push vs. polling architecture, webhook design | `src/webhook.py` |
| 19 | Tool-use inside decision-making, real LangChain tool-calling loops | `src/agentic_decide.py` |
| 20 | LLM provider abstraction -- running for free on a local model instead of a paid API | `src/llm_provider.py` |

See `docs/LEARNING_LOG.md` — fill this in per phase with what tripped you up
and what clicked. That log is often more convincing to a hiring engineer than
the code itself.

## Review dashboard

Escalated/low-confidence emails shouldn't just sit in a Gmail label nobody
checks. Run:

```
python -m src.dashboard
```

then open `http://127.0.0.1:5001` to see stats, the live review queue
(anything labeled `NeedsReview`), and a log of recent decisions with
reasoning. Approve/dismiss actions still route through `guardrails.py` --
a human clicking "approve" doesn't bypass the sensitive-content check.

## Two decision modes

`build_graph(agentic=False)` (default): one fixed-context LLM call per
email — cheap, predictable, good enough for most mail.

`build_graph(agentic=True)`: the model can call tools itself
(`search_emails`, sender history lookup) mid-reasoning if an email is
ambiguous, via a real tool-calling loop (capped at `MAX_TOOL_ITERATIONS`,
default 3 — a model that never commits to a decision escalates instead of
looping forever). Costs 1-3 LLM calls instead of always 1. Switch to this
if `observability.py` shows escalations correlating with under-information
rather than genuine risk.

## Push notifications (alternative to polling)

`src/webhook.py` receives Gmail change notifications via Google Cloud
Pub/Sub instead of polling every N seconds — lower latency, fewer wasted
API calls on a quiet inbox. Requires real Google Cloud infrastructure
(Pub/Sub topic + a public HTTPS endpoint), so full setup is a manual step
— see the docstring at the top of `src/webhook.py` for exact commands.
What's tested here (no live infra needed): correct parsing of the Pub/Sub
push envelope, and the webhook route's behavior on both valid and
malformed pushes.

## Scheduler (continuous mode)

```
python -m src.scheduler
```

Polls for new mail every `POLL_INTERVAL_SECONDS` (default 300s), processing
each email through the full agent graph. Per-email errors are caught and
logged individually so one bad email doesn't take down the whole run.

## Deployment (Docker)

```
docker-compose up -d
```

Runs the scheduler and dashboard as two services sharing `memory_store/`.
`credentials.json`/`token.json`/`.env` are mounted as volumes, never baked
into the image — an image with secrets in it is one you can never safely
share or push to a registry.

**Note:** this Dockerfile was written and reviewed but not build-tested,
since Docker isn't available in the environment it was built in. Before
relying on it, run `docker-compose build` locally and confirm it starts
cleanly — flag anything that breaks.

## Cost: this runs for free by default

`LLM_PROVIDER=ollama` is the default — the agent runs on a local model via
[Ollama](https://ollama.com), with zero per-token cost. You're spending your
own machine's compute, not paying an API bill. The only requirement is
installing Ollama and pulling a model once (see Setup below).

Set `LLM_PROVIDER=anthropic` in `.env` instead if you want Claude's answer
quality and are fine paying Anthropic's per-token rate (see
`src/llm_provider.py` for the trade-offs — local models are not a drop-in
quality replacement, particularly for reliably returning strict JSON, which
`decide_node` depends on).

Everything else in this project is free regardless of which LLM provider
you choose: the Gmail API, the TF-IDF/neural embedding backends, GitHub
Actions CI, and running the agent itself all have no cost beyond your own
machine and (optionally) hosting if you deploy it somewhere persistent.

## Setup (run locally — Gmail OAuth needs a real browser + your own Google Cloud project)

1. Create a Google Cloud project → enable Gmail API → create OAuth desktop
   credentials → download as `credentials.json` into project root.
2. `python -m venv venv && source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. `pip install -r requirements.txt`
4. Copy `.env.example` → `.env`.
   - **Free path (default):** install [Ollama](https://ollama.com), run
     `ollama pull llama3.1`, leave `LLM_PROVIDER=ollama` as-is. No API key
     needed at all.
   - **Paid path:** set `LLM_PROVIDER=anthropic` and add your
     `ANTHROPIC_API_KEY`.
5. `python main.py --dry-run` — first run authenticates via browser and runs
   in dry-run mode (no real sends/archives, just logs intended actions)
6. Once you trust it: `python main.py` for live autonomous mode

## Safety-first design

This agent defaults to `DRY_RUN=true`. Autonomous send/archive/label only
activates once you explicitly disable dry-run in `.env` — and even then,
`guardrails.py` escalates anything below a confidence threshold to a
`NEEDS_REVIEW` label instead of acting on it. Read `src/guardrails.py` first.
