# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python 3.12+ monorepo AI teammate orchestration suite: FastAPI + LangGraph state machine with SSE streaming, cost tracking, and PII scrubbing.

## Commands

```bash
uv sync                    # Install
ruff check . && ruff format .  # Lint (120 char)
pytest tests/ -v           # Test (pytest-asyncio auto mode)
# Run specific tests: pytest tests/test_privacy.py -v

# Run locally
# Start dependencies: docker-compose up -d db redis
cd apps/liaison_brain && uvicorn app.main:app --reload --port 8000
cd apps/workspace_ui && uvicorn app:app --reload --port 5001

# Docker
docker-compose up -d  # liaison:8000, ui:5001 (host:8501), db:5432, redis:6379
```

**Debug:** Set `_USE_MEMORY_CHECKPOINTER = True` in `apps/liaison_brain/app/graph.py:693`

## Architecture

```
apps/liaison_brain/     → LangGraph orchestrator (port 8000)
apps/workspace_ui/      → FastHTML 3-panel UI (port 5001)
packages/skillhive/     → MCP tools + SYSTEM_PERSONA
packages/sentry_foundation/ → PII scrubber + cost tracker
```

### LangGraph Nodes

```
START → liaison_node → budget_check_node
              ↓              ↓
        tools_node    human_intervention → END
              ↓
        liaison_node (loop)
```

### Key Files

| File | Purpose |
|------|---------|
| `apps/liaison_brain/app/main.py` | FastAPI + SSE endpoints (`/chat/input`, `/chat/stream/{thread_id}`) |
| `apps/liaison_brain/app/graph.py` | State machine, nodes, LLM factory, MCP tools |
| `apps/liaison_brain/app/state.py` | TeammateState TypedDict |
| `apps/liaison_brain/app/callbacks.py` | ThoughtStreamCallback for token streaming |
| `apps/liaison_brain/app/shield.py` | SentryShieldCallback, PII/cost event formatters |
| `apps/workspace_ui/app.py` | FastHTML UI, HTMX SSE proxy |
| `packages/sentry_foundation/src/sentry/privacy.py` | PIIScrubber |
| `packages/sentry_foundation/src/sentry/economy.py` | UniversalCostTracker (litellm) |
| `packages/skillhive/src/skillhive/persona.py` | SYSTEM_PERSONA |

### SSE Events

`GET /chat/stream/{thread_id}` emits: `thought`, `message`, `cost`, `intervene`, `ping`, `error`

## State Fields

```python
task_id, task_description, task_budget, messages, current_step
current_context, sub_agents_spawned, requires_approval, approval_granted
total_cost, cost_limit, cost_history
privacy_violation, intervention_reason
```

## Security

- **PII:** Detects redacts emails, phones, credit cards (Luhn), API keys
- **Cost:** litellm-based tracking, budget circuit breaker (default $0.50)
- **Approval:** `/chat/approval/{thread_id}` for human-in-the-loop

## Environment

```bash
DATABASE_URL=postgresql://...@db:5432/teammate_memory
REDIS_URL=redis://redis:6379
CHAT_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=...
COST_LIMIT=0.50
# See .env.example for all variables
```
