# Teammate Suite: 

## 1. Project Architecture
This is a 2026-standard **Agentic Enterprise Monorepo** managed by `uv`.
- **Liaison Brain (`apps/liaison_brain`)**: FastAPI + LangGraph state machine. The "Mind."
- **Workspace UI (`apps/workspace_ui`)**: Streamlit/FastHTML interface. The "Desk."
- **SkillHive (`packages/skillhive`)**: MCP Tool registry and Specialist configs. The "Talent."
- **Sentry Foundation (`packages/sentry_foundation`)**: Security, PII, and Budgeting. The "Shield."

## 2. Tech Stack & Commands
- **Environment**: Python 3.12+ managed by `uv`
- **Orchestration**: LangGraph (Stateful graphs) + DeepAgents (High-reasoning)
- **Tooling**: Model Context Protocol (MCP)
- **Lint/Format**: Ruff
- **Type Check**: Pyright

### Critical Commands
- **Install All**: `uv sync`
- **Run Brain**: `uv run -m apps.liaison_brain.app.main`
- **Run UI**: `uv run streamlit run apps/workspace_ui/app.py`
- **Lint**: `uv run ruff check . --fix`
- **Type Check**: `uv run pyright`
- **Add Dependency**: `uv add <package>` (Use `-p <package_name>` for specific workspace members)

## 3. Coding Guidelines
- **Python Style**: Follow PEP 8 with 2026 modernisms (Strict typing, Pydantic v2).
- **State Management**: All agent state must reside in `TeammateState` (TypedDict) within LangGraph. Never use global variables for agent memory.
- **Error Handling**: Use the `Sentry Foundation` to wrap all external LLM and Tool calls for PII scrubbing and cost tracking.
- **Tools**: New tools must be defined as **MCP Servers** in `packages/skillhive/mcp/` or as LangChain `@tool` decorated functions with clear Docstrings.
- **Memory**: Implement "Memory Sync" using the `PostgresSaver` checkpointer for session persistence.

## 4. Teammate Specific Rules
- **The "Liaison" Node**: This is the primary orchestrator. It must always check the `Task Budget` before spawning a sub-agent.
- **Human-in-the-Loop**: Any action involving "Financial Approval" or "External Email" must trigger an `interrupt_before` in the LangGraph.
- **Deep Research**: Use the `deepagents` package for nodes requiring >5 steps of iterative reasoning.

## 5. Directory Mapping for Claude
- Edit Brain Logic: `apps/liaison_brain/app/graph.py`
- Edit Security Rules: `packages/sentry_foundation/src/sentry/guardrails/`
- Edit Agent Skills: `packages/skillhive/src/skillhive/registry.py`

## 6. Teammate Monorepo Blueprint
teammate-suite/
├── .python-version          # Python 3.12+
├── pyproject.toml           # UV Workspace: defines shared build logic & ruff rules
├── uv.lock                  # Unified lockfile for reproducible 2026 builds
├── .env.example             # Template for ANTHROPIC_API_KEY, DB_URL, etc.
│
├── apps/
│   ├── liaison_brain/       # THE "MIND" (Orchestration & State)
│   │   ├── app/
│   │   │   ├── main.py      # FastAPI entry (Websocket streaming for UI)
│   │   │   ├── graph.py     # LangGraph State Machine (Nodes & Edges)
│   │   │   ├── state.py     # Pydantic models for Agentic State
│   │   │   └── memory.py    # Postgres Checkpointer logic for "Resume-ability"
│   │   ├── Dockerfile
│   │   └── pyproject.toml   # Deps: langgraph, fastapi, deepagents
│   │
│   └── workspace_ui/        # THE "DESK" (Collaborative UI)
│       ├── app.py           # Streamlit or FastHTML (Python-native interface)
│       └── components/      # UI components for "Human-in-the-loop" approval
│
├── packages/
│   ├── skillhive/           # THE "TALENT" (Tooling & RAG)
│   │   ├── src/skillhive/
│   │   │   ├── mcp/         # MCP Server connectors (SAP, GitHub, Slack)
│   │   │   ├── prompts/     # Version-controlled "System Instructions"
│   │   │   └── vectors/     # RAG embedding & retrieval logic
│   │   └── pyproject.toml
│   │
│   └── sentry_foundation/   # THE "SHIELD" (Security & Governance)
│       ├── src/sentry/
│       │   ├── redaction.py # Regex/NER-based PII scrubbing
│       │   ├── budget.py    # Real-time token/cost tracking gatekeeper
│       │   └── audit.py     # Logging "Chain of Thought" for legal/HR
│       └── pyproject.toml
│
└── docker-compose.yml       # Orchestrates the "Digital Office" services
