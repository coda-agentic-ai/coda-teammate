"""Liaison Brain FastAPI application with SSE voice streaming."""

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .graph import get_graph
from .state import TeammateStateModel
from .shield import (
    format_cost_event,
    format_intervene_event,
    format_thought_event,
)


# === Request/Response Models ===

class ChatInput(BaseModel):
    """Request model for chat input."""
    message: str
    task_description: str | None = None
    task_budget: int = 1000
    cost_limit: float = 0.50


class ChatResponse(BaseModel):
    """Response model for chat input."""
    status: str
    thread_id: str


class ApprovalRequest(BaseModel):
    """Request model for approving an intervention."""
    approved: bool


# === Lifespan Context ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for initialization."""
    # Startup: ensure graph is loaded
    try:
        await get_graph()
        print("Liaison Brain initialized successfully")
    except Exception as e:
        print(f"Warning: Graph initialization deferred - {e}")
    yield
    # Shutdown cleanup if needed
    pass


# === FastAPI App ===

app = FastAPI(
    title="Liaison Brain",
    description="Agentic workflow orchestrator with SSE voice streaming",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for workspace UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Helper Functions ===

async def generate_state(
    thread_id: str,
    message: str,
    task_description: str | None = None,
    task_budget: int = 1000,
    cost_limit: float = 0.50,
) -> dict:
    """Create initial state for a new conversation."""
    return {
        "task_id": thread_id,
        "task_description": task_description or message[:100],
        "task_budget": task_budget,
        "messages": [message],
        "current_context": "",
        "current_step": "init",
        "sub_agents_spawned": [],
        "requires_approval": False,
        "approval_granted": False,
        "total_cost": 0.0,
        "cost_limit": cost_limit,
        "cost_history": [],
        "privacy_violation": False,
        "intervention_reason": "",
    }


async def event_generator(
    thread_id: str,
    channel: str = "WEB_CANVAS",
) -> AsyncGenerator[dict, None]:
    """Generate SSE events from the LangGraph stream.

    Maps LangGraph events to SSE payloads:
    - on_chat_model_stream -> "thought" events
    - on_tool_start -> "action" events
    - on_tool_end -> "observation" events
    - on_chain_end -> "result" events
    - Periodic cost updates
    - 15-second pings for keep-alive
    """
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Create initial input state
    initial_state = await graph.aget_state(config)
    if not initial_state or not initial_state.values.get("messages"):
        yield {"event": "error", "data": json.dumps({"message": "No conversation found"})}
        return

    last_ping_time = time.time()
    last_cost_time = time.time()

    # Stream events from the graph
    async for event in graph.astream_events(
        initial_state.values,
        config,
        version="v2",
    ):
        event_type = event.get("event")
        data = event.get("data", {})

        # === Thought Events (from LLM streaming) ===
        if event_type == "on_chat_model_stream":
            chunk = data.get("chunk", {})
            # Handle both AIMessageChunk and Content blocks
            if hasattr(chunk, "content"):
                content = chunk.content
            elif isinstance(chunk, dict):
                content = chunk.get("content", "")
            else:
                content = str(chunk)

            if content:
                # Format and yield thought event
                event_data = format_thought_event(content, channel)
                yield event_data

        # === Action Events (tool start) ===
        elif event_type == "on_tool_start":
            tool_name = data.get("name", "unknown")
            yield {
                "event": "action",
                "data": json.dumps({
                    "tool": tool_name,
                    "status": "started",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            }

        # === Observation Events (tool end) ===
        elif event_type == "on_tool_end":
            tool_name = data.get("name", "unknown")
            output = data.get("output", "")
            yield {
                "event": "observation",
                "data": json.dumps({
                    "tool": tool_name,
                    "status": "completed",
                    "output_preview": str(output)[:200] if output else "",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            }

        # === Result Events (chain end) ===
        elif event_type == "on_chain_end" and "output" in data:
            output = data.get("output", {})
            if isinstance(output, dict):
                yield {
                    "event": "result",
                    "data": json.dumps({
                        "output": output,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                }

        # === Check State for Interventions ===
        try:
            current_state = await graph.aget_state(config)
            if current_state:
                state_values = current_state.values
                current_step = state_values.get("current_step", "")

                # Budget exceeded intervention
                if current_step == "budget_exceeded":
                    yield format_intervene_event(
                        reason=state_values.get("intervention_reason", "Budget limit reached"),
                        current_cost=state_values.get("total_cost", 0.0),
                    )

                # Privacy violation intervention
                elif current_step == "privacy_intervention":
                    yield format_intervene_event(
                        reason=state_values.get("intervention_reason", "PII violation detected"),
                        current_cost=state_values.get("total_cost", 0.0),
                    )

        except Exception:
            # State check failed - continue streaming
            pass

        # === Periodic Cost Events (every ~5 seconds) ===
        current_time = time.time()
        if current_time - last_cost_time > 5:
            try:
                current_state = await graph.aget_state(config)
                if current_state:
                    total_cost = current_state.values.get("total_cost", 0.0)
                    yield format_cost_event(total_cost)
            except Exception:
                pass
            last_cost_time = current_time

        # === Ping Events (every 15 seconds for proxy keep-alive) ===
        if current_time - last_ping_time > 15:
            yield {"event": "ping", "data": ""}
            last_ping_time = current_time


# === API Endpoints ===

@app.post("/chat/input", response_model=ChatResponse)
async def chat_input(thread_id: str, input_data: ChatInput):
    """Receive user message and start the graph execution.

    This endpoint initiates a conversation by:
    1. Creating initial state with the user's message
    2. Invoking the graph to process the message
    3. Returning immediately with thread_id for streaming
    """
    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": thread_id}}

        # Create initial state
        initial_state = await generate_state(
            thread_id=thread_id,
            message=input_data.message,
            task_description=input_data.task_description,
            task_budget=input_data.task_budget,
            cost_limit=input_data.cost_limit,
        )

        # Validate state
        state_model = TeammateStateModel(**initial_state)

        # Start graph execution
        await graph.ainvoke(state_model.to_typeddict(), config=config)

        return ChatResponse(status="started", thread_id=thread_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/chat/stream/{thread_id}")
async def chat_stream(
    thread_id: str,
    channel: str = "WEB_CANVAS",
):
    """SSE endpoint for streaming thought/cost/intervene events.

    Streams events from the LangGraph execution:

    Event Types:
    - thought: LLM reasoning chunks (INTERNAL_MONOLOGUE)
    - action: Tool execution starts (ACTION_SIGNAL)
    - observation: Tool execution completes
    - result: Final output from chain
    - cost: Periodic cost updates (every ~5 seconds)
    - intervene: Budget/PII intervention events
    - ping: Keep-alive signal (every 15 seconds)

    Query Parameters:
    - channel: Target channel ("WEB_CANVAS" or "SLACK")
              WEB_CANVAS: Full JSON with PII flags
              SLACK: Aggregated thoughts, action/result updates only
    """
    valid_channels = ["WEB_CANVAS", "SLACK"]
    if channel not in valid_channels:
        channel = "WEB_CANVAS"

    return EventSourceResponse(
        event_generator(thread_id, channel),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/chat/approval/{thread_id}")
async def submit_approval(thread_id: str, approval: ApprovalRequest):
    """Submit approval for a pending intervention."""
    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": thread_id}}

        # Get current state
        current_state = await graph.aget_state(config)
        if not current_state:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Update approval status
        await graph.update_state(
            config,
            {
                "approval_granted": approval.approved,
                "requires_approval": False,
            }
        )

        return {
            "status": "approved" if approval.approved else "denied",
            "thread_id": thread_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "Liaison Brain",
        "version": "0.1.0",
        "endpoints": {
            "POST /chat/input": "Send message and start graph execution",
            "GET /chat/stream/{thread_id}": "Stream SSE events",
            "POST /chat/approval/{thread_id}": "Submit intervention approval",
            "GET /health": "Health check",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
