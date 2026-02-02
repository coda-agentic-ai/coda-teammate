"""Liaison Brain FastAPI application with SSE voice streaming."""

import asyncio
import json
import time
import traceback
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
from .callbacks import get_thought_callback, remove_thought_callback
from langchain_core.messages import HumanMessage, AIMessage


# === Request/Response Models ===

class ChatInput(BaseModel):
    """Request model for chat input."""
    message: str
    thread_id: str | None = None
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

# Global lock to serialize all graph/checkpointer operations
# This prevents PostgreSQL "command in progress" errors from concurrent access
_GRAPH_LOCK = asyncio.Lock()


async def graph_aget_state(graph, config):
    """Thread-safe graph.aget_state with retry logic."""
    async with _GRAPH_LOCK:
        try:
            return await graph.aget_state(config)
        except Exception as e:
            if "another command is already in progress" in str(e).lower():
                await asyncio.sleep(0.1)
                return await graph.aget_state(config)
            raise


async def graph_ainvoke(graph, state, config):
    """Thread-safe graph.ainvoke with retry logic."""
    async with _GRAPH_LOCK:
        try:
            return await graph.ainvoke(state, config)
        except Exception as e:
            if "another command is already in progress" in str(e).lower():
                await asyncio.sleep(0.1)
                return await graph.ainvoke(state, config)
            raise


def is_ai_message(msg) -> bool:
    """Check if message is an AI message (handles dict deserialization)."""
    if isinstance(msg, AIMessage):
        return True
    # Check dict representation (after PostgreSQL deserialization)
    if isinstance(msg, dict):
        return msg.get("type") == "ai" or msg.get("type") == "ai_message"
    return False


def has_tool_call_id(msg) -> bool:
    """Check if message has tool_call_id (handles dict deserialization)."""
    if hasattr(msg, "tool_call_id"):
        return True
    if isinstance(msg, dict):
        return "tool_call_id" in msg and msg["tool_call_id"] is not None
    return False


def get_message_content(msg) -> str:
    """Extract content from a message (handles dict deserialization)."""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    if hasattr(msg, "content"):
        return msg.content
    return str(msg)


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
        "messages": [HumanMessage(content=message)],
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
    print(f"=== EVENT GENERATOR START: thread_id={thread_id} ===")

    # Initialize variables for the try block
    current_state = None
    last_message_count = 0
    last_ping_time = time.time()
    last_cost_time = time.time()
    thought_callback = None

    # Wrap initialization in try/except to prevent crashes on graph/state errors
    try:
        print("TEST PRINT BEFORE GRAPH")
        graph = await get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        print(f"event_generator: thread_id={thread_id}, config={config}")

        # Get current state to check for messages (thread-safe)
        current_state = await graph_aget_state(graph, config)
        if current_state:
            print(f"event_generator: state exists, values keys: {list(current_state.values.keys())}")

        last_message_count = len(current_state.values.get("messages", [])) if current_state else 0

        # Send existing AI messages that haven't been sent yet
        messages = current_state.values.get("messages", []) if current_state else []
        for msg in messages:
            content = get_message_content(msg)
            if content and is_ai_message(msg) and not has_tool_call_id(msg):
                print(f"event_generator: yielding existing AI message, content length: {len(content)}")
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "role": "assistant",
                        "content": content,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                }

        # Yield cost event immediately if we have state
        if current_state:
            print("event_generator: yielding cost event, current_state exists")
            total_cost = current_state.values.get("total_cost", 0.0)
            yield format_cost_event(total_cost)

        # Thought callback will be fetched in the loop (handles timing issue)

    except Exception as e:
        print(f"event_generator: initialization error: {type(e).__name__}: {e}")
        traceback.print_exc()
        yield {"event": "error", "data": json.dumps({"message": f"Initialization error: {str(e)}"})}
        return  # Exit early if initialization failed

    # Stream indefinitely, checking for new state
    while True:
        try:
            # Get latest state (thread-safe with retry logic)
            state = await graph_aget_state(graph, config)
            if not state:
                # Thread not created yet - wait and retry instead of killing stream
                print(f"[SSE] Thread {thread_id} not found yet, waiting for creation...")
                await asyncio.sleep(0.5)
                continue

            # Thread found - log state exists
            print(f"[SSE] Thread {thread_id} found, state exists")

            messages = state.values.get("messages", [])
            print(f"event_generator: messages count = {len(messages)}, last_message_count = {last_message_count}")
            # Debug: print message types
            if len(messages) != last_message_count:
                print(f"event_generator: message types: {[type(m).__name__ for m in messages]}")
                for i, m in enumerate(messages):
                    if hasattr(m, 'content'):
                        print(f"  [{i}] {type(m).__name__}: {repr(m.content)[:80]}")
                    else:
                        print(f"  [{i}] {type(m).__name__}: {repr(str(m))[:80]}")

            # Check for new messages to display
            if len(messages) > last_message_count:
                # New messages have been added
                new_messages = messages[last_message_count:]
                print(f"event_generator: {len(new_messages)} new messages")
                for msg in new_messages:
                    # Extract content using helper (handles both dict and AIMessage)
                    content = get_message_content(msg)
                    is_ai = is_ai_message(msg)
                    has_tool_id = has_tool_call_id(msg)
                    print(f"event_generator: message type: {type(msg).__name__}, is_ai: {is_ai}, has_tool_call_id: {has_tool_id}, content length: {len(content)}")
                    # Skip tool messages (have tool_call_id) and only show assistant responses
                    # Also skip if thought callback exists (thought events will represent streaming content)
                    if content and not has_tool_id and is_ai:
                        # Check if thought callback is active for this thread
                        thought_cb = await get_thought_callback(thread_id)
                        if thought_cb is None:
                            # No active thought streaming, safe to emit message event
                            print("event_generator: yielding message event")
                            yield {
                                "event": "message",
                                "data": json.dumps({
                                    "role": "assistant",
                                    "content": content,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                })
                            }
                        else:
                            print("event_generator: skipping message event (thought events will stream)")
            # Always update last_message_count to current count
            last_message_count = len(messages)

            # Check current step for interventions
            current_step = state.values.get("current_step", "")
            if current_step == "budget_exceeded":
                yield format_intervene_event(
                    reason=state.values.get("intervention_reason", "Budget limit reached"),
                    current_cost=state.values.get("total_cost", 0.0),
                )
            elif current_step == "privacy_intervention":
                yield format_intervene_event(
                    reason=state.values.get("intervention_reason", "PII violation detected"),
                    current_cost=state.values.get("total_cost", 0.0),
                )

            # Poll for thought tokens from the callback
            # Re-check callback each iteration to handle timing issue
            # (liaison_node registers callback after event_generator starts)
            thought_callback = await get_thought_callback(thread_id)
            if thought_callback:
                try:
                    # Non-blocking check for available thought tokens
                    while not thought_callback.queue.empty():
                        node_name, token = thought_callback.queue.get_nowait()
                        if token is not None:
                            # Yield thought event with node name and token
                            event = format_thought_event(token, channel, node_name)
                            print(f"[SSE] Yielding thought event: node={node_name}, token_len={len(token)}, event={event}")
                            yield event
                        else:
                            # None token signals callback should be removed
                            # (either stream ended or replaced by new callback)
                            await remove_thought_callback(thread_id)
                            thought_callback = None
                            break
                except asyncio.QueueEmpty:
                    pass

            # Periodic cost events
            current_time = time.time()
            if current_time - last_cost_time > 5:
                total_cost = state.values.get("total_cost", 0.0)
                yield format_cost_event(total_cost)
                last_cost_time = current_time

            # Ping for keep-alive
            if current_time - last_ping_time > 15:
                yield {"event": "ping", "data": ""}
                last_ping_time = current_time

            # Small delay before next poll (reduced to 0.1s for faster thought streaming)
            await asyncio.sleep(0.1)

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
            break


# === API Endpoints ===

@app.post("/chat/input", response_model=ChatResponse)
async def chat_input(input_data: ChatInput):
    """Receive user message and start the graph execution asynchronously.

    This endpoint initiates a conversation by:
    1. Creating initial state with the user's message
    2. Starting graph execution as background task
    3. Returning immediately (< 200ms) with thread_id for streaming
    """
    import uuid
    thread_id = input_data.thread_id or str(uuid.uuid4())
    print(f"[chat_input] thread_id={thread_id}, message={input_data.message[:50]}")

    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": thread_id}}

        # Check if state already exists for this thread (conversational memory)
        existing_state = await graph.aget_state(config)

        if existing_state:
            # Append new message to existing conversation history
            existing_messages = existing_state.values.get("messages", [])
            new_messages = list(existing_messages) + [HumanMessage(content=input_data.message)]

            # Create state update preserving existing state
            initial_state = {
                "task_id": thread_id,
                "task_description": input_data.task_description or existing_state.values.get("task_description", input_data.message[:100]),
                "task_budget": input_data.task_budget,
                "messages": new_messages,  # Include history!
                "current_context": existing_state.values.get("current_context", ""),
                "current_step": "liaison",  # Continue from liaison node
                "sub_agents_spawned": existing_state.values.get("sub_agents_spawned", []),
                "requires_approval": False,
                "approval_granted": False,
                "total_cost": existing_state.values.get("total_cost", 0.0),
                "cost_limit": input_data.cost_limit,
                "cost_history": existing_state.values.get("cost_history", []),
                "privacy_violation": False,
                "intervention_reason": "",
            }
            print(f"[chat_input] Continuing existing conversation, messages: {len(new_messages)}")
        else:
            # Create fresh state for new conversation
            initial_state = await generate_state(
                thread_id=thread_id,
                message=input_data.message,
                task_description=input_data.task_description,
                task_budget=input_data.task_budget,
                cost_limit=input_data.cost_limit,
            )
            print("[chat_input] Starting new conversation")

        # Validate state
        state_model = TeammateStateModel(**initial_state)

        async def run_graph():
            """Background task to execute the graph."""
            print(f"[chat_input] starting graph.ainvoke as background task for thread {thread_id}")
            try:
                # Use a longer timeout for background execution (e.g., 300 seconds)
                await asyncio.wait_for(
                    graph_ainvoke(graph, state_model.to_typeddict(), config=config),
                    timeout=300.0
                )
                print(f"[chat_input] graph.ainvoke completed for thread {thread_id}")
            except asyncio.TimeoutError:
                # Graph invocation timed out, but that's okay - state may still be saved
                print(f"[chat_input] graph.ainvoke timed out for thread {thread_id}")
            except Exception as e:
                # Log errors but don't crash the background task
                print(f"[chat_input] Error in background graph execution: {type(e).__name__}: {e}")
                traceback.print_exc()

        # Start graph execution as background task
        asyncio.create_task(run_graph())

        # Return immediately - graph execution continues in background
        return ChatResponse(status="started", thread_id=thread_id)

    except Exception as e:
        print(f"Error in /chat/input: {type(e).__name__}: {e}")
        print("Traceback:")
        traceback.print_exc()
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
