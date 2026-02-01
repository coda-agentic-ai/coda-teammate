"""LangGraph state machine with MCP tool integration and LLM orchestration."""

from pathlib import Path
from typing import Any, Literal
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic_settings import BaseSettings
from pydantic import SecretStr
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage

from .memory import get_async_saver, get_sync_saver, close_async_saver
from .state import TeammateState, TeammateStateModel
import asyncio
import threading


# === Settings for LLM Configuration ===
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    CHAT_MODEL: str = "deepseek/deepseek-chat"
    DEEPSEEK_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    COST_LIMIT: float = 0.50  # Default cost limit in USD


settings = Settings()


# === Cost Tracking Callback ===
class CostTrackingCallback(BaseCallbackHandler):
    """LangChain callback handler for tracking LLM costs.

    This callback listens for LLM response events and calculates the cost
    using the UniversalCostTracker from sentry_foundation.

    Usage:
        callback = CostTrackingCallback(state_ref)
        llm = ChatAnthropic(callbacks=[callback])
    """

    def __init__(self, state_ref: dict):
        """Initialize the callback with a reference to the state dict.

        Args:
            state_ref: A mutable dict reference to TeammateState for updating costs.
        """
        self.state_ref = state_ref
        self._model_name = None

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        """Capture the model name at the start of an LLM call."""
        # Extract model name from the serialized LLM config
        self._model_name = serialized.get("name", settings.CHAT_MODEL)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Calculate and record the cost after LLM response."""
        if self._model_name is None:
            return

        try:
            from sentry_foundation.economy import UniversalCostTracker

            # Calculate the cost using litellm
            cost = UniversalCostTracker.calculate_cost(response, self._model_name)

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, "usage"):
                token_usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

            # Record the cost
            cost_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": self._model_name,
                "cost_usd": cost,
                **token_usage,
            }

            # Update the state reference
            current_total = self.state_ref.get("total_cost", 0.0)
            self.state_ref["total_cost"] = current_total + cost

            cost_history = self.state_ref.get("cost_history", [])
            cost_history.append(cost_record)
            self.state_ref["cost_history"] = cost_history

        except ImportError:
            # sentry_foundation might not be available in all contexts
            pass
        except Exception:
            # Silently handle cost calculation errors to not disrupt workflow
            pass


# === LLM Factory ===
def create_llm(provider: str, api_key: str | SecretStr) -> Any:
    """Create an LLM instance based on the provider.

    Args:
        provider: The LLM provider name (deepseek, anthropic, openai)
        api_key: The API key for the provider

    Returns:
        Configured LLM instance
    """
    provider = provider.lower()

    # Convert string to SecretStr if needed
    api_key_value = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)

    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(model="deepseek-chat", api_key=api_key_value)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=api_key_value)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", api_key=api_key_value)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    return llm


# === MCP Tool Loading ===
# Global state for MCP connection
_mcp_tools: list[Any] | None = None


async def get_mcp_tools() -> tuple[list[Any], Any]:
    """Load MCP tools from the research server.

    Starts the research.py MCP server as a subprocess and loads tools.
    Returns tools that can be used with the LLM.
    """
    global _mcp_tools

    if _mcp_tools is not None:
        return _mcp_tools, None

    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_mcp_adapters.sessions import StdioConnection

    # Get the path to the research.py script
    research_script = (
        Path(__file__).parent.parent.parent.parent
        / "packages"
        / "skillhive"
        / "src"
        / "skillhive"
        / "mcp"
        / "research.py"
    )

    # Create a connection config
    connection: StdioConnection = {
        "transport": "stdio",
        "command": "uv",
        "args": ["run", "--script", str(research_script)],
    }

    # Load tools using connection (tools will manage their own sessions)
    tools = await load_mcp_tools(session=None, connection=connection)
    _mcp_tools = tools

    return tools, None


async def close_mcp_connection():
    """Close the MCP connection."""
    global _mcp_tools
    _mcp_tools = None


# === Initialize LLM with Tools ===
_llm_with_tools: Any | None = None
_bound_mcp_tools: list[Any] = []


async def get_llm_with_tools() -> Any:
    """Get or create the LLM instance bound with MCP tools."""
    global _llm_with_tools, _bound_mcp_tools

    if _llm_with_tools is None:
        # Parse CHAT_MODEL to get provider (format: "provider/model")
        model_config = settings.CHAT_MODEL
        if "/" in model_config:
            provider, _ = model_config.split("/", 1)
        else:
            provider = model_config

        # Get the appropriate API key
        provider_api_keys = {
            "deepseek": settings.DEEPSEEK_API_KEY,
            "anthropic": settings.ANTHROPIC_API_KEY,
            "openai": settings.OPENAI_API_KEY,
        }
        api_key = provider_api_keys.get(provider, "")
        print(f"get_llm_with_tools: provider={provider}, api_key present={bool(api_key)}")

        if not api_key:
            raise ValueError(f"API key not found for provider: {provider}")

        # Create LLM and bind MCP tools
        print(f"get_llm_with_tools: creating LLM for provider {provider}")
        llm = create_llm(provider, api_key)
        print(f"get_llm_with_tools: LLM created, type: {type(llm).__name__}")
        mcp_tools, _ = await get_mcp_tools()  # Unpack tuple, ignore session
        print(f"get_llm_with_tools: loaded {len(mcp_tools)} MCP tools")
        _bound_mcp_tools = mcp_tools  # Store tools separately
        _llm_with_tools = llm.bind_tools(mcp_tools)
        print(f"get_llm_with_tools: LLM bound with tools")

    return _llm_with_tools


def get_bound_tools() -> list[Any]:
    """Get the list of MCP tools bound to the LLM."""
    global _bound_mcp_tools
    return _bound_mcp_tools


# === Define Graph Nodes ===

# System prompt for the liaison node
LIAISON_SYSTEM_PROMPT = """You are the Liaison - the primary orchestrator for the teammate suite.

## Your Role
You are the "Brain" of the teammate, responsible for:
1. Understanding user requests and breaking them down into actionable tasks
2. Coordinating with other agents and tools to accomplish goals
3. Managing task budgets and ensuring efficient resource usage

## Available Tools
You have access to research tools for market and company data. Use the research tools whenever the user asks for market or company data (e.g., stock prices, company information, financial metrics).

## Guidelines
- Always consider the task budget before spawning sub-agents
- For financial approvals or external emails, request human approval
- For complex research tasks (>5 steps), delegate to the research node
- Maintain context across the conversation using the state messages

## Current Context
Task: {task_description}
Budget: {task_budget} tokens remaining
Current Step: {current_step}
"""


async def liaison_node(state: TeammateState) -> TeammateState:
    """Primary orchestrator node - the 'Brain' of the teammate.

    Analyzes the task and routes to appropriate next steps.
    Invokes the LLM with bound MCP tools to handle user requests.
    """
    import traceback
    messages = state.get("messages", [])
    print(f"liaison_node: messages count = {len(messages)}")

    if not messages:
        return {
            **state,
            "current_step": "liaison",
        }

    # Get LLM with tools bound (initializes the LLM and MCP tools)
    llm = await get_llm_with_tools()
    print(f"liaison_node: LLM obtained, type: {type(llm).__name__}")

    # Convert any string messages to HumanMessage objects
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, str):
            formatted_messages.append(HumanMessage(content=msg))
        elif hasattr(msg, "content"):
            formatted_messages.append(msg)
        else:
            formatted_messages.append(msg)

    # Invoke the LLM with the FULL message history (including tool calls/results)
    print(f"liaison_node: invoking LLM with {len(formatted_messages)} messages")
    try:
        response = await llm.ainvoke(formatted_messages)
    except Exception as e:
        print(f"liaison_node: LLM invocation failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        # Return error message as AIMessage instead of crashing
        error_msg = f"Error processing request: {str(e)}"
        response = AIMessage(content=error_msg)

    print(f"liaison_node: response type: {type(response).__name__}")
    if hasattr(response, 'content'):
        print(f"liaison_node: response content: {response.content[:200] if response.content else 'empty'}")

    # Add the response to messages
    updated_messages = messages + [response]

    # Check if response contains tool calls
    has_tool_calls = (
        hasattr(response, "tool_calls") and response.tool_calls
    ) or (
        hasattr(response, "additional_kwargs") and
        response.additional_kwargs.get("tool_calls")
    )

    if has_tool_calls:
        return {
            **state,
            "messages": updated_messages,
            "current_step": "tools",
        }

    # No tool calls - return final response
    return {
        **state,
        "messages": updated_messages,
        "current_step": "liaison",
    }


async def tools_node(state: TeammateState) -> TeammateState:
    """Execute tool calls using the LLM.

    This node handles tool invocations when the liaison node determines
    that tools need to be called. It executes the pending tool calls and
    returns the results.
    """
    messages = state.get("messages", [])

    if not messages:
        return state

    # Get the last message (should be an AIMessage with tool_calls)
    last_message = messages[-1]

    # Get available tools
    available_tools = get_bound_tools()
    if not available_tools:
        return {
            **state,
            "current_step": "liaison",
            "intervention_reason": "No tools available",
        }

    from langchain_core.messages import ToolMessage

    # Check if last message has tool calls
    tool_calls = getattr(last_message, "tool_calls", None) or \
                 getattr(last_message, "additional_kwargs", {}).get("tool_calls", [])

    if not tool_calls:
        return {
            **state,
            "current_step": "liaison",
        }

    # Execute tool calls - use the actual IDs from the LLM's response
    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        # Find the tool and execute it
        result_content = "Tool not found"
        for tool in available_tools:
            if tool.name == tool_name:
                try:
                    if hasattr(tool, "ainvoke"):
                        result = await tool.ainvoke(tool_args)
                    else:
                        result = tool.invoke(tool_args)
                    result_content = str(result)
                except Exception:
                    result_content = f"Error: Tool execution failed"
                break

        tool_results.append(
            ToolMessage(
                content=result_content,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    # Add tool results to messages
    updated_messages = messages + tool_results

    return {
        **state,
        "messages": updated_messages,
        "current_step": "liaison",  # Loop back for LLM to process results
    }


def research_node(state: TeammateState) -> TeammateState:
    """Deep research node for complex queries (>5 steps reasoning)."""
    # TODO: Implement research logic using deepagents
    return {
        **state,
        "current_step": "research",
    }


def approval_node(state: TeammateState) -> TeammateState:
    """Human-in-the-loop approval gate for sensitive operations."""
    # TODO: Implement approval workflow
    return {
        **state,
        "current_step": "approval",
    }


def execute_node(state: TeammateState) -> TeammateState:
    """Execution node for tool/agent invocation."""
    # TODO: Implement execution logic
    return {
        **state,
        "current_step": "execute",
    }


def budget_check_node(state: TeammateState) -> TeammateState:
    """Circuit breaker: Check accumulated cost against budget limit.

    This node acts as a gatekeeper, routing to human_intervention if the
    accumulated cost exceeds the configured cost_limit. This prevents
    runaway spending on expensive model calls.

    Returns:
        Updated state with requires_approval=True if budget exceeded.
    """
    total_cost = state.get("total_cost", 0.0)
    cost_limit = state.get("cost_limit", settings.COST_LIMIT)

    if total_cost >= cost_limit:
        return {
            **state,
            "current_step": "budget_exceeded",
            "requires_approval": True,
            "approval_granted": False,
            "intervention_reason": "Budget limit reached",
        }

    return {
        **state,
        "current_step": "budget_ok",
    }


def privacy_intervention_node(state: TeammateState) -> TeammateState:
    """Circuit breaker: Handle PII violations detected in LLM responses.

    This node is triggered when the SentryShieldCallback detects PII in
    an LLM response. It halts the workflow and requires human approval
    to proceed.

    Returns:
        Updated state with requires_approval=True and intervention_reason set.
    """
    return {
        **state,
        "current_step": "privacy_intervention",
        "requires_approval": True,
        "approval_granted": False,
        "intervention_reason": state.get("intervention_reason", "PII violation detected"),
    }


# === Define Graph Edges ===

def should_continue(state: TeammateState) -> Literal["research_node", "execute_node", "approval_node", "tools_node", "budget_check_node"]:
    """Routing logic after liaison processing.

    Routes to budget_check first to verify we haven't exceeded cost limits.
    """
    # Always check budget before proceeding
    return "budget_check_node"


def should_budget_check(state: TeammateState) -> Literal["liaison_node", "research_node", "execute_node", "approval_node", "tools_node", "human_intervention"]:
    """Routing logic after budget check.

    If budget exceeded, route to human_intervention for approval.
    If there are pending tool calls, route to tools_node.
    Otherwise, continue with normal routing.
    """
    # Check if budget exceeded
    if state.get("requires_approval", False) and not state.get("approval_granted", False):
        return "human_intervention"

    # Check if there are pending tool calls to execute
    messages = state.get("messages", [])
    print(f"[DEBUG] should_budget_check: len(messages)={len(messages)}")
    if messages:
        last_message = messages[-1]
        # Check if last message has tool_calls
        tool_calls = getattr(last_message, "tool_calls", None) or \
                     getattr(last_message, "additional_kwargs", {}).get("tool_calls", [])
        if tool_calls:
            return "tools_node"
    else:
        last_message = None

    # Budget check for high-value tasks
    if state.get("task_budget", 0) > 5000:
        return "research_node"

    # If we have at least one AI response and no tool calls, execute to end
    if len(messages) >= 2 and last_message and isinstance(last_message, AIMessage):
        print(f"[DEBUG] should_budget_check: routing to execute_node, AI response present")
        return "execute_node"

    # If we have more than 4 messages, assume task is complete
    if len(messages) > 4:
        print(f"[DEBUG] should_budget_check: routing to execute_node, len={len(messages)}")
        return "execute_node"

    return "liaison_node"  # Continue normal flow


def should_approve(state: TeammateState) -> Literal["liaison_node", "execute_node", "end"]:
    """Routing after approval node."""
    if state.get("approval_granted", False):
        return "execute_node"
    return "end"


# === Build the Graph ===

def create_graph(checkpointer: AsyncPostgresSaver) -> Any:
    """Create the LangGraph with state schema and checkpointer."""

    # Define the state schema
    builder = StateGraph(TeammateStateModel)

    # Add nodes
    builder.add_node("liaison_node", liaison_node)
    builder.add_node("tools_node", tools_node)
    builder.add_node("research_node", research_node)
    builder.add_node("approval_node", approval_node)
    builder.add_node("execute_node", execute_node)
    builder.add_node("budget_check_node", budget_check_node)
    builder.add_node("privacy_intervention_node", privacy_intervention_node)
    builder.add_node("human_intervention", approval_node)  # Reuse approval_node

    # Define flow
    builder.add_edge(START, "liaison_node")

    # First routing to budget check
    builder.add_conditional_edges(
        "liaison_node",
        should_continue,
        {
            "budget_check_node": "budget_check_node",
        }
    )

    # Budget check routing
    builder.add_conditional_edges(
        "budget_check_node",
        should_budget_check,
        {
            "liaison_node": "liaison_node",
            "research_node": "research_node",
            "execute_node": "execute_node",
            "approval_node": "approval_node",
            "tools_node": "tools_node",
            "human_intervention": "human_intervention",
        }
    )

    builder.add_edge("tools_node", "liaison_node")  # Loop back after tool execution
    builder.add_edge("research_node", "liaison_node")  # Loop back for iteration
    builder.add_conditional_edges(
        "human_intervention",
        should_approve,
        {
            "execute_node": "execute_node",
            "end": END,
        }
    )
    builder.add_edge("execute_node", END)

    # Compile with PostgreSQL checkpointer
    graph = builder.compile(checkpointer=checkpointer)

    return graph


# === Graph Accessor ===

_graph: Any | None = None
_checkpointer: Any | None = None
_graph_lock = asyncio.Lock()
_graph_sync_lock = threading.Lock()


async def get_graph() -> Any:
    """Get or create the compiled graph (async singleton).

    The checkpointer is stored globally to prevent connection closure.
    Use close_graph() during shutdown to cleanup.
    """
    global _graph, _checkpointer
    if _graph is None:
        async with _graph_lock:
            # Double-check after acquiring lock
            if _graph is None:
                # Create checkpointer without context manager (keeps connection alive)
                _checkpointer = await get_async_saver()
                _graph = create_graph(_checkpointer)
    return _graph


def get_graph_sync() -> Any:
    """Synchronous graph accessor (for testing)."""
    global _graph, _checkpointer
    if _graph is None:
        with _graph_sync_lock:
            # Double-check after acquiring lock
            if _graph is None:
                # Create checkpointer without context manager (keeps connection alive)
                _checkpointer = get_sync_saver()
                _graph = create_graph(_checkpointer)
    return _graph


async def close_graph() -> None:
    """Close the graph and checkpointer connection gracefully."""
    global _graph, _checkpointer
    async with _graph_lock:
        if _checkpointer is not None:
            await close_async_saver(_checkpointer)
        _checkpointer = None
        _graph = None
