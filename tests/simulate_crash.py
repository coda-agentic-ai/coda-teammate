"""Crash & Recovery Test for LangGraph PostgreSQL Checkpointer."""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env
_dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(_dotenv_path)


async def session_1_store_secret():
    """Session 1: Store a secret code, then crash."""
    from apps.liaison_brain.app.memory import setup_database, get_async_saver_context
    from apps.liaison_brain.app.state import TeammateStateModel

    print("=" * 60)
    print("SESSION 1: Storing secret code...")
    print("=" * 60)

    # Setup schema if needed
    await setup_database()

    thread_id = "test-resilience-001"

    async with get_async_saver_context() as checkpointer:
        from apps.liaison_brain.app.graph import create_graph

        graph = create_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # Store the secret
        state = TeammateStateModel(
            task_id=thread_id,
            task_description="Hello, I am User A. Remember that my secret code is 9988",
            task_budget=1000,
        )

        print(f"Invoking graph with thread_id: {thread_id}")
        result = await graph.ainvoke(state.to_typeddict(), config=config)
        print(f"Graph result: {result}")

    print("\n[CRASH] Simulating process termination after Session 1...")
    print("State saved to PostgreSQL. Process exiting.\n")

    # Return the thread_id for Session 2
    return thread_id


async def session_2_retrieve_secret(thread_id: str):
    """Session 2: Recover state and retrieve the secret."""
    from apps.liaison_brain.app.memory import get_async_saver_context
    from apps.liaison_brain.app.state import TeammateStateModel

    print("=" * 60)
    print("SESSION 2: Recovering from crash and retrieving secret...")
    print("=" * 60)

    async with get_async_saver_context() as checkpointer:
        from apps.liaison_brain.app.graph import create_graph

        graph = create_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # Retrieve checkpoint from database
        from langgraph.checkpoint.postgres import PostgresSaver
        with PostgresSaver.from_conn_string(os.getenv("DATABASE_URL")) as saver:
            checkpoint = saver.get(config)
            if checkpoint:
                print(f"Checkpoint recovered: {checkpoint['id']}")
                print(f"Recovered state: {checkpoint.get('channel_values', {})}")
            else:
                print("[ERROR] No checkpoint found!")
                return None

        # Now ask about the secret
        print(f"\nAsking: 'What was my secret code?'")
        state = TeammateStateModel(
            task_id=thread_id,
            task_description="What was my secret code?",
            task_budget=1000,
        )

        result = await graph.ainvoke(state.to_typeddict(), config=config)
        print(f"Response: {result.get('current_context', 'No response')}")

    return result


async def main():
    """Run the crash & recovery test."""
    # Session 1: Store secret and crash
    thread_id = await session_1_store_secret()

    print("\n[INFO] Process killed. Simulating restart...\n")

    # Session 2: Recover and retrieve
    result = await session_2_retrieve_secret(thread_id)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
