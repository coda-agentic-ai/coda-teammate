"""Test LangGraph PostgreSQL checkpointer."""

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root before importing modules
_dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(_dotenv_path)


@pytest.mark.asyncio
async def test_checkpoint_persistence():
    """Test that thread_id is saved to database."""
    from apps.liaison_brain.app.memory import setup_database, get_async_saver_context
    from apps.liaison_brain.app.state import TeammateStateModel

    # Setup schema
    print("\nSetting up database schema...")
    await setup_database()
    print("Schema created!")

    # Create and run a simple graph with one "thought" node
    async with get_async_saver_context() as checkpointer:
        from apps.liaison_brain.app.graph import create_graph

        # Create graph
        print("Creating graph...")
        graph = create_graph(checkpointer)

        # Run with test input
        thread_id = "test-thread-001"
        config = {"configurable": {"thread_id": thread_id}}

        state = TeammateStateModel(
            task_id=thread_id,
            task_description="Test task for checkpoint persistence",
            task_budget=1000,
        )

        print(f"Invoking graph with thread_id: {thread_id}")
        # Invoke graph
        result = await graph.ainvoke(state.to_typeddict(), config=config)
        print(f"Graph result: {result}")

        # Verify checkpoint was saved
        from langgraph.checkpoint.postgres import PostgresSaver

        print("Verifying checkpoint was saved...")
        with PostgresSaver.from_conn_string(os.getenv("DATABASE_URL")) as saver:
            checkpoint = saver.get(config)
            assert checkpoint is not None, "Checkpoint not found in database!"
            print(f"Checkpoint found: ID={checkpoint['id']}")
            print(f"Checkpoint timestamp: {checkpoint.get('ts', 'N/A')}")
            print(f"Channel values: {checkpoint.get('channel_values', {})}")

    print("\n[PASS] Test passed! Thread checkpoint successfully persisted to PostgreSQL.")


if __name__ == "__main__":
    asyncio.run(test_checkpoint_persistence())
