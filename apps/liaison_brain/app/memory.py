"""PostgreSQL checkpointer configuration for LangGraph state persistence."""

import os
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator

from dotenv import load_dotenv

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import connect as sync_connect
from psycopg import sql
from psycopg.rows import dict_row

# Load .env from project root (for local dev) or skip if not found (Docker injects via env_file)
_dotenv_path = Path(__file__).resolve().parents[2] / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)

# Read DATABASE_URL from environment (injected by docker-compose or .env)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")


@contextmanager
def get_sync_saver_context() -> PostgresSaver:
    """Get synchronous PostgresSaver as a context manager.

    Usage:
        with get_sync_saver_context() as saver:
            # use saver
    """
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        yield saver


@asynccontextmanager
async def get_async_saver_context() -> AsyncIterator[AsyncPostgresSaver]:
    """Get asynchronous PostgresSaver as an async context manager.

    Usage:
        async with get_async_saver_context() as saver:
            # use saver
    """
    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as saver:
        yield saver


async def get_async_saver() -> AsyncPostgresSaver:
    """Get asynchronous PostgresSaver with persistent connection.

    Creates a persistent connection that won't close until explicitly closed.
    Use close_async_saver() to cleanup when done.
    """
    from psycopg import AsyncConnection

    # Create connection and keep it open
    conn = await AsyncConnection.connect(
        DATABASE_URL, autocommit=True, prepare_threshold=0
    )
    return AsyncPostgresSaver(conn=conn, serde=None)


def get_sync_saver() -> PostgresSaver:
    """Get synchronous PostgresSaver with persistent connection.

    Creates a persistent connection that won't close until explicitly closed.
    Use close_sync_saver() to cleanup when done.
    """
    conn = sync_connect(DATABASE_URL, autocommit=True)
    return PostgresSaver(conn=conn)


async def close_async_saver(saver: AsyncPostgresSaver) -> None:
    """Close the async saver and its connection."""
    if saver is not None and hasattr(saver, 'conn') and saver.conn is not None:
        try:
            await saver.conn.aclose()
        except Exception:
            pass


def close_sync_saver(saver: PostgresSaver) -> None:
    """Close the sync saver and its connection."""
    if saver is not None and hasattr(saver, 'conn') and saver.conn is not None:
        try:
            saver.conn.close()
        except Exception:
            pass


async def setup_database() -> None:
    """Initialize the database schema (run once on startup)."""
    saver = await get_async_saver()
    try:
        await saver.setup()
    finally:
        await close_async_saver(saver)


def setup_database_sync() -> None:
    """Synchronous database schema initialization."""
    saver = get_sync_saver()
    try:
        saver.setup()
    finally:
        close_sync_saver(saver)
