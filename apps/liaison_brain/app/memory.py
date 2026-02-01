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
import asyncio


# Module-level lock for serializing ALL database operations
# This prevents concurrent access issues with PostgreSQL connections
_DB_LOCK = asyncio.Lock()


class SerializedAsyncPostgresSaver(AsyncPostgresSaver):
    """AsyncPostgresSaver with serialized access using a module-level lock."""
    async def aget_tuple(self, *args, **kwargs):
        async with _DB_LOCK:
            return await super().aget_tuple(*args, **kwargs)

    async def aset_tuple(self, *args, **kwargs):
        async with _DB_LOCK:
            return await super().aset_tuple(*args, **kwargs)

    async def aget_state(self, *args, **kwargs):
        async with _DB_LOCK:
            return await super().aget_state(*args, **kwargs)

    async def aset_state(self, *args, **kwargs):
        async with _DB_LOCK:
            return await super().aset_state(*args, **kwargs)

    async def alist(self, *args, **kwargs):
        async with _DB_LOCK:
            return await super().alist(*args, **kwargs)

    async def adelete(self, *args, **kwargs):
        async with _DB_LOCK:
            return await super().adelete(*args, **kwargs)

    async def asetup(self, *args, **kwargs):
        async with _DB_LOCK:
            return await super().asetup(*args, **kwargs)

    async def aclose(self, *args, **kwargs):
        async with _DB_LOCK:
            return await super().aclose(*args, **kwargs)


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
    import traceback
    print(f"[DEBUG] get_async_saver called from {traceback.extract_stack()[-2].name}")
    from psycopg import AsyncConnection
    import asyncio

    max_retries = 3
    base_delay = 1.0  # seconds

    for attempt in range(max_retries):
        try:
            # Create connection and keep it open
            conn = await AsyncConnection.connect(
                DATABASE_URL, autocommit=True
            )
            print(f"Database connection established (attempt {attempt + 1}/{max_retries})")
            return SerializedAsyncPostgresSaver(conn=conn, serde=None)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to connect to database after {max_retries} attempts: {e}")
                raise
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            print(f"Database connection failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)


def get_sync_saver() -> PostgresSaver:
    """Get synchronous PostgresSaver with persistent connection.

    Creates a persistent connection that won't close until explicitly closed.
    Use close_sync_saver() to cleanup when done.
    """
    import time

    max_retries = 3
    base_delay = 1.0  # seconds

    for attempt in range(max_retries):
        try:
            conn = sync_connect(DATABASE_URL, autocommit=True)
            print(f"Sync database connection established (attempt {attempt + 1}/{max_retries})")
            return PostgresSaver(conn=conn)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to connect to database synchronously after {max_retries} attempts: {e}")
                raise
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            print(f"Sync database connection failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)


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
