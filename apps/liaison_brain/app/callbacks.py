"""Thought Stream Callbacks for LLM token streaming.

Provides callback handlers and registry functions for capturing LLM thought tokens
and making them available to the SSE event stream in main.py.
"""

import asyncio
from typing import AsyncGenerator, Optional

from langchain_core.callbacks.base import BaseCallbackHandler


# === Thread-safe Callback Registry ===
_callback_registries: dict[str, "ThoughtStreamCallback"] = {}
_registry_lock = asyncio.Lock()


async def register_thought_callback(thread_id: str, callback: "ThoughtStreamCallback") -> None:
    """Register a thought callback for a specific thread.

    If a callback already exists for this thread, it is replaced after signaling
    it to end (puts an end marker in its queue). This ensures clean callback
    lifecycle management when a new LLM call starts.

    Args:
        thread_id: The unique identifier for the conversation thread.
        callback: The ThoughtStreamCallback instance to register.
    """
    async with _registry_lock:
        # Signal any existing callback to end and remove it
        existing = _callback_registries.get(thread_id)
        if existing:
            # Signal end by putting None marker
            try:
                existing.queue.put_nowait((existing._node_name, None))
            except asyncio.QueueFull:
                pass
        _callback_registries[thread_id] = callback


async def get_thought_callback(thread_id: str) -> Optional["ThoughtStreamCallback"]:
    """Get the registered thought callback for a thread.

    Args:
        thread_id: The unique identifier for the conversation thread.

    Returns:
        The registered ThoughtStreamCallback, or None if not found.
    """
    async with _registry_lock:
        return _callback_registries.get(thread_id)


async def remove_thought_callback(thread_id: str) -> Optional["ThoughtStreamCallback"]:
    """Remove and return the registered thought callback for a thread.

    Args:
        thread_id: The unique identifier for the conversation thread.

    Returns:
        The removed ThoughtStreamCallback, or None if not found.
    """
    async with _registry_lock:
        return _callback_registries.pop(thread_id, None)


# === Thought Stream Callback ===

class ThoughtStreamCallback(BaseCallbackHandler):
    """LangChain callback handler for streaming LLM thought tokens.

    This callback captures tokens from `on_chat_model_stream` events and stores
    them in an asyncio.Queue for consumption by the SSE event stream.

    Usage:
        >>> callback = ThoughtStreamCallback(thread_id="abc123")
        >>> llm = ChatDeepSeek(callbacks=[callback])
        >>> await llm.ainvoke(messages)
        >>> # Tokens are now available via callback.queue
    """

    def __init__(self, thread_id: str):
        """Initialize the thought stream callback.

        Args:
            thread_id: The unique identifier for the conversation thread.
        """
        super().__init__()
        self.thread_id = thread_id
        self.queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()
        self._node_name: str = "liaison"
        print(f"[ThoughtStreamCallback] Created for thread={thread_id}")

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list,
        **kwargs,
    ) -> None:
        """Called when the chat model starts processing."""
        print(f"[ThoughtStreamCallback] LLM started for thread={self.thread_id}, node={self._node_name}")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called for each token chunk from the LLM stream.

        Args:
            token: The token string from the LLM stream.
        """
        print(f"[ThoughtStreamCallback] TOKEN: thread={self.thread_id}, node={self._node_name}, token='{token[:20]}...' (len={len(token)})")

        # Stream ALL content as thoughts (raw LLM output)
        # This captures internal reasoning and final response as it streams
        # The right panel shows the raw streaming output from the LLM

        if token:
            # Put (node_name, token) into the queue
            try:
                self.queue.put_nowait((self._node_name, token))
            except asyncio.QueueFull:
                # Queue is full, skip this token (shouldn't happen normally)
                pass

    def on_llm_end(self, response, **kwargs) -> None:
        """Called when the chat model finishes processing."""
        # Signal the end of the stream with None
        try:
            self.queue.put_nowait((self._node_name, ""))
        except asyncio.QueueFull:
            pass
        print(f"[ThoughtStreamCallback] LLM finished for thread={self.thread_id}, stream ending")

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        """Called when the chat model encounters an error."""
        # Signal error with special marker
        try:
            self.queue.put_nowait((self._node_name, f"[ERROR: {type(error).__name__}]"))
        except asyncio.QueueFull:
            pass
        print(f"[ThoughtStreamCallback] LLM error for thread={self.thread_id}: {error}")

    async def stream_thoughts(self) -> AsyncGenerator[tuple[str, str | None], None]:
        """Async generator that yields thought tokens from the queue.

        Yields:
            Tuples of (node_name, token_content). Token content is None when
            the stream should be terminated.

        Note:
            The generator will wait indefinitely for new tokens.
            The caller should check for None content to detect stream end.
        """
        while True:
            try:
                node_name, content = await self.queue.get()
                yield (node_name, content)
                # If content is empty, the stream has ended
                if not content:
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ThoughtStreamCallback] Stream error: {e}")
                break
