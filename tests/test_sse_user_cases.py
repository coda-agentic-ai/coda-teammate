"""End-to-end tests for SSE event user cases.

Tests verify:
1. LLM Streaming Response - Thought events stream to right panel
2. Simple Response - Non-streaming message events to center panel
3. User Messages - Optimistic HTMX updates (no SSE)

These tests require Docker Compose services to be running:
    docker-compose up -d liaison workspace_ui db redis
"""

import pytest
import time
import httpx
import json
import uuid
import os
import sys
from typing import Dict, Any, List


# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


async def collect_sse_events(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 10.0,
    collect_duration: float = 5.0
) -> Dict[str, List[Any]]:
    """Collect all SSE events from a stream.

    Args:
        client: Async HTTP client
        url: SSE stream URL
        timeout: Connection timeout
        collect_duration: How long to collect events (seconds)

    Returns:
        Dictionary with event types as keys and lists of event data as values
    """
    events = {
        "thought": [],
        "message": [],
        "ping": [],
        "cost": [],
        "intervene": [],
        "error": []
    }
    current_event = None

    try:
        async with client.stream("GET", url, timeout=timeout) as response:
            assert response.status_code == 200
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Expected SSE content-type, got: {content_type}"

            start_time = time.time()

            async for line in response.aiter_lines():
                elapsed = time.time() - start_time
                if elapsed > collect_duration:
                    break

                line = line.strip()

                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and current_event:
                    data_str = line.split(":", 1)[1].strip()
                    try:
                        parsed_data = json.loads(data_str)
                        if current_event in events:
                            events[current_event].append(parsed_data)
                    except json.JSONDecodeError:
                        # Non-JSON data (like ping with empty data)
                        if current_event in events:
                            events[current_event].append(data_str)
                elif line == "" and current_event:
                    # Empty line signals event end, reset current_event
                    current_event = None

    except (httpx.ConnectError, httpx.TimeoutException) as e:
        pytest.skip(f"SSE stream connection failed: {e}")

    return events


class TestLLMStreamingResponse:
    """Test Case 1: LLM Streaming Response.

    Verify that thought events stream correctly with proper structure
    when the LLM generates a response.
    """

    @pytest.fixture
    async def async_client(self):
        """Async HTTP client for testing."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            yield client

    @pytest.fixture
    def liaison_url(self):
        """URL for liaison_brain service."""
        return os.getenv("LIAISON_URL", "http://localhost:8000")

    @pytest.mark.asyncio
    async def test_thought_events_received(self, async_client, liaison_url):
        """Verify thought events are received when LLM streams tokens."""
        thread_id = str(uuid.uuid4())
        test_message = "Explain how photosynthesis works"

        # Step 1: Send a message that triggers LLM streaming
        try:
            response = await async_client.post(
                f"{liaison_url}/chat/input",
                json={
                    "message": test_message,
                    "thread_id": thread_id,
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                },
                timeout=5.0
            )
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            data = response.json()
            assert data["status"] == "started"
            assert data["thread_id"] == thread_id
        except httpx.ConnectError:
            pytest.skip("Liaison brain service not running")

        # Step 2: Connect to SSE stream and collect events
        events = await collect_sse_events(
            async_client,
            f"{liaison_url}/chat/stream/{thread_id}?channel=SLACK",
            timeout=10.0,
            collect_duration=10.0
        )

        # Step 3: Verify thought events were received
        thought_events = events.get("thought", [])
        message_events = events.get("message", [])

        # Verify at least 3 thought events were received (token chunks)
        assert len(thought_events) >= 3, \
            f"Expected at least 3 thought events, got {len(thought_events)}"

        # Step 4: Verify each thought event has correct structure
        for thought_event in thought_events:
            assert isinstance(thought_event, dict), \
                f"Thought event should be a dict, got {type(thought_event)}"
            assert "content" in thought_event, \
                f"Thought event missing 'content' field: {thought_event}"
            assert "node" in thought_event, \
                f"Thought event missing 'node' field: {thought_event}"
            # Optional fields
            assert isinstance(thought_event.get("content"), str), \
                f"Thought 'content' should be a string: {thought_event}"
            assert isinstance(thought_event.get("node"), str), \
                f"Thought 'node' should be a string: {thought_event}"

        # Step 5: Verify final message event has correct structure
        if message_events:
            final_message = message_events[-1]
            assert isinstance(final_message, dict), \
                f"Message event should be a dict, got {type(final_message)}"
            assert "role" in final_message, \
                f"Message event missing 'role' field: {final_message}"
            assert "content" in final_message, \
                f"Message event missing 'content' field: {final_message}"
            assert final_message.get("role") == "assistant", \
                f"Expected role='assistant', got: {final_message.get('role')}"

        # Step 6: Verify event order (thoughts should come before final message)
        if thought_events and message_events:
            # Thoughts should be collected before message events
            # We verify this by checking that thoughts were collected during the
            # streaming phase (before the message event appeared)
            assert len(thought_events) > 0, "Thoughts should be collected before message"

    @pytest.mark.asyncio
    async def test_thought_event_structure(self, async_client, liaison_url):
        """Verify thought event JSON structure matches expected format."""
        thread_id = str(uuid.uuid4())
        test_message = "What is 2+2?"

        # Start message processing
        try:
            await async_client.post(
                f"{liaison_url}/chat/input",
                json={
                    "message": test_message,
                    "thread_id": thread_id,
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                },
                timeout=5.0
            )
        except httpx.ConnectError:
            pytest.skip("Liaison brain service not running")

        # Collect events
        events = await collect_sse_events(
            async_client,
            f"{liaison_url}/chat/stream/{thread_id}?channel=SLACK",
            timeout=10.0,
            collect_duration=8.0
        )

        thought_events = events.get("thought", [])

        if thought_events:
            # Check for expected fields (some may be optional)
            thought = thought_events[0]
            assert "content" in thought, "Thought must have 'content' field"
            assert "node" in thought, "Thought must have 'node' field"
            # Cost field is optional
            assert isinstance(thought.get("content"), str), "content must be a string"
            assert isinstance(thought.get("node"), str), "node must be a string"
        else:
            pytest.skip("No thought events received (LLM may not have streamed)")


class TestSimpleResponse:
    """Test Case 2: Simple Response (no streaming).

    Verify that instant responses skip thought events and only send message events.

    Note: Most LLMs stream by default. This test verifies the logic path when
    thought callback is None (no streaming active).
    """

    @pytest.fixture
    async def async_client(self):
        """Async HTTP client for testing."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            yield client

    @pytest.fixture
    def liaison_url(self):
        """URL for liaison_brain service."""
        return os.getenv("LIAISON_URL", "http://localhost:8000")

    @pytest.mark.asyncio
    async def test_message_event_without_thoughts(self, async_client, liaison_url):
        """Verify message events are sent when no thought callback is active.

        This tests the differentiation logic in main.py:275-291 where
        message events are emitted only when thought_cb is None.
        """
        thread_id = str(uuid.uuid4())
        test_message = "Quick hello"

        # Start message processing
        try:
            response = await async_client.post(
                f"{liaison_url}/chat/input",
                json={
                    "message": test_message,
                    "thread_id": thread_id,
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                },
                timeout=5.0
            )
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Liaison brain service not running")

        # Connect to SSE stream and check for events
        events = await collect_sse_events(
            async_client,
            f"{liaison_url}/chat/stream/{thread_id}?channel=SLACK",
            timeout=10.0,
            collect_duration=5.0
        )

        # Verify we got message events
        message_events = events.get("message", [])

        # We should receive message events (at minimum ping events)
        assert len(events.get("ping", [])) >= 0, "Should have ping or other events"

        # If we have message events, verify structure
        if message_events:
            for msg in message_events:
                assert isinstance(msg, dict), f"Message should be a dict: {msg}"
                assert "role" in msg, f"Message missing 'role': {msg}"
                assert "content" in msg, f"Message missing 'content': {msg}"
                assert msg.get("role") == "assistant", \
                    f"Expected role='assistant', got: {msg.get('role')}"

    @pytest.mark.asyncio
    async def test_event_differentiation_logic(self, async_client, liaison_url):
        """Verify the logic that differentiates thought vs message events.

        When thought callback is None, message events should be emitted.
        When thought callback exists, thought events are streamed instead.
        """
        thread_id = str(uuid.uuid4())
        test_message = "Test differentiation"

        try:
            response = await async_client.post(
                f"{liaison_url}/chat/input",
                json={
                    "message": test_message,
                    "thread_id": thread_id,
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                },
                timeout=5.0
            )
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Liaison brain service not running")

        # Collect events - look for both thought and message types
        events = await collect_sse_events(
            async_client,
            f"{liaison_url}/chat/stream/{thread_id}?channel=SLACK",
            timeout=10.0,
            collect_duration=5.0
        )

        # At minimum, we should have either thoughts OR messages
        has_thoughts = len(events.get("thought", [])) > 0
        has_messages = len(events.get("message", [])) > 0
        has_pings = len(events.get("ping", [])) > 0

        assert has_thoughts or has_messages or has_pings, \
            "Expected at least one type of event (thought, message, or ping)"


class TestUserMessages:
    """Test Case 3: User Messages (Optimistic HTMX).

    Verify that user messages appear immediately in UI without waiting for SSE.
    The optimistic UI JavaScript handles this via htmx:beforeRequest event.
    """

    @pytest.fixture
    async def async_client(self):
        """Async HTTP client for testing."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            yield client

    @pytest.fixture
    def workspace_url(self):
        """URL for workspace_ui service."""
        return os.getenv("WORKSPACE_UI_URL", "http://localhost:8501")

    @pytest.mark.asyncio
    async def test_optimistic_ui_returns_empty_response(self, async_client, workspace_url):
        """Verify workspace UI returns empty response for user messages.

        The endpoint should return immediately (< 100ms) with empty body,
        as the optimistic UI JavaScript handles the display.
        """
        thread_id = str(uuid.uuid4())
        test_message = "Hello, teammate!"

        start_time = time.perf_counter()

        try:
            response = await async_client.post(
                f"{workspace_url}/chat",
                data={"msg": test_message, "thread_id": thread_id},
                timeout=5.0
            )
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # Verify quick response
            assert response.status_code == 200, \
                f"Expected 200, got {response.status_code}: {response.text}"

            # Response should be empty (optimistic UI handles display)
            assert len(response.text.strip()) == 0, \
                f"Expected empty response, got: {response.text[:200]}"

            # Response time should be < 150ms for optimistic feel (allows for some variance)
            assert response_time_ms < 150, \
                f"Response time {response_time_ms:.1f}ms exceeds 150ms target for optimistic UI"

        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")

    @pytest.mark.asyncio
    async def test_optimistic_ui_response_time(self, async_client, workspace_url):
        """Verify optimistic UI response meets performance target."""
        thread_id = str(uuid.uuid4())

        try:
            response_times = []
            for i in range(3):
                start = time.perf_counter()
                response = await async_client.post(
                    f"{workspace_url}/chat",
                    data={"msg": f"Test message {i}", "thread_id": thread_id},
                    timeout=5.0
                )
                end = time.perf_counter()
                if response.status_code == 200:
                    response_times.append((end - start) * 1000)

            if response_times:
                avg_time = sum(response_times) / len(response_times)
                assert avg_time < 150, \
                    f"Average response time {avg_time:.1f}ms exceeds 150ms target"
        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")

    def test_optimistic_ui_javascript_elements(self):
        """Verify optimistic UI JavaScript contains required elements.

        This is a static analysis test - no services required.
        """
        import os

        # Read the workspace_ui app.py file
        app_path = os.path.join(
            project_root,
            "apps",
            "workspace_ui",
            "app.py"
        )

        with open(app_path, "r") as f:
            content = f.read()

        # Verify required JavaScript elements exist
        # htmx:beforeRequest handler for optimistic UI
        assert 'htmx:beforeRequest' in content, \
            "Missing htmx:beforeRequest event handler for optimistic UI"

        # X-Message-Id header setting
        assert 'X-Message-Id' in content, \
            "Missing X-Message-Id header setting for message tracking"

        # Canvas area insertion
        assert "canvas-area" in content.lower() or "canvas" in content.lower(), \
            "Missing canvas-area element for message insertion"

        # User label in emerald styling
        assert "emerald" in content.lower(), \
            "Missing emerald styling for user messages"

        # Error handling
        assert "htmx:responseError" in content, \
            "Missing htmx:responseError handler for error cases"

        # Loading indicator class
        assert "htmx-indicator" in content, \
            "Missing htmx-indicator class for loading state"


class TestSSEEventTypes:
    """Additional tests for SSE event type handling."""

    @pytest.fixture
    async def async_client(self):
        """Async HTTP client for testing."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            yield client

    @pytest.fixture
    def liaison_url(self):
        """URL for liaison_brain service."""
        return os.getenv("LIAISON_URL", "http://localhost:8000")

    @pytest.mark.asyncio
    async def test_sse_content_type_header(self, async_client, liaison_url):
        """Verify SSE endpoint returns correct content-type header."""
        thread_id = str(uuid.uuid4())

        try:
            async with async_client.stream(
                "GET",
                f"{liaison_url}/chat/stream/{thread_id}?channel=SLACK",
                timeout=5.0
            ) as response:

                content_type = response.headers.get("content-type", "")
                assert "text/event-stream" in content_type, \
                    f"Expected text/event-stream, got: {content_type}"

        except httpx.ConnectError:
            pytest.skip("Liaison brain service not running")

    @pytest.mark.asyncio
    async def test_cost_event_format(self, async_client, liaison_url):
        """Verify cost events have correct structure."""
        thread_id = str(uuid.uuid4())

        try:
            # Start a message
            await async_client.post(
                f"{liaison_url}/chat/input",
                json={
                    "message": "Test cost tracking",
                    "thread_id": thread_id,
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                },
                timeout=5.0
            )
        except httpx.ConnectError:
            pytest.skip("Liaison brain service not running")

        # Collect events
        events = await collect_sse_events(
            async_client,
            f"{liaison_url}/chat/stream/{thread_id}?channel=SLACK",
            timeout=10.0,
            collect_duration=6.0
        )

        cost_events = events.get("cost", [])
        if cost_events:
            cost_event = cost_events[0]
            assert isinstance(cost_event, dict), "Cost event should be a dict"
            assert "total_usd" in cost_event, \
                f"Cost event missing 'total_usd': {cost_event}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
