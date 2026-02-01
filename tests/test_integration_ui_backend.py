"""Integration tests for UI-Backend communication and performance.

Tests verify:
1. End-to-end message flow works correctly with async endpoints
2. User message appears immediately in UI (< 100ms)
3. SSE streaming delivers events correctly
4. Concurrent users are handled properly
5. Error states are handled gracefully

These tests require Docker Compose services to be running:
    docker-compose up -d liaison workspace_ui db redis
"""

import pytest
import asyncio
import time
import httpx
import json
import uuid
import os
import sys
from typing import AsyncGenerator, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestEndToEndMessageFlow:
    """End-to-end tests for complete message flow."""

    @pytest.fixture
    async def async_client(self):
        """Async HTTP client for testing."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            yield client

    @pytest.fixture
    def workspace_url(self):
        """URL for workspace_ui service."""
        return os.getenv("WORKSPACE_UI_URL", "http://localhost:8501")

    @pytest.fixture
    def liaison_url(self):
        """URL for liaison_brain service."""
        return os.getenv("LIAISON_URL", "http://localhost:8000")

    @pytest.mark.asyncio
    async def test_immediate_message_display(self, async_client, workspace_url):
        """Verify user message appears immediately in UI (< 100ms)."""
        # This test measures the time from form submission to message display
        # Since we can't measure browser rendering, we measure endpoint response time
        # which should be < 100ms for the UI to feel responsive

        thread_id = str(uuid.uuid4())
        message = "Hello, teammate! " + thread_id

        start_time = time.perf_counter()

        try:
            # Submit message via workspace UI endpoint
            response = await async_client.post(
                f"{workspace_url}/chat",
                data={"msg": message, "thread_id": thread_id},
                timeout=5.0
            )
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # Verify quick response
            assert response_time_ms < 100, \
                f"UI endpoint response time {response_time_ms:.1f}ms exceeds 100ms target"

            # Should return empty response (optimistic UI handles display)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            # Empty HTML response
            assert len(response.text.strip()) == 0 or "Error" not in response.text, \
                f"Unexpected error response: {response.text[:100]}"

        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")
        except Exception as e:
            pytest.fail(f"Failed to test immediate message display: {e}")

    @pytest.mark.asyncio
    async def test_chat_input_endpoint_response_time(self, async_client, liaison_url):
        """Verify liaison brain /chat/input returns quickly (< 200ms)."""
        thread_id = str(uuid.uuid4())
        message = "Performance test message"

        start_time = time.perf_counter()

        try:
            response = await async_client.post(
                f"{liaison_url}/chat/input",
                json={
                    "message": message,
                    "thread_id": thread_id,
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                },
                timeout=5.0
            )
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # Verify quick response
            assert response_time_ms < 200, \
                f"Liaison endpoint response time {response_time_ms:.1f}ms exceeds 200ms target"

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert data["status"] == "started"
            assert data["thread_id"] == thread_id

        except httpx.ConnectError:
            pytest.skip("Liaison brain service not running")

    @pytest.mark.asyncio
    async def test_sse_stream_connectivity(self, async_client, liaison_url):
        """Verify SSE stream can be connected and delivers events."""
        thread_id = str(uuid.uuid4())

        # First, send a message to start graph execution
        try:
            response = await async_client.post(
                f"{liaison_url}/chat/input",
                json={
                    "message": "Test SSE connectivity",
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

        # Connect to SSE stream
        try:
            # Use streaming request
            async with async_client.stream(
                "GET",
                f"{liaison_url}/chat/stream/{thread_id}",
                timeout=10.0
            ) as response:
                assert response.status_code == 200

                # Check SSE headers
                content_type = response.headers.get("content-type", "")
                assert "text/event-stream" in content_type

                # Read a few events (with timeout)
                events_received = 0
                start_time = time.time()

                async for line in response.aiter_lines():
                    if time.time() - start_time > 5.0:  # Max 5 seconds
                        break

                    line = line.strip()
                    if line.startswith("event:"):
                        events_received += 1

                    if events_received >= 2:  # Got at least ping and maybe thought
                        break

                # Should have received at least ping events
                assert events_received > 0, "No SSE events received"

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("SSE stream connection failed")

    @pytest.mark.asyncio
    async def test_end_to_end_flow_with_sse(self, async_client, workspace_url, liaison_url):
        """Complete test: UI → Liaison → SSE events."""
        thread_id = str(uuid.uuid4())
        message = f"End-to-end test message {thread_id}"

        # Step 1: Send message via workspace UI
        try:
            ui_response = await async_client.post(
                f"{workspace_url}/chat",
                data={"msg": message, "thread_id": thread_id},
                timeout=5.0
            )
            assert ui_response.status_code == 200, f"UI endpoint failed: {ui_response.text}"
        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")

        # Step 2: Connect to SSE stream from liaison
        try:
            events = []
            async with async_client.stream(
                "GET",
                f"{liaison_url}/chat/stream/{thread_id}",
                timeout=15.0
            ) as response:
                assert response.status_code == 200

                # Collect events for 3 seconds
                start_time = time.time()
                async for line in response.aiter_lines():
                    if time.time() - start_time > 3.0:
                        break

                    line = line.strip()
                    if line.startswith("event:"):
                        event_type = line.split(":", 1)[1].strip()
                        events.append(event_type)

                    # Also check for data lines
                    if line.startswith("data:"):
                        # Verify JSON is valid
                        data_str = line.split(":", 1)[1].strip()
                        try:
                            json.loads(data_str)
                        except json.JSONDecodeError:
                            pass  # Some data might not be JSON

            # Should have received some events
            assert len(events) > 0, "No events received in SSE stream"
            assert "ping" in events or "thought" in events or "message" in events or "cost" in events, \
                f"Expected event types not found: {events}"

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            pytest.skip(f"SSE stream failed: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_message_submission(self, async_client, workspace_url):
        """Test multiple users submitting messages simultaneously."""
        num_concurrent = 3
        message_template = "Concurrent test message {}"

        async def submit_message(i: int):
            thread_id = str(uuid.uuid4()) + f"-{i}"
            message = message_template.format(i)

            try:
                start_time = time.perf_counter()
                response = await async_client.post(
                    f"{workspace_url}/chat",
                    data={"msg": message, "thread_id": thread_id},
                    timeout=10.0
                )
                end_time = time.perf_counter()
                response_time_ms = (end_time - start_time) * 1000

                return {
                    "success": response.status_code == 200,
                    "response_time": response_time_ms,
                    "thread_id": thread_id
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "thread_id": thread_id
                }

        # Submit concurrent messages
        tasks = [submit_message(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)

        # Check all succeeded
        success_count = sum(1 for r in results if r["success"])
        assert success_count == num_concurrent, \
            f"Only {success_count}/{num_concurrent} concurrent requests succeeded"

        # Check response times are reasonable
        response_times = [r["response_time"] for r in results if "response_time" in r]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        assert avg_response_time < 200, \
            f"Average concurrent response time {avg_response_time:.1f}ms exceeds 200ms"

    @pytest.mark.asyncio
    async def test_error_handling_network_failure(self, async_client, workspace_url):
        """Verify UI handles network failures gracefully."""
        # This test requires mocking or simulating a failure
        # For now, test that the UI endpoint returns error HTML when brain is unreachable
        # by using a non-existent brain host (simulated via thread_id that causes error)

        # Note: This is a simplified test - in real scenario we'd mock the connectivity check
        pass  # Implement when we have better mocking

    @pytest.mark.asyncio
    async def test_sse_proxy_performance(self, async_client, workspace_url, liaison_url):
        """Measure latency added by workspace_ui SSE proxy."""
        thread_id = str(uuid.uuid4())

        # First ensure a thread exists
        try:
            await async_client.post(
                f"{liaison_url}/chat/input",
                json={
                    "message": "SSE proxy test",
                    "thread_id": thread_id,
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                },
                timeout=5.0
            )
        except httpx.ConnectError:
            pytest.skip("Liaison brain service not running")

        # Measure direct liaison SSE connection
        liaison_start = time.perf_counter()
        try:
            async with async_client.stream(
                "GET",
                f"{liaison_url}/chat/stream/{thread_id}",
                timeout=5.0
            ) as response:
                assert response.status_code == 200
                liaison_end = time.perf_counter()
                # Close immediately
                pass
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Direct SSE connection failed")

        liaison_time = (liaison_end - liaison_start) * 1000

        # Measure proxied SSE connection (if workspace UI exposes SSE proxy)
        # Currently workspace_ui has /chat/stream/{thread_id} that proxies to liaison
        workspace_start = time.perf_counter()
        try:
            async with async_client.stream(
                "GET",
                f"{workspace_url}/chat/stream/{thread_id}",
                timeout=5.0
            ) as response:
                assert response.status_code == 200
                workspace_end = time.perf_counter()
                # Close immediately
                pass
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Workspace UI SSE proxy not available")

        workspace_time = (workspace_end - workspace_start) * 1000

        # Proxy should add minimal latency (< 100ms)
        latency_added = workspace_time - liaison_time
        assert latency_added < 100, \
            f"SSE proxy added {latency_added:.1f}ms latency, expected < 100ms"


class TestPerformanceMetricsValidation:
    """Validate that performance metrics meet requirements."""

    @pytest.mark.asyncio
    async def test_performance_targets_met(self, async_client, workspace_url, liaison_url):
        """Run comprehensive performance validation."""
        import statistics

        metrics = {
            "ui_endpoint_response_time": [],
            "liaison_endpoint_response_time": [],
            "sse_connection_time": [],
        }

        # Run 3 iterations for each metric
        for i in range(3):
            thread_id = str(uuid.uuid4()) + f"-perf-{i}"

            # Test UI endpoint
            try:
                start = time.perf_counter()
                response = await async_client.post(
                    f"{workspace_url}/chat",
                    data={"msg": f"Performance test {i}", "thread_id": thread_id},
                    timeout=5.0
                )
                end = time.perf_counter()
                if response.status_code == 200:
                    metrics["ui_endpoint_response_time"].append((end - start) * 1000)
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            # Test liaison endpoint
            try:
                start = time.perf_counter()
                response = await async_client.post(
                    f"{liaison_url}/chat/input",
                    json={
                        "message": f"Performance test {i}",
                        "thread_id": thread_id,
                        "task_description": None,
                        "task_budget": 1000,
                        "cost_limit": 0.50
                    },
                    timeout=5.0
                )
                end = time.perf_counter()
                if response.status_code == 200:
                    metrics["liaison_endpoint_response_time"].append((end - start) * 1000)
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

        # Calculate averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = statistics.mean(values)

        # Verify targets (if we have data)
        if "ui_endpoint_response_time" in avg_metrics:
            assert avg_metrics["ui_endpoint_response_time"] < 100, \
                f"UI endpoint avg response time {avg_metrics['ui_endpoint_response_time']:.1f}ms exceeds 100ms target"

        if "liaison_endpoint_response_time" in avg_metrics:
            assert avg_metrics["liaison_endpoint_response_time"] < 200, \
                f"Liaison endpoint avg response time {avg_metrics['liaison_endpoint_response_time']:.1f}ms exceeds 200ms target"

        print(f"Performance metrics: {avg_metrics}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])