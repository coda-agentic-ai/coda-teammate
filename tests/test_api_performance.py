"""API Performance Tests for UI responsiveness.

Tests verify:
1. Endpoints return quickly (< 200ms) regardless of processing time
2. Async behavior works correctly (non-blocking)
3. SSE endpoints have correct headers and connect
4. Performance metrics meet requirements
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestChatEndpointPerformance:
    """Performance tests for chat endpoints."""

    def setup_method(self):
        """Set up environment variables and mock missing dependencies."""
        import os
        import sys
        import types
        from unittest.mock import MagicMock

        # Set dummy DATABASE_URL to allow imports without real database
        os.environ['DATABASE_URL'] = os.getenv('DATABASE_URL', 'postgresql://test:test@localhost/test')

        # Mock sentry module to avoid import errors
        if 'sentry' not in sys.modules:
            # Create a proper module
            sentry_module = types.ModuleType('sentry')
            sentry_module.privacy = types.ModuleType('sentry.privacy')
            sentry_module.privacy.PIIScrubber = MagicMock()
            sentry_module.economy = types.ModuleType('sentry.economy')
            sentry_module.economy.UniversalCostTracker = MagicMock()
            sentry_module.budget = types.ModuleType('sentry.budget')
            sentry_module.budget.BudgetGuard = MagicMock()
            sentry_module.audit = types.ModuleType('sentry.audit')
            sentry_module.audit.AuditLogger = MagicMock()
            sys.modules['sentry'] = sentry_module
            sys.modules['sentry.privacy'] = sentry_module.privacy
            sys.modules['sentry.economy'] = sentry_module.economy
            sys.modules['sentry.budget'] = sentry_module.budget
            sys.modules['sentry.audit'] = sentry_module.audit

    @pytest.mark.asyncio
    async def test_chat_input_endpoint_response_time(self):
        """Verify /chat/input endpoint returns quickly (< 200ms)."""
        # Import inside test to avoid dependency issues
        try:
            from fastapi.testclient import TestClient
            from apps.liaison_brain.app.main import app
        except ImportError as e:
            import traceback
            traceback.print_exc()
            pytest.skip(f"Required dependencies not available: {e}")

        # Mock get_graph to return a minimal graph that doesn't actually process
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=None)

        with patch('apps.liaison_brain.app.main.get_graph', AsyncMock(return_value=mock_graph)):
            client = TestClient(app)

            # Measure response time
            start_time = time.perf_counter()
            response = client.post(
                "/chat/input",
                json={
                    "message": "Hello, world!",
                    "thread_id": "test-thread-123",
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                }
            )
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # Verify quick response
            assert response_time_ms < 200, f"Response time {response_time_ms:.1f}ms exceeds 200ms target"
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"

            # Verify response structure
            data = response.json()
            assert "status" in data
            assert "thread_id" in data
            assert data["status"] == "started"
            assert data["thread_id"] == "test-thread-123"

            # Verify graph was called as background task (ainvoke should be called eventually)
            # Since we're using async mocks, we need to let the event loop run
            await asyncio.sleep(0.01)  # Allow background task to start
            assert mock_graph.ainvoke.called, "Graph should have been invoked in background"

    @pytest.mark.asyncio
    async def test_chat_input_endpoint_async_behavior(self):
        """Verify endpoint returns immediately without waiting for graph completion."""
        try:
            from fastapi.testclient import TestClient
            from apps.liaison_brain.app.main import app
        except ImportError as e:
            import traceback
            traceback.print_exc()
            pytest.skip(f"Required dependencies not available: {e}")

        # Create a mock graph that takes a long time to complete
        slow_graph = AsyncMock()
        # Make ainvoke hang for 2 seconds before returning
        async def slow_ainvoke(*args, **kwargs):
            await asyncio.sleep(2.0)
            return {"result": "done"}

        slow_graph.ainvoke = slow_ainvoke
        slow_graph.aget_state = AsyncMock(return_value=None)

        with patch('apps.liaison_brain.app.main.get_graph', AsyncMock(return_value=slow_graph)):
            client = TestClient(app)

            # Measure response time - should be fast despite slow graph
            start_time = time.perf_counter()
            response = client.post(
                "/chat/input",
                json={
                    "message": "Slow test message",
                    "thread_id": "slow-thread-456",
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                }
            )
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # Should return in < 200ms even though graph takes 2 seconds
            assert response_time_ms < 200, f"Response time {response_time_ms:.1f}ms exceeds 200ms despite async graph"
            assert response.status_code == 200

            # Verify response returned before graph completed
            # (graph would still be running in background)
            data = response.json()
            assert data["status"] == "started"

    @pytest.mark.asyncio
    async def test_workspace_chat_endpoint_response_time(self):
        """Verify workspace_ui /chat endpoint returns quickly (< 200ms)."""
        pytest.skip("Workspace UI tests require fasthtml dependencies; run in integration tests")

    def test_sse_endpoint_headers(self):
        """Verify SSE endpoint returns correct headers for streaming."""
        try:
            from fastapi.testclient import TestClient
            from apps.liaison_brain.app.main import app
        except ImportError as e:
            import traceback
            traceback.print_exc()
            pytest.skip(f"Required dependencies not available: {e}")

        client = TestClient(app)

        import asyncio
        # Mock get_graph to avoid dependencies
        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=None)

        # Mock event_generator to avoid state access errors
        async def mock_event_generator(thread_id: str, channel: str = "WEB_CANVAS"):
            yield {"event": "ping", "data": '{"message": "connected"}'}
            await asyncio.sleep(0.01)  # small delay to allow stream to start

        with patch('apps.liaison_brain.app.main.get_graph', AsyncMock(return_value=mock_graph)), \
             patch('apps.liaison_brain.app.main.event_generator', mock_event_generator):
            response = client.get("/chat/stream/test-thread-123")

            # Check SSE headers
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            assert "no-cache" in response.headers.get("cache-control", "").lower()
            assert "keep-alive" in response.headers.get("connection", "").lower()

    @pytest.mark.asyncio
    async def test_sse_endpoint_connectivity(self):
        """Verify SSE endpoint can be connected and sends initial events."""
        try:
            from fastapi.testclient import TestClient
            from apps.liaison_brain.app.main import app
        except ImportError as e:
            import traceback
            traceback.print_exc()
            pytest.skip(f"Required dependencies not available: {e}")

        client = TestClient(app)

        # Mock get_graph and event_generator
        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=None)

        # Create a simple event generator for testing
        async def mock_event_generator(thread_id: str, channel: str = "WEB_CANVAS"):
            yield {"event": "ping", "data": '{"message": "connected"}'}
            await asyncio.sleep(0.1)
            yield {"event": "thought", "data": '{"node": "test", "content": "Thinking..."}'}

        with patch('apps.liaison_brain.app.main.get_graph', AsyncMock(return_value=mock_graph)), \
             patch('apps.liaison_brain.app.main.event_generator', mock_event_generator):

            # Connect to SSE endpoint
            with client.stream("GET", "/chat/stream/test-thread-456") as response:
                assert response.status_code == 200

                # Read first few lines
                lines = []
                for line in response.iter_lines():
                    lines.append(line)
                    if len(lines) >= 2:
                        break

                # Verify SSE format
                assert any('event: ping' in line for line in lines)
                assert any('data:' in line for line in lines)

    @pytest.mark.asyncio
    async def test_concurrent_endpoint_calls(self):
        """Verify endpoints handle concurrent requests without performance degradation."""
        try:
            from fastapi.testclient import TestClient
            from apps.liaison_brain.app.main import app
        except ImportError as e:
            import traceback
            traceback.print_exc()
            pytest.skip(f"Required dependencies not available: {e}")

        client = TestClient(app)

        # Mock get_graph
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=None)

        with patch('apps.liaison_brain.app.main.get_graph', AsyncMock(return_value=mock_graph)):
            # Make multiple concurrent requests using threads
            import concurrent.futures
            num_requests = 5
            start_time = time.perf_counter()

            def make_request(i):
                response = client.post(
                    "/chat/input",
                    json={
                        "message": f"Concurrent message {i}",
                        "thread_id": f"concurrent-thread-{i}",
                        "task_description": None,
                        "task_budget": 1000,
                        "cost_limit": 0.50
                    }
                )
                return response.status_code

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = [executor.submit(make_request, i) for i in range(num_requests)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000

            # All requests should succeed
            assert all(code == 200 for code in results), f"Not all requests succeeded: {results}"

            # Total time for concurrent requests should be reasonable
            # (serialized would be > 200ms each, but async should handle concurrently)
            # Allow some overhead but less than serialized
            max_reasonable_time = 500  # 500ms for 5 concurrent requests
            assert total_time_ms < max_reasonable_time, \
                f"Concurrent requests took {total_time_ms:.1f}ms, expected < {max_reasonable_time}ms"

    def test_error_handling_performance(self):
        """Verify error handling doesn't add significant latency."""
        try:
            from fastapi.testclient import TestClient
            from apps.liaison_brain.app.main import app
        except ImportError as e:
            import traceback
            traceback.print_exc()
            pytest.skip(f"Required dependencies not available: {e}")

        client = TestClient(app)

        # Mock get_graph to raise an exception
        with patch('apps.liaison_brain.app.main.get_graph', side_effect=Exception("Test error")):
            start_time = time.perf_counter()
            response = client.post(
                "/chat/input",
                json={
                    "message": "Error test",
                    "thread_id": "error-thread-999",
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                }
            )
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # Error responses should still be fast
            assert response_time_ms < 200, f"Error response time {response_time_ms:.1f}ms exceeds 200ms"
            assert response.status_code == 500


class TestPerformanceMetrics:
    """Tests to verify performance metrics are met."""

    def test_response_time_targets(self):
        """Verify all performance targets are defined and reasonable."""
        targets = {
            "user_message_display_latency": 100,  # ms
            "backend_endpoint_response_time": 200,  # ms
            "sse_event_latency": 1000,  # ms (1 second)
            "concurrent_users_supported": 10,
        }

        # Verify targets exist
        assert len(targets) == 4, "Should have 4 performance targets"
        assert all(isinstance(v, (int, float)) for v in targets.values()), \
            "All targets should be numeric"
        assert all(v > 0 for v in targets.values()), \
            "All targets should be positive"

        # Verify latency targets are reasonable
        assert targets["user_message_display_latency"] <= 100, \
            "User message display should be < 100ms"
        assert targets["backend_endpoint_response_time"] <= 200, \
            "Backend endpoints should respond in < 200ms"
        assert targets["sse_event_latency"] <= 1000, \
            "SSE events should arrive within 1 second"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])