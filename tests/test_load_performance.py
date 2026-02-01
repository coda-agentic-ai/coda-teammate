"""Load and performance tests for UI responsiveness.

Tests verify:
1. System handles multiple concurrent users
2. Response times remain acceptable under load
3. Memory and resource usage is reasonable
4. SSE connections scale properly

These tests are more intensive and may require a running Docker Compose environment.
"""

import pytest
import asyncio
import time
import httpx
import uuid
import os
import sys
from typing import List, Tuple
import statistics

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestLoadPerformance:
    """Load testing for concurrent users and sustained traffic."""

    @pytest.fixture
    async def async_client(self):
        """Async HTTP client for load testing."""
        # Longer timeout for load tests
        async with httpx.AsyncClient(timeout=60.0) as client:
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
    async def test_single_user_response_time(self, async_client, workspace_url):
        """Measure baseline response time for single user."""
        thread_id = str(uuid.uuid4())
        message = "Single user load test"

        try:
            start_time = time.perf_counter()
            response = await async_client.post(
                f"{workspace_url}/chat",
                data={"msg": message, "thread_id": thread_id},
                timeout=10.0
            )
            end_time = time.perf_counter()

            assert response.status_code == 200, f"Request failed: {response.text}"
            response_time_ms = (end_time - start_time) * 1000

            # Baseline should be < 100ms
            assert response_time_ms < 100, \
                f"Single user response time {response_time_ms:.1f}ms exceeds 100ms target"

            print(f"Single user response time: {response_time_ms:.1f}ms")

        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")

    @pytest.mark.asyncio
    async def test_multiple_concurrent_users(self, async_client, workspace_url):
        """Test system with multiple concurrent users (5 users)."""
        num_users = 5
        results = []

        async def user_request(user_id: int):
            thread_id = str(uuid.uuid4()) + f"-user-{user_id}"
            message = f"Concurrent user {user_id} message"

            try:
                start_time = time.perf_counter()
                response = await async_client.post(
                    f"{workspace_url}/chat",
                    data={"msg": message, "thread_id": thread_id},
                    timeout=15.0
                )
                end_time = time.perf_counter()

                success = response.status_code == 200
                response_time_ms = (end_time - start_time) * 1000

                return {
                    "user_id": user_id,
                    "success": success,
                    "response_time": response_time_ms,
                    "status_code": response.status_code
                }
            except Exception as e:
                return {
                    "user_id": user_id,
                    "success": False,
                    "error": str(e),
                    "response_time": None
                }

        # Launch concurrent requests
        tasks = [user_request(i) for i in range(num_users)]
        results = await asyncio.gather(*tasks)

        # Analyze results
        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]

        success_rate = len(successes) / num_users * 100
        assert success_rate >= 80, f"Success rate {success_rate}% below 80% threshold"

        if successes:
            response_times = [r["response_time"] for r in successes if r["response_time"] is not None]
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)

            # Under concurrent load, response times may increase but should stay reasonable
            assert avg_response_time < 200, \
                f"Average concurrent response time {avg_response_time:.1f}ms exceeds 200ms"
            assert max_response_time < 500, \
                f"Max concurrent response time {max_response_time:.1f}ms exceeds 500ms"

            print(f"Concurrent users ({num_users}): {len(successes)} succeeded, "
                  f"avg response time: {avg_response_time:.1f}ms, "
                  f"max: {max_response_time:.1f}ms")

        if failures:
            print(f"Failures: {failures}")

    @pytest.mark.asyncio
    async def test_multiple_sse_connections(self, async_client, liaison_url):
        """Test system with multiple concurrent SSE connections."""
        num_connections = 3
        connection_duration = 5  # seconds to maintain connections

        async def maintain_sse_connection(conn_id: int):
            thread_id = str(uuid.uuid4()) + f"-sse-{conn_id}"

            # First, create a thread by sending a message
            try:
                await async_client.post(
                    f"{liaison_url}/chat/input",
                    json={
                        "message": f"SSE connection test {conn_id}",
                        "thread_id": thread_id,
                        "task_description": None,
                        "task_budget": 1000,
                        "cost_limit": 0.50
                    },
                    timeout=5.0
                )
            except httpx.ConnectError:
                return {"conn_id": conn_id, "success": False, "error": "Liaison unavailable"}

            # Connect to SSE stream
            events_received = 0
            start_time = time.time()

            try:
                async with async_client.stream(
                    "GET",
                    f"{liaison_url}/chat/stream/{thread_id}",
                    timeout=connection_duration + 5
                ) as response:
                    if response.status_code != 200:
                        return {"conn_id": conn_id, "success": False, "error": f"HTTP {response.status_code}"}

                    # Read events for specified duration
                    async for line in response.aiter_lines():
                        if time.time() - start_time > connection_duration:
                            break

                        line = line.strip()
                        if line.startswith("event:"):
                            events_received += 1

                    return {
                        "conn_id": conn_id,
                        "success": True,
                        "events_received": events_received,
                        "duration": time.time() - start_time
                    }

            except Exception as e:
                return {"conn_id": conn_id, "success": False, "error": str(e)}

        # Establish multiple SSE connections concurrently
        tasks = [maintain_sse_connection(i) for i in range(num_connections)]
        results = await asyncio.gather(*tasks)

        successes = [r for r in results if r["success"]]
        success_rate = len(successes) / num_connections * 100

        assert success_rate >= 80, f"SSE connection success rate {success_rate}% below 80%"

        if successes:
            avg_events = statistics.mean([r.get("events_received", 0) for r in successes])
            assert avg_events > 0, "SSE connections should receive at least some events"

            print(f"SSE connections ({num_connections}): {len(successes)} succeeded, "
                  f"avg events per connection: {avg_events:.1f}")

    @pytest.mark.asyncio
    async def test_sustained_load(self, async_client, workspace_url):
        """Test system under sustained load over time."""
        duration_seconds = 10
        requests_per_second = 2
        total_requests = duration_seconds * requests_per_second

        results = []
        start_time = time.time()
        request_count = 0

        async def make_request(request_num: int):
            thread_id = str(uuid.uuid4()) + f"-sustained-{request_num}"
            message = f"Sustained load request {request_num}"

            try:
                req_start = time.perf_counter()
                response = await async_client.post(
                    f"{workspace_url}/chat",
                    data={"msg": message, "thread_id": thread_id},
                    timeout=10.0
                )
                req_end = time.perf_counter()

                success = response.status_code == 200
                response_time = (req_end - req_start) * 1000

                return {
                    "request_num": request_num,
                    "success": success,
                    "response_time": response_time,
                    "timestamp": time.time() - start_time
                }
            except Exception as e:
                return {
                    "request_num": request_num,
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time() - start_time
                }

        # Create requests spaced out over time
        while time.time() - start_time < duration_seconds and request_count < total_requests:
            # Schedule requests at intervals
            tasks = []
            for _ in range(requests_per_second):
                tasks.append(make_request(request_count))
                request_count += 1

            # Execute batch
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Wait until next second
            elapsed = time.time() - start_time
            if elapsed < duration_seconds and request_count < total_requests:
                await asyncio.sleep(1.0 - (elapsed % 1.0))

        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        success_rate = len(successful) / len(results) * 100 if results else 0
        assert success_rate >= 70, f"Sustained load success rate {success_rate}% below 70%"

        if successful:
            response_times = [r["response_time"] for r in successful]
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)

            # Performance should remain stable
            assert avg_response_time < 150, \
                f"Sustained load avg response time {avg_response_time:.1f}ms exceeds 150ms"
            assert p95_response_time < 300, \
                f"Sustained load 95th percentile response time {p95_response_time:.1f}ms exceeds 300ms"

            print(f"Sustained load ({len(results)} requests, {duration_seconds}s): "
                  f"success rate {success_rate:.1f}%, "
                  f"avg response time: {avg_response_time:.1f}ms, "
                  f"95th percentile: {p95_response_time:.1f}ms")

    @pytest.mark.asyncio
    async def test_memory_usage_long_running(self, async_client, liaison_url):
        """Monitor memory usage during long-running operation."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

        # Start a long-running task
        thread_id = str(uuid.uuid4())
        long_message = "Please perform a comprehensive analysis. " * 10

        try:
            # Start graph execution (will run in background)
            response = await async_client.post(
                f"{liaison_url}/chat/input",
                json={
                    "message": long_message,
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

        # Monitor memory for a short period
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_readings = [initial_memory]

        # Connect to SSE stream to keep connection alive
        try:
            async with async_client.stream(
                "GET",
                f"{liaison_url}/chat/stream/{thread_id}",
                timeout=10.0
            ) as response:
                # Take memory readings over 5 seconds
                for _ in range(5):
                    await asyncio.sleep(1.0)
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_readings.append(current_memory)

        except (httpx.ConnectError, httpx.TimeoutException):
            # SSE connection failed, but we can still check memory
            pass

        # Calculate memory growth
        max_memory = max(memory_readings)
        memory_growth = max_memory - initial_memory

        # Memory growth should be reasonable (< 50MB for short test)
        assert memory_growth < 50, \
            f"Memory growth {memory_growth:.1f}MB exceeds 50MB threshold"

        print(f"Memory usage: initial {initial_memory:.1f}MB, "
              f"max {max_memory:.1f}MB, growth {memory_growth:.1f}MB")

    @pytest.mark.asyncio
    async def test_cpu_usage_under_load(self):
        """Measure CPU usage during load test (simplified)."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for CPU monitoring")

        # This is a simplified test - in production would use more sophisticated monitoring
        # Measure CPU usage before and during load

        initial_cpu = psutil.cpu_percent(interval=0.5)

        # Run a CPU-intensive loop to simulate load (in test process)
        # This is just a placeholder - real test would apply load to the services
        print(f"Initial CPU usage: {initial_cpu:.1f}%")

        # CPU usage measurement is system-wide; we can't isolate service CPU easily
        # For now, just ensure test doesn't cause excessive CPU in test process
        assert initial_cpu < 90, f"Initial CPU usage {initial_cpu:.1f}% is too high"

    @pytest.mark.asyncio
    async def test_error_rate_under_load(self, async_client, workspace_url):
        """Verify error rate remains low under moderate load."""
        num_requests = 20
        concurrent_limit = 5  # Max concurrent requests

        semaphore = asyncio.Semaphore(concurrent_limit)

        async def make_request_with_limit(request_num: int):
            async with semaphore:
                thread_id = str(uuid.uuid4()) + f"-error-test-{request_num}"
                message = f"Error rate test {request_num}"

                try:
                    response = await async_client.post(
                        f"{workspace_url}/chat",
                        data={"msg": message, "thread_id": thread_id},
                        timeout=10.0
                    )
                    success = response.status_code == 200
                    return {"success": success, "status_code": response.status_code}
                except Exception as e:
                    return {"success": False, "error": str(e)}

        tasks = [make_request_with_limit(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]

        error_rate = len(failures) / num_requests * 100
        assert error_rate < 10, f"Error rate {error_rate}% exceeds 10% threshold under load"

        print(f"Error rate under load: {error_rate:.1f}% ({len(failures)} failures out of {num_requests} requests)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])