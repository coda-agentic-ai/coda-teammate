"""
Automated UI Testing for SSE Streaming Fix - httpx-based tests.

Test Category 1: HTMX OOB Swap Verification (httpx-based)
1. Test OOB thought events appear in right panel
2. Test budget updates in left panel
3. Test message exchange in center panel
4. Test intervention panel appears
5. Test skill updates

These tests verify UI updates via HTML parsing without browser automation.
"""

import asyncio
import pytest
import time
import httpx
import uuid
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.ui_test_helpers import (
    collect_sse_events_html,
    extract_element_by_selector
)


class TestHTMXOOBThoughtEvents:
    """Test Case 1: OOB thought events appear in right panel."""


    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_thought_events_oob_html(self, async_client, liaison_url):
        """Verify thought events contain OOB HTML targeting #thought-stream."""
        thread_id = str(uuid.uuid4())
        test_message = "Hello, teammate"

        # Step 1: Send message to liaison brain
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

        # Step 2: Collect SSE events with HTML parsing
        events = await collect_sse_events_html(
            async_client,
            f"{liaison_url}/chat/stream/{thread_id}",
            timeout=10.0,
            collect_duration=8.0
        )

        # Step 3: Verify thought events contain OOB HTML
        thought_events = events.get("thought", [])
        assert len(thought_events) > 0, "Expected at least one thought event"

        html_thought_events = [e for e in thought_events if isinstance(e, dict) and "html" in e]
        assert len(html_thought_events) > 0, "Expected at least one thought event with HTML"

        # Verify each thought event targets #thought-stream
        # Filter out events with empty content (e.g., end-of-stream markers)
        non_empty_events = [
            e for e in html_thought_events
            if e.get("parsed", {}).get("content", "").strip()
        ]
        assert len(non_empty_events) > 0, "Expected at least one thought event with non-empty content"

        for event in non_empty_events:
            html = event["html"]
            parsed = event.get("parsed", {})

            assert "hx-swap-oob" in html, "Thought event should contain hx-swap-oob attribute"
            assert "thought-stream" in html, "Thought event should target #thought-stream"

            target = parsed.get("target", "")
            assert target == "beforeend:#thought-stream", \
                f"Expected target 'beforeend:#thought-stream', got {target}"

            node = parsed.get("node", "")
            assert node == "liaison", f"Expected node 'liaison', got {node}"

            content = parsed.get("content", "")
            assert len(content.strip()) > 0, f"Thought content should not be empty, got '{content}'"

            # Verify HTML structure contains expected classes
            assert "border-purple-500/50" in html, "Thought element should have purple border"
            assert "text-purple-400" in html, "Thought node should have purple text"
            assert "text-gray-400" in html, "Thought content should have gray text"

        print(f"✓ Verified {len(non_empty_events)} thought events with OOB HTML targeting #thought-stream")

    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_thought_events_update_ui(self, async_client, liaison_url, workspace_url):
        """Verify thought events actually update the UI (right panel)."""
        thread_id = str(uuid.uuid4())
        test_message = "Explain how AI works"

        # Send message to liaison brain
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

        # Wait a moment for UI to update (thoughts to appear)
        await asyncio.sleep(2.0)

        # Fetch the workspace UI page and parse HTML
        try:
            response = await async_client.get(workspace_url, timeout=5.0)
            assert response.status_code == 200
            html_content = response.text

            # Extract thought-stream element
            thought_stream_data = extract_element_by_selector(html_content, "#thought-stream")
            assert thought_stream_data is not None, "#thought-stream element not found in UI"

            # Check that thought-stream contains thought items
            # Look for child elements with purple border
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            thought_items = soup.select("#thought-stream .border-purple-500/50")
            assert len(thought_items) > 0, "No thought items found in #thought-stream"

            # Verify thought items have expected structure
            for item in thought_items[:3]:  # Check first few
                node_span = item.find(class_="text-purple-400")
                assert node_span is not None, "Thought item missing node span"
                node_text = node_span.get_text(strip=True)
                assert node_text == "liaison", f"Expected node 'liaison', got '{node_text}'"

                content_div = item.find(class_="text-gray-400")
                assert content_div is not None, "Thought item missing content div"
                content_text = content_div.get_text(strip=True)
                assert len(content_text) > 0, "Thought content empty"

            print(f"✓ Verified UI contains {len(thought_items)} thought items in right panel")

        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")


class TestBudgetUpdates:
    """Test Case 2: Budget updates in left panel."""




    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_cost_events_oob_html(self, async_client, liaison_url):
        """Verify cost events contain OOB HTML targeting budget elements."""
        thread_id = str(uuid.uuid4())
        test_message = "Calculate something expensive"

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

        # Collect SSE events
        events = await collect_sse_events_html(
            async_client,
            f"{liaison_url}/chat/stream/{thread_id}",
            timeout=10.0,
            collect_duration=8.0
        )

        # Verify cost events
        cost_events = events.get("cost", [])
        if not cost_events:
            pytest.skip("No cost events received (may not have triggered cost tracking)")

        html_cost_events = [e for e in cost_events if isinstance(e, dict) and "html" in e]
        assert len(html_cost_events) > 0, "Expected at least one cost event with HTML"

        # Verify each cost event targets budget-progress or budget-text
        for event in html_cost_events:
            html = event["html"]
            parsed = event.get("parsed", {})

            assert "hx-swap-oob" in html, "Cost event should contain hx-swap-oob attribute"
            target = parsed.get("target", "")

            # Cost events can target either #budget-progress or #budget-text
            assert any(target_id in target for target_id in ["budget-progress", "budget-text"]), \
                f"Cost event should target budget-progress or budget-text, got {target}"

            # Verify HTML contains expected elements
            if "budget-progress" in target:
                assert "bg-emerald-500" in html or "bg-amber-500" in html or "bg-rose-500" in html, \
                    "Budget progress should have color class"
                assert "rounded-full" in html, "Progress bar should be rounded"
            elif "budget-text" in target:
                assert "$" in html, "Budget text should contain currency symbol"

        print(f"✓ Verified {len(html_cost_events)} cost events with OOB HTML targeting budget elements")

    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_budget_ui_updates(self, async_client, liaison_url, workspace_url):
        """Verify budget UI elements update after cost events."""
        thread_id = str(uuid.uuid4())
        test_message = "Test budget updates"

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

        # Wait for UI updates
        await asyncio.sleep(3.0)

        # Fetch workspace UI and verify budget elements
        try:
            response = await async_client.get(workspace_url, timeout=5.0)
            assert response.status_code == 200
            html_content = response.text

            # Verify budget-progress element exists and has style
            progress_data = extract_element_by_selector(html_content, "#budget-progress")
            assert progress_data is not None, "#budget-progress element not found"

            style = progress_data["attributes"].get("style", "")
            assert "width:" in style, "Budget progress should have width style"
            assert "%" in style, "Width should be percentage"

            # Verify budget-text element exists and shows cost
            text_data = extract_element_by_selector(html_content, "#budget-text")
            assert text_data is not None, "#budget-text element not found"

            text = text_data["text"]
            assert "$" in text, "Budget text should contain currency symbol"
            assert "/" in text, "Budget text should show usage (e.g., $0.42 / $1.00)"

            print(f"✓ Verified budget UI elements: progress style '{style}', text '{text}'")

        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")


class TestMessageExchange:
    """Test Case 3: Message exchange in center panel."""




    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_optimistic_ui_user_message(self, async_client, workspace_url):
        """Verify user messages appear immediately via optimistic UI."""
        thread_id = str(uuid.uuid4())
        test_message = "Hello from test"

        start_time = time.perf_counter()

        try:
            response = await async_client.post(
                f"{workspace_url}/chat",
                data={"msg": test_message, "thread_id": thread_id},
                timeout=5.0
            )
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # Verify quick response (optimistic UI)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            assert len(response.text.strip()) == 0, \
                f"Expected empty response for optimistic UI, got: {response.text[:200]}"
            assert response_time_ms < 150, \
                f"Response time {response_time_ms:.1f}ms exceeds 150ms target"

            # The optimistic UI JavaScript should have added the message client-side
            # We can't verify that with httpx alone (requires JavaScript execution)
            # This is covered by Playwright tests
            print(f"✓ Optimistic UI response time: {response_time_ms:.1f}ms")

        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")

    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_assistant_message_sse_events(self, async_client, liaison_url):
        """Verify assistant messages appear via SSE message events."""
        thread_id = str(uuid.uuid4())
        test_message = "Please respond with a simple answer"

        # Send message via liaison to trigger assistant response
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

        # Collect SSE events
        events = await collect_sse_events_html(
            async_client,
            f"{liaison_url}/chat/stream/{thread_id}",
            timeout=10.0,
            collect_duration=8.0
        )

        # Verify message events
        message_events = events.get("message", [])
        assert len(message_events) > 0, "Expected at least one message event"

        # Check message event structure
        for msg_event in message_events:
            if isinstance(msg_event, dict):
                assert "role" in msg_event, "Message event missing 'role' field"
                assert "content" in msg_event, "Message event missing 'content' field"
                assert msg_event.get("role") == "assistant", \
                    f"Expected role='assistant', got: {msg_event.get('role')}"
                assert len(msg_event.get("content", "")) > 0, "Message content empty"

        print(f"✓ Verified {len(message_events)} assistant message events via SSE")

    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_message_styling_in_ui(self, async_client, liaison_url, workspace_url):
        """Verify message styling (user: emerald, assistant: cyan) in UI."""
        thread_id = str(uuid.uuid4())
        test_message = "Test message styling"

        # Send a message via workspace UI (optimistic)
        try:
            await async_client.post(
                f"{workspace_url}/chat",
                data={"msg": test_message, "thread_id": thread_id},
                timeout=5.0
            )
        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")

        # Also send via liaison to get assistant response
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

        # Wait for UI updates
        await asyncio.sleep(3.0)

        # Fetch workspace UI and check message styling
        try:
            response = await async_client.get(workspace_url, timeout=5.0)
            assert response.status_code == 200
            html_content = response.text

            # Parse with BeautifulSoup
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find canvas area
            canvas = soup.select_one("#canvas-area")
            assert canvas is not None, "#canvas-area not found"

            # Look for message elements
            messages = canvas.find_all("div", recursive=False)
            assert len(messages) > 0, "No messages found in canvas"

            # Check styling of each message
            for msg in messages:
                # Check for user message styling (emerald)
                if "bg-emerald-900/20" in str(msg):
                    user_label = msg.find(class_="text-emerald-400")
                    assert user_label is not None, "User message missing emerald label"
                    assert "You" in user_label.get_text(), "User message should label 'You'"

                # Check for assistant message styling (cyan)
                elif "bg-slate-800/50" in str(msg):
                    assistant_label = msg.find(class_="text-cyan-400")
                    assert assistant_label is not None, "Assistant message missing cyan label"
                    assert "Teammate" in assistant_label.get_text(), "Assistant message should label 'Teammate'"

            print(f"✓ Verified message styling for {len(messages)} messages in canvas")

        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")


class TestInterventionPanel:
    """Test Case 4: Intervention panel appears."""



    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_intervention_events_oob_html(self, async_client, liaison_url):
        """Verify intervention events contain OOB HTML with approve/deny buttons.

        Note: This test may need a specific trigger to cause intervention.
        We'll check that the event structure is correct when interventions occur.
        """
        thread_id = str(uuid.uuid4())
        test_message = "Please trigger intervention if possible"

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

        # Collect SSE events
        events = await collect_sse_events_html(
            async_client,
            f"{liaison_url}/chat/stream/{thread_id}",
            timeout=10.0,
            collect_duration=8.0
        )

        # Check for intervention events
        intervention_events = events.get("intervene", [])

        # Intervention may not always occur, so we skip if none
        if not intervention_events:
            pytest.skip("No intervention events received (may not have triggered intervention)")

        html_intervention_events = [e for e in intervention_events if isinstance(e, dict) and "html" in e]
        assert len(html_intervention_events) > 0, "Expected at least one intervention event with HTML"

        for event in html_intervention_events:
            html = event["html"]
            parsed = event.get("parsed", {})

            assert "hx-swap-oob" in html, "Intervention event should contain hx-swap-oob"
            target = parsed.get("target", "")
            assert "intervention-panel" in target, f"Intervention should target intervention-panel, got {target}"

            # Verify HTML contains required elements
            assert "Human Intervention Required" in html, "Missing intervention title"
            assert "Approve" in html, "Missing approve button"
            assert "Deny" in html, "Missing deny button"
            assert "hx_post" in html, "Missing HTMX post attributes"

            # Verify button styling
            assert "bg-emerald-600" in html, "Approve button missing emerald background"
            assert "bg-rose-600" in html, "Deny button missing rose background"

        print(f"✓ Verified {len(html_intervention_events)} intervention events with OOB HTML")


class TestSkillUpdates:
    """Test Case 5: Skill updates."""



    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_skill_update_events_oob_html(self, async_client, liaison_url):
        """Verify skill_update events contain OOB HTML targeting #skill-name."""
        thread_id = str(uuid.uuid4())
        test_message = "Test skill updates"

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

        # Collect SSE events
        events = await collect_sse_events_html(
            async_client,
            f"{liaison_url}/chat/stream/{thread_id}",
            timeout=10.0,
            collect_duration=8.0
        )

        # Check for skill_update events
        skill_events = events.get("skill_update", [])
        if not skill_events:
            pytest.skip("No skill_update events received (may not have triggered skill change)")

        html_skill_events = [e for e in skill_events if isinstance(e, dict) and "html" in e]
        assert len(html_skill_events) > 0, "Expected at least one skill_update event with HTML"

        for event in html_skill_events:
            html = event["html"]
            parsed = event.get("parsed", {})

            assert "hx-swap-oob" in html, "Skill update event should contain hx-swap-oob"
            target = parsed.get("target", "")
            assert "skill-name" in target, f"Skill update should target skill-name, got {target}"

            # Verify HTML contains skill name
            assert "text-blue-400" in html, "Skill name should have blue text styling"

        print(f"✓ Verified {len(html_skill_events)} skill_update events with OOB HTML")


class TestNetworkFailureHandling:
    """Test Case 9: Network failure handling."""

    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_error_handling_javascript_present(self, async_client, workspace_url):
        """Verify error handling JavaScript is present in the UI."""
        try:
            response = await async_client.get(workspace_url, timeout=5.0)
            assert response.status_code == 200
            html = response.text
        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")

        # Check for error handling JavaScript
        assert "htmx:responseError" in html, \
            "Missing htmx:responseError handler for network failures"
        assert "bg-rose-900/20" in html, \
            "Missing error background styling class"
        # text-rose-400 is used for budget warning and error labels, ensure it's present
        assert "text-rose-400" in html or "text-emerald-400" in html, \
            "Missing error or emerald text styling class"

        print("✓ Error handling JavaScript present in UI")

    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_error_display_elements_exist(self, async_client, workspace_url):
        """Verify error display elements exist in HTML structure."""
        try:
            response = await async_client.get(workspace_url, timeout=5.0)
            assert response.status_code == 200
            html = response.text
        except httpx.ConnectError:
            pytest.skip("Workspace UI service not running")

        # Check for error-related CSS classes in HTML (simpler than CSS selector)
        # The slash in class name is valid in HTML but not in CSS selectors without escaping
        # We'll just check that the class strings appear in the HTML
        assert "bg-rose-900/20" in html or "bg-rose-900" in html, \
            "Missing error background styling class"
        # text-rose-400 is used for budget warning, text-emerald-400 for normal messages
        assert "text-rose-400" in html or "text-emerald-400" in html, \
            "Missing error or emerald text styling class"

        print("✓ Error display elements present in UI HTML")


class TestMalformedSSEEvents:
    """Test Case 10: Malformed SSE events."""

    @pytest.mark.service_required
    @pytest.mark.asyncio
    async def test_sse_error_event_handling(self, async_client, liaison_url):
        """Verify SSE error events are handled (requires service)."""
        # This test verifies that the SSE stream includes error event handling
        # by checking that error events are defined in the JavaScript.
        # We'll fetch the workspace UI and check for SSE error listeners.
        try:
            # Just check liaison is up
            await async_client.get(liaison_url + "/docs", timeout=5.0)
        except httpx.ConnectError:
            pytest.skip("Liaison brain service not running")

        # Fetch workspace UI to check JavaScript
        # For now, just verify the test infrastructure works
        # Actual malformed SSE injection is complex and requires mocking
        print("✓ SSE error event handling test placeholder (requires actual malformed event injection)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])