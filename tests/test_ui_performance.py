"""Unit tests for UI performance optimizations.

Tests verify:
1. Optimistic UI message display (JavaScript adds message immediately on form submission)
2. HTMX form attributes are correctly set for non-blocking behavior
3. SSE event handlers are properly configured
4. Loading indicators work correctly during async operations
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add project root to path to import app modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestOptimisticUIMessageDisplay:
    """Tests for optimistic UI JavaScript behavior."""

    def test_optimistic_ui_javascript_included(self):
        """Verify that the main_page HTML includes optimistic UI JavaScript."""
        # Import after path setup
        from apps.workspace_ui.app import main_page

        # Generate the HTML page
        html_content = str(main_page())

        # Check for key optimistic UI components
        assert 'htmx:beforeRequest' in html_content, "Optimistic UI event listener missing"
        assert 'insertAdjacentHTML' in html_content, "Dynamic message insertion missing"
        assert 'Sending...' in html_content, "Loading indicator text missing"
        assert 'X-Message-Id' in html_content, "Message ID tracking missing"

        # Check for error handling
        assert 'htmx:responseError' in html_content, "Error handling missing"
        assert 'Failed - try again' in html_content, "Error state handling missing"

    def test_htmx_form_attributes(self):
        """Verify HTMX attributes are correctly set for non-blocking behavior."""
        from apps.workspace_ui.app import main_page

        html_content = str(main_page())

        # Check for HTMX attributes on the form
        assert 'hx-post="/chat"' in html_content, "HTMX POST endpoint missing"
        assert 'hx-indicator' in html_content, "Loading indicator attribute missing"
        assert 'hx-swap' in html_content, "Swap strategy attribute missing"
        assert 'hx-target' in html_content or 'hx-swap="none"' in html_content, \
            "Target or swap-none missing"

        # Verify form has proper structure
        assert '<form' in html_content, "Form element missing"
        assert 'msg' in html_content, "Message input field missing"

    def test_sse_event_handlers(self):
        """Verify SSE event handlers are properly configured in JavaScript."""
        from apps.workspace_ui.app import main_page

        html_content = str(main_page())

        # Check for EventSource initialization
        assert 'EventSource(' in html_content, "EventSource initialization missing"
        assert '/chat/stream/' in html_content, "SSE endpoint URL missing"

        # Check for all required event listeners
        assert 'addEventListener(\'thought\'' in html_content, "Thought event handler missing"
        assert 'addEventListener(\'message\'' in html_content, "Message event handler missing"
        assert 'addEventListener(\'cost\'' in html_content, "Cost event handler missing"
        assert 'addEventListener(\'intervene\'' in html_content, "Intervene event handler missing"
        assert 'addEventListener(\'error\'' in html_content, "Error event handler missing"

        # Check for error handling on EventSource
        assert 'eventSource.onerror' in html_content, "EventSource error handler missing"

    def test_loading_indicator_configuration(self):
        """Verify loading indicators are properly configured."""
        from apps.workspace_ui.app import main_page

        html_content = str(main_page())

        # Check for HTMX indicator class
        assert 'htmx-indicator' in html_content, "HTMX indicator class missing"

        # Check for spinner elements
        assert 'animate-spin' in html_content, "Spinner animation class missing"

        # Check for specific loading indicator in optimistic UI
        assert 'Sending...' in html_content, "Sending text missing in optimistic UI"

        # Check that indicator is removed on successful response
        assert 'elem.querySelector(\'.htmx-indicator\').remove()' in html_content, \
            "Indicator removal logic missing"


class TestChatEndpointAsyncBehavior:
    """Tests for async behavior of chat endpoints."""

    @pytest.mark.asyncio
    async def test_chat_endpoint_returns_immediately(self):
        """Verify /chat endpoint returns quickly regardless of processing time."""
        # This test will be expanded after implementing async endpoints
        # For now, just verify the endpoint exists and has correct signature
        from apps.workspace_ui.app import chat_post
        import inspect

        # Check function signature
        sig = inspect.signature(chat_post)
        params = list(sig.parameters.keys())
        assert 'msg' in params, "Message parameter missing"
        assert 'thread_id' in params, "Thread ID parameter missing"

        # Note: Actual performance test will be in integration tests
        # with timing measurements

    def test_error_handling_in_optimistic_ui(self):
        """Verify error states are handled gracefully in optimistic UI."""
        from apps.workspace_ui.app import main_page

        html_content = str(main_page())

        # Check for error state handling in JavaScript
        assert 'text-rose-400' in html_content, "Error color class missing"
        assert 'bg-rose-900/20' in html_content, "Error background class missing"
        assert 'Failed - try again' in html_content, "Error message missing"

        # Check that error state updates existing optimistic message
        assert 'elem.querySelector(\'.text-emerald-400\').textContent = \'Error\'' in html_content, \
            "Error state update logic missing"


class TestSSEStreamConfiguration:
    """Tests for SSE stream configuration and performance."""

    def test_sse_endpoint_proxied_correctly(self):
        """Verify SSE endpoint properly proxies to liaison brain."""
        # Check that the SSE endpoint exists in the app
        from apps.workspace_ui.app import chat_stream
        import inspect

        sig = inspect.signature(chat_stream)
        params = list(sig.parameters.keys())
        assert 'thread_id' in params, "Thread ID parameter missing"

        # Function should exist and be callable (it returns EventSourceResponse)
        assert callable(chat_stream), "chat_stream should be callable"

    def test_sse_headers_correct(self):
        """Verify SSE endpoint returns correct headers for streaming."""
        # This will be tested in integration tests
        pass


class TestPerformanceRequirements:
    """Tests to verify performance requirements are met."""

    def test_no_synchronous_blocks_in_ui(self):
        """Verify UI JavaScript doesn't have synchronous blocking operations."""
        from apps.workspace_ui.app import main_page
        import re

        html_content = str(main_page())

        # Extract inline script content (not external scripts)
        # Look for <script> tags without src attribute
        script_pattern = r'<script(?:\s+[^>]*)?>(.*?)</script>'
        scripts = re.findall(script_pattern, html_content, re.DOTALL)

        # Filter out empty scripts and external scripts (those with src)
        inline_scripts = []
        for script in scripts:
            # Check if the script tag contains src attribute (external)
            # We can't easily correlate with regex capture, but we can check
            # if script content is minimal (like just whitespace) - assume external
            if script.strip() and len(script.strip()) > 10:
                inline_scripts.append(script)

        # Should have at least one inline script
        assert inline_scripts, "No inline JavaScript found in HTML"

        # Combine all inline scripts
        js_snippet = '\n'.join(inline_scripts)

        # Check for common blocking patterns
        blocking_patterns = [
            'while(true)',  # Infinite loops
            'alert(',  # Blocking dialogs (except for intervene)
            'confirm(',
            'prompt(',
            'syncRequest',  # Synchronous XMLHttpRequest
            '.send()',  # Without async flag
        ]

        for pattern in blocking_patterns:
            # alert is used for intervene events, which is acceptable
            if pattern == 'alert(':
                # Count alerts - should only be for intervene events
                alert_count = js_snippet.count('alert(')
                intervene_count = js_snippet.count('intervene')
                assert alert_count <= intervene_count, \
                    f"Unexpected alert() calls found: {alert_count}"
            elif pattern in js_snippet:
                # Other blocking patterns should not exist
                assert False, f"Blocking pattern found in UI JavaScript: {pattern}"

    def test_message_display_latency_targets(self):
        """Verify UI is configured to meet latency targets."""
        # Check that optimistic UI updates happen before request
        from apps.workspace_ui.app import main_page

        html_content = str(main_page())

        # Verify message is added on 'htmx:beforeRequest' (before network call)
        assert 'messageForm.addEventListener(\'htmx:beforeRequest\'' in html_content, \
            "Message should be added before request"

        # Verify loading indicators are shown immediately
        assert 'Sending...' in html_content, \
            "Loading indicator should show immediately"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])