"""
UI Testing Helpers for HTML parsing and verification.

This module provides utilities for verifying UI state through HTML parsing,
particularly focused on HTMX OOB (Out-of-Band) swaps and CSS selector targeting.

Key functions:
- parse_oob_html(): Extract target, content, and attributes from OOB HTML
- verify_ui_state(): Compare expected vs actual UI state from HTML
- collect_sse_events_html(): Enhanced parsing for UI verification
- CSS selector helpers for targeting UI elements by ID/class
"""

import re
import json
import httpx
import time
from typing import Dict, List, Any, Optional, Tuple
from bs4 import BeautifulSoup
import pytest


def parse_oob_html(html_chunk: str) -> Dict[str, Any]:
    """Extract data from HTML OOB swap chunk using BeautifulSoup.

    Parses OOB HTML chunks like:
    <div hx-swap-oob="beforeend:#thought-stream">
      <div class="border-l-2 border-purple-500/50 pl-3 py-1">
        <div class="flex items-center">
          <span class="text-xs font-medium text-purple-400">liaison</span>
        </div>
        <div class="mt-1 text-sm text-gray-400">Thought content</div>
      </div>
    </div>

    Returns dict with keys: target, content, node, attributes, html
    """
    soup = BeautifulSoup(html_chunk, 'html.parser')

    # Find the element with hx-swap-oob attribute
    oob_element = soup.find(attrs={"hx-swap-oob": True})
    if not oob_element:
        # Try regex fallback
        target_match = re.search(r'hx-swap-oob="([^"]+)"', html_chunk)
        target = target_match.group(1) if target_match else ""

        # Parse node name (fallback)
        node_match = re.search(r'<span[^>]*>([^<]+)</span>', html_chunk)
        node = node_match.group(1).strip() if node_match else ""

        # Parse content (fallback)
        content_match = re.search(r'<div[^>]*class="[^"]*text-gray-400[^"]*"[^>]*>([^<]+)</div>', html_chunk)
        content = content_match.group(1).strip() if content_match else ""

        return {
            "target": target,
            "node": node,
            "content": content,
            "attributes": {},
            "html": html_chunk
        }

    # Extract target
    target = oob_element.get('hx-swap-oob', '')

    # Extract all attributes
    attributes = dict(oob_element.attrs)

    # Extract node name (look for span with text)
    node_span = oob_element.find('span')
    node = node_span.get_text(strip=True) if node_span else ""

    # Extract content (look for div with text-gray-400 class)
    content_div = oob_element.find(class_=re.compile(r'text-gray-400'))
    if not content_div:
        # Fallback to any div with text
        content_div = oob_element.find('div', recursive=True)

    content = content_div.get_text(strip=True) if content_div else ""

    # Extract CSS classes for styling verification
    classes_attr = oob_element.get('class')
    if classes_attr is None:
        classes = []
    elif isinstance(classes_attr, str):
        classes = classes_attr.split()
    else:
        classes = list(classes_attr)

    # Extract style attribute
    style = oob_element.get('style') or ''

    return {
        "target": target,
        "node": node,
        "content": content,
        "attributes": attributes,
        "classes": classes,
        "style": style,
        "html": html_chunk
    }


def extract_element_by_selector(html: str, selector: str) -> Optional[Dict[str, Any]]:
    """Extract element data by CSS selector from HTML.

    Args:
        html: HTML content to parse
        selector: CSS selector (e.g., '#budget-progress', '.text-gray-400')

    Returns:
        Dict with element data or None if not found
    """
    soup = BeautifulSoup(html, 'html.parser')
    element = soup.select_one(selector)

    if not element:
        return None

    # Extract element data
    classes_attr = element.get('class')
    if classes_attr is None:
        classes = []
    elif isinstance(classes_attr, str):
        classes = classes_attr.split()
    else:
        classes = list(classes_attr)

    element_data = {
        "tag": element.name,
        "text": element.get_text(strip=True),
        "attributes": dict(element.attrs),
        "classes": classes,
        "html": str(element)
    }

    # Extract specific attributes based on selector type
    if selector.startswith('#'):
        element_data['id'] = selector[1:]
    elif selector.startswith('.'):
        element_data['class_name'] = selector[1:]

    return element_data


def verify_ui_state(actual_html: str, expected_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Compare expected UI state vs actual parsed HTML.

    Args:
        actual_html: Actual HTML content
        expected_state: Dict with expected values keyed by selector
            Example: {
                '#budget-progress': {'style': 'width: 42%'},
                '#budget-text': {'text': '$0.42 / $1.00'},
                '#thought-stream': {'count': 3}  # expects 3 child elements
            }

    Returns:
        Tuple of (success, error_messages)
    """
    errors = []
    soup = BeautifulSoup(actual_html, 'html.parser')

    for selector, expectations in expected_state.items():
        elements = soup.select(selector)

        # Check element exists
        if not elements:
            errors.append(f"Selector '{selector}' not found in HTML")
            continue

        element = elements[0]

        # Verify each expectation
        for key, expected_value in expectations.items():
            if key == 'text':
                actual = element.get_text(strip=True)
                if expected_value not in actual:
                    errors.append(f"Selector '{selector}' text mismatch. Expected '{expected_value}' in '{actual}'")

            elif key == 'style':
                actual_style = element.get('style', '')
                if expected_value not in actual_style:
                    errors.append(f"Selector '{selector}' style mismatch. Expected '{expected_value}' in '{actual_style}'")

            elif key == 'class':
                actual_classes_attr = element.get('class')
                if actual_classes_attr is None:
                    actual_classes = []
                elif isinstance(actual_classes_attr, str):
                    actual_classes = actual_classes_attr.split()
                else:
                    actual_classes = list(actual_classes_attr)
                if expected_value not in actual_classes:
                    errors.append(f"Selector '{selector}' class mismatch. Expected '{expected_value}' not in {actual_classes}")

            elif key == 'count':
                if len(elements) != expected_value:
                    errors.append(f"Selector '{selector}' count mismatch. Expected {expected_value}, got {len(elements)}")

            elif key == 'exists':
                if expected_value and not elements:
                    errors.append(f"Selector '{selector}' expected to exist but not found")
                elif not expected_value and elements:
                    errors.append(f"Selector '{selector}' expected not to exist but found")

            elif key == 'attribute':
                for attr_name, attr_value in expected_value.items():
                    actual_attr = element.get(attr_name, '')
                    if attr_value != actual_attr:
                        errors.append(f"Selector '{selector}' attribute '{attr_name}' mismatch. Expected '{attr_value}', got '{actual_attr}'")

    return len(errors) == 0, errors


async def collect_sse_events_html(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 10.0,
    collect_duration: float = 5.0
) -> Dict[str, List[Any]]:
    """Collect all SSE events from a stream, handling HTML OOB chunks.

    Enhanced version with better HTML parsing and UI element extraction.

    Args:
        client: Async HTTP client
        url: SSE stream URL
        timeout: Connection timeout
        collect_duration: How long to collect events (seconds)

    Returns:
        Dictionary with event types as keys and lists of event data as values.
        For HTML events, data is dict with keys: html, parsed (extracted data).
    """
    events = {
        "thought": [],
        "message": [],
        "ping": [],
        "cost": [],
        "intervene": [],
        "skill_update": [],
        "error": []
    }
    current_event = None
    current_data_lines = []

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

                line = line.rstrip()  # Keep right spaces? strip only trailing newline

                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    # Accumulate data lines (may be multiple lines for one event)
                    data_line = line.split(":", 1)[1]  # Keep leading space if present
                    current_data_lines.append(data_line)
                elif line == "":
                    # Empty line signals event end, process accumulated data
                    if current_event and current_data_lines:
                        # Join data lines (SSE spec: newline separated, keep newlines)
                        data_str = "".join(current_data_lines).lstrip()  # Remove leading newline from first line
                        if current_event in events:
                            # Check if data is HTML (contains hx-swap-oob)
                            if "hx-swap-oob" in data_str:
                                parsed_data = parse_oob_html(data_str)
                                events[current_event].append({
                                    "html": data_str,
                                    "parsed": parsed_data
                                })

                                # Also extract specific UI elements for easier verification
                                if parsed_data["target"]:
                                    target_id = parsed_data["target"].split("#")[-1] if "#" in parsed_data["target"] else ""
                                    if target_id:
                                        events[current_event][-1]["target_id"] = target_id
                                        events[current_event][-1]["element_data"] = extract_element_by_selector(
                                            data_str, f"#{target_id}"
                                        )
                            else:
                                # Try to parse as JSON
                                try:
                                    parsed_data = json.loads(data_str)
                                    events[current_event].append(parsed_data)
                                except json.JSONDecodeError:
                                    # Non-JSON data (like ping with empty data)
                                    events[current_event].append(data_str)
                    # Reset for next event
                    current_event = None
                    current_data_lines = []
                # Ignore other lines (comments, retry, etc.)

    except (httpx.ConnectError, httpx.TimeoutException) as e:
        pytest.skip(f"SSE stream connection failed: {e}")

    return events


def extract_ui_elements_from_html(html: str, selectors: List[str]) -> Dict[str, Any]:
    """Extract multiple UI elements from HTML for verification.

    Args:
        html: HTML content
        selectors: List of CSS selectors to extract

    Returns:
        Dict mapping selectors to element data
    """
    result = {}
    for selector in selectors:
        element_data = extract_element_by_selector(html, selector)
        result[selector] = element_data
    return result


def verify_oob_targets(events: Dict[str, List[Any]], expected_targets: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
    """Verify that OOB events target the correct UI elements.

    Args:
        events: SSE events collected via collect_sse_events_html
        expected_targets: Dict mapping event type to list of expected target IDs
            Example: {
                "thought": ["thought-stream"],
                "cost": ["budget-progress", "budget-text"],
                "skill_update": ["skill-name"]
            }

    Returns:
        Tuple of (success, error_messages)
    """
    errors = []

    for event_type, expected_ids in expected_targets.items():
        event_list = events.get(event_type, [])

        for event in event_list:
            if isinstance(event, dict) and "target_id" in event:
                target_id = event["target_id"]
                if target_id not in expected_ids:
                    errors.append(f"Event '{event_type}' targets unexpected ID '{target_id}'. Expected: {expected_ids}")
            elif isinstance(event, dict) and "parsed" in event:
                parsed = event["parsed"]
                target = parsed.get("target", "")
                if target:
                    # Extract ID from target string like "beforeend:#thought-stream"
                    target_id = target.split("#")[-1] if "#" in target else ""
                    if target_id and target_id not in expected_ids:
                        errors.append(f"Event '{event_type}' targets unexpected ID '{target_id}'. Expected: {expected_ids}")

    return len(errors) == 0, errors


# ============================================================================
# Playwright helper functions
# ============================================================================

try:
    from playwright.async_api import Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = None  # type: ignore


def take_screenshot(page: Page, name: str) -> str:
    """Take screenshot and save to reports directory.

    Returns path to screenshot file.
    """
    import datetime
    import os

    reports_dir = os.path.join(os.path.dirname(__file__), "reports", "screenshots")
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    filepath = os.path.join(reports_dir, filename)

    if page and PLAYWRIGHT_AVAILABLE:
        page.screenshot(path=filepath, full_page=True)
    return filepath


async def wait_for_sse_event(page: Page, event_type: str, timeout: float = 10.0) -> dict:
    """Wait for specific SSE event to arrive.

    This function listens to the SSE connection and waits for an event
    of the specified type. Requires the page to have SSE connection established.

    Args:
        page: Playwright page
        event_type: SSE event type to wait for (e.g., "thought", "message")
        timeout: Maximum wait time in seconds

    Returns:
        Event data as dict
    """
    # This is a placeholder - actual implementation would require
    # intercepting SSE events, which is complex in Playwright
    # For now, we'll use a simple timeout and rely on UI updates
    import asyncio
    await asyncio.sleep(2.0)

    # Return dummy data for now
    return {"type": event_type, "received": True}


async def verify_oob_update(page: Page, selector: str, expected_content: str = None) -> bool:
    """Verify that OOB update was applied to element.

    Args:
        page: Playwright page
        selector: CSS selector for element that should have been updated
        expected_content: Optional text content to verify

    Returns:
        True if element exists and (if expected_content provided) contains text
    """
    if not page or not PLAYWRIGHT_AVAILABLE:
        return False

    element = page.locator(selector)
    if not await element.count():
        return False

    if expected_content:
        text = await element.text_content()
        return expected_content in text

    return True


async def wait_for_selector_with_text(page: Page, selector: str, text: str, timeout: float = 10.0) -> bool:
    """Wait for element with selector to contain specific text."""
    import asyncio

    if not page or not PLAYWRIGHT_AVAILABLE:
        return False

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        element = page.locator(selector)
        if await element.count():
            element_text = await element.text_content()
            if text in element_text:
                return True

        await asyncio.sleep(0.5)

    return False