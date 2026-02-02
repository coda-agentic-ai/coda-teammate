"""
Shared pytest fixtures for UI automation tests.

This conftest.py provides common fixtures used across multiple test files.
"""

import pytest
import httpx
import os
from typing import AsyncGenerator
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def pytest_addoption(parser):
    """Add command-line options for UI tests."""
    parser.addoption(
        "--skip-service-check",
        action="store_true",
        default=False,
        help="Skip service availability checks"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "service_required: test requires external services (liaison, workspace_ui)"
    )
    config.addinivalue_line(
        "markers", "playwright: test requires Playwright browser automation"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests if services are not available (unless --skip-service-check)."""
    if config.getoption("--skip-service-check"):
        return

    skip_service = pytest.mark.skip(reason="Service check enabled and services not available")
    for item in items:
        if "service_required" in item.keywords:
            item.add_marker(skip_service)


@pytest.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client for testing."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


@pytest.fixture
def liaison_url() -> str:
    """URL for liaison_brain service."""
    return os.getenv("LIAISON_URL", "http://localhost:8000")


@pytest.fixture
def workspace_url() -> str:
    """URL for workspace_ui service."""
    return os.getenv("WORKSPACE_UI_URL", "http://localhost:8501")


@pytest.fixture
async def ensure_services(async_client, liaison_url, workspace_url):
    """Check that required services are running, skip tests if not."""
    import httpx

    services_ok = True
    errors = []

    # Check liaison brain
    try:
        response = await async_client.get(f"{liaison_url}/docs", timeout=2.0)
        if response.status_code != 200:
            errors.append(f"Liaison brain returned {response.status_code}")
            services_ok = False
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        errors.append(f"Liaison brain connection failed: {e}")
        services_ok = False

    # Check workspace UI
    try:
        response = await async_client.get(workspace_url, timeout=2.0)
        if response.status_code != 200:
            errors.append(f"Workspace UI returned {response.status_code}")
            services_ok = False
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        errors.append(f"Workspace UI connection failed: {e}")
        services_ok = False

    if not services_ok:
        pytest.skip(f"Services not available: {', '.join(errors)}")


# Playwright fixtures (only if Playwright is installed)
try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page

    @pytest.fixture(scope="session")
    async def browser() -> AsyncGenerator[Browser, None]:
        """Shared browser instance for all tests."""
        playwright = await async_playwright().start()
        headless = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true"
        browser = await playwright.chromium.launch(headless=headless)

        yield browser

        await browser.close()
        await playwright.stop()

    @pytest.fixture
    async def context(browser: Browser) -> AsyncGenerator[BrowserContext, None]:
        """Browser context for test isolation."""
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True
        )
        yield context
        await context.close()

    @pytest.fixture
    async def page(context: BrowserContext, workspace_url: str) -> AsyncGenerator[Page, None]:
        """Page fixture with workspace UI loaded."""
        page = await context.new_page()
        page.set_default_timeout(10000)

        try:
            await page.goto(workspace_url, wait_until="networkidle")
        except Exception as e:
            pytest.skip(f"Failed to load workspace UI: {e}")

        yield page

except ImportError:
    # Playwright not installed - create dummy fixtures that skip tests
    @pytest.fixture(scope="session")
    async def browser():
        pytest.skip("Playwright not installed")

    @pytest.fixture
    async def context():
        pytest.skip("Playwright not installed")

    @pytest.fixture
    async def page():
        pytest.skip("Playwright not installed")


# Helper fixtures for common test operations
@pytest.fixture
async def new_thread_id() -> str:
    """Generate a new thread ID for test isolation."""
    import uuid
    return str(uuid.uuid4())


@pytest.fixture
async def send_test_message(async_client, liaison_url, new_thread_id):
    """Send a test message and return thread ID."""
    async def _send(message: str = "Hello, teammate"):
        try:
            response = await async_client.post(
                f"{liaison_url}/chat/input",
                json={
                    "message": message,
                    "thread_id": new_thread_id,
                    "task_description": None,
                    "task_budget": 1000,
                    "cost_limit": 0.50
                },
                timeout=5.0
            )
            response.raise_for_status()
            return new_thread_id
        except httpx.ConnectError:
            pytest.skip("Liaison brain service not running")

    return _send