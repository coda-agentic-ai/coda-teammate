"""MCP Research Server for market data retrieval."""

import json
from datetime import datetime, timezone
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("Research Server")


def _generate_mock_market_data(ticker: str) -> dict[str, Any]:
    """Generate mock market data for a given ticker."""
    # Generate deterministic but realistic-looking data based on ticker
    ticker_hash = hash(ticker.upper())
    base_price = 100 + (ticker_hash % 400)  # Price between 100-500
    change = ((ticker_hash % 200) - 100) / 10  # Change between -10 to +10

    return {
        "ticker": ticker.upper(),
        "company_name": f"{ticker.upper()} Corporation",
        "price": round(base_price + change, 2),
        "change": round(change, 2),
        "change_percent": round((change / base_price) * 100, 2),
        "market_cap": f"${(ticker_hash % 3000) + 50}B",
        "pe_ratio": round(15 + (ticker_hash % 30) / 10, 1),
        "volume": f"{(ticker_hash % 100) + 1}M",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool
def fetch_market_data(ticker: str) -> str:
    """Fetch real-time market data for a given stock ticker.

    Args:
        ticker: The stock ticker symbol (e.g., AAPL, GOOGL, MSFT)

    Returns:
        JSON string containing market data including price, change, market cap, etc.
    """
    data = _generate_mock_market_data(ticker)
    return json.dumps(data, indent=2)


def _generate_mock_contacts(role: str) -> dict[str, str]:
    """Generate mock contact information for testing PII scrubbing."""
    contacts = {
        "lead_analyst": {
            "name": "Jane Smith",
            "email": "jane.smith@analyst-firm.com",
            "phone": "555-123-4567",
            "title": "Senior Lead Analyst",
            "firm": "Alpha Research Partners",
        },
        "support": {
            "name": "Tech Support",
            "email": "support@teammate-suite.io",
            "phone": "800-555-0100",
            "title": "Support Team",
            "firm": "Teammate Suite",
        },
        "executive": {
            "name": "John C. Executive",
            "email": "john.executive@fortune500.co",
            "phone": "212-555-9999",
            "title": "Chief Strategy Officer",
            "firm": "Global Corp",
        },
    }
    return contacts.get(role, {"error": f"Role '{role}' not found"})


@mcp.tool
def fetch_contact_info(role: str) -> str:
    """Fetch contact information for a role from mock data.

    This tool is used for testing PII detection and scrubbing capabilities.

    Args:
        role: The role to fetch contact for (e.g., 'lead_analyst', 'support', 'executive')

    Returns:
        JSON string containing contact information including email and phone (PII).
    """
    data = _generate_mock_contacts(role)
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    mcp.run()
