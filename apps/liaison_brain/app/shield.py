"""Sentry Shield - Security wrapper for LLM responses.

Provides PII scrubbing, cost tracking, and intervention detection
for the Liaison Brain streaming pipeline.
"""

import json
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler

from sentry.privacy import PIIScrubber
from sentry.economy import UniversalCostTracker


class SentryShieldCallback(BaseCallbackHandler):
    """Callback handler that wraps LLM responses with PII and cost checks.

    This callback:
    1. Tracks cost accumulation from LLM responses
    2. Detects PII violations in LLM output
    3. Updates state with privacy violations and intervention flags

    Usage:
        >>> from apps.liaison_brain.app.shield import SentryShieldCallback
        >>> callback = SentryShieldCallback(state_ref, model_name="anthropic/claude-3-5-sonnet")
        >>> llm = ChatAnthropic(callbacks=[callback])
    """

    def __init__(self, state_ref: dict, model_name: str):
        """Initialize the shield callback.

        Args:
            state_ref: A mutable dict reference to the graph state.
                      The callback will update this dict in-place.
            model_name: The model identifier for cost calculation.
        """
        super().__init__()
        self.state_ref = state_ref
        self.model_name = model_name
        self._scrubber = PIIScrubber()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM finishes generating a response.

        Updates cost and checks for PII violations.
        """
        # Calculate and accumulate cost
        cost = UniversalCostTracker.calculate_cost(response, self.model_name)
        current_total = self.state_ref.get("total_cost", 0.0)
        self.state_ref["total_cost"] = current_total + cost

        # Record cost history
        cost_record = {
            "model": self.model_name,
            "cost_usd": cost,
            "total_cost_usd": self.state_ref["total_cost"],
        }
        cost_history = self.state_ref.get("cost_history", [])
        cost_history.append(cost_record)
        self.state_ref["cost_history"] = cost_history

        # Check for PII violations in the response content
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                _, has_violation = self._scrubber.scrub_with_violation_report(content)
                if has_violation:
                    self.state_ref["privacy_violation"] = True
                    self.state_ref["intervention_reason"] = "PII detected in LLM response"
            elif isinstance(content, list):
                # Handle structured content (e.g., tool calls with content)
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        _, has_violation = self._scrubber.scrub_with_violation_report(item["text"])
                        if has_violation:
                            self.state_ref["privacy_violation"] = True
                            self.state_ref["intervention_reason"] = "PII detected in LLM response"
                            break


def scrub_text_for_stream(text: str) -> tuple[str, bool]:
    """Scrub text and report violation status for streaming.

    This is a convenience function for use in the SSE event stream
    to sanitize content before sending to clients.

    Args:
        text: The text content to scrub.

    Returns:
        A tuple of (scrubbed_text, has_violation).
    """
    scrubber = PIIScrubber()
    return scrubber.scrub_with_violation_report(text)


def format_thought_event(content: str, channel: str = "WEB_CANVAS") -> dict:
    """Format a thought chunk for the SSE stream.

    Args:
        content: The raw thought content from the LLM.
        channel: The target channel ("WEB_CANVAS" or "SLACK").

    Returns:
        A dict formatted for SSE emission.
    """
    scrubbed, has_violation = scrub_text_for_stream(content)

    if channel == "SLACK":
        # Slack gets aggregated thoughts - just return the scrubbed content
        return {
            "event": "thought",
            "data": json.dumps({"content": scrubbed, "complete": False})
        }

    # Web Canvas gets full JSON with violation flag
    return {
        "event": "thought",
        "data": json.dumps({
            "content": scrubbed,
            "complete": False,
            "pii_detected": has_violation
        })
    }


def format_cost_event(total_cost: float) -> dict:
    """Format a cost update event for the SSE stream.

    Args:
        total_cost: The current total cost in USD.

    Returns:
        A dict formatted for SSE emission.
    """
    return {
        "event": "cost",
        "data": json.dumps({
            "total_usd": round(total_cost, 4),
            "currency": "USD"
        })
    }


def format_intervene_event(
    reason: str,
    current_cost: float,
    requires_approval: bool = True
) -> dict:
    """Format an intervention event for the SSE stream.

    Args:
        reason: The reason for intervention (e.g., "budget exceeded", "PII detected").
        current_cost: The current accumulated cost.
        requires_approval: Whether human approval is required.

    Returns:
        A dict formatted for SSE emission.
    """
    return {
        "event": "intervene",
        "data": json.dumps({
            "reason": reason,
            "current_cost_usd": round(current_cost, 4),
            "requires_approval": requires_approval
        })
    }
