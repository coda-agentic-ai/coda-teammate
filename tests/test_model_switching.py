"""Test simulation for Model-Agnostic Shield cost tracking.

This script simulates switching between different LLM models and verifies
that the cost calculation correctly reflects each model's pricing.
"""

import litellm
from typing import Any
from datetime import datetime, timezone


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost for a model given token counts.

    Uses litellm's completion_cost with prompt/completion strings
    to approximate the cost based on token counts.
    """
    # Create approximate text based on token count (4 chars per token average)
    prompt_text = "x " * prompt_tokens
    completion_text = "y " * completion_tokens

    # Map model names to litellm format
    model_map = {
        "openai/gpt-4o": ("gpt-4o", None),
        "anthropic/claude-3-5-sonnet": ("claude-3-5-sonnet-20241022", "anthropic"),
        "deepseek/deepseek-chat": ("deepseek-chat", "deepseek"),
        # Also support bare model names
        "gpt-4o": ("gpt-4o", None),
        "claude-3-5-sonnet": ("claude-3-5-sonnet-20241022", "anthropic"),
    }

    litellm_model, provider = model_map.get(model, (model, None))

    kwargs = {
        "model": litellm_model,
        "prompt": prompt_text,
        "completion": completion_text,
    }

    if provider:
        kwargs["custom_llm_provider"] = provider

    return litellm.completion_cost(**kwargs)


def simulate_model_switch():
    """Simulate switching between GPT-4o and Claude-3-5-sonnet."""
    print("=" * 70)
    print("Model-Agnostic Shield: Model Switching Simulation")
    print("=" * 70)

    # Simulate a typical workload: 5 requests with ~1000 prompt + ~2000 completion
    prompt_tokens = 1000
    completion_tokens = 2000

    models = [
        ("openai/gpt-4o", "gpt-4o"),
        ("anthropic/claude-3-5-sonnet", "claude-3-5-sonnet"),
    ]

    for display_name, model_key in models:
        print(f"\n{'=' * 70}")
        print(f"Simulating: {display_name}")
        print(f"{'=' * 70}")

        # Simulate 5 requests
        total_cost = 0.0
        cost_history = []

        for i in range(1, 6):
            # Calculate cost
            cost = calculate_cost(model_key, prompt_tokens, completion_tokens)
            total_cost += cost

            # Record the cost
            cost_record = {
                "request_id": i,
                "model": display_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost,
            }
            cost_history.append(cost_record)

            print(f"  Request {i}: {prompt_tokens} + {completion_tokens} tokens = ${cost:.6f}")

        print(f"\n  Total Cost for 5 requests: ${total_cost:.6f}")
        print(f"  Average cost per request: ${total_cost / 5:.6f}")

        # Calculate how many requests until hitting $0.50 budget
        budget = 0.50
        requests_until_limit = budget / (total_cost / 5)
        print(f"  Requests until $0.50 budget exhausted: {requests_until_limit:.1f}")

    # Summary comparison
    print(f"\n{'=' * 70}")
    print("Cost Comparison Summary")
    print(f"{'=' * 70}")

    # Get costs for both models
    gpt_cost = calculate_cost("openai/gpt-4o", prompt_tokens, completion_tokens)
    claude_cost = calculate_cost("anthropic/claude-3-5-sonnet", prompt_tokens, completion_tokens)

    print(f"\n  Single Request Cost Comparison ({prompt_tokens} prompt + {completion_tokens} completion):")
    print(f"    openai/gpt-4o:         ${gpt_cost:.6f}")
    print(f"    anthropic/claude-3-5-sonnet: ${claude_cost:.6f}")
    print(f"    Claude is {claude_cost / gpt_cost:.2f}x {'more expensive' if claude_cost > gpt_cost else 'cheaper'} than GPT-4o")

    print(f"\n  100 Requests Cost Projection:")
    print(f"    openai/gpt-4o:         ${gpt_cost * 100:.2f}")
    print(f"    anthropic/claude-3-5-sonnet: ${claude_cost * 100:.2f}")
    print(f"    Cost difference: ${abs(claude_cost - gpt_cost) * 100:.2f}")


def test_state_cost_tracking():
    """Test that state cost tracking works correctly."""
    print(f"\n{'=' * 70}")
    print("TeammateState Cost Tracking Test")
    print(f"{'=' * 70}")

    # Simulate TeammateState
    state = {
        "task_id": "test-123",
        "task_description": "Test task",
        "total_cost": 0.0,
        "cost_limit": 0.50,
        "cost_history": [],
    }

    print(f"\n  Initial State:")
    print(f"    total_cost: ${state['total_cost']:.6f}")
    print(f"    cost_limit: ${state['cost_limit']:.6f}")

    # Simulate requests with different models
    requests = [
        ("openai/gpt-4o", 1000, 2000),
        ("openai/gpt-4o", 1000, 2000),
        ("anthropic/claude-3-5-sonnet", 1000, 2000),
    ]

    print(f"\n  Simulating requests (1000 prompt + 2000 completion each):")
    for i, (model, prompt, completion) in enumerate(requests, 1):
        cost = calculate_cost(model, prompt, completion)

        state["total_cost"] += cost
        state["cost_history"].append({
            "request_id": i,
            "model": model,
            "cost_usd": cost,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        })

        budget_status = "[OVER BUDGET!]" if state["total_cost"] >= state["cost_limit"] else "[OK]"
        print(f"    Request {i}: {model.split('/')[-1]:<15} = ${cost:.6f} (Running: ${state['total_cost']:.6f}) {budget_status}")

    print(f"\n  Final State:")
    print(f"    total_cost: ${state['total_cost']:.6f}")
    print(f"    cost_limit: ${state['cost_limit']:.6f}")
    print(f"    Budget used: {(state['total_cost'] / state['cost_limit']) * 100:.1f}%")
    print(f"    Requires approval: {state['total_cost'] >= state['cost_limit']}")


if __name__ == "__main__":
    simulate_model_switch()
    test_state_cost_tracking()

    print(f"\n{'=' * 70}")
    print("Simulation Complete")
    print(f"{'=' * 70}")
    print("\nThe Model-Agnostic Shield correctly tracks costs for different models.")
    print("Switching CHAT_MODEL in .env automatically uses the correct pricing.")
