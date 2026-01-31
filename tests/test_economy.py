"""Test script for Model-Agnostic Shield cost tracking.

This script demonstrates how the UniversalCostTracker calculates costs
differently for different LLM models.
"""

import litellm
from typing import Any


def calculate_cost_with_litellm(model: str, prompt: str, completion: str) -> float:
    """Calculate cost using litellm directly.

    Handles different model name formats for different providers.
    """
    # Map of models to their litellm-compatible names and providers
    model_map = {
        "openai/gpt-4o": ("gpt-4o", None),
        "anthropic/claude-3-5-sonnet": ("claude-3-5-sonnet-20241022", "anthropic"),
        "deepseek/deepseek-chat": ("deepseek-chat", "deepseek"),
    }

    model_name, provider = model_map.get(model, (model, None))

    kwargs = {
        "model": model_name,
        "prompt": prompt,
        "completion": completion,
    }

    if provider:
        kwargs["custom_llm_provider"] = provider

    return litellm.completion_cost(**kwargs)


def test_cost_calculation():
    """Test cost calculation for different models."""
    print("=" * 70)
    print("Model-Agnostic Shield: Cost Calculation Test")
    print("=" * 70)

    # Test models with different pricing (using full model identifiers)
    models = [
        ("openai/gpt-4o", "openai/gpt-4o"),
        ("anthropic/claude-3-5-sonnet", "anthropic/claude-3-5-sonnet"),
        ("deepseek/deepseek-chat", "deepseek/deepseek-chat"),
    ]

    # Simulate typical usage (1000 prompt tokens, 2000 completion tokens)
    # We'll use a text approximation for token counting
    prompt_text = "Analyze the following data and provide insights. " * 70  # ~1000 tokens
    completion_text = "Based on the analysis, the key findings are: First, there is significant growth in the market segment. Second, customer satisfaction has improved. Third, operational efficiency has increased. " * 50  # ~2000 tokens

    print(f"\nSimulated Usage: ~1000 prompt tokens + ~2000 completion tokens")
    print("-" * 70)
    print(f"{'Model':<35} {'Cost/1K Tokens':<15} {'Total Cost':<12}")
    print("-" * 70)

    for display_name, model_name in models:
        try:
            # Calculate cost using litellm
            cost = calculate_cost_with_litellm(model_name, prompt_text, completion_text)

            # Get approximate cost per 1K tokens
            cost_per_1k = cost / 3.0  # ~3000 tokens total

            print(f"{display_name:<35} ~${cost_per_1k:.4f}        ${cost:.6f}")
        except Exception as e:
            print(f"{display_name:<35} ERROR: {str(e)[:30]}")

    print("-" * 70)

    # Demonstrate budget impact
    print("\nBudget Impact Simulation (10 requests at ~3000 tokens each):")
    print("-" * 70)

    budget = 0.50  # Default COST_LIMIT

    for display_name, model_name in models:
        try:
            cost_per_request = calculate_cost_with_litellm(model_name, prompt_text, completion_text)
            total_for_10 = cost_per_request * 10
            budget_hit = total_for_10 >= budget

            print(f"{display_name:<35} 10 requests: ${total_for_10:.4f}    {'[OVER BUDGET!]' if budget_hit else '[OK]'}")
        except Exception:
            print(f"{display_name:<35} 10 requests: N/A")

    print("-" * 70)
    print(f"\nBudget Limit: ${budget:.2f}")

    # Calculate savings
    print("\n" + "=" * 70)
    print("Cost Comparison: DeepSeek vs Expensive Models")
    print("=" * 70)

    try:
        expensive = calculate_cost_with_litellm("openai/gpt-4o", prompt_text, completion_text)
        cheap = calculate_cost_with_litellm("deepseek/deepseek-chat", prompt_text, completion_text)

        if cheap > 0:
            savings_ratio = expensive / cheap
            print(f"DeepSeek is {savings_ratio:.1f}x cheaper than GPT-4o")
            print(f"For 100 requests: GPT-4o = ${expensive * 100:.2f}, DeepSeek = ${cheap * 100:.2f}")
            print(f"Savings: ${(expensive - cheap) * 100:.2f} per 100 requests")
    except Exception as e:
        print(f"Could not calculate savings: {e}")


if __name__ == "__main__":
    test_cost_calculation()
