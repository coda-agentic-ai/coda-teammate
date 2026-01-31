"""Economy module for model-agnostic cost tracking.

Uses litellm to calculate exact costs across different LLM providers
(OpenAI, Anthropic, DeepSeek, MiniMax, local models, etc.).
"""

from typing import Any

import litellm


class UniversalCostTracker:
    """Model-agnostic cost tracker using litellm's universal pricing.

    This class provides a unified interface for calculating the exact USD cost
    of LLM responses regardless of the underlying provider or model.

    Example:
        >>> from sentry_foundation.economy import UniversalCostTracker
        >>> cost = UniversalCostTracker.calculate_cost(response, "anthropic/claude-3-5-sonnet")
        >>> print(f"Cost: ${cost:.4f}")
    """

    @staticmethod
    def calculate_cost(response: Any, model_name: str) -> float:
        """Calculate the exact USD cost of an LLM response.

        Args:
            response: The LLM response object (from langchain or litellm)
            model_name: The model identifier (e.g., "anthropic/claude-3-5-sonnet",
                        "openai/gpt-4o", "deepseek/deepseek-chat")

        Returns:
            The exact cost in USD as a float.

        Note:
            litellm maintains an internal pricing table for all major providers.
            The model_name should match litellm's expected format (provider/model).
        """
        return litellm.completion_cost(
            completion_response=response,
            model=model_name
        )

    @staticmethod
    def calculate_token_cost(
        prompt_tokens: int,
        completion_tokens: int,
        model_name: str
    ) -> float:
        """Calculate cost based on token counts directly.

        This is useful when you have token counts but not the full response object.

        Args:
            prompt_tokens: Number of tokens in the prompt/input
            completion_tokens: Number of tokens in the completion/output
            model_name: The model identifier

        Returns:
            The exact cost in USD as a float.
        """
        # Get pricing for the model
        pricing = UniversalCostTracker.get_model_pricing(model_name)
        input_cost_per_token = pricing.get("input_cost_per_token", 0)
        output_cost_per_token = pricing.get("output_cost_per_token", 0)

        return (prompt_tokens * input_cost_per_token) + (completion_tokens * output_cost_per_token)

    @staticmethod
    def get_model_pricing(model_name: str) -> dict[str, float]:
        """Get the pricing information for a specific model.

        Args:
            model_name: The model identifier

        Returns:
            A dictionary with 'input_cost_per_token' and 'output_cost_per_token'
        """
        try:
            # litellm stores model info in model_prices_and_context_length_json
            model_info = litellm.model_prices_and_context_length_json.get(
                model_name.lower(), {}
            )
            return {
                "input_cost_per_token": model_info.get("input_cost_per_token", 0),
                "output_cost_per_token": model_info.get("output_cost_per_token", 0),
            }
        except AttributeError:
            # Fallback for older litellm versions
            return {"input_cost_per_token": 0, "output_cost_per_token": 0}
