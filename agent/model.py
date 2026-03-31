import os
from smolagents import LiteLLMModel


def build_model() -> LiteLLMModel:
    """
    Instantiate the LLM backend using Anthropic Claude Haiku via LiteLLM.

    Claude Haiku is the recommended model for this benchmark:
      - Reliable tool use with smolagents CodeAgent
      - Fast enough to complete 20 questions without timeouts
      - Cost: ~$0.05-0.15 for a full 20-question benchmark run

    Upgrade to claude-sonnet-4-20250514 for higher accuracy if needed.
    Requires ANTHROPIC_API_KEY to be set in environment or .env file.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your HF Space secrets or .env file. "
            "Get a key at https://console.anthropic.com"
        )

    return LiteLLMModel(
        model_id="anthropic/claude-haiku-4-5-20251001",
        api_key=api_key,
    )