"""Concrete providers: Groq + OpenRouter (free), and Anthropic (paid, opt-in)."""
from __future__ import annotations

from app.config import settings
from app.constants import LLMProviderName
from app.services.llm.anthropic_client import messages_json
from app.services.llm.base import LLMProvider
from app.services.llm.openai_compatible import chat_json

_GROQ_BASE = "https://api.groq.com/openai/v1"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class GroqProvider(LLMProvider):
    def __init__(self, model: str, name: str):
        self.model = model
        self.name = name

    def available(self) -> bool:
        return bool(settings.groq_api_key)

    def generate_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1500) -> dict:
        return chat_json(
            base_url=_GROQ_BASE,
            api_key=settings.groq_api_key,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            reasoning_effort="low" if "gpt-oss" in self.model else None,
        )


class OpenRouterProvider(LLMProvider):
    name = LLMProviderName.OPENROUTER

    def __init__(self, model: str):
        self.model = model

    def available(self) -> bool:
        return bool(settings.openrouter_api_key) and self.model.endswith(":free")

    def generate_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1500) -> dict:
        return chat_json(
            base_url=_OPENROUTER_BASE,
            api_key=settings.openrouter_api_key,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "LinkedIn Connections Intelligence",
            },
        )


class AnthropicProvider(LLMProvider):
    """Paid. Only enters the chain when ENABLE_PAID_LLM=true and a key is set."""

    name = LLMProviderName.ANTHROPIC

    def __init__(self, model: str):
        self.model = model

    def available(self) -> bool:
        return settings.enable_paid_llm and bool(settings.anthropic_api_key)

    def generate_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1500) -> dict:
        return messages_json(
            api_key=settings.anthropic_api_key,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            workspace_id=settings.anthropic_workspace_id,
        )


def default_chain() -> list[LLMProvider]:
    free = [
        GroqProvider(settings.groq_primary_model, LLMProviderName.GROQ_PRIMARY),
        GroqProvider(settings.groq_fallback_model, LLMProviderName.GROQ_FALLBACK),
        OpenRouterProvider(settings.openrouter_model),
    ]
    paid = AnthropicProvider(settings.anthropic_model)
    if not paid.available():
        return free
    # opt-in: Anthropic first (fast, reliable) with the free tier as the safety net,
    # unless the operator wants free-first and paid only as a last resort.
    return [paid, *free] if settings.anthropic_first else [*free, paid]
