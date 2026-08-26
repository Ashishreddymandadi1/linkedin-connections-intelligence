"""LLM provider abstraction (spec §20).

The app never depends on a concrete provider — only on ``LLMProvider`` and the
router. ``generate_json`` returns raw parsed JSON; schema validation happens in
the router so every provider benefits from the same retry-on-invalid logic.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


class LLMError(RuntimeError):
    pass


class LLMRateLimited(LLMError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class LLMUnavailable(LLMError):
    """Transient: timeout, 5xx, connection error, model cold."""


class LLMBadOutput(LLMError):
    """Response was not valid JSON / not usable."""


@dataclass
class LLMResult:
    data: dict
    provider: str
    model: str


class LLMProvider(abc.ABC):
    name: str
    model: str

    @abc.abstractmethod
    def available(self) -> bool:
        """True if this provider is configured (has a key)."""

    @abc.abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1500) -> dict:
        """Return parsed JSON dict. Raise an ``LLMError`` subclass on failure."""
