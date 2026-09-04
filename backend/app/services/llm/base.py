"""LLM provider abstraction (spec §20, V4 §6).

The app never depends on a concrete provider — only on ``LLMProvider`` and the
router. ``generate_json`` returns raw parsed JSON; schema validation happens in
the router so every provider benefits from the same retry-on-invalid logic.

Error taxonomy (V4 §6) — every failure is one of six categories so the router
knows whether retrying the SAME provider can possibly help:

  authentication_error    LLMAuthError      401/403                     no retry
  configuration_error     LLMConfigError    bad workspace / model / 400 no retry
  rate_limited            LLMRateLimited    429                         retry w/ Retry-After
  temporarily_unavailable LLMUnavailable    500/502/503/504/529         retry w/ backoff
  transport_error         LLMTransport      timeout / connection reset  retry w/ backoff
  bad_output              LLMBadOutput      non-JSON / schema mismatch  retry (prompt-local)
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


class LLMError(RuntimeError):
    #: stable machine-readable category (V4 §6) — used for logging + circuit breaker
    category = "error"
    #: True when trying the SAME provider again could plausibly succeed
    retryable = False


class LLMRateLimited(LLMError):
    category = "rate_limited"
    retryable = True

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class LLMUnavailable(LLMError):
    """Transient server-side: 5xx, 529 overloaded, model cold."""

    category = "temporarily_unavailable"
    retryable = True


class LLMTransport(LLMUnavailable):
    """Client-side transport failure: timeout, DNS, connection reset."""

    category = "transport_error"
    retryable = True


class LLMAuthError(LLMError):
    """401 / 403 — the key is wrong or lacks access. Retrying is pointless."""

    category = "authentication_error"
    retryable = False


class LLMConfigError(LLMError):
    """Invalid workspace id, unknown model, malformed request (400/404). The
    same request will fail identically on retry — move to the next provider and
    cool this one down hard (V4 §7/§8)."""

    category = "configuration_error"
    retryable = False


class LLMBadOutput(LLMError):
    """Response was not valid JSON / did not match the schema. Often prompt- or
    truncation-specific, so a retry can help, but it is not a provider fault —
    the circuit breaker is NOT tripped for this."""

    category = "bad_output"
    retryable = True


class LLMOutputTruncated(LLMBadOutput):
    """The model stopped because it hit ``max_tokens`` before producing
    complete JSON (Anthropic ``stop_reason == "max_tokens"`` / OpenAI-style
    ``finish_reason == "length"``). Retrying the IDENTICAL request wastes a
    call — it will truncate again, on ANY provider, since the problem is the
    request/expected-response SIZE, not this particular provider. The router
    does not try the next provider with the same payload — it returns
    "truncated" straight back to the caller (a batched judge/audit) so the
    ADAPTIVE SPLITTER can shrink the request first; the smaller request then
    goes through the normal provider chain from the top."""

    category = "output_truncated"


class LLMRequestTooLarge(LLMError):
    """413 (or equivalent) — the INPUT payload itself was too large for this
    provider, independent of whether the model even started generating.
    Request-specific, not a provider-health signal: a smaller request should
    be allowed to try the SAME provider immediately, so this never trips the
    circuit breaker. Treated like ``LLMOutputTruncated`` for control flow —
    the router returns immediately so the caller can shrink the request
    rather than replaying the identical oversized payload elsewhere."""

    category = "request_too_large"
    retryable = True


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
    def generate_json(
        self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1500, timeout: float | None = None,
    ) -> dict:
        """Return parsed JSON dict. Raise an ``LLMError`` subclass on failure.
        ``timeout`` (hardening PART 17) overrides the provider's default HTTP
        timeout — used to keep query interpretation from holding a search
        indefinitely on a slow-but-not-failing provider."""
