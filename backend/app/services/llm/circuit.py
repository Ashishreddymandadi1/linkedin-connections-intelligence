"""In-process provider circuit breaker (V4 §8).

Stops the router from hammering a provider that clearly will not work right now
— an Anthropic key with a bad workspace id would otherwise be retried once per
profile across a 1000-profile backfill.

Scope: process-local, never persisted (V4 §8). A restart, or waiting out the
cooldown, re-arms the provider. Cooldowns:

  configuration_error / authentication_error -> long (a human must fix it)
  rate_limited / temporarily_unavailable / transport_error -> short, and only
      after several consecutive hits (a single 429 is normal, not a fault)
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from app.config import settings

log = logging.getLogger("app.llm.circuit")

#: consecutive transient failures before a provider is cooled down
_TRANSIENT_TRIP_THRESHOLD = 3


@dataclass
class _State:
    open_until: float = 0.0
    consecutive_failures: int = 0
    last_category: str = ""


_states: dict[str, _State] = {}
_lock = threading.Lock()


def is_open(provider: str) -> bool:
    """True when the provider is cooling down and should be skipped."""
    with _lock:
        st = _states.get(provider)
        if not st:
            return False
        if st.open_until and time.monotonic() < st.open_until:
            return True
        st.open_until = 0.0
        return False


def cooldown_remaining(provider: str) -> float:
    with _lock:
        st = _states.get(provider)
        if not st or not st.open_until:
            return 0.0
        return max(0.0, st.open_until - time.monotonic())


def record_success(provider: str) -> None:
    with _lock:
        _states.pop(provider, None)


def record_failure(provider: str, category: str) -> None:
    """Update the breaker after a provider attempt failed for good (all retries
    exhausted, or a non-retryable category)."""
    with _lock:
        st = _states.setdefault(provider, _State())
        st.consecutive_failures += 1
        st.last_category = category

        if category in ("configuration_error", "authentication_error"):
            secs = float(settings.anthropic_config_failure_cooldown_seconds)
            st.open_until = time.monotonic() + secs
            log.warning("circuit OPEN %s (%s) cooldown %.0fs", provider, category, secs)
        elif st.consecutive_failures >= _TRANSIENT_TRIP_THRESHOLD:
            secs = float(settings.llm_provider_cooldown_seconds)
            st.open_until = time.monotonic() + secs
            log.warning(
                "circuit OPEN %s (%s x%d) cooldown %.0fs",
                provider, category, st.consecutive_failures, secs,
            )


def reset_all() -> None:
    """Test helper — clear every breaker."""
    with _lock:
        _states.clear()


def snapshot() -> dict[str, dict]:
    """Debug view for /health and search traces. No secrets."""
    now = time.monotonic()
    with _lock:
        return {
            name: {
                "open": bool(st.open_until and now < st.open_until),
                "cooldown_remaining_s": round(max(0.0, st.open_until - now), 1),
                "consecutive_failures": st.consecutive_failures,
                "last_category": st.last_category,
            }
            for name, st in _states.items()
        }
