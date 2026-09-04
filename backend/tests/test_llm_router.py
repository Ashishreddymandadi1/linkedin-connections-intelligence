"""Provider routing / fallback / circuit-breaker tests (V4 §13 TEST A–J, §66).

The router must:
  * try Anthropic first whenever a key is configured (regardless of enable_paid_llm)
  * stop the chain the moment a provider returns validated output
  * fall through on ANY failure, fast for unretryable ones (401 / bad workspace)
  * cool a provider down after an unretryable or repeated-transient failure
  * return None (→ local search) only when every provider is exhausted
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.constants import LLMProviderName
from app.services.llm import circuit
from app.services.llm.base import (
    LLMAuthError,
    LLMBadOutput,
    LLMConfigError,
    LLMOutputTruncated,
    LLMProvider,
    LLMRateLimited,
    LLMRequestTooLarge,
    LLMTransport,
    LLMUnavailable,
)
from app.services.llm.providers import default_chain
from app.services.llm.router import generate_structured


class Out(BaseModel):
    answer: str
    score: int


@pytest.fixture(autouse=True)
def _clean_breakers():
    circuit.reset_all()
    yield
    circuit.reset_all()


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr("app.services.llm.router.settings.llm_max_retries", 1)


class FakeProvider(LLMProvider):
    def __init__(self, name, behavior):
        self.name = name
        self.model = f"model-{name}"
        self.behavior = behavior
        self.calls = 0

    def available(self) -> bool:
        return True

    def generate_json(self, system_prompt, user_prompt, *, max_tokens=1500, timeout=None):  # noqa: ARG002
        self.calls += 1
        b = self.behavior
        if b == "ok":
            return {"answer": "hi", "score": 5}
        if b == "429":
            raise LLMRateLimited("429", retry_after=0.01)
        if b == "unavailable":
            raise LLMUnavailable("503")
        if b == "transport":
            raise LLMTransport("connection reset")
        if b == "auth":
            raise LLMAuthError("401")
        if b == "config":
            raise LLMConfigError("anthropic workspace configuration error")
        if b == "badjson":
            raise LLMBadOutput("not json")
        if b == "badschema":
            return {"answer": "hi"}  # missing score
        if b == "truncated":
            raise LLMOutputTruncated("hit max_tokens")
        if b == "413":
            raise LLMRequestTooLarge("payload too large")
        raise AssertionError(b)


def _run(chain):
    return generate_structured("s", "u", Out, chain=chain, operation="test")


# ─────────────────────────── TEST A ───────────────────────────
def test_A_anthropic_success_stops_chain():
    anth = FakeProvider(LLMProviderName.ANTHROPIC, "ok")
    groq_p = FakeProvider(LLMProviderName.GROQ_PRIMARY, "ok")
    groq_f = FakeProvider(LLMProviderName.GROQ_FALLBACK, "ok")
    res = _run([anth, groq_p, groq_f])
    assert res and res[1] == LLMProviderName.ANTHROPIC
    assert anth.calls == 1
    assert groq_p.calls == 0 and groq_f.calls == 0


# ─────────────────────────── TEST B ───────────────────────────
def test_B_anthropic_rate_limited_falls_to_groq_primary():
    anth = FakeProvider(LLMProviderName.ANTHROPIC, "429")
    groq_p = FakeProvider(LLMProviderName.GROQ_PRIMARY, "ok")
    res = _run([anth, groq_p])
    assert res and res[1] == LLMProviderName.GROQ_PRIMARY
    assert anth.calls == 2  # initial + 1 retry, then fall through


# ─────────────────────────── TEST C ───────────────────────────
def test_C_anthropic_401_no_pointless_retries():
    anth = FakeProvider(LLMProviderName.ANTHROPIC, "auth")
    groq_p = FakeProvider(LLMProviderName.GROQ_PRIMARY, "ok")
    res = _run([anth, groq_p])
    assert res and res[1] == LLMProviderName.GROQ_PRIMARY
    assert anth.calls == 1  # 401 is not retried
    assert circuit.is_open(LLMProviderName.ANTHROPIC)


# ─────────────────────────── TEST D ───────────────────────────
def test_D_anthropic_workspace_config_error_falls_through():
    anth = FakeProvider(LLMProviderName.ANTHROPIC, "config")
    groq_p = FakeProvider(LLMProviderName.GROQ_PRIMARY, "ok")
    res = _run([anth, groq_p])
    assert res and res[1] == LLMProviderName.GROQ_PRIMARY
    assert anth.calls == 1
    assert circuit.is_open(LLMProviderName.ANTHROPIC)


# ─────────────────────────── TEST E ───────────────────────────
def test_E_anthropic_and_groq_primary_fail_then_groq_fallback():
    anth = FakeProvider(LLMProviderName.ANTHROPIC, "unavailable")
    groq_p = FakeProvider(LLMProviderName.GROQ_PRIMARY, "unavailable")
    groq_f = FakeProvider(LLMProviderName.GROQ_FALLBACK, "ok")
    res = _run([anth, groq_p, groq_f])
    assert res and res[1] == LLMProviderName.GROQ_FALLBACK


# ─────────────────────────── TEST F ───────────────────────────
def test_F_all_but_openrouter_fail():
    chain = [
        FakeProvider(LLMProviderName.ANTHROPIC, "transport"),
        FakeProvider(LLMProviderName.GROQ_PRIMARY, "unavailable"),
        FakeProvider(LLMProviderName.GROQ_FALLBACK, "429"),
        FakeProvider(LLMProviderName.OPENROUTER, "ok"),
    ]
    res = _run(chain)
    assert res and res[1] == LLMProviderName.OPENROUTER


# ─────────────────────────── TEST G ───────────────────────────
def test_G_no_anthropic_key_means_groq_first(monkeypatch):
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "")
    monkeypatch.setattr("app.config.settings.groq_api_key", "x")
    monkeypatch.setattr("app.config.settings.openrouter_api_key", "")
    names = [p.name for p in default_chain()]
    assert LLMProviderName.ANTHROPIC not in names
    assert names[0] == LLMProviderName.GROQ_PRIMARY


# ─────────────────────────── TEST H ───────────────────────────
def test_H_no_provider_available_returns_none():
    class Unconfigured(FakeProvider):
        def available(self) -> bool:
            return False

    assert _run([Unconfigured("x", "ok")]) is None


# ─────────────────────────── TEST I ───────────────────────────
def test_I_configured_key_used_even_when_enable_paid_llm_false(monkeypatch):
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr("app.config.settings.groq_api_key", "x")
    monkeypatch.setattr("app.config.settings.enable_paid_llm", False)
    names = [p.name for p in default_chain()]
    assert names[0] == LLMProviderName.ANTHROPIC


# ─────────────────────────── TEST J ───────────────────────────
def test_J_returned_provider_is_a_real_name():
    for behavior_chain, expected in [
        ([(LLMProviderName.ANTHROPIC, "ok")], LLMProviderName.ANTHROPIC),
        ([(LLMProviderName.ANTHROPIC, "auth"), (LLMProviderName.GROQ_PRIMARY, "ok")], LLMProviderName.GROQ_PRIMARY),
    ]:
        circuit.reset_all()
        chain = [FakeProvider(n, b) for n, b in behavior_chain]
        res = _run(chain)
        assert res and res[1] == expected and res[1] != "deterministic"


# ─────────────────────── default_chain ordering ───────────────────────
def test_default_chain_full_priority_order(monkeypatch):
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "sk-ant")
    monkeypatch.setattr("app.config.settings.groq_api_key", "gsk")
    monkeypatch.setattr("app.config.settings.openrouter_api_key", "or")
    monkeypatch.setattr("app.config.settings.openrouter_model", "meta/x:free")
    assert [p.name for p in default_chain()] == [
        LLMProviderName.ANTHROPIC,
        LLMProviderName.GROQ_PRIMARY,
        LLMProviderName.GROQ_FALLBACK,
        LLMProviderName.OPENROUTER,
    ]


# ─────────────────────── circuit breaker ───────────────────────
def test_circuit_skips_cooled_provider_then_success_resets():
    anth = FakeProvider(LLMProviderName.ANTHROPIC, "auth")
    groq_p = FakeProvider(LLMProviderName.GROQ_PRIMARY, "ok")
    _run([anth, groq_p])
    assert circuit.is_open(LLMProviderName.ANTHROPIC)

    anth2 = FakeProvider(LLMProviderName.ANTHROPIC, "ok")
    groq2 = FakeProvider(LLMProviderName.GROQ_PRIMARY, "ok")
    res = _run([anth2, groq2])
    assert res and res[1] == LLMProviderName.GROQ_PRIMARY
    assert anth2.calls == 0  # never attempted while cooling down


def test_circuit_trips_after_repeated_transient_failures(monkeypatch):
    monkeypatch.setattr("app.services.llm.router.settings.llm_max_retries", 0)
    name = LLMProviderName.GROQ_PRIMARY
    for _ in range(3):
        _run([FakeProvider(name, "unavailable"), FakeProvider(LLMProviderName.GROQ_FALLBACK, "ok")])
    assert circuit.is_open(name)


def test_bad_output_does_not_trip_circuit(monkeypatch):
    monkeypatch.setattr("app.services.llm.router.settings.llm_max_retries", 0)
    name = LLMProviderName.ANTHROPIC
    for _ in range(5):
        _run([FakeProvider(name, "badjson"), FakeProvider(LLMProviderName.GROQ_PRIMARY, "ok")])
    assert not circuit.is_open(name)  # prompt-local, not a provider fault


# ─────────────────────── meta / schema validation ───────────────────────
def test_return_meta_records_attempts():
    chain = [
        FakeProvider(LLMProviderName.ANTHROPIC, "429"),
        FakeProvider(LLMProviderName.GROQ_PRIMARY, "ok"),
    ]
    model, name, model_id, meta = generate_structured(
        "s", "u", Out, chain=chain, operation="query_interpretation", return_meta=True
    )
    assert name == LLMProviderName.GROQ_PRIMARY
    assert meta["operation"] == "query_interpretation"
    assert meta["selected_provider"] == LLMProviderName.GROQ_PRIMARY
    assert meta["attempts"][0] == {"provider": LLMProviderName.ANTHROPIC, "status": "rate_limited"}
    assert meta["attempts"][-1] == {"provider": LLMProviderName.GROQ_PRIMARY, "status": "success"}


def test_schema_validation_failure_moves_on():
    bad = FakeProvider("bad", "badschema")
    good = FakeProvider("good", "ok")
    res = _run([bad, good])
    assert res and res[1] == "good"


# ─────────────────────── hardening PART 5/22 — truncation returns to caller ───────────────────────


def test_truncation_returns_to_caller_immediately_not_the_next_provider():
    """The core live-failure fix: an oversized-output truncation must NOT fall
    through to the next provider with the identical oversized request (that
    provider would very likely truncate/413 the same way, wasting a round trip
    that the caller should instead spend on a SMALLER, split request). The
    router returns None right away and the next provider is never called."""
    anth = FakeProvider(LLMProviderName.ANTHROPIC, "truncated")
    groq_p = FakeProvider(LLMProviderName.GROQ_PRIMARY, "ok")
    res = _run([anth, groq_p])
    assert res is None
    assert anth.calls == 1
    assert groq_p.calls == 0  # never attempted with the same oversized payload


def test_truncation_does_not_trip_the_circuit():
    """Output truncation is REQUEST-specific (this particular batch was too
    big), not a provider-health signal — it must never cool the provider down;
    a smaller request must be allowed to try the SAME provider right away."""
    name = LLMProviderName.ANTHROPIC
    _run([FakeProvider(name, "truncated")])
    assert not circuit.is_open(name)
    # the same provider, now with a small enough request, succeeds immediately
    res = _run([FakeProvider(name, "ok")])
    assert res and res[1] == name


# ─────────────────────── hardening PART 6/23 — 413 is request-specific ───────────────────────


def test_413_returns_to_caller_immediately_not_the_next_provider():
    groq_p = FakeProvider(LLMProviderName.GROQ_PRIMARY, "413")
    groq_f = FakeProvider(LLMProviderName.GROQ_FALLBACK, "ok")
    res = _run([groq_p, groq_f])
    assert res is None
    assert groq_p.calls == 1
    assert groq_f.calls == 0  # never attempted with the same oversized payload


def test_413_does_not_trip_the_circuit_and_a_smaller_request_succeeds_right_after():
    """The exact scenario the mission called out: '413 is request-specific. It
    does NOT mean Groq is down.' One oversized 413 must not globally disable
    the provider for 15 minutes — the very next (smaller) request to the SAME
    provider must be attempted normally, not skipped as circuit-open."""
    name = LLMProviderName.GROQ_PRIMARY
    large = FakeProvider(name, "413")
    _run([large])
    assert large.calls == 1
    assert not circuit.is_open(name)

    small = FakeProvider(name, "ok")
    res = _run([small])
    assert res and res[1] == name
    assert small.calls == 1  # the smaller request was actually attempted, not skipped


def test_413_after_repeated_hits_still_does_not_trip_the_circuit():
    """Unlike transient transport/rate-limit failures (which trip after 3
    consecutive hits), request_too_large must NEVER trip the circuit no matter
    how many times it happens — each one is a fact about that specific
    request's size, not the provider's health."""
    name = LLMProviderName.GROQ_PRIMARY
    for _ in range(5):
        _run([FakeProvider(name, "413")])
    assert not circuit.is_open(name)
