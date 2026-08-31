from __future__ import annotations

from pydantic import BaseModel

from app.services.llm.base import LLMBadOutput, LLMProvider, LLMRateLimited, LLMUnavailable
from app.services.llm.providers import default_chain
from app.services.llm.router import generate_structured


class Out(BaseModel):
    answer: str
    score: int


class FakeProvider(LLMProvider):
    def __init__(self, name, behavior):
        self.name = name
        self.model = f"model-{name}"
        self.behavior = behavior
        self.calls = 0

    def available(self) -> bool:
        return True

    def generate_json(self, system_prompt, user_prompt, *, max_tokens=1500):  # noqa: ARG002
        self.calls += 1
        b = self.behavior
        if b == "ok":
            return {"answer": "hi", "score": 5}
        if b == "429":
            raise LLMRateLimited("429", retry_after=0.01)
        if b == "down":
            raise LLMUnavailable("503")
        if b == "badjson":
            raise LLMBadOutput("not json")
        if b == "badschema":
            return {"answer": "hi"}  # missing score
        raise AssertionError(b)


def test_primary_success():
    p = FakeProvider("primary", "ok")
    res = generate_structured("s", "u", Out, chain=[p])
    assert res is not None
    model, name, _ = res
    assert model.answer == "hi" and name == "primary"


def test_falls_back_on_rate_limit(monkeypatch):
    monkeypatch.setattr("app.services.llm.router.settings.llm_max_retries", 1)
    primary = FakeProvider("primary", "429")
    fallback = FakeProvider("fallback", "ok")
    res = generate_structured("s", "u", Out, chain=[primary, fallback])
    assert res is not None
    _, name, _ = res
    assert name == "fallback"
    assert primary.calls == 2  # initial + 1 retry, then give up


def test_falls_back_on_unavailable_then_openrouter():
    a = FakeProvider("groq120", "down")
    b = FakeProvider("groq20", "down")
    c = FakeProvider("openrouter", "ok")
    res = generate_structured("s", "u", Out, chain=[a, b, c])
    assert res and res[1] == "openrouter"


def test_all_exhausted_returns_none():
    chain = [FakeProvider("a", "down"), FakeProvider("b", "429"), FakeProvider("c", "badjson")]
    assert generate_structured("s", "u", Out, chain=chain) is None


def test_schema_validation_failure_moves_on():
    bad = FakeProvider("bad", "badschema")
    good = FakeProvider("good", "ok")
    res = generate_structured("s", "u", Out, chain=[bad, good])
    assert res and res[1] == "good"


def test_paid_provider_only_enters_chain_when_opted_in(monkeypatch):
    from app.constants import LLMProviderName

    monkeypatch.setattr("app.config.settings.groq_api_key", "x")
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "sk-ant-test")

    monkeypatch.setattr("app.config.settings.enable_paid_llm", False)
    assert LLMProviderName.ANTHROPIC not in [p.name for p in default_chain()]

    monkeypatch.setattr("app.config.settings.enable_paid_llm", True)
    monkeypatch.setattr("app.config.settings.anthropic_first", True)
    chain = [p.name for p in default_chain()]
    assert chain[0] == LLMProviderName.ANTHROPIC
    assert LLMProviderName.GROQ_PRIMARY in chain  # free tier is still the fallback

    monkeypatch.setattr("app.config.settings.anthropic_first", False)
    assert [p.name for p in default_chain()][-1] == LLMProviderName.ANTHROPIC

    # no key -> not in chain even when the flag is on
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "")
    assert LLMProviderName.ANTHROPIC not in [p.name for p in default_chain()]


def test_unavailable_provider_skipped():
    class Unconfigured(FakeProvider):
        def available(self) -> bool:
            return False

    skipped = Unconfigured("skipme", "ok")
    used = FakeProvider("used", "ok")
    res = generate_structured("s", "u", Out, chain=[skipped, used])
    assert res and res[1] == "used"
    assert skipped.calls == 0
