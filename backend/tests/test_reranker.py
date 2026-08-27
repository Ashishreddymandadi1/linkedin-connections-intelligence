from __future__ import annotations

from app.services import reranker


def test_cross_encode_disabled_is_noop(monkeypatch):
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", False)
    scores = reranker.cross_encode("python data engineer", ["a", "b", "c"])
    assert scores == [0.0, 0.0, 0.0]


def test_cross_encode_empty():
    assert reranker.cross_encode("x", []) == []


def test_cross_encode_normalises_to_unit_range(monkeypatch):
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", True)

    class _FakeModel:
        def predict(self, pairs, **kw):  # noqa: ANN001, ARG002
            return [(-3.0), 1.0, 5.0][: len(pairs)]

    monkeypatch.setattr(reranker, "_model", _FakeModel())
    out = reranker.cross_encode("q", ["p1", "p2", "p3"])
    assert out[0] == 0.0 and out[2] == 1.0 and 0.4 < out[1] < 0.6


def test_llm_rerank_disabled_returns_none(monkeypatch):
    monkeypatch.setattr("app.services.reranker.settings.llm_rerank_enabled", False)
    assert reranker.llm_rerank("q", [{"person_id": "p1", "line": "x"}]) is None


def test_relevance_component_present_in_score_breakdown(monkeypatch):
    """With relevance_weight > 0, every scored candidate gets the component and
    criteria weights renormalise so the total still caps at 100."""
    monkeypatch.setattr("app.services.scoring.settings.relevance_weight", 20.0)
    from app.schemas import ParsedSearchQuery, SearchCriterion
    from app.services.scoring import ProfileFacts, score_candidate
    from tests.test_search import _Person, _Skill

    parsed = ParsedSearchQuery(criteria=[SearchCriterion(id="s", type="skill", value="Rust", weight=100)])
    facts = ProfileFacts(
        person=_Person(), experiences=[], education=[], skills=[_Skill("Rust")], semantic={}, embedding=None
    )
    scored = score_candidate(facts, parsed)
    ids = {c.criterion_id for c in scored.components}
    assert "relevance" in ids
    crit = next(c for c in scored.components if c.criterion_id == "s")
    assert crit.weight == 80.0  # 100 * (1 - 20/100)
    assert scored.match_score <= 100
