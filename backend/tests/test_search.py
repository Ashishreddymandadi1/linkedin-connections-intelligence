from __future__ import annotations

from pathlib import Path

import pytest

from app.services.query_interpreter import _deterministic_parse, interpret_query
from app.services.scoring import ProfileFacts, score_candidate

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


# ─────────────────────── query interpreter ───────────────────────


@pytest.mark.parametrize(
    "query,ctype,value",
    [
        ("People who previously worked at Amazon", "past_company", "Amazon"),
        ("People currently at Google", "current_company", "Google"),
        ("People who studied at Georgia Tech", "education", "Georgia Tech"),
        ("Senior Java backend engineers", "skill", "java"),
    ],
)
def test_deterministic_parse_shapes(query, ctype, value):
    parsed = _deterministic_parse(query)
    assert any(c.type == ctype and value.lower() in c.value.lower() for c in parsed.criteria), parsed.model_dump()


def test_weights_always_sum_to_100():
    for q in [
        "People who previously worked at Amazon, know AWS and Java, distributed systems",
        "People currently at Google who know machine learning",
        "People who studied at Stanford",
        "random text with no structure at all here",
    ]:
        parsed = _deterministic_parse(q)
        assert abs(sum(c.weight for c in parsed.criteria) - 100.0) < 0.5


def test_interpret_query_fallback_is_deterministic(client):  # client -> LLM disabled in conftest
    parsed, provider, model = interpret_query("People who previously worked at Amazon and know AWS")
    assert provider == "deterministic" and model is None
    types = {c.type for c in parsed.criteria}
    assert "past_company" in types and "skill" in types


def test_required_flag_from_who_clause():
    parsed = _deterministic_parse("People currently at Google who know Java")
    google = next(c for c in parsed.criteria if c.value.lower() == "google")
    assert google.required is True


# ─────────────────────── scoring ───────────────────────


class _Exp:
    def __init__(self, position, company, sy, ey, current, desc=None, skills=None):
        self.position, self.company_name, self.start_year, self.end_year = position, company, sy, ey
        self.is_current, self.description, self.skills_json = current, desc, skills or []
        self.company_linkedin_url = self.location = self.duration_text = None
        self.employment_type = self.workplace_type = None


class _Skill:
    def __init__(self, name, inferred=False, source="linkedin_profile_skill", conf=1.0):
        self.skill_name = name
        self.skill_name_norm = name.lower()
        self.is_inferred = inferred
        self.source = source
        self.confidence = conf
        self.evidence = None


class _Person:
    def __init__(self, **kw):
        self.id = "p1"
        self.full_name = kw.get("full_name", "Test Person")
        self.current_title = kw.get("current_title")
        self.current_company = kw.get("current_company")
        self.headline = kw.get("headline")
        self.about = kw.get("about")
        self.location_text = kw.get("location_text")
        self.city = self.state = self.country = None
        self.profile_completeness = kw.get("completeness", 80)
        self.linkedin_url = "https://www.linkedin.com/in/test"
        self.profile_picture_url = None


def _facts(person, exps=None, edu=None, skills=None, semantic=None):
    return ProfileFacts(
        person=person,
        experiences=exps or [],
        education=edu or [],
        skills=skills or [],
        semantic=semantic or {},
        embedding=None,
    )


def test_exact_past_company_beats_partial_title():
    from app.schemas import ParsedSearchQuery, SearchCriterion

    parsed = ParsedSearchQuery(
        criteria=[
            SearchCriterion(id="past_amazon", type="past_company", value="Amazon", weight=60, required=False),
            SearchCriterion(id="title", type="title", value="engineer", weight=40, required=False),
        ]
    )
    p = _Person(current_title="Senior Software Engineer", current_company="Google")
    exps = [
        _Exp("Senior Software Engineer", "Google", 2024, None, True),
        _Exp("SDE II", "Amazon", 2021, 2023, False),
    ]
    scored = score_candidate(_facts(p, exps=exps), parsed)
    amazon = next(c for c in scored.components if c.criterion_id == "past_amazon")
    assert amazon.match_strength == 1.0 and amazon.score == 60
    assert scored.match_score > 90


def test_required_criterion_excludes_non_match():
    from app.schemas import ParsedSearchQuery, SearchCriterion

    parsed = ParsedSearchQuery(
        criteria=[
            SearchCriterion(id="cur_google", type="current_company", value="Google", weight=70, required=True),
            SearchCriterion(id="java", type="skill", value="Java", weight=30, required=False),
        ]
    )
    p = _Person(current_company="Meta", current_title="SWE")
    scored = score_candidate(_facts(p, skills=[_Skill("Java")]), parsed)
    assert scored.excluded_reason is not None
    assert scored.match_score == 0.0


def test_inferred_skill_scores_below_explicit():
    from app.schemas import ParsedSearchQuery, SearchCriterion

    parsed = ParsedSearchQuery(criteria=[SearchCriterion(id="k8s", type="skill", value="Kubernetes", weight=100)])
    explicit = score_candidate(_facts(_Person(), skills=[_Skill("Kubernetes")]), parsed)
    inferred = score_candidate(
        _facts(_Person(), semantic={"inferred_skills": [{"skill": "Kubernetes", "confidence": 0.8, "evidence": "ran k8s clusters"}]}),
        parsed,
    )
    assert explicit.match_score == 100
    assert inferred.match_score < explicit.match_score
    assert inferred.match_score > 0


# ─────────────────────── end to end ───────────────────────


def _enriched_dataset(client) -> str:
    r = client.post("/datasets", files={"file": ("c.csv", FIXTURE_CSV.read_bytes(), "text/csv")})
    ds_id = r.json()["dataset"]["dataset_id"]
    client.post(f"/datasets/{ds_id}/enrich")
    return ds_id


def test_search_endpoint_top20_and_ordering(client):
    ds_id = _enriched_dataset(client)
    r = client.post("/search", json={"dataset_id": ds_id, "query": "People who previously worked at Amazon and know AWS"})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["external"]["searched"] is False
    assert body["connections"]["returned"] <= 20
    scores = [x["match_score"] for x in body["connections"]["results"]]
    assert scores == sorted(scores, reverse=True)

    top = body["connections"]["results"][0]
    assert top["is_connection"] is True
    assert top["match_score"] > 0
    assert top["data_confidence"] == top["data_confidence"]  # present
    assert sum(c["weight"] for c in body["interpreted_query"]["criteria"]) == pytest.approx(100, abs=0.5)
    # every score component points at real evidence or scored 0
    for comp in top["score_breakdown"]:
        if comp["match_strength"] > 0.2:
            assert comp["evidence"], comp

    # Jane (ex-Amazon, AWS + Java) should rank at/near the top
    names = [x["name"] for x in body["connections"]["results"][:3]]
    assert "Jane Smith" in names or "Sofia Rossi" in names


def test_search_is_retrievable_by_id(client):
    ds_id = _enriched_dataset(client)
    sid = client.post("/search", json={"dataset_id": ds_id, "query": "Senior engineers at Microsoft"}).json()["search_id"]
    again = client.get(f"/search/{sid}")
    assert again.status_code == 200
    assert again.json()["query"] == "Senior engineers at Microsoft"


def test_search_never_calls_apify(client, monkeypatch):
    called = {"n": 0}

    def boom(*a, **k):  # noqa: ARG001
        called["n"] += 1
        raise AssertionError("Apify must not be called during search")

    monkeypatch.setattr("app.services.apify_client.scrape_profiles", boom)
    ds_id = _enriched_dataset(client)
    monkeypatch.setattr("app.services.enrichment_runner.scrape_profiles", boom)
    client.post("/search", json={"dataset_id": ds_id, "query": "AWS distributed systems"})
    assert called["n"] == 0
