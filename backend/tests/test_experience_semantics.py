"""Experience-level semantics (V4 PART C — §11, §13, §14, §35).

  * role_function is tracked SEPARATELY from employer_industry
  * semantic assertions / experience_semantics reference REAL normalized ids;
    hallucinated ids are dropped by _ground()
"""
from __future__ import annotations

from app.constants import CriterionType, Scope
from app.schemas import ParsedSearchQuery, ProfileSemanticData, SearchCriterion
from app.services.scoring import ProfileFacts, score_candidate
from tests.test_search import _Exp, _Person


def _facts(person, exps, sem):
    return ProfileFacts(person=person, experiences=exps, education=[], skills=[],
                        semantic=sem, embedding=None)


_ACCOUNTANT_AT_GOOGLE = {
    "experience_semantics": [{
        "experience_id": "e1", "role_function": "accounting",
        "professional_domain": "finance", "employer_industries": ["technology"],
        "employer_categories": ["big tech"], "confidence": 0.9,
    }],
}
_SWE_AT_JPM = {
    "experience_semantics": [{
        "experience_id": "e1", "role_function": "software engineering",
        "professional_domain": "software systems",
        "employer_industries": ["financial services"], "confidence": 0.9,
    }],
}


def _score(sem, ctype, concept):
    crit = SearchCriterion(id="c", type=ctype, concept=concept, weight=100, scope=Scope.ANY_EXPERIENCE)
    exps = [_Exp("role", "co", 2019, None, True, id="e1")]
    return score_candidate(_facts(_Person(), exps, sem), ParsedSearchQuery(criteria=[crit]))


def _strength(scored):
    c = next((x for x in scored.components if x.criterion_id == "c"), None)
    return c.match_strength if c else 0.0


def test_accountant_at_google_is_tech_industry_not_technical_role():
    industry = _score(_ACCOUNTANT_AT_GOOGLE, CriterionType.INDUSTRY_EXPERIENCE, "technology industry")
    role = _score(_ACCOUNTANT_AT_GOOGLE, CriterionType.ROLE_FUNCTION, "software engineering")
    assert _strength(industry) > 0.5
    assert _strength(role) < 0.4


def test_swe_at_jpmorgan_is_technical_role_not_tech_industry():
    role = _score(_SWE_AT_JPM, CriterionType.ROLE_FUNCTION, "software engineering")
    industry = _score(_SWE_AT_JPM, CriterionType.INDUSTRY_EXPERIENCE, "technology industry")
    assert _strength(role) > 0.5
    assert _strength(industry) < 0.5


def test_swe_at_jpmorgan_matches_financial_services_industry():
    assert _strength(_score(_SWE_AT_JPM, CriterionType.INDUSTRY_EXPERIENCE, "financial services")) > 0.5


# ─────────────────── §14 id grounding ───────────────────


def _compact():
    return {
        "experience": [{"experience_id": "exp-real", "company": "Google", "title": "SWE"}],
        "education": [{"education_id": "edu-real", "school": "MIT"}],
        "certifications": [{"certification_id": "cert-real", "name": "AWS SA"}],
    }


def test_ground_drops_hallucinated_experience_ids():
    from app.services.semantic_llm import _ground

    data = ProfileSemanticData(
        semantic_assertions=[{
            "concept": "technology industry experience", "category": "industry_experience",
            "scope": "past", "confidence": 0.9,
            "experience_ids": ["exp-real", "fake-123"],
            "education_ids": ["edu-real", "bogus"],
            "evidence": ["SWE at Google"],
        }],
        experience_semantics=[
            {"experience_id": "exp-real", "role_function": "software engineering"},
            {"experience_id": "ghost-9", "role_function": "wizardry"},
        ],
    )
    out = _ground(data, _compact())
    a = out.semantic_assertions[0]
    assert a.experience_ids == ["exp-real"]
    assert a.education_ids == ["edu-real"]
    assert [es.experience_id for es in out.experience_semantics] == ["exp-real"]


def test_ground_keeps_assertion_grounded_only_by_a_valid_id():
    from app.services.semantic_llm import _ground

    data = ProfileSemanticData(semantic_assertions=[{
        "concept": "cloud expertise", "category": "domain_expertise", "scope": "career",
        "confidence": 0.8, "experience_ids": ["exp-real"], "evidence": [],
    }])
    out = _ground(data, _compact())
    assert len(out.semantic_assertions) == 1 and out.semantic_assertions[0].experience_ids == ["exp-real"]


def test_ground_drops_assertion_with_no_evidence_and_no_valid_id():
    from app.services.semantic_llm import _ground

    data = ProfileSemanticData(semantic_assertions=[{
        "concept": "made up", "category": "leadership", "scope": "career",
        "confidence": 0.9, "experience_ids": ["nope"], "evidence": [],
    }])
    assert _ground(data, _compact()).semantic_assertions == []
