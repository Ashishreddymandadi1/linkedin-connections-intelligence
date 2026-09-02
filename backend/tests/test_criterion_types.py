"""SearchCriterion type validation (V4 §6/§10/§35).

An unsupported / semantic-ish type must NEVER silently become `keyword` — it
maps to a real first-class professional type, or to professional_concept.
"""
from __future__ import annotations

from app.constants import CriterionType
from app.schemas import SearchCriterion


def _t(type_str: str) -> str:
    return SearchCriterion(id="x", type=type_str, weight=10).type


def test_known_first_class_types_pass_through():
    for t in ("industry_experience", "role_function", "professional_concept",
              "company_category", "career_transition", "years_experience"):
        assert _t(t) == t


def test_common_llm_aliases_map_to_real_types():
    assert _t("industry") == CriterionType.INDUSTRY_EXPERIENCE
    assert _t("role") == CriterionType.ROLE_FUNCTION
    assert _t("job_function") == CriterionType.ROLE_FUNCTION
    assert _t("leadership") == CriterionType.PROFESSIONAL_CONCEPT
    assert _t("years_of_experience") == CriterionType.YEARS_EXPERIENCE
    assert _t("career_change") == CriterionType.CAREER_TRANSITION


def test_unknown_semantic_type_becomes_professional_concept_not_keyword():
    assert _t("technology_leadership_capability") == CriterionType.PROFESSIONAL_CONCEPT
    assert _t("some_made_up_thing") == CriterionType.PROFESSIONAL_CONCEPT
    assert _t("startup_founder_signal") == CriterionType.PROFESSIONAL_CONCEPT


def test_explicit_text_types_still_allow_keyword():
    assert _t("keyword") == "keyword"
    assert _t("text") == "keyword"
    assert _t("phrase") == "keyword"


def test_hyphens_and_spaces_normalised():
    assert _t("role-function") == CriterionType.ROLE_FUNCTION
    assert _t("Industry Experience") == CriterionType.INDUSTRY_EXPERIENCE
