"""Match qualification tiers (V4 PART E — §22-26, §36, §37).

EXACT_MATCH  = every required criterion TRUE
POSSIBLE     = no required FALSE, but a required semantic is UNKNOWN
NOT_MATCH    = a required criterion is confidently FALSE  (kept out of results)

Tier is ranked BEFORE match_score — a verified TRUE always beats an UNKNOWN with
a higher embedding score.
"""
from __future__ import annotations

from app.constants import CriterionType, Qualification, Scope
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.company_intel import company_key
from app.services.scoring import ProfileFacts, ScoringContext, score_candidate
from tests.test_search import _Exp, _Person


def _facts(person, exps=None, sem=None):
    return ProfileFacts(person=person, experiences=exps or [], education=[], skills=[],
                        semantic=sem or {}, embedding=None)


def _amazon_startup_plan():
    return ParsedSearchQuery(criteria=[
        SearchCriterion(id="past", type=CriterionType.PAST_COMPANY, value="Amazon",
                        weight=40, required=True, scope=Scope.PAST_COMPANY),
        SearchCriterion(id="startup", type=CriterionType.COMPANY_CATEGORY, concept="startup",
                        scope=Scope.CURRENT_COMPANY, weight=60, required=True),
    ])


# ─────────────────── §36 Amazon -> startup ───────────────────


def test_A_verified_startup_is_exact_match():
    ctx = ScoringContext(company_class={
        company_key("77", "ExampleAI"): {"is_startup": True, "confidence": 0.9,
                                         "provenance": "ai_company_inference", "reason": "early-stage",
                                         "industries": [], "categories": []},
    })
    p = _Person(current_company="ExampleAI")
    exps = [_Exp("Founding Engineer", "ExampleAI", 2023, None, True, company_id="77"),
            _Exp("SDE II", "Amazon", 2019, 2022, False, company_id="1586")]
    s = score_candidate(_facts(p, exps), _amazon_startup_plan(), ctx)
    assert s.qualification == Qualification.EXACT_MATCH
    assert s.excluded_reason is None


def test_B_advises_startups_at_microsoft_is_not_match():
    ctx = ScoringContext(company_class={
        company_key("99", "Microsoft"): {"is_startup": False, "confidence": 0.97,
                                         "provenance": "ai_company_inference", "reason": "large public co",
                                         "industries": ["technology"], "categories": []},
    })
    p = _Person(current_company="Microsoft", about="I advise startups on the side")
    exps = [_Exp("Principal PM", "Microsoft", 2018, None, True, company_id="99"),
            _Exp("SDE", "Amazon", 2013, 2017, False, company_id="1586")]
    s = score_candidate(_facts(p, exps), _amazon_startup_plan(), ctx)
    assert s.qualification == Qualification.NOT_MATCH
    assert "startup" in " ".join(s.unmet_required).lower()


def test_C_unknown_company_is_possible_match():
    p = _Person(current_company="ObscureCo")
    exps = [_Exp("Engineer", "ObscureCo", 2022, None, True, company_id="555"),
            _Exp("SDE", "Amazon", 2018, 2021, False, company_id="1586")]
    s = score_candidate(_facts(p, exps), _amazon_startup_plan(), ScoringContext())
    assert s.qualification == Qualification.POSSIBLE_MATCH
    assert s.excluded_reason is None


def test_A_ranks_above_C_regardless_of_embedding():
    from app.services import search_service

    ctx_true = ScoringContext(company_class={
        company_key("77", "ExampleAI"): {"is_startup": True, "confidence": 0.9,
                                         "provenance": "ai_company_inference", "reason": "x",
                                         "industries": [], "categories": []},
    })
    a = score_candidate(
        _facts(_Person(current_company="ExampleAI"),
               [_Exp("Eng", "ExampleAI", 2023, None, True, company_id="77"),
                _Exp("SDE", "Amazon", 2019, 2022, False, company_id="1586")]),
        _amazon_startup_plan(), ctx_true)
    c = score_candidate(
        _facts(_Person(current_company="ObscureCo"),
               [_Exp("Eng", "ObscureCo", 2022, None, True, company_id="555"),
                _Exp("SDE", "Amazon", 2018, 2021, False, company_id="1586")]),
        _amazon_startup_plan(), ScoringContext())
    assert a.qualification == Qualification.EXACT_MATCH
    assert c.qualification == Qualification.POSSIBLE_MATCH
    c.match_score = 99.0  # pretend the embedding loved C
    a.match_score = 40.0
    ranked = sorted([c, a], key=search_service._tier_key)
    assert ranked[0].qualification == Qualification.EXACT_MATCH  # tier beats score


# ─────────────────── §37 CXO event, no real CXO ───────────────────


def test_cxo_event_with_no_cxo_yields_zero_exact_matches():
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="loc", type=CriterionType.LOCATION, values=["Memphis", "Nashville"],
                        operator="ANY_OF", weight=50, required=True),
        SearchCriterion(id="cxo", type=CriterionType.SENIORITY, value="cxo", weight=50, required=True),
    ])
    director_nashville = score_candidate(
        _facts(_Person(current_title="Director of Ops", location_text="Nashville, Tennessee")), plan)
    ceo_atlanta = score_candidate(
        _facts(_Person(current_title="CEO", location_text="Atlanta, Georgia")), plan)
    assert director_nashville.qualification == Qualification.NOT_MATCH
    assert ceo_atlanta.qualification == Qualification.NOT_MATCH
