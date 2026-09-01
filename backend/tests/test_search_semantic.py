"""Regression tests for the v3 semantic search pipeline (spec §34, §35).

Deterministic — no live LLM. They exercise the SCORING/PLAN behavior: a
semantic concept must be satisfiable without the literal word, a company
category must be judged from the actual employer, OR must not be AND, and the
keyword traps must not create false positives.
"""
from __future__ import annotations

from app.constants import TriState
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.company_intel import company_key
from app.services.query_interpreter import _deterministic_parse
from app.services.scoring import ProfileFacts, ScoringContext, score_candidate
from tests.test_search import _Exp, _Person, _Skill


def _facts(person, exps=None, skills=None, semantic=None):
    return ProfileFacts(
        person=person, experiences=exps or [], education=[], skills=skills or [],
        semantic=semantic or {}, embedding=None,
    )


# ─────────────────── TEST 1 — "people who worked in tech" ───────────────────


def test_tech_concept_matches_google_swe_without_the_word_tech():
    """A Software Engineer at Google whose profile never says 'tech'/'technology'
    still satisfies the technology-industry concept via a semantic assertion."""
    p = _Person(current_title="Software Engineer", current_company="Google")
    sem = {
        "industries": ["big tech"],
        "semantic_assertions": [
            {"concept": "technology industry experience", "category": "industry_experience",
             "confidence": 0.95, "evidence": ["Software Engineer at Google"]},
        ],
    }
    crit = SearchCriterion(id="tech", type="semantic_concept",
                           concept="professional experience in the technology industry",
                           weight=100, required=True, scope="any_experience")
    scored = score_candidate(_facts(p, exps=[_Exp("Software Engineer", "Google", 2022, None, True)], semantic=sem),
                             ParsedSearchQuery(criteria=[crit]))
    assert scored.excluded_reason is None
    tech = next(c for c in scored.components if c.criterion_id == "tech")
    assert tech.match_strength >= 0.7


def test_technology_transformation_phrase_does_not_qualify_as_tech_industry():
    """The false-positive trap (spec §34): a consultant whose profile mentions
    'technology transformation' but whose career is consulting must NOT match
    the technology-industry concept as strongly as the real Google engineer."""
    consultant = _Person(current_title="Senior Consultant", current_company="Big Consulting Firm")
    sem = {
        "industries": ["consulting", "retail"],
        "semantic_assertions": [
            {"concept": "consulting industry experience", "category": "industry_experience",
             "confidence": 0.97, "evidence": ["Senior Consultant"]},
            {"concept": "technology transformation expertise", "category": "domain_expertise",
             "confidence": 0.9, "evidence": ["Led a technology transformation"]},
        ],
    }
    crit = SearchCriterion(id="tech", type="semantic_concept",
                           concept="professional experience in the technology industry", weight=100)
    s_consultant = score_candidate(_facts(consultant, semantic=sem), ParsedSearchQuery(criteria=[crit]))

    engineer = _Person(current_title="Software Engineer", current_company="Google")
    eng_sem = {"semantic_assertions": [
        {"concept": "technology industry experience", "category": "industry_experience",
         "confidence": 0.95, "evidence": ["Software Engineer at Google"]}]}
    s_engineer = score_candidate(_facts(engineer, semantic=eng_sem), ParsedSearchQuery(criteria=[crit]))

    assert s_engineer.match_score > s_consultant.match_score


# ─────────────────── TEST 2 — "Former Amazon people now at startups" ───────────────────


def _amazon_startup_plan():
    return ParsedSearchQuery(criteria=[
        SearchCriterion(id="past_amazon", type="past_company", value="Amazon", weight=50, required=True, scope="past_company"),
        SearchCriterion(id="cur_startup", type="company_category", concept="startup",
                        scope="current_company", weight=50, required=True),
    ])


def test_ex_amazon_now_at_classified_startup_matches_without_literal_word():
    ctx = ScoringContext(company_class={
        company_key("777", "Nimbus AI"): {"is_startup": True, "confidence": 0.9, "reason": "early-stage",
                                          "provenance": "ai_company_inference", "industries": [], "categories": []},
    })
    p = _Person(current_title="Founding Engineer", current_company="Nimbus AI")
    exps = [
        _Exp("Founding Engineer", "Nimbus AI", 2023, None, True, company_id="777"),
        _Exp("SDE II", "Amazon", 2019, 2022, False, company_id="1586"),
    ]
    scored = score_candidate(_facts(p, exps=exps), _amazon_startup_plan(), ctx)
    assert scored.excluded_reason is None
    assert scored.match_score > 60
    assert any(e.type == "company_inference" for e in scored.evidence)


def test_ex_amazon_now_at_established_company_is_excluded_even_saying_i_like_startups():
    ctx = ScoringContext(company_class={
        company_key("999", "MegaBank"): {"is_startup": False, "confidence": 0.96,
                                         "provenance": "ai_company_inference", "industries": ["banking"], "categories": []},
    })
    p = _Person(current_title="Director", current_company="MegaBank",
                about="I really enjoy advising startups on the side")
    exps = [
        _Exp("Director", "MegaBank", 2015, None, True, company_id="999"),
        _Exp("SDE", "Amazon", 2010, 2014, False, company_id="1586"),
    ]
    scored = score_candidate(_facts(p, exps=exps), _amazon_startup_plan(), ctx)
    assert scored.excluded_reason is not None  # current company confidently NOT a startup


def test_unclassified_current_company_is_not_hard_excluded():
    """UNKNOWN != FALSE (spec §15/§28) — no classification yet must not exclude."""
    p = _Person(current_title="Engineer", current_company="Obscure Co")
    exps = [
        _Exp("Engineer", "Obscure Co", 2022, None, True, company_id="555"),
        _Exp("SDE", "Amazon", 2018, 2021, False, company_id="1586"),
    ]
    scored = score_candidate(_facts(p, exps=exps), _amazon_startup_plan(), ScoringContext())
    assert scored.excluded_reason is None  # not excluded, just no credit for the startup criterion


# ─────────────────── TEST 3 — location OR + CXO ───────────────────


def test_location_or_matches_either_and_excludes_neither_region():
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="loc", type="location", values=["Memphis", "Nashville"],
                        operator="ANY_OF", weight=100, required=True),
    ])
    nash = score_candidate(_facts(_Person(location_text="Nashville, Tennessee, United States")), plan)
    memp = score_candidate(_facts(_Person(location_text="Memphis, Tennessee, United States")), plan)
    atl = score_candidate(_facts(_Person(location_text="Atlanta, Georgia, United States")), plan)
    assert nash.match_score > 0 and memp.match_score > 0
    assert atl.excluded_reason is not None


def test_cxo_matches_a_real_ceo_title_but_not_someone_who_sells_to_cxos():
    plan = ParsedSearchQuery(criteria=[SearchCriterion(id="sen", type="seniority", value="CXO", weight=100)])
    ceo = score_candidate(_facts(_Person(current_title="CEO")), plan)
    cto = score_candidate(_facts(_Person(current_title="Chief Technology Officer")), plan)
    seller = score_candidate(_facts(_Person(current_title="Enterprise Account Executive",
                                            about="I sell to CXO customers and C-suite buyers")), plan)
    assert ceo.match_score > 50 and cto.match_score > 50
    assert seller.match_score < 20


# ─────────────────── TEST 6 — "former Google or Meta engineers" ───────────────────


def test_past_company_any_of_matches_either_not_both():
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="co", type="past_company", values=["Google", "Meta"],
                        operator="ANY_OF", weight=100, required=True),
    ])
    ex_google = score_candidate(_facts(_Person(), exps=[_Exp("SWE", "Google", 2018, 2021, False)]), plan)
    ex_meta = score_candidate(_facts(_Person(), exps=[_Exp("SWE", "Meta", 2019, 2022, False)]), plan)
    neither = score_candidate(_facts(_Person(), exps=[_Exp("SWE", "Oracle", 2019, 2022, False)]), plan)
    assert ex_google.match_score > 0 and ex_meta.match_score > 0
    assert neither.excluded_reason is not None


# ─────────────────── §35 keyword traps ───────────────────


def test_amazon_api_mention_is_not_amazon_employment():
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="amz", type="past_company", value="Amazon", weight=100, required=True),
    ])
    p = _Person(current_company="Acme")
    exp = _Exp("Backend Engineer", "Acme", 2020, None, True,
               desc="Built an integration with the Amazon Selling Partner API")
    scored = score_candidate(_facts(p, exps=[exp]), plan)
    assert scored.excluded_reason is not None  # description mention != an employment row


def test_startup_weekend_attendance_is_not_working_at_a_startup():
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="su", type="company_category", concept="startup",
                        scope="current_company", weight=100, required=True),
    ])
    ctx = ScoringContext(company_class={
        company_key("111", "Deloitte"): {"is_startup": False, "confidence": 0.95,
                                         "provenance": "ai_company_inference", "industries": ["consulting"], "categories": []},
    })
    p = _Person(current_company="Deloitte", about="Attended Startup Weekend 2021, loved it")
    scored = score_candidate(_facts(p, exps=[_Exp("Consultant", "Deloitte", 2019, None, True, company_id="111")]), plan, ctx)
    assert scored.excluded_reason is not None


# ─────────────────── plan shape (deterministic fallback) ───────────────────


def test_deterministic_fallback_preserves_location_or_when_llm_is_down():
    p = _deterministic_parse("Who should I invite to a CXO event in Memphis or Nashville?")
    loc = next(c for c in p.criteria if c.type == "location")
    assert set(loc.values) == {"Memphis", "Nashville"}
    assert loc.operator == "ANY_OF" and loc.required is True


def test_deterministic_fallback_startup_is_company_category_not_a_company():
    p = _deterministic_parse("Former Amazon people now at startups")
    types = {c.type for c in p.criteria}
    assert "company_category" in types
    assert not any(c.type == "current_company" and "startup" in (c.value or "").lower() for c in p.criteria)
