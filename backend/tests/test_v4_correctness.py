"""V4 PART 1 correctness fixes (spec steps 1-11) + code-review findings #1-#3, #5, #9."""
from __future__ import annotations

from app.constants import CriterionType, Operator, Qualification, Scope, TriState
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.company_intel import company_key
from app.services.query_interpreter import interpret_query
from app.services.scoring import ProfileFacts, ScoringContext, _score_one, score_candidate
from tests.test_search import _Exp, _Person


def _facts(person=None, exps=None, sem=None):
    return ProfileFacts(person=person or _Person(), experiences=exps or [], education=[],
                        skills=[], semantic=sem or {}, embedding=None)


def _crit(**kw):
    kw.setdefault("weight", 100)
    return SearchCriterion(**kw)


# ─────────────────── §1/§2 strict current / past scope ───────────────────


def _former_amazon_plan():
    return ParsedSearchQuery(criteria=[_crit(id="a", type=CriterionType.PAST_COMPANY,
                                             value="Amazon", required=True, scope=Scope.PAST_COMPANY)])


def test_current_amazon_only_fails_former_amazon():
    p = _facts(exps=[_Exp("SDE", "Amazon", 2021, None, True, id="e1")])
    assert score_candidate(p, _former_amazon_plan()).qualification == Qualification.NOT_MATCH


def test_past_amazon_and_current_microsoft_matches_former_amazon():
    p = _facts(exps=[_Exp("SDE", "Amazon", 2016, 2020, False, id="e1"),
                     _Exp("PM", "Microsoft", 2020, None, True, id="e2")])
    assert score_candidate(p, _former_amazon_plan()).qualification == Qualification.EXACT_MATCH


def test_past_and_current_amazon_satisfies_past_criterion():
    p = _facts(exps=[_Exp("SDE", "Amazon", 2015, 2019, False, id="e1"),
                     _Exp("Sr SDE", "Amazon", 2019, None, True, id="e2")])
    assert score_candidate(p, _former_amazon_plan()).qualification == Qualification.EXACT_MATCH


def test_past_amazon_only_fails_currently_at_amazon():
    plan = ParsedSearchQuery(criteria=[_crit(id="c", type=CriterionType.CURRENT_COMPANY,
                                             value="Amazon", required=True, scope=Scope.CURRENT_COMPANY)])
    p = _facts(exps=[_Exp("SDE", "Amazon", 2015, 2019, False, id="e1")])
    assert score_candidate(p, plan).qualification == Qualification.NOT_MATCH


# ─────────────────── §2 experience-semantics scope ───────────────────


def test_current_role_function_not_satisfied_by_a_past_role():
    sem = {"experience_semantics": [
        {"experience_id": "e1", "role_function": "software engineering", "confidence": 0.9},
        {"experience_id": "e2", "role_function": "product management", "confidence": 0.9},
    ]}
    exps = [_Exp("SWE", "Co", 2018, 2021, False, id="e1"),
            _Exp("PM", "Co", 2022, None, True, id="e2")]
    crit = _crit(id="r", type=CriterionType.ROLE_FUNCTION, concept="software engineering",
                 scope=Scope.CURRENT, required=True)
    assert score_candidate(_facts(exps=exps, sem=sem),
                           ParsedSearchQuery(criteria=[crit])).qualification != Qualification.EXACT_MATCH

    crit_past = _crit(id="r", type=CriterionType.ROLE_FUNCTION, concept="software engineering",
                      scope=Scope.PAST, required=True)
    assert score_candidate(_facts(exps=exps, sem=sem),
                           ParsedSearchQuery(criteria=[crit_past])).qualification == Qualification.EXACT_MATCH


# ─────────────────── §4 multi-value semantic type preservation ───────────────────


def test_multi_value_role_function_keeps_type():
    sem = {"experience_semantics": [
        {"experience_id": "e1", "role_function": "cloud engineering", "confidence": 0.9}]}
    exps = [_Exp("Cloud Eng", "Co", 2020, None, True, id="e1")]
    crit = _crit(id="r", type=CriterionType.ROLE_FUNCTION,
                 values=["security engineering", "cloud engineering"], operator=Operator.ANY_OF,
                 scope=Scope.CAREER)
    _s, _e, status = _score_one(_facts(exps=exps, sem=sem), crit, ScoringContext())
    assert status == TriState.TRUE


# ─────────────────── §5 tri-state semantic NOT ───────────────────


def test_semantic_not_true_becomes_false():
    sem = {"semantic_assertions": [{"concept": "frontend development focus", "confidence": 0.9,
                                    "evidence": ["Frontend Engineer"], "scope": "career"}]}
    crit = _crit(id="n", type=CriterionType.PROFESSIONAL_CONCEPT, value="frontend development focus",
                 operator=Operator.NOT, required=True)
    _s, _e, status = _score_one(_facts(sem=sem), crit, ScoringContext())
    assert status == TriState.FALSE


def test_semantic_not_unknown_stays_unknown():
    crit = _crit(id="n", type=CriterionType.PROFESSIONAL_CONCEPT, value="underwater basket weaving",
                 operator=Operator.NOT, required=True)
    _s, _e, status = _score_one(_facts(sem={}), crit, ScoringContext())
    assert status == TriState.UNKNOWN


# ─────────────────── §7 company tri-state confidence ───────────────────


def _startup_now_plan():
    return ParsedSearchQuery(criteria=[_crit(id="s", type=CriterionType.COMPANY_CATEGORY,
                                             concept="startup", scope=Scope.CURRENT_COMPANY, required=True)])


def test_low_confidence_startup_true_is_not_exact():
    ctx = ScoringContext(company_class={
        company_key("7", "MaybeCo"): {"is_startup": True, "confidence": 0.4,
                                      "provenance": "ai_company_inference", "industries": [], "categories": []}})
    p = _facts(exps=[_Exp("Eng", "MaybeCo", 2022, None, True, id="e1", company_id="7")])
    assert score_candidate(p, _startup_now_plan(), ctx).qualification == Qualification.POSSIBLE_MATCH


def test_high_confidence_startup_true_is_exact():
    ctx = ScoringContext(company_class={
        company_key("7", "RealCo"): {"is_startup": True, "confidence": 0.9,
                                     "provenance": "ai_company_inference", "industries": [], "categories": []}})
    p = _facts(exps=[_Exp("Eng", "RealCo", 2022, None, True, id="e1", company_id="7")])
    assert score_candidate(p, _startup_now_plan(), ctx).qualification == Qualification.EXACT_MATCH


# ─────────────────── §8 company classification cache-only during search ───────────────────


def test_pool_company_class_never_calls_the_llm(monkeypatch):
    from app.services import search_service

    monkeypatch.setattr("app.services.company_intel.get_or_classify",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not classify during search")))
    import app.repositories as repo
    monkeypatch.setattr(repo, "get_company_semantics", lambda db, keys: {})
    plan = ParsedSearchQuery(criteria=[_crit(id="s", type=CriterionType.COMPANY_CATEGORY,
                                             concept="startup", scope=Scope.CURRENT_COMPANY)])
    exp_by_person = {"p1": [_Exp("Eng", "Unclassified Co", 2022, None, True, id="e1", company_id="999")]}
    out = search_service._pool_company_class(object(), plan, exp_by_person)
    assert out == {}


# ─────────────────── §9 confidence cap not overwritten ───────────────────


def test_validator_confidence_cap_survives_finalize():
    parsed, _prov, _ = interpret_query("people who like cats or something entirely vague")
    assert parsed.interpretation_confidence <= 0.5
    assert parsed.interpretation_confidence_cap <= 0.5


# ─────────────────── §10 conservative missing-date chronology ───────────────────


def test_transition_with_missing_dates_is_unknown_not_false():
    from app.services.career_chronology import score_transition

    exps = [_Exp("Consultant", "Bain", None, None, False, id="e1", desc="strategy consulting"),
            _Exp("Software Engineer", "Datadog", None, None, True, id="e2", desc="backend at a tech company")]
    crit = _crit(id="t", type=CriterionType.CAREER_TRANSITION, concept="from consulting to technology")
    _s, _e, status = score_transition(_facts(exps=exps), crit)
    assert status == TriState.UNKNOWN


def test_transition_with_contradicting_dates_is_false():
    from app.services.career_chronology import score_transition

    exps = [_Exp("Technology Lead", "Datadog", 2015, 2019, False, id="e1", desc="technology platform work"),
            _Exp("Consulting Partner", "Bain", 2020, 2023, False, id="e2", desc="strategy consulting engagements")]
    crit = _crit(id="t", type=CriterionType.CAREER_TRANSITION, concept="from consulting to technology")
    _s, _e, status = score_transition(_facts(exps=exps), crit)
    assert status == TriState.FALSE
