"""Hard-fact viability gate (V4 PART 3 §4–§7, §45, §49).

The gate rejects a candidate before the LLM judge ONLY on a verified
contradiction the judge could not reasonably overturn. It must NEVER reject for
a weak local semantic signal, a missing enrichment, or an UNKNOWN classification.
"""
from __future__ import annotations

from app.constants import CriterionType, Modality, Operator, Scope, TriState
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.candidate_gate import hard_gate
from app.services.company_intel import company_key
from app.services.scoring import ProfileFacts, ScoringContext
from tests.test_search import _Exp, _Person


def _facts(person, exps=None, edu=None, sem=None):
    return ProfileFacts(person=person, experiences=exps or [], education=edu or [],
                        skills=[], semantic=sem or {}, embedding=None)


def _plan(*crits):
    return ParsedSearchQuery(criteria=list(crits))


def _c(**kw):
    kw.setdefault("weight", 100)
    kw.setdefault("required", True)
    return SearchCriterion(**kw)


# ─────────────────────── §5 — safe hard rejections ───────────────────────


def test_required_location_verified_different_is_rejected():
    p = _Person(location_text="Atlanta, Georgia, United States")
    d = hard_gate(_facts(p), _plan(_c(id="loc", type=CriterionType.LOCATION, value="Nashville")),
                  ScoringContext())
    assert d.viable is False and "location" in d.rejection_reason.lower()


def test_required_location_missing_data_stays_viable():
    p = _Person()  # no location at all
    d = hard_gate(_facts(p), _plan(_c(id="loc", type=CriterionType.LOCATION, value="Nashville")),
                  ScoringContext())
    assert d.viable is True and d.hard_fact_statuses["loc"] == TriState.UNKNOWN


def test_required_current_company_verified_different_is_rejected():
    p = _Person(current_company="Microsoft")
    exps = [_Exp("PM", "Microsoft", 2020, None, True, id="e1")]
    d = hard_gate(_facts(p, exps),
                  _plan(_c(id="cur", type=CriterionType.CURRENT_COMPANY, value="Google",
                           scope=Scope.CURRENT_COMPANY)), ScoringContext())
    assert d.viable is False


def test_not_current_company_verified_present_is_rejected():
    p = _Person(current_company="Amazon")
    exps = [_Exp("SDE", "Amazon", 2021, None, True, id="e1")]
    d = hard_gate(_facts(p, exps),
                  _plan(_c(id="not_amz", type=CriterionType.CURRENT_COMPANY, value="Amazon",
                           operator=Operator.NOT, scope=Scope.CURRENT_COMPANY)), ScoringContext())
    assert d.viable is False


def test_required_past_company_absent_from_history_is_rejected():
    p = _Person(completeness=85)
    exps = [_Exp("SWE", "Stripe", 2019, 2022, False, id="e1"),
            _Exp("SWE", "Datadog", 2022, None, True, id="e2")]
    d = hard_gate(_facts(p, exps),
                  _plan(_c(id="past", type=CriterionType.PAST_COMPANY, value="Amazon",
                           scope=Scope.PAST_COMPANY)), ScoringContext())
    assert d.viable is False


def test_required_past_company_no_history_stays_viable():
    p = _Person(completeness=0)
    d = hard_gate(_facts(p, exps=[]),
                  _plan(_c(id="past", type=CriterionType.PAST_COMPANY, value="Amazon",
                           scope=Scope.PAST_COMPANY)), ScoringContext())
    assert d.viable is True


def test_high_confidence_false_company_category_is_rejected():
    ctx = ScoringContext(company_class={
        company_key("99", "Microsoft"): {"is_startup": False, "confidence": 0.96,
                                         "provenance": "ai_company_inference", "industries": [], "categories": []},
    })
    p = _Person(current_company="Microsoft")
    exps = [_Exp("Principal PM", "Microsoft", 2018, None, True, id="e1", company_id="99")]
    d = hard_gate(_facts(p, exps),
                  _plan(_c(id="su", type=CriterionType.COMPANY_CATEGORY, concept="startup",
                           scope=Scope.CURRENT_COMPANY)), ctx)
    assert d.viable is False and d.hard_fact_statuses["su"] == TriState.FALSE


def test_career_transition_chronology_contradiction_is_rejected():
    exps = [_Exp("Technology Lead", "Datadog", 2015, 2019, False, id="e1", desc="technology platform work"),
            _Exp("Consulting Partner", "Bain", 2020, 2023, False, id="e2", desc="strategy consulting engagements")]
    d = hard_gate(_facts(_Person(), exps),
                  _plan(_c(id="t", type=CriterionType.CAREER_TRANSITION,
                           concept="from consulting to technology")), ScoringContext())
    assert d.viable is False


def test_career_transition_incomplete_dates_stays_viable():
    exps = [_Exp("Consultant", "Bain", None, None, False, id="e1", desc="strategy consulting"),
            _Exp("Software Engineer", "Datadog", None, None, True, id="e2", desc="backend at a tech company")]
    d = hard_gate(_facts(_Person(), exps),
                  _plan(_c(id="t", type=CriterionType.CAREER_TRANSITION,
                           concept="from consulting to technology")), ScoringContext())
    assert d.viable is True


# ─────────────────────── §6 — things that MUST NOT pre-reject ───────────────────────


def test_weak_semantic_signal_never_rejects():
    p = _Person(current_title="Software Engineer", current_company="SomeCo")
    exps = [_Exp("Software Engineer", "SomeCo", 2020, None, True, id="e1")]
    plan = _plan(_c(id="mentor", type=CriterionType.PROFESSIONAL_CONCEPT,
                    concept="evidence of mentoring, coaching and people leadership"))
    assert hard_gate(_facts(p, exps), plan, ScoringContext()).viable is True


def test_unknown_company_category_stays_viable():
    p = _Person(current_company="Obscure Co")
    exps = [_Exp("Engineer", "Obscure Co", 2022, None, True, id="e1", company_id="555")]
    d = hard_gate(_facts(p, exps),
                  _plan(_c(id="su", type=CriterionType.COMPANY_CATEGORY, concept="startup",
                           scope=Scope.CURRENT_COMPANY)), ScoringContext())
    assert d.viable is True and d.hard_fact_statuses["su"] == TriState.UNKNOWN


def test_hipaa_not_literally_mentioned_stays_viable():
    p = _Person(current_title="Security Engineer", current_company="HealthCo")
    exps = [_Exp("Security Engineer", "HealthCo", 2021, None, True, id="e1",
                 desc="patient data protection and access controls")]
    plan = _plan(_c(id="hipaa", type=CriterionType.PROFESSIONAL_CONCEPT,
                    concept="HIPAA compliance experience", required=True))
    assert hard_gate(_facts(p, exps), plan, ScoringContext()).viable is True


# ─────────────────────── §7 — modality never hard-rejects ───────────────────────


def test_modality_possible_never_rejects_even_if_required_true():
    p = _Person(current_title="SWE", current_company="Co")
    plan = _plan(SearchCriterion(id="maybe", type=CriterionType.PROFESSIONAL_CONCEPT,
                                 concept="HIPAA compliance experience", weight=100,
                                 required=True, modality=Modality.POSSIBLE))
    assert hard_gate(_facts(p, [_Exp("SWE", "Co", 2020, None, True, id="e1")]),
                     plan, ScoringContext()).viable is True


# ─────────────────────── §45 — hard-gate unit test ───────────────────────


def test_hard_gate_100_candidates_50_location_rejected():
    plan = _plan(_c(id="loc", type=CriterionType.LOCATION, value="Nashville"))
    decisions = []
    for i in range(100):
        loc = "Nashville, Tennessee, United States" if i % 2 == 0 else "Atlanta, Georgia, United States"
        p = _Person(location_text=loc)
        p.id = f"p{i}"
        decisions.append(hard_gate(_facts(p), plan, ScoringContext()))

    rejected = [d for d in decisions if not d.viable]
    viable = [d for d in decisions if d.viable]
    assert len(rejected) == 50 and len(viable) == 50
    assert all("atlanta" in d.rejection_reason.lower() for d in rejected)
