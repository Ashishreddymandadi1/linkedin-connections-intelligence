"""Career chronology + transitions + years-experience (V4 PART D — §18-21, §38, §39)."""
from __future__ import annotations

from app.constants import CriterionType, Scope, TriState
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.career_chronology import (
    ordered_experiences,
    score_transition,
    score_years_experience,
    total_years_matching,
)
from app.services.scoring import ProfileFacts, score_candidate
from tests.test_search import _Exp, _Person


def _facts(exps, sem=None):
    return ProfileFacts(person=_Person(), experiences=exps, education=[], skills=[],
                        semantic=sem or {}, embedding=None)


# ─────────────────── §18 ordering ───────────────────


def test_ordered_experiences_oldest_first():
    a = _Exp("new", "C", 2021, None, True, id="a")
    b = _Exp("old", "B", 2015, 2018, False, id="b")
    c = _Exp("mid", "A", 2019, 2021, False, id="c")
    assert [e.id for e in ordered_experiences([a, b, c])] == ["b", "c", "a"]


def test_total_years_merges_overlapping_roles():
    ft = _Exp("Engineer", "C", 2015, 2020, False, id="ft")
    advisory = _Exp("Advisor", "D", 2018, 2019, False, id="ad")  # fully inside ft
    # the concurrent advisory role adds NOTHING — no double counting (V4 §21)
    assert total_years_matching([ft, advisory], lambda e: True) == total_years_matching([ft], lambda e: True)


# ─────────────────── §38 transition verifies order ───────────────────

_CONSULTING_THEN_TECH = [
    _Exp("Management Consultant", "Bain & Company", 2017, 2020, False, id="c1",
         desc="strategy consulting engagements"),
    _Exp("Software Engineer", "Datadog", 2021, None, True, id="c2",
         desc="backend services at a technology company"),
]
_TECH_THEN_SIDE_CONSULTING = [
    _Exp("Software Engineer", "Datadog", 2017, None, True, id="d1",
         desc="backend at a technology company"),
    _Exp("Consulting Advisor", "Self-employed", 2022, 2022, False, id="d2",
         desc="occasional consulting on the side"),
]


def _transition_crit():
    return SearchCriterion(id="t", type=CriterionType.CAREER_TRANSITION,
                           concept="from consulting to technology", scope=Scope.CAREER, weight=100)


def test_consulting_then_tech_is_a_true_transition():
    s, ev, status = score_transition(_facts(_CONSULTING_THEN_TECH), _transition_crit())
    assert status == TriState.TRUE and s > 0.7 and ev


def test_tech_then_side_consulting_is_not_the_same_transition():
    s, ev, status = score_transition(_facts(_TECH_THEN_SIDE_CONSULTING), _transition_crit())
    assert status != TriState.TRUE


def test_transition_through_score_candidate_ranks_correctly():
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="t", type=CriterionType.CAREER_TRANSITION,
                        concept="from consulting to technology", weight=100, required=True),
    ])
    good = score_candidate(_facts(_CONSULTING_THEN_TECH), plan)
    bad = score_candidate(_facts(_TECH_THEN_SIDE_CONSULTING), plan)
    assert good.match_score > bad.match_score
    assert bad.excluded_reason is not None


# ─────────────────── §39 years experience ───────────────────


def test_years_experience_counts_only_relevant_roles():
    exps = [
        _Exp("Backend Engineer", "A", 2013, 2018, False, id="b1", desc="backend systems"),
        _Exp("Backend Engineer", "B", 2018, None, True, id="b2", desc="backend platform"),
        _Exp("Barista", "Coffee Co", 2010, 2013, False, id="x1", desc="made coffee"),
    ]
    crit = SearchCriterion(id="y", type=CriterionType.YEARS_EXPERIENCE, value="10",
                           concept="at least 10 years in backend", weight=100)
    s, ev, status = score_years_experience(_facts(exps), crit)
    assert status == TriState.TRUE and s > 0.5


def test_years_experience_falls_short():
    exps = [_Exp("Backend Engineer", "A", 2020, None, True, id="b1", desc="backend")]
    crit = SearchCriterion(id="y", type=CriterionType.YEARS_EXPERIENCE, value="10",
                           concept="at least 10 years in backend", weight=100)
    _s, _ev, status = score_years_experience(_facts(exps), crit)
    assert status == TriState.FALSE


def test_years_experience_no_matching_role_is_unknown_not_false():
    exps = [_Exp("Barista", "Coffee Co", 2010, None, True, id="x1", desc="coffee")]
    crit = SearchCriterion(id="y", type=CriterionType.YEARS_EXPERIENCE, value="10",
                           concept="at least 10 years in backend engineering", weight=100)
    _s, _ev, status = score_years_experience(_facts(exps), crit)
    assert status == TriState.UNKNOWN
