"""V4 PART 3.6 — negative-evidence authority + packet-limit correctness.

  §1/§10/§11  the LLM cannot declare source data complete — an absence-based
              semantic FALSE needs the BACKEND completeness policy behind it
  §3/§14      PAST_COMPANY NOT uses the SAME strong work-history rule as the gate
  §4/§12      a FALSE must have SCOPE-appropriate contradicting evidence
  §5/§13      a career-wide FALSE needs career-wide coverage, not one stray ref
  §6          assertion refs carry scope derived from their linked experiences
  §7/§15      _fit_size() hard-enforces the per-packet char cap
  §8/§16      an oversized single packet never becomes an oversized request
"""
from __future__ import annotations

from app.constants import CriterionType, Operator, Qualification, Scope, TriState
from app.schemas import (
    JudgeBatch,
    JudgeCriterionVerdict,
    JudgePersonVerdict,
    ParsedSearchQuery,
    SearchCriterion,
)
from app.services import judge_packet, semantic_judge
from app.services.judge_packet import _size, build_packets, packet_refs
from app.services.judge_validator import validate_person
from app.services.scoring import ProfileFacts, ScoringContext, score_candidate
from tests.test_search import _Exp, _Person


def _pf(person, exps=None, sem=None, edu=None, pubs=None):
    return ProfileFacts(person=person, experiences=exps or [], education=edu or [], skills=[],
                        semantic=sem or {}, embedding=None, publications=pubs or [])


def _crit(**kw):
    kw.setdefault("weight", 100)
    return SearchCriterion(**kw)


def _plan(*c):
    return ParsedSearchQuery(criteria=list(c))


class _Row:
    def __init__(self, _id=None, **kw):
        self.id = _id
        for k, v in kw.items():
            setattr(self, k, v)


# ─────────────────────── §12 — wrong-scope FALSE ───────────────────────


def _scoped_packet():
    return {"person_id": "p0",
            "current": {"experience_id": "cur_pm", "title": "Product Manager", "is_current": True},
            "past": [{"experience_id": "past_pm", "title": "Product Manager", "is_current": False}],
            "education": [], "experience_semantics": [], "company_classifications": [],
            "skills": [], "certifications": [], "semantic_assertions": []}


def test_current_scoped_false_cited_by_a_past_role_is_unknown():
    crit = _crit(id="r", type=CriterionType.ROLE_FUNCTION, concept="software engineering",
                 scope=Scope.CURRENT, required=True)
    facts = _pf(_Person(), [_Exp("Product Manager", "Co", 2016, 2020, False, id="past_pm")])
    raw = {"r": {"criterion_id": "r", "status": "false", "match_strength": 0.1,
                 "contradicting_evidence_refs": ["exp:past_pm"],
                 "reason": "was a Product Manager, not a software engineer"}}
    out = validate_person(raw, _scoped_packet(), _plan(crit), facts, ScoringContext())
    assert out["r"]["status"] == TriState.UNKNOWN
    assert any("scope" in n.lower() for n in out["r"]["validation"]["notes"])


def test_current_scoped_false_cited_by_the_current_role_stays_false():
    crit = _crit(id="r", type=CriterionType.ROLE_FUNCTION, concept="software engineering",
                 scope=Scope.CURRENT, required=True)
    facts = _pf(_Person(current_title="Product Manager"),
                [_Exp("Product Manager", "Co", 2021, None, True, id="cur_pm")])
    raw = {"r": {"criterion_id": "r", "status": "false", "match_strength": 0.1,
                 "contradicting_evidence_refs": ["exp:cur_pm"],
                 "reason": "current role is Product Manager"}}
    out = validate_person(raw, _scoped_packet(), _plan(crit), facts, ScoringContext())
    assert out["r"]["status"] == TriState.FALSE


# ─────────────────────── §13 — career FALSE needs coverage ───────────────────────


def test_career_false_grounded_by_one_ref_is_unknown():
    crit = _crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="career-long mentoring experience", scope=Scope.CAREER, required=True)
    pkt = {"person_id": "p0", "current": {"experience_id": "e5", "is_current": True},
           "past": [{"experience_id": f"e{i}", "is_current": False} for i in range(1, 5)],
           "education": [], "experience_semantics": [], "company_classifications": [],
           "skills": [], "certifications": [], "semantic_assertions": []}
    facts = _pf(_Person(completeness=80),
                [_Exp("SWE", "Co", 2010 + i, 2011 + i, i == 4, id=f"e{i + 1}") for i in range(5)])
    raw = {"m": {"criterion_id": "m", "status": "false", "match_strength": 0.2,
                 "contradicting_evidence_refs": ["exp:e1"],
                 "reason": "this one early role was a pure individual contributor"}}
    out = validate_person(raw, pkt, _plan(crit), facts, ScoringContext())
    assert out["m"]["status"] == TriState.UNKNOWN
    assert any("career-wide" in n.lower() for n in out["m"]["validation"]["notes"])


def test_career_false_with_broad_coverage_may_stand():
    crit = _crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="career-long mentoring experience", scope=Scope.CAREER, required=True)
    pkt = {"person_id": "p0", "current": {"experience_id": "e3", "is_current": True},
           "past": [{"experience_id": "e1", "is_current": False},
                    {"experience_id": "e2", "is_current": False}],
           "education": [], "experience_semantics": [], "company_classifications": [],
           "skills": [], "certifications": [], "semantic_assertions": []}
    facts = _pf(_Person(completeness=80),
                [_Exp("SWE", "A", 2014, 2017, False, id="e1"),
                 _Exp("SWE", "B", 2017, 2020, False, id="e2"),
                 _Exp("SWE", "C", 2020, None, True, id="e3")])
    raw = {"m": {"criterion_id": "m", "status": "false", "match_strength": 0.2,
                 "contradicting_evidence_refs": ["exp:e1", "exp:e2", "exp:e3"],
                 "reason": "all three roles are individual-contributor with no reports"}}
    out = validate_person(raw, pkt, _plan(crit), facts, ScoringContext())
    assert out["m"]["status"] == TriState.FALSE


# ─────────────────────── §6 — assertion ref scope ───────────────────────


def test_current_scoped_true_supported_only_by_a_past_assertion_is_unknown():
    crit = _crit(id="r", type=CriterionType.ROLE_FUNCTION, concept="software engineering",
                 scope=Scope.CURRENT, required=True)
    pkt = {"person_id": "p0", "current": {"experience_id": "cur", "is_current": True},
           "past": [{"experience_id": "old", "is_current": False}],
           "education": [], "experience_semantics": [], "company_classifications": [],
           "skills": [], "certifications": [],
           "semantic_assertions": [{"concept": "software engineering", "scope": "past",
                                    "experience_ids": ["old"], "confidence": 0.9, "evidence": []}]}
    facts = _pf(_Person(), [_Exp("Manager", "Co", 2022, None, True, id="cur"),
                            _Exp("SWE", "Co", 2018, 2021, False, id="old")])
    raw = {"r": {"criterion_id": "r", "status": "true", "match_strength": 0.8,
                 "supporting_evidence_refs": ["assertion:0"], "reason": "an assertion says software engineering"}}
    out = validate_person(raw, pkt, _plan(crit), facts, ScoringContext())
    assert out["r"]["status"] == TriState.UNKNOWN


# ─────────────────────── §14 — PAST_COMPANY NOT completeness ───────────────────────


def _not_amazon():
    return _plan(_crit(id="np", type=CriterionType.PAST_COMPANY, value="Amazon",
                       operator=Operator.NOT, scope=Scope.PAST_COMPANY, required=True))


def _roles(n, *, dated=True, current_last=True):
    out = []
    for i in range(n):
        is_cur = current_last and i == n - 1
        sy = 2010 + i if dated else None
        ey = None if is_cur else (2011 + i if dated else None)
        out.append(_Exp("SWE", f"Co{i}", sy, ey, is_cur, id=f"e{i}"))
    return out


def test_not_past_amazon_one_role_is_unknown():
    assert score_candidate(_pf(_Person(completeness=90), _roles(1)),
                           _not_amazon()).qualification == Qualification.POSSIBLE_MATCH


def test_not_past_amazon_two_roles_is_unknown():
    assert score_candidate(_pf(_Person(completeness=90), _roles(2)),
                           _not_amazon()).qualification == Qualification.POSSIBLE_MATCH


def test_not_past_amazon_three_dated_roles_amazon_absent_is_true():
    assert score_candidate(_pf(_Person(completeness=90), _roles(3)),
                           _not_amazon()).qualification == Qualification.EXACT_MATCH


def test_not_past_amazon_three_roles_one_undated_is_unknown():
    exps = _roles(3)
    exps[1].start_year = None
    assert score_candidate(_pf(_Person(completeness=90), exps),
                           _not_amazon()).qualification == Qualification.POSSIBLE_MATCH


def test_not_past_amazon_present_is_false_regardless_of_completeness():
    exps = [_Exp("SDE", "Amazon", 2018, 2021, False, id="e0"),
            _Exp("SWE", "Co", 2021, None, True, id="e1")]
    assert score_candidate(_pf(_Person(completeness=10), exps),
                           _not_amazon()).qualification == Qualification.NOT_MATCH


# ─────────────────────── §15 — packet size invariant ───────────────────────


def _rich_bundle():
    p = _Person(current_title="Principal Engineer", current_company="BigCo",
                headline="Principal Engineer | " + "many words " * 40)
    p.id = "rich0"
    long = "x" * 1400
    exps = [_Exp(f"Role {i}", f"Company {i}", 2000 + i, 2001 + i, i == 19, id=f"e{i}", desc=long)
            for i in range(20)]
    sem = {"career_summary": "y" * 900,
           "semantic_assertions": [{"concept": f"concept {i}", "category": "domain_expertise",
                                    "scope": "career", "confidence": 0.8,
                                    "evidence": ["ev " * 40, "ev2 " * 40]} for i in range(12)]}
    facts = _pf(p, exps, sem=sem,
                pubs=[_Row(f"pub_{i}", title=f"Paper {i} " * 10, description="d" * 300) for i in range(8)])
    facts.skills = [_Row(skill_name=f"skill {i}") for i in range(40)]
    facts.certifications = [_Row(f"cert_{i}", name=f"Certification number {i}") for i in range(15)]
    vols = [_Row(f"vol_{i}", role="Mentor", organization=f"Org {i}", description="v" * 300) for i in range(8)]
    recs = [_Row(f"rec_{i}", text="r" * 400) for i in range(5)]
    return [(p, facts, {"volunteering": vols, "recommendations": recs})]


def test_fit_size_hard_enforces_the_packet_cap(monkeypatch):
    monkeypatch.setattr(judge_packet.settings, "semantic_judge_max_packet_chars", 1800)
    plan = _plan(_crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT,
                       concept="mentoring and leadership", required=True))
    pkt = build_packets(_rich_bundle(), plan, ScoringContext(), query="career mentors")[0]

    assert _size(pkt) <= 1800, f"packet is {_size(pkt)} chars, cap 1800"
    assert pkt["person_id"] == "rich0"
    assert pkt.get("current") and pkt["current"].get("experience_id")
    for r in packet_refs(pkt):
        assert r.split(":", 1)[0] in {"exp", "edu", "cert", "skill", "assertion", "company", "pub", "vol", "rec"}


def test_fit_size_flags_packet_too_large_when_impossible(monkeypatch):
    monkeypatch.setattr(judge_packet.settings, "semantic_judge_max_packet_chars", 60)
    plan = _plan(_crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", required=True))
    pkt = build_packets(_rich_bundle(), plan, ScoringContext(), query="x")[0]
    assert pkt.get("_packet_too_large") is True


# ─────────────────────── §16 — oversized single packet safety ───────────────────────


def test_oversized_single_packet_does_not_crash_and_stays_unknown(monkeypatch):
    small = {"person_id": "small", "current": {"experience_id": "e1", "is_current": True},
             "past": [], "education": [], "experience_semantics": [], "company_classifications": [],
             "skills": [], "certifications": [], "semantic_assertions": []}
    huge = dict(small, person_id="huge", blob="z" * 200_000)

    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge, "build_packets", lambda *a, **k: [small, huge])

    seen = []

    def fake_batch(payload, packets):
        seen.extend(p["person_id"] for p in packets)
        return ("ok", JudgeBatch(people=[JudgePersonVerdict(
            person_id=p["person_id"], overall_fit="moderate",
            criteria=[JudgeCriterionVerdict(criterion_id=c["id"], status="true", match_strength=0.7,
                                            supporting_evidence_refs=["exp:e1"], experience_ids=["e1"])
                      for c in payload["criteria_to_judge"]])
            for p in packets]), "mock", "m1")

    monkeypatch.setattr(semantic_judge, "_call_judge", fake_batch)

    plan = _plan(_crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT, concept="mentoring", required=True))
    bundle = [(_Person(), _pf(_Person()), {}), (_Person(), _pf(_Person()), {})]
    run = semantic_judge.run_judge("mentors", plan, bundle, ScoringContext(),
                                   network_size=2, pool_size=2, hard_rejected_count=0)

    assert "small" in seen and "huge" not in seen
    assert run.metadata.oversized_packets == 1
    assert run.metadata.judge_status == "partial"
    assert run.verdicts["huge"]["m"].get("judge_missing") is True
    assert run.verdicts["huge"]["m"]["status"] == TriState.UNKNOWN
    assert run.verdicts["small"]["m"]["status"] == "true"
