"""V4 PART 3.5 — exhaustive judge correctness hardening.

  §1  a semantic FALSE must be GROUNDED (contradiction / deterministic / explicit
      complete-data negative) — missing evidence is UNKNOWN, never FALSE, and an
      unsupported FALSE can never bury a deterministic TRUE
  §2  publications / volunteering / recommendations carry validatable refs
      (pub: / vol: / rec:)
  §3  factual NOT is tri-state — absence of data is UNKNOWN, not a satisfied NOT
  §4  absence-based hard gating is conservative (see test_candidate_gate.py)
"""
from __future__ import annotations

from app.constants import CriterionType, Operator, Qualification, Scope, TriState
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.judge_packet import build_packets, packet_refs
from app.services.judge_validator import validate_person
from app.services.scoring import ProfileFacts, ScoringContext, score_candidate
from tests.test_search import _Exp, _Person


def _pf(person, exps=None, sem=None, edu=None, pubs=None):
    return ProfileFacts(person=person, experiences=exps or [], education=edu or [], skills=[],
                        semantic=sem or {}, embedding=None, publications=pubs or [])


def _crit(**kw):
    kw.setdefault("weight", 100)
    return SearchCriterion(**kw)


def _plan(*crits):
    return ParsedSearchQuery(criteria=list(crits))


# ─────────────────────── §1 — grounded FALSE ───────────────────────


def _mentor_crit():
    return _crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="evidence of mentoring and people leadership", required=True,
                 scope=Scope.CAREER)


def _packet_min(pid="p0"):
    return {
        "person_id": pid,
        "current": {"experience_id": "e1", "title": "Software Engineer", "company": "Co", "is_current": True},
        "past": [], "education": [], "experience_semantics": [], "company_classifications": [],
        "skills": [], "certifications": [], "semantic_assertions": [], "publications": [],
    }


def _facts_plain():
    p = _Person(current_title="Software Engineer", current_company="Co")
    return _pf(p, [_Exp("Software Engineer", "Co", 2020, None, True, id="e1")])


def test_semantic_false_with_no_evidence_downgrades_to_unknown():
    raw = {"m": {"criterion_id": "m", "status": "false", "match_strength": 0.8,
                 "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
                 "reason": "did not see mentoring evidence"}}
    out = validate_person(raw, _packet_min(), _plan(_mentor_crit()), _facts_plain(), ScoringContext())
    assert out["m"]["status"] == TriState.UNKNOWN
    assert any("missing" in n.lower() for n in out["m"]["validation"]["notes"])


def test_semantic_false_with_incomplete_profile_downgrades_to_unknown():
    p = _Person(current_title="Engineer", completeness=15)
    facts = _pf(p, [_Exp("Engineer", "Co", 2022, None, True, id="e1")])
    raw = {"m": {"criterion_id": "m", "status": "false", "match_strength": 0.6,
                 "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
                 "reason": "profile is thin, no sign of mentoring"}}
    out = validate_person(raw, _packet_min(), _plan(_mentor_crit()), facts, ScoringContext())
    assert out["m"]["status"] == TriState.UNKNOWN


def test_current_scoped_false_with_a_valid_current_contradiction_stays_false():
    crit = _crit(id="r", type=CriterionType.ROLE_FUNCTION, concept="software engineering",
                 scope=Scope.CURRENT, required=True)
    pkt = {"person_id": "p0",
           "current": {"experience_id": "e1", "title": "Product Manager", "company": "Co", "is_current": True},
           "past": [], "education": [], "experience_semantics": [], "company_classifications": [],
           "skills": [], "certifications": [], "semantic_assertions": []}
    facts = _pf(_Person(current_title="Product Manager"),
                [_Exp("Product Manager", "Co", 2021, None, True, id="e1")])
    raw = {"r": {"criterion_id": "r", "status": "false", "match_strength": 0.1,
                 "supporting_evidence_refs": [], "contradicting_evidence_refs": ["exp:e1"],
                 "reason": "current role is Product Manager, not software engineering"}}
    out = validate_person(raw, pkt, _plan(crit), facts, ScoringContext())
    assert out["r"]["status"] == TriState.FALSE


def test_explicit_complete_phrase_needs_backend_authoritative_history():
    # §10 — thin profile: the model's "reviewed the full history" is worthless
    thin = validate_person(
        {"m": {"criterion_id": "m", "status": "false", "match_strength": 0.2,
               "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
               "reason": "reviewed the full work history and it clearly lacks any people-management role"}},
        _packet_min(), _plan(_mentor_crit()), _facts_plain(), ScoringContext())
    assert thin["m"]["status"] == TriState.UNKNOWN

    # §11 — >=3 dated roles + completeness 75: an absence-based negative may stand
    p = _Person(completeness=75)
    strong = _pf(p, [_Exp("SWE", "A", 2014, 2017, False, id="e1"),
                     _Exp("SWE", "B", 2017, 2020, False, id="e2"),
                     _Exp("Staff SWE", "C", 2020, None, True, id="e3")])
    pkt = {"person_id": p.id, "current": {"experience_id": "e3", "is_current": True},
           "past": [{"experience_id": "e1", "is_current": False},
                    {"experience_id": "e2", "is_current": False}],
           "education": [], "experience_semantics": [], "company_classifications": [],
           "skills": [], "certifications": [], "semantic_assertions": []}
    out = validate_person(
        {"m": {"criterion_id": "m", "status": "false", "match_strength": 0.2,
               "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
               "reason": "reviewed every role in the full work history — none show people management or mentoring"}},
        pkt, _plan(_mentor_crit()), strong, ScoringContext())
    assert out["m"]["status"] == TriState.FALSE


def test_unsupported_false_cannot_override_a_deterministic_true():
    sem = {"semantic_assertions": [
        {"concept": "technology industry experience", "category": "industry_experience",
         "confidence": 0.95, "evidence": ["Software Engineer at Google"], "scope": "career"}]}
    p = _Person(current_title="Software Engineer", current_company="Google")
    facts = _pf(p, [_Exp("Software Engineer", "Google", 2021, None, True, id="e1")], sem=sem)
    crit = _crit(id="tech", type=CriterionType.SEMANTIC_CONCEPT,
                 concept="technology industry experience", required=True, scope=Scope.CAREER)
    raw = {"tech": {"criterion_id": "tech", "status": "false", "match_strength": 0.1,
                    "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
                    "reason": "not obviously a tech company employee"}}
    out = validate_person(raw, _packet_min(), _plan(crit), facts, ScoringContext())
    assert out["tech"]["status"] == TriState.UNKNOWN
    assert any("deterministic" in n.lower() for n in out["tech"]["validation"]["notes"])

    ctx = ScoringContext()
    ctx.judge_results["p0"] = out
    facts.person.id = "p0"
    scored = score_candidate(facts, _plan(crit), ctx)
    assert scored.qualification == Qualification.EXACT_MATCH


# ─────────────────────── §2 — pub / vol / rec refs ───────────────────────


class _Row:
    def __init__(self, _id, **kw):
        self.id = _id
        for k, v in kw.items():
            setattr(self, k, v)


def _bundle_one(pid="p0", *, pubs=(), vols=(), recs=()):
    p = _Person(current_title="Engineering Manager", current_company="Co")
    p.id = pid
    facts = _pf(p, [_Exp("Engineering Manager", "Co", 2019, None, True, id="e1",
                         desc="led and mentored a team of engineers")], pubs=list(pubs))
    return [(p, facts, {"volunteering": list(vols), "recommendations": list(recs)})]


def test_packet_exposes_pub_vol_rec_refs():
    pubs = [_Row("pub_1", title="A study of distributed consensus", description="research paper")]
    vols = [_Row("vol_1", role="STEM Mentor", organization="Local School",
                 description="mentored high-school students in programming")]
    recs = [_Row("rec_1", text="She personally mentored me into my first management role.",
                 relationship="reported to")]
    plan = _plan(_crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT,
                       concept="evidence of mentoring and coaching others", required=True))
    packets = build_packets(_bundle_one(pubs=pubs, vols=vols, recs=recs), plan, ScoringContext(),
                            query="career mentors")
    refs = packet_refs(packets[0])
    assert "pub:pub_1" in refs and "vol:vol_1" in refs and "rec:rec_1" in refs


def test_mentor_true_grounded_only_by_a_recommendation_is_accepted():
    pkt = {"person_id": "p0", "current": {"experience_id": "e1", "is_current": True},
           "past": [], "education": [], "experience_semantics": [], "company_classifications": [],
           "skills": [], "certifications": [], "semantic_assertions": [], "publications": [],
           "recommendations_received": [{"ref": "rec:rec_1",
                                         "text": "He mentored three of us into senior roles."}]}
    crit = _crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="evidence of mentoring and coaching others", required=True, scope=Scope.CAREER)
    raw = {"m": {"criterion_id": "m", "status": "true", "match_strength": 0.85,
                 "supporting_evidence_refs": ["rec:rec_1"],
                 "reason": "a direct report states he mentored several people into senior roles"}}
    out = validate_person(raw, pkt, _plan(crit), _facts_plain(), ScoringContext())
    assert out["m"]["status"] == TriState.TRUE and out["m"]["supporting_evidence_refs"] == ["rec:rec_1"]


def test_mentor_true_grounded_by_volunteering_is_accepted():
    pkt = {"person_id": "p0", "current": {"experience_id": "e1", "is_current": True},
           "past": [], "education": [], "experience_semantics": [], "company_classifications": [],
           "skills": [], "certifications": [], "semantic_assertions": [], "publications": [],
           "volunteering": [{"ref": "vol:vol_1", "role": "Coding Mentor", "organization": "Code Club",
                             "description": "personally mentored ten students through their first projects"}]}
    crit = _crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="evidence of mentoring and coaching others", required=True, scope=Scope.CAREER)
    raw = {"m": {"criterion_id": "m", "status": "true", "match_strength": 0.8,
                 "supporting_evidence_refs": ["vol:vol_1"],
                 "reason": "volunteers as a coding mentor and personally mentored ten students"}}
    out = validate_person(raw, pkt, _plan(crit), _facts_plain(), ScoringContext())
    assert out["m"]["status"] == TriState.TRUE


def test_research_true_grounded_by_a_publication_is_accepted():
    pkt = {"person_id": "p0", "current": {"experience_id": "e1", "is_current": True},
           "past": [], "education": [], "experience_semantics": [], "company_classifications": [],
           "skills": [], "certifications": [], "semantic_assertions": [],
           "publications": [{"ref": "pub:pub_1", "title": "Neural methods for X",
                             "description": "peer-reviewed research"}]}
    crit = _crit(id="r", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="hands-on research experience", required=True, scope=Scope.CAREER)
    raw = {"r": {"criterion_id": "r", "status": "true", "match_strength": 0.8,
                 "supporting_evidence_refs": ["pub:pub_1"],
                 "reason": "authored a peer-reviewed research publication"}}
    out = validate_person(raw, pkt, _plan(crit), _facts_plain(), ScoringContext())
    assert out["r"]["status"] == TriState.TRUE


def test_invented_pub_ref_is_removed_and_true_downgraded():
    pkt = {"person_id": "p0", "current": {"experience_id": "e1", "is_current": True},
           "past": [], "education": [], "experience_semantics": [], "company_classifications": [],
           "skills": [], "certifications": [], "semantic_assertions": [],
           "publications": [{"ref": "pub:pub_1", "title": "Real paper"}]}
    crit = _crit(id="r", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="hands-on research experience", required=True, scope=Scope.CAREER)
    raw = {"r": {"criterion_id": "r", "status": "true", "match_strength": 0.8,
                 "supporting_evidence_refs": ["pub:pub_999"], "reason": "cited a publication"}}
    out = validate_person(raw, pkt, _plan(crit), _facts_plain(), ScoringContext())
    assert "pub:pub_999" not in out["r"]["supporting_evidence_refs"]
    assert out["r"]["status"] == TriState.UNKNOWN


def test_faculty_appointment_supported_only_by_publication_is_downgraded():
    pkt = {"person_id": "p0", "current": {"experience_id": "e1", "is_current": True},
           "past": [], "education": [], "experience_semantics": [], "company_classifications": [],
           "skills": [], "certifications": [], "semantic_assertions": [],
           "publications": [{"ref": "pub:pub_1", "title": "A paper"}]}
    crit = _crit(id="f", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="a professor / faculty appointment at a university", required=True)
    raw = {"f": {"criterion_id": "f", "status": "true", "match_strength": 0.7,
                 "supporting_evidence_refs": ["pub:pub_1"], "reason": "has published academic work"}}
    out = validate_person(raw, pkt, _plan(crit), _facts_plain(), ScoringContext())
    assert out["f"]["status"] == TriState.UNKNOWN


# ─────────────────────── §3 — factual NOT tri-state ───────────────────────


def _not_amazon_plan():
    return _plan(_crit(id="na", type=CriterionType.CURRENT_COMPANY, value="Amazon",
                       operator=Operator.NOT, scope=Scope.CURRENT_COMPANY, required=True))


def test_not_current_company_verified_present_is_false():
    p = _Person(current_company="Amazon")
    exps = [_Exp("SDE", "Amazon", 2021, None, True, id="e1")]
    assert score_candidate(_pf(p, exps), _not_amazon_plan()).qualification == Qualification.NOT_MATCH


def test_not_current_company_verified_different_is_true():
    p = _Person(current_company="Microsoft")
    exps = [_Exp("PM", "Microsoft", 2020, None, True, id="e1")]
    assert score_candidate(_pf(p, exps), _not_amazon_plan()).qualification == Qualification.EXACT_MATCH


def test_not_current_company_missing_employer_is_unknown():
    p = _Person()  # no current_company, no current experience
    assert score_candidate(_pf(p, exps=[]), _not_amazon_plan()).qualification == Qualification.POSSIBLE_MATCH


def test_not_location_missing_is_unknown():
    plan = _plan(_crit(id="nl", type=CriterionType.LOCATION, value="Nashville",
                       operator=Operator.NOT, required=True))
    assert score_candidate(_pf(_Person()), plan).qualification == Qualification.POSSIBLE_MATCH


def test_not_location_verified_present_is_false():
    plan = _plan(_crit(id="nl", type=CriterionType.LOCATION, value="Nashville",
                       operator=Operator.NOT, required=True))
    scored = score_candidate(_pf(_Person(location_text="Nashville, Tennessee, United States")), plan)
    assert scored.qualification == Qualification.NOT_MATCH
