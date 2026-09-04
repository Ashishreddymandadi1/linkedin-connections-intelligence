"""V4 PART 5.5 — final audit completeness + grounding hardening.

  §1/§2/§4  a review per required criterion is enforced in code; APPROVED needs
            complete grounded coverage; omissions -> audit status PARTIAL
  §3        llm_verified means FULL grounded APPROVED coverage
  §5-§9     an audit "unsupported" of a required criterion is validated with the
            SAME negative-grounding rules as a first-pass semantic FALSE
  §10/§11   a "supported" review must be grounded (scope-valid ref OR
            deterministic / code-authoritative TRUE)
  §20       one authoritative result count = TOP_CONNECTIONS
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.constants import (
    AuditDecision,
    AuditStatus,
    CriterionType,
    Qualification,
    Scope,
    TriState,
)
from app.schemas import (
    FinalAuditBatch,
    FinalAuditCriterionReview,
    FinalAuditPersonDecision,
    ParsedSearchQuery,
    ScoreComponent,
    SearchCriterion,
)
from app.services import final_auditor
from app.services.company_intel import company_key
from app.services.final_audit_validator import validate_audit
from app.services.scoring import ProfileFacts, ScoringContext
from tests.test_search import _Exp, _Person


def _pf(person, exps=None, sem=None, edu=None):
    return ProfileFacts(person=person, experiences=exps or [], education=edu or [], skills=[],
                        semantic=sem or {}, embedding=None)


def _crit(**kw):
    kw.setdefault("weight", 100)
    return SearchCriterion(**kw)


def _plan(*c, **kw):
    return ParsedSearchQuery(criteria=list(c), **kw)


def _packet(pid="p0", *, current=None, past=(), education=()):
    return {"person_id": pid,
            "current": current or {"experience_id": "cur", "is_current": True},
            "past": list(past), "education": list(education),
            "experience_semantics": [], "company_classifications": [], "skills": [],
            "certifications": [], "semantic_assertions": []}


@dataclass
class _Scored:
    person: object
    match_score: float = 40.0
    components: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    matched_criteria: list = field(default_factory=list)
    excluded_reason: str | None = None
    qualification: str = Qualification.POSSIBLE_MATCH
    unmet_required: list = field(default_factory=list)
    uncertain_required: list = field(default_factory=list)


def _comp(cid, *, required=True):
    return ScoreComponent(criterion=cid, criterion_id=cid, type=CriterionType.PROFESSIONAL_CONCEPT,
                          weight=50, match_strength=0.9, score=45, required=required, evidence=[])


# ─────────────────────── §12 — required coverage ───────────────────────


def test_approved_with_a_missing_required_review_is_not_approved():
    plan = _plan(_crit(id="cyber", type=CriterionType.PROFESSIONAL_CONCEPT,
                       concept="cybersecurity experience", required=True, scope=Scope.CAREER),
                 _crit(id="health", type=CriterionType.INDUSTRY_EXPERIENCE,
                       concept="healthcare industry experience", required=True, scope=Scope.CAREER))
    facts = _pf(_Person(current_title="Security Engineer"),
                [_Exp("Security Engineer", "SaaSCo", 2020, None, True, id="cur")])
    ctx = ScoringContext()
    ctx.judge_results["p0"] = {"cyber": {"status": TriState.TRUE}, "health": {"status": TriState.TRUE}}
    raw = {"person_id": "p0", "decision": "approved", "confidence": 0.9, "reason": "looks good",
           "criteria": [{"criterion_id": "cyber", "status_review": "supported",
                         "supporting_evidence_refs": ["exp:cur"]}]}  # healthcare omitted
    out = validate_audit(raw, _packet(), plan, facts, ctx,
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["decision"] != AuditDecision.APPROVED
    assert out["applied_qualification"] == Qualification.POSSIBLE_MATCH
    assert out["llm_verified"] is False
    assert out["missing_required_reviews"] >= 1
    assert any(r["criterion_id"] == "health" and r.get("audit_missing") for r in out["criteria"])


# ─────────────────────── §13 — approved with an uncertain required review ───────────────────────


def test_approved_with_uncertain_required_review_is_not_verified():
    plan = _plan(_crit(id="c", type=CriterionType.PROFESSIONAL_CONCEPT, concept="mentoring",
                       required=True, scope=Scope.CAREER))
    facts = _pf(_Person(), [_Exp("Eng", "Co", 2020, None, True, id="cur")])
    ctx = ScoringContext()
    ctx.judge_results["p0"] = {"c": {"status": TriState.TRUE}}
    raw = {"person_id": "p0", "decision": "approved", "confidence": 0.9, "reason": "ok",
           "criteria": [{"criterion_id": "c", "status_review": "uncertain",
                         "reason": "not enough to be sure"}]}
    out = validate_audit(raw, _packet(), plan, facts, ctx,
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["llm_verified"] is False
    assert out["applied_qualification"] == Qualification.POSSIBLE_MATCH


# ─────────────────────── §14 — wrong-scope unsupported ───────────────────────


def test_wrong_scope_unsupported_review_is_not_grounded():
    plan = _plan(_crit(id="r", type=CriterionType.ROLE_FUNCTION, concept="software engineering",
                       scope=Scope.CURRENT, required=True))
    facts = _pf(_Person(current_title="Product Manager"),
                [_Exp("Software Engineer", "Co", 2015, 2019, False, id="past_pm"),
                 _Exp("Product Manager", "Co", 2019, None, True, id="cur")])
    pkt = _packet(current={"experience_id": "cur", "is_current": True},
                  past=[{"experience_id": "past_pm", "is_current": False}])
    raw = {"person_id": "p0", "decision": "incorrect", "confidence": 0.8,
           "reason": "past role was PM", "criteria": [
               {"criterion_id": "r", "status_review": "unsupported",
                "reason": "a past Product Manager role", "contradicting_evidence_refs": ["exp:past_pm"]}]}
    out = validate_audit(raw, pkt, plan, facts, ScoringContext(),
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["applied_qualification"] != Qualification.NOT_MATCH
    assert any(r["criterion_id"] == "r" and r["status_review"] == "uncertain" for r in out["criteria"])


# ─────────────────────── §15 — career-wide unsupported needs coverage ───────────────────────


def test_career_wide_unsupported_from_one_ref_is_not_grounded():
    plan = _plan(_crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT,
                       concept="career-long mentoring experience", scope=Scope.CAREER, required=True))
    facts = _pf(_Person(completeness=80),
                [_Exp("SWE", f"Co{i}", 2010 + i, 2011 + i, i == 4, id=f"e{i}") for i in range(5)])
    pkt = _packet(current={"experience_id": "e4", "is_current": True},
                  past=[{"experience_id": f"e{i}", "is_current": False} for i in range(4)])
    ctx = ScoringContext()
    ctx.judge_results["p0"] = {"m": {"status": TriState.TRUE}}
    raw = {"person_id": "p0", "decision": "incorrect", "confidence": 0.7,
           "reason": "one early IC role", "criteria": [
               {"criterion_id": "m", "status_review": "unsupported",
                "reason": "this one role was IC", "contradicting_evidence_refs": ["exp:e0"]}]}
    out = validate_audit(raw, pkt, plan, facts, ctx,
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["applied_qualification"] != Qualification.NOT_MATCH


# ─────────────────────── §16 — deterministic TRUE cannot be argued away ───────────────────────


def test_verified_current_company_cannot_be_marked_unsupported():
    plan = _plan(_crit(id="g", type=CriterionType.CURRENT_COMPANY, value="Google", required=True,
                       scope=Scope.CURRENT_COMPANY))
    facts = _pf(_Person(current_company="Google"),
                [_Exp("SWE", "Google", 2021, None, True, id="cur", company_id="1441")])
    raw = {"person_id": "p0", "decision": "incorrect", "confidence": 0.9,
           "reason": "does not feel like Google", "criteria": [
               {"criterion_id": "g", "status_review": "unsupported",
                "contradicting_evidence_refs": ["exp:cur"]}]}
    out = validate_audit(raw, _packet(), plan, facts, ScoringContext(),
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["applied_qualification"] == Qualification.EXACT_MATCH
    assert not out["failed_required"]


def test_locked_company_false_still_removes_even_if_audit_approves():
    ctx = ScoringContext(company_class={
        company_key("99", "Microsoft"): {"is_startup": False, "confidence": 0.97,
                                         "provenance": "ai_company_inference", "industries": [], "categories": []},
    })
    plan = _plan(_crit(id="su", type=CriterionType.COMPANY_CATEGORY, concept="startup",
                       scope=Scope.CURRENT_COMPANY, required=True))
    facts = _pf(_Person(current_company="Microsoft"),
                [_Exp("PM", "Microsoft", 2019, None, True, id="cur", company_id="99")])
    raw = {"person_id": "p0", "decision": "approved", "confidence": 0.9, "reason": "startup vibes",
           "criteria": [{"criterion_id": "su", "status_review": "supported"}]}
    out = validate_audit(raw, _packet(), plan, facts, ctx,
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["applied_qualification"] == Qualification.NOT_MATCH


# ─────────────────────── §17 — supported without evidence ───────────────────────


def test_supported_review_with_no_evidence_becomes_uncertain():
    plan = _plan(_crit(id="m", type=CriterionType.PROFESSIONAL_CONCEPT, concept="mentoring",
                       required=True, scope=Scope.CAREER))
    facts = _pf(_Person(), [_Exp("Eng", "Co", 2020, None, True, id="cur")])
    raw = {"person_id": "p0", "decision": "approved", "confidence": 0.9, "reason": "trust me",
           "criteria": [{"criterion_id": "m", "status_review": "supported", "reason": "clearly a mentor"}]}
    out = validate_audit(raw, _packet(), plan, facts, ScoringContext(),
                         first_pass_qualification=Qualification.POSSIBLE_MATCH)
    assert out["llm_verified"] is False
    assert any(r["criterion_id"] == "m" and r["status_review"] == "uncertain" for r in out["criteria"])


# ─────────────────────── §18 — full grounded coverage -> verified ───────────────────────


def test_full_grounded_coverage_stays_approved_and_verified():
    plan = _plan(_crit(id="cyber", type=CriterionType.PROFESSIONAL_CONCEPT,
                       concept="cybersecurity experience", required=True, scope=Scope.CAREER),
                 _crit(id="health", type=CriterionType.INDUSTRY_EXPERIENCE,
                       concept="healthcare industry experience", required=True, scope=Scope.CAREER))
    facts = _pf(_Person(current_title="Security Engineer at a hospital"),
                [_Exp("Security Engineer", "General Hospital", 2019, None, True, id="cur",
                      desc="patient-data security in a healthcare setting")])
    ctx = ScoringContext()
    ctx.judge_results["p0"] = {"cyber": {"status": TriState.TRUE}, "health": {"status": TriState.TRUE}}
    raw = {"person_id": "p0", "decision": "approved", "confidence": 0.93,
           "reason": "direct cybersecurity work in a healthcare environment",
           "criteria": [
               {"criterion_id": "cyber", "status_review": "supported",
                "supporting_evidence_refs": ["exp:cur"]},
               {"criterion_id": "health", "status_review": "supported",
                "supporting_evidence_refs": ["exp:cur"]}]}
    out = validate_audit(raw, _packet(), plan, facts, ctx,
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["decision"] == AuditDecision.APPROVED
    assert out["applied_qualification"] == Qualification.EXACT_MATCH
    assert out["llm_verified"] is True
    assert out["missing_required_reviews"] == 0


# ─────────────────────── §19 — omitted required reviews -> PARTIAL ───────────────────────


def _pool(n):
    out = []
    for i in range(n):
        p = _Person()
        p.id = f"p{i}"
        out.append(_Scored(person=p, qualification=Qualification.EXACT_MATCH,
                           components=[_comp("a"), _comp("b")]))
    return out


def test_audit_status_partial_when_required_reviews_are_omitted(monkeypatch):
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", True)
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_batch_size", 10)

    def fake(payload, packets, first_pass_by_id):
        people = [FinalAuditPersonDecision(
            person_id=pkt["person_id"], decision="approved", confidence=0.9, reason="ok",
            criteria=[FinalAuditCriterionReview(criterion_id="a", status_review="supported",
                                                supporting_evidence_refs=["exp:cur"])])
            for pkt in packets]
        return "ok", FinalAuditBatch(people=people), "mock", "m1"

    monkeypatch.setattr(final_auditor, "_call_audit", fake)

    plan = _plan(_crit(id="a", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", required=True),
                 _crit(id="b", type=CriterionType.PROFESSIONAL_CONCEPT, concept="y", required=True))
    pool = _pool(6)
    facts_by_id = {sc.person.id: _pf(sc.person, [_Exp("Eng", "Co", 2020, None, True, id="cur")]) for sc in pool}
    bundle_by_id = {sc.person.id: (sc.person, facts_by_id[sc.person.id],
                                   {"volunteering": [], "recommendations": []}) for sc in pool}

    run = final_auditor.run_final_audit("q", plan, pool, ScoringContext(), bundle_by_id=bundle_by_id)
    assert run.metadata.status == AuditStatus.FULL

    validated = {}
    for sc in pool:
        validated[sc.person.id] = validate_audit(
            run.decisions[sc.person.id], run.packets_by_id.get(sc.person.id, {}), plan,
            facts_by_id[sc.person.id], ScoringContext(),
            first_pass_qualification=sc.qualification)
    final_auditor.finalize(run, validated)

    assert run.metadata.status == AuditStatus.PARTIAL
    assert run.metadata.missing_required_reviews == 6
    assert run.metadata.candidates_with_incomplete_reviews == 6


# ─────────────────────── §20 — TOP_CONNECTIONS is authoritative ───────────────────────


def test_top_connections_is_the_only_result_count(client, monkeypatch):
    from tests.test_search import _enriched_dataset
    from app.services import semantic_judge

    monkeypatch.setattr("app.services.search_service.settings.top_connections", 3)
    monkeypatch.setattr("app.services.search_service.settings.final_result_audit_top_n", 20)
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", True)
    monkeypatch.setattr(semantic_judge, "_call_judge", lambda *a, **k: ("failed", None, None, None))

    def fake_audit(payload, packets, first_pass_by_id):
        people = [FinalAuditPersonDecision(person_id=p["person_id"], decision="approved", confidence=0.8,
                                           reason="ok", criteria=[]) for p in packets]
        return "ok", FinalAuditBatch(people=people), "mock", "m1"

    monkeypatch.setattr(final_auditor, "_call_audit", fake_audit)

    ds = _enriched_dataset(client)
    body = client.post("/search", json={"dataset_id": ds, "query": "engineers"}).json()
    assert body["connections"]["returned"] <= 3
    assert len(body["connections"]["results"]) <= 3
