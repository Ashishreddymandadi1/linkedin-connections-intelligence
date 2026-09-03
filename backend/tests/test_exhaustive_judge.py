"""Exhaustive semantic judge + validator (V4 PART 3 §46–§57).

No live LLM — ``semantic_judge._judge_batch`` (or the router) is mocked. These
prove: EVERY viable candidate is judged in all_viable mode, batching is real,
partial failure is survivable, a missing verdict is UNKNOWN, the universal
SearchPlan (intent / relational context / modality / boolean structure) reaches
the judge, and the fact-consistency validator rejects invented / out-of-scope /
locked-fact-contradicting verdicts.
"""
from __future__ import annotations

from app.constants import CriterionType, JudgeStatus, Modality, Operator, Scope, TriState
from app.schemas import (
    JudgeBatch,
    JudgeCriterionVerdict,
    JudgePersonVerdict,
    ParsedSearchQuery,
    SearchCriterion,
)
from app.services import semantic_judge
from app.services.company_intel import company_key
from app.services.judge_validator import validate_person
from app.services.scoring import ProfileFacts, ScoringContext
from tests.test_search import _Exp, _Person


# ─────────────────────── fixtures / helpers ───────────────────────


def _person(i: int, **kw) -> _Person:
    p = _Person(**kw)
    p.id = f"p{i}"
    return p


def _pf(person, exps=None, sem=None) -> ProfileFacts:
    return ProfileFacts(person=person, experiences=exps or [], education=[], skills=[],
                        semantic=sem or {}, embedding=None)


def _bundle(n: int) -> list[tuple]:
    out = []
    for i in range(n):
        p = _person(i, headline="Engineer", current_company="Co", current_title="Engineer")
        exps = [_Exp("Engineer", "Co", 2020, None, True, id=f"e{i}",
                     desc="led a team and mentored engineers")]
        out.append((p, _pf(p, exps), {"volunteering": [], "recommendations": []}))
    return out


def _mentor_plan(*, extra_crits=()) -> ParsedSearchQuery:
    crits = [
        SearchCriterion(id="mentor_evidence", type=CriterionType.PROFESSIONAL_CONCEPT,
                        concept="evidence of mentoring, coaching, advising or people leadership",
                        scope=Scope.CAREER, weight=60, required=True),
        *extra_crits,
    ]
    return ParsedSearchQuery(
        intent="mentor_recommendation", criteria=crits,
        target_person_context={"field": "backend engineering", "goal": "engineering management"},
    )


def _fake_judge(*, true_ids=None, none_after=None, omit_people=(), omit_criteria=None, capture=None):
    """Fake ``_judge_batch``. ``none_after`` -> batch index from which the call fails."""
    state = {"batch": 0}

    def fake(payload, packets):
        idx = state["batch"]
        state["batch"] += 1
        if capture is not None:
            capture.append({"payload": payload, "packets": packets})
        if none_after is not None and idx >= none_after:
            return None
        people = []
        for pkt in packets:
            pid = pkt["person_id"]
            if pid in omit_people:
                continue
            crit_ids = [c["id"] for c in payload["criteria_to_judge"]]
            want_true = true_ids if true_ids is not None else crit_ids
            verdicts = []
            for cid in crit_ids:
                if omit_criteria and (pid, cid) in omit_criteria:
                    continue
                first_exp = (pkt.get("current") or {}).get("experience_id") \
                    or (pkt["past"][0]["experience_id"] if pkt.get("past") else None)
                verdicts.append(JudgeCriterionVerdict(
                    criterion_id=cid,
                    status="true" if cid in want_true else "unknown",
                    match_strength=0.9 if cid in want_true else 0.0,
                    confidence=0.9,
                    reason="clear evidence in the role description",
                    supporting_evidence_refs=[f"exp:{first_exp}"] if first_exp else [],
                    experience_ids=[first_exp] if first_exp else [],
                ))
            people.append(JudgePersonVerdict(person_id=pid, criteria=verdicts, overall_fit="strong"))
        return JudgeBatch(people=people), "mock:provider", "mock-model-1"

    return fake


def _raise_judge(*a, **k):
    raise AssertionError("_judge_batch must not be called")


# ─────────────────────── §46 — all viable candidates are judged ───────────────────────


def test_all_viable_candidates_are_judged_in_batches(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 10)
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_max_candidates", 0)
    calls = []
    monkeypatch.setattr(semantic_judge, "_judge_batch", _fake_judge(capture=calls))

    run = semantic_judge.run_judge(
        "career mentors", _mentor_plan(), _bundle(100), ScoringContext(),
        network_size=991, pool_size=991, hard_rejected_count=0,
    )
    assert run.metadata.judge_candidate_count == 100
    assert run.metadata.judge_batch_count == 10           # NOT 60, NOT ambiguity-band
    assert len(calls) == 10
    assert sum(len(c["packets"]) for c in calls) == 100
    assert run.metadata.judge_status == JudgeStatus.FULL
    assert len(run.verdicts) == 100
    assert all("mentor_evidence" in v for v in run.verdicts.values())


def test_no_hidden_60_person_cap(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_max_candidates", 0)
    monkeypatch.setattr(semantic_judge, "_judge_batch", _fake_judge())
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(75), ScoringContext(),
                                   network_size=75, pool_size=75, hard_rejected_count=0)
    assert run.metadata.judge_candidate_count == 75 and not run.metadata.capped


def test_explicit_cap_is_respected_and_reported(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_max_candidates", 20)
    monkeypatch.setattr(semantic_judge, "_judge_batch", _fake_judge())
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(50), ScoringContext(),
                                   network_size=50, pool_size=50, hard_rejected_count=0)
    assert run.metadata.judge_candidate_count == 20
    assert run.metadata.capped is True and run.metadata.cap_limit == 20


# ─────────────────────── §52 — partial batch failure ───────────────────────


def test_partial_batch_failure_keeps_earlier_verdicts(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 10)
    monkeypatch.setattr(semantic_judge, "_judge_batch", _fake_judge(none_after=6))
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(100), ScoringContext(),
                                   network_size=100, pool_size=100, hard_rejected_count=0)
    judged = [pid for pid, v in run.verdicts.items()
              if not v["mentor_evidence"].get("judge_missing")]
    unjudged = [pid for pid, v in run.verdicts.items()
                if v["mentor_evidence"].get("judge_missing")]
    assert len(judged) == 60 and len(unjudged) == 40
    assert run.metadata.judge_status == JudgeStatus.PARTIAL
    assert run.metadata.judge_successful_batches == 6 and run.metadata.judge_failed_batches == 4
    assert all(run.verdicts[pid]["mentor_evidence"]["status"] == TriState.UNKNOWN for pid in unjudged)


def test_all_batches_fail_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge, "_judge_batch", _fake_judge(none_after=0))
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(30), ScoringContext(),
                                   network_size=30, pool_size=30, hard_rejected_count=0)
    assert run.metadata.judge_status == JudgeStatus.UNAVAILABLE
    assert all(v["mentor_evidence"]["status"] == TriState.UNKNOWN for v in run.verdicts.values())


# ─────────────────────── §53 — missing person / criterion ───────────────────────


def test_missing_person_and_criteria_become_unknown(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 10)
    plan = _mentor_plan(extra_crits=(
        SearchCriterion(id="ai", type=CriterionType.INDUSTRY_EXPERIENCE, concept="AI industry",
                        weight=20, required=True),
        SearchCriterion(id="hc", type=CriterionType.INDUSTRY_EXPERIENCE, concept="healthcare industry",
                        weight=20, required=True),
    ))
    monkeypatch.setattr(semantic_judge, "_judge_batch",
                        _fake_judge(omit_people={"p0"}, omit_criteria={("p1", "ai"), ("p1", "hc")}))
    run = semantic_judge.run_judge("x", plan, _bundle(10), ScoringContext(),
                                   network_size=10, pool_size=10, hard_rejected_count=0)
    for cid in ("mentor_evidence", "ai", "hc"):
        assert run.verdicts["p0"][cid].get("judge_missing")
        assert run.verdicts["p0"][cid]["status"] == TriState.UNKNOWN
    assert run.verdicts["p1"]["ai"].get("judge_missing") and run.verdicts["p1"]["hc"].get("judge_missing")
    assert run.verdicts["p1"]["mentor_evidence"]["status"] == TriState.TRUE
    assert run.metadata.omitted_people == 1
    assert run.metadata.omitted_criteria == 2
    assert run.metadata.judge_status == JudgeStatus.PARTIAL


# ─────────────────────── §54 — provider routing (through the router) ───────────────────────


def test_judge_batch_uses_the_router(monkeypatch):
    seen = {}

    def fake_generate(system, user, schema, **kw):
        seen["operation"] = kw.get("operation")
        return JudgeBatch(people=[]), "anthropic:paid", "claude-haiku-4-5"

    monkeypatch.setattr(semantic_judge, "generate_structured", fake_generate)
    res = semantic_judge._judge_batch({"criteria_to_judge": []}, [{"person_id": "p0"}])
    assert res is not None and res[1] == "anthropic:paid"
    assert seen["operation"] == "semantic_judge"


def test_judge_batch_none_when_router_exhausted(monkeypatch):
    monkeypatch.setattr(semantic_judge, "generate_structured", lambda *a, **k: None)
    assert semantic_judge._judge_batch({"criteria_to_judge": []}, [{"person_id": "p0"}]) is None


# ─────────────────────── §55/§56/§57 — the SearchPlan reaches the judge ───────────────────────


def test_query_intent_and_relational_context_reach_the_judge(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    calls = []
    monkeypatch.setattr(semantic_judge, "_judge_batch", _fake_judge(capture=calls))
    semantic_judge.run_judge("Who could mentor a backend engineer moving into management?",
                             _mentor_plan(), _bundle(3), ScoringContext(),
                             network_size=3, pool_size=3, hard_rejected_count=0)
    payload = calls[0]["payload"]
    assert payload["intent"] == "mentor_recommendation"
    assert payload["target_person_context"]["field"] == "backend engineering"
    assert payload["target_person_context"]["goal"] == "engineering management"


def test_modality_reaches_the_judge(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    calls = []
    monkeypatch.setattr(semantic_judge, "_judge_batch", _fake_judge(capture=calls))
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="hipaa", type=CriterionType.PROFESSIONAL_CONCEPT,
                        concept="HIPAA compliance experience", weight=100,
                        required=False, modality=Modality.POSSIBLE),
    ])
    semantic_judge.run_judge("might have HIPAA experience", plan, _bundle(2), ScoringContext(),
                             network_size=2, pool_size=2, hard_rejected_count=0)
    crit = calls[0]["payload"]["criteria_to_judge"][0]
    assert crit["modality"] == "possible" and crit["required"] is False


def test_boolean_structure_reaches_the_judge(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    calls = []
    monkeypatch.setattr(semantic_judge, "_judge_batch", _fake_judge(capture=calls))
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="cyber", type=CriterionType.PROFESSIONAL_CONCEPT,
                        concept="cybersecurity experience", weight=33, required=True),
        SearchCriterion(id="health", type=CriterionType.INDUSTRY_EXPERIENCE,
                        concept="healthcare industry experience", weight=33, required=True),
        SearchCriterion(id="skills", type=CriterionType.ROLE_FUNCTION,
                        values=["security engineering", "cloud engineering"],
                        operator=Operator.ANY_OF, weight=34, required=False),
    ])
    semantic_judge.run_judge("cybersecurity AND healthcare", plan, _bundle(2), ScoringContext(),
                             network_size=2, pool_size=2, hard_rejected_count=0)
    crits = {c["id"]: c for c in calls[0]["payload"]["criteria_to_judge"]}
    assert crits["cyber"]["required"] and crits["health"]["required"]      # two required dimensions
    assert crits["skills"]["operator"] == "ANY_OF"
    assert crits["skills"]["values"] == ["security engineering", "cloud engineering"]  # not flattened


# ─────────────────────── §50/§51/§18 — fact-consistency validator ───────────────────────


def _crit(**kw):
    kw.setdefault("weight", 100)
    return SearchCriterion(**kw)


def _packet(pid="p0"):
    return {
        "person_id": pid,
        "current": {"experience_id": "e2", "title": "Product Manager", "company": "Co", "is_current": True},
        "past": [{"experience_id": "e1", "title": "Software Engineer", "company": "Co", "is_current": False,
                  "description": "backend services"}],
        "education": [{"education_id": "ed1", "school": "Stanford", "degree": "MS", "field": "CS"}],
        "experience_semantics": [], "company_classifications": [], "skills": [],
        "certifications": [], "semantic_assertions": [],
    }


def _facts0():
    p = _person(0, current_title="Product Manager", current_company="Co")
    return _pf(p, [_Exp("Software Engineer", "Co", 2018, 2021, False, id="e1"),
                   _Exp("Product Manager", "Co", 2022, None, True, id="e2")])


def test_invalid_evidence_ref_downgrades_unsupported_true():
    raw = {"mentor": {"criterion_id": "mentor", "status": "true", "match_strength": 0.9,
                      "supporting_evidence_refs": ["exp:e999"], "reason": "led people"}}
    crit = _crit(id="mentor", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="evidence of mentoring and people leadership", required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert out["mentor"]["status"] == TriState.UNKNOWN
    assert any("invalid evidence ref" in n for n in out["mentor"]["validation"]["notes"])


def test_wrong_scope_evidence_downgrades_current_claim():
    raw = {"role": {"criterion_id": "role", "status": "true", "match_strength": 0.9,
                    "supporting_evidence_refs": ["exp:e1"], "experience_ids": ["e1"],
                    "reason": "was a software engineer"}}
    crit = _crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="software engineering",
                 scope=Scope.CURRENT, required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert out["role"]["status"] == TriState.UNKNOWN
    assert any("not current" in n for n in out["role"]["validation"]["notes"])


def test_professor_from_education_only_is_downgraded():
    raw = {"prof": {"criterion_id": "prof", "status": "true", "match_strength": 0.8,
                    "supporting_evidence_refs": ["edu:ed1"], "reason": "studied at Stanford"}}
    crit = _crit(id="prof", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="a professor / faculty appointment at a university", required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert out["prof"]["status"] == TriState.UNKNOWN


def test_locked_company_classification_forces_false():
    ctx = ScoringContext(company_class={
        company_key("99", "Microsoft"): {"is_startup": False, "confidence": 0.95,
                                         "provenance": "ai_company_inference", "industries": [], "categories": []},
    })
    p = _person(0, current_company="Microsoft")
    facts = _pf(p, [_Exp("Principal PM", "Microsoft", 2018, None, True, id="e1", company_id="99")])
    key = company_key("99", "Microsoft")
    pkt = {"person_id": "p0",
           "current": {"experience_id": "e1", "company": "Microsoft", "is_current": True},
           "past": [], "company_classifications": [
               {"ref": f"company:{key}", "company": "Microsoft", "is_startup": False}],
           "education": [], "experience_semantics": [], "skills": [], "certifications": [],
           "semantic_assertions": []}
    raw = {"su": {"criterion_id": "su", "status": "true", "match_strength": 0.8,
                  "supporting_evidence_refs": [f"company:{key}"], "reason": "feels like a startup"}}
    crit = _crit(id="su", type=CriterionType.COMPANY_CATEGORY, concept="startup",
                 scope=Scope.CURRENT_COMPANY, required=True)
    out = validate_person(raw, pkt, ParsedSearchQuery(criteria=[crit]), facts, ctx)
    assert out["su"]["status"] == TriState.FALSE


def test_well_grounded_true_survives_validation():
    raw = {"mentor": {"criterion_id": "mentor", "status": "true", "match_strength": 0.88,
                      "supporting_evidence_refs": ["exp:e1"], "experience_ids": ["e1"],
                      "reason": "explicitly mentored five engineers per the role description"}}
    crit = _crit(id="mentor", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="evidence of mentoring and people leadership", scope=Scope.CAREER, required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert out["mentor"]["status"] == TriState.TRUE
    assert out["mentor"]["supporting_evidence_refs"] == ["exp:e1"]


def test_non_judgeable_criterion_verdict_is_dropped():
    raw = {"tr": {"criterion_id": "tr", "status": "true", "match_strength": 0.9,
                  "supporting_evidence_refs": ["exp:e1"]}}
    crit = _crit(id="tr", type=CriterionType.CAREER_TRANSITION, concept="from consulting to tech",
                 required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert "tr" not in out  # chronology stays code-authoritative (§16)


# ─────────────────────── mode = off / no judgeable criteria ───────────────────────


def test_mode_off_never_judges(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "off")
    monkeypatch.setattr(semantic_judge, "_judge_batch", _raise_judge)
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(5), ScoringContext(),
                                   network_size=5, pool_size=5, hard_rejected_count=0)
    assert run.metadata.judge_status == JudgeStatus.NOT_USED and run.verdicts == {}


def test_no_judgeable_criteria_is_not_used(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge, "_judge_batch", _raise_judge)
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="c", type=CriterionType.CURRENT_COMPANY, value="Google", weight=100, required=True),
    ])
    run = semantic_judge.run_judge("at Google", plan, _bundle(5), ScoringContext(),
                                   network_size=5, pool_size=5, hard_rejected_count=0)
    assert run.metadata.judge_status == JudgeStatus.NOT_USED


# ─────────────────────── §22/§47/§48/§59 — end-to-end through /search ───────────────────────


def _all_true_batch(payload, packets):
    people = []
    for pkt in packets:
        first_exp = (pkt.get("current") or {}).get("experience_id") \
            or (pkt["past"][0]["experience_id"] if pkt.get("past") else None)
        people.append(JudgePersonVerdict(
            person_id=pkt["person_id"], overall_fit="strong",
            criteria=[JudgeCriterionVerdict(
                criterion_id=c["id"], status="true", match_strength=0.9, confidence=0.9,
                reason="the role descriptions show mentoring and leadership",
                supporting_evidence_refs=[f"exp:{first_exp}"] if first_exp else [],
                experience_ids=[first_exp] if first_exp else [],
            ) for c in payload["criteria_to_judge"]],
        ))
    return JudgeBatch(people=people), "mock:provider", "mock-model-1"


def test_search_judges_every_viable_candidate_and_min_score_does_not_prefilter(client, monkeypatch):
    from tests.test_search import _enriched_dataset

    # §59 — normal search never classifies companies
    import app.services.company_intel as ci
    monkeypatch.setattr(ci, "get_or_classify",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no classify during search")))

    seen = []
    monkeypatch.setattr(semantic_judge, "_judge_batch",
                        lambda payload, packets: (seen.append(len(packets)), _all_true_batch(payload, packets))[1])
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_max_candidates", 0)

    ds = _enriched_dataset(client)
    body = client.post("/search", json={
        "dataset_id": ds,
        "query": "engineers who could mentor someone moving into management",
    }).json()

    jm = body["judge_metadata"]
    assert jm is not None and jm["mode"] == "all_viable"
    assert jm["hard_fact_rejected_count"] == 0
    assert jm["judge_candidate_count"] == jm["viable_candidate_count"] >= 3
    assert jm["judge_candidate_count"] == sum(seen)  # every viable candidate was in a packet
    # candidates whose DETERMINISTIC mentor score is ~0 still appear — proof
    # MIN_MATCH_SCORE no longer pre-filters and the judge verdict lifted them
    assert body["connections"]["returned"] >= 2
    assert body["connections"]["exact_match_count"] >= 1


def test_search_survives_judge_unavailable(client, monkeypatch):
    from tests.test_search import _enriched_dataset

    monkeypatch.setattr(semantic_judge, "_judge_batch", lambda *a, **k: None)
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    ds = _enriched_dataset(client)
    body = client.post("/search", json={
        "dataset_id": ds, "query": "people with mentoring and leadership experience",
    }).json()
    assert body["judge_metadata"]["status"] in ("unavailable", "not_used")
    # search still returns — deterministic scoring stands
    assert "connections" in body
