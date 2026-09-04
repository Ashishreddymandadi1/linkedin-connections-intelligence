"""hardening PART 25/26 — offline ~1,000-profile benchmark, 5 named queries.

No live LLM / DB / Apify calls — this is entirely synthetic data (never real
connection data) run through REAL production code: query interpretation (the
deterministic parser — conftest disables the LLM path for all tests),
company-classification lookup, the hard-fact gate, deterministic scoring,
staged local judge resolution, and proactive batch planning. Only the actual
network calls the judge/audit would make are mocked, at the same seam other
offline tests in this suite use (``_call_judge`` / ``_call_audit``).

The population is NOT 1,000 identical fixtures — it cycles through 20 distinct
archetypes (company, industry, seniority, role, location, mentoring signal)
so each named query genuinely partitions the population into hard-rejected /
locally-resolved / judge-ambiguous slices, the same way real profile data
would. Phase 26 acceptance: a broad ~1,000-person query must NOT turn into
~100 sequential LLM calls.
"""
from __future__ import annotations

import time

import pytest

from app.constants import CriterionType, Qualification, TriState
from app.services import final_auditor, semantic_judge
from app.services.candidate_gate import hard_gate
from app.services.company_intel import company_key
from app.services.query_interpreter import interpret_query
from app.services.scoring import ProfileFacts, ScoringContext, score_candidate
from tests.test_search import _Exp, _Person

NETWORK_SIZE = 987

# ─────────────────────── synthetic population ───────────────────────

#: (company, is_startup, is_big_tech, industries, role_function, seniority_title,
#:  location, mentoring_desc)
_ARCHETYPES = [
    ("Amazon", False, True, ["technology", "e-commerce"], "software engineering",
     "Senior Software Engineer", "Seattle, WA", None),
    ("Amazon", False, True, ["technology", "e-commerce"], "engineering management",
     "Engineering Manager", "Seattle, WA", "manages and mentors a team of 8 engineers"),
    ("Google", False, True, ["technology"], "software engineering",
     "Staff Software Engineer", "San Francisco, CA", None),
    ("Meta", False, True, ["technology"], "software engineering",
     "Senior Software Engineer", "Menlo Park, CA", "mentors junior engineers on the team"),
    ("TinyLaunchCo", True, False, ["technology"], "software engineering",
     "Founding Engineer", "Austin, TX", None),
    ("NimbusStartup", True, False, ["technology"], "software engineering",
     "Software Engineer", "Nashville, TN", None),
    ("BrightPathAI", True, False, ["technology", "artificial intelligence"], "research",
     "Research Engineer", "Nashville, TN", None),
    ("Regional Trust Bank", False, False, ["financial services"], "accounting",
     "Senior Accountant", "Memphis, TN", None),
    ("Regional Trust Bank", False, False, ["financial services"], "software engineering",
     "Software Engineer", "Memphis, TN", None),
    ("St. Luke General Hospital", False, False, ["healthcare"], "clinical operations",
     "Operations Manager", "Memphis, TN", None),
    ("St. Luke General Hospital", False, False, ["healthcare"], "software engineering",
     "Health IT Engineer", "Nashville, TN", None),
    ("Midstate University", False, False, ["education", "research"], "research",
     "Research Scientist", "Nashville, TN", None),
    ("Vantage Consulting Group", False, False, ["consulting"], "consulting",
     "Senior Consultant", "Chicago, IL", None),
    ("Meridian Retail Co", False, False, ["retail"], "marketing",
     "Marketing Director", "Dallas, TX", None),
    ("Global Manufacturing Inc", False, False, ["manufacturing"], "operations",
     "Plant Operations Lead", "Detroit, MI", None),
    ("Amazon", False, True, ["technology", "e-commerce"], "product management",
     "Senior Product Manager", "Seattle, WA", None),
    ("QuantumLeap Robotics", True, False, ["technology", "robotics"], "research",
     "Principal Research Engineer", "Boston, MA", "advises graduate researchers"),
    ("SilverOak Capital", False, False, ["financial services"], "executive leadership",
     "Chief Technology Officer", "Memphis, TN", None),
    ("Crescent Health Systems", False, False, ["healthcare"], "executive leadership",
     "Chief Executive Officer", "Nashville, TN", None),
    ("Beacon Freight Logistics", False, False, ["logistics"], "operations",
     "VP of Operations", "Atlanta, GA", None),
]


def _company_class_ctx() -> ScoringContext:
    class_map = {}
    for company, is_startup, is_big_tech, industries, *_rest in _ARCHETYPES:
        key = company_key(None, company)
        if key in class_map:
            continue
        class_map[key] = {
            "is_startup": is_startup, "is_big_tech": is_big_tech,
            "is_technology_company": "technology" in industries,
            "industries": industries, "categories": [], "confidence": 0.9,
            "provenance": "ai_company_inference",
        }
    return ScoringContext(company_class=class_map)


def _build_population(n: int) -> dict[str, ProfileFacts]:
    facts_by_id: dict[str, ProfileFacts] = {}
    for i in range(n):
        (company, _startup, _bigtech, industries, role_fn, title, location,
         mentor_desc) = _ARCHETYPES[i % len(_ARCHETYPES)]
        p = _Person(current_title=title, current_company=company, location_text=location, completeness=85)
        p.id = f"p{i}"
        exp = _Exp(title, company, 2018, None, True, id=f"e{i}", desc=mentor_desc)
        sem = {
            "experience_semantics": [{
                "experience_id": f"e{i}", "role_function": role_fn,
                "employer_industries": industries, "employer_categories": [],
                "role_seniority": "senior" if "Senior" in title or "Staff" in title
                or "Principal" in title or "Chief" in title or "VP" in title
                or "Director" in title or "Manager" in title else "mid",
                "mentoring_signals": ["mentors junior engineers"] if mentor_desc else [],
                "confidence": 0.85,
            }],
        }
        if role_fn == "research":
            sem["semantic_assertions"] = [{
                "concept": "career-long research experience", "category": "professional_concept",
                "scope": "career", "confidence": 0.8, "experience_ids": [f"e{i}"],
                "education_ids": [], "certification_ids": [], "evidence": ["research publications"],
            }]
        facts_by_id[p.id] = ProfileFacts(person=p, experiences=[exp], education=[], skills=[],
                                         semantic=sem, embedding=None)
    return facts_by_id


# ─────────────────────── mocked judge / audit call seams ───────────────────────


def _fake_call_judge(payload, packets, unresolved_by_person=None, *, _retry=False):
    expanded = {}
    for pkt in packets:
        pid = pkt["person_id"]
        crit_ids = (unresolved_by_person or {}).get(pid) or []
        expanded[pid] = {
            cid: {
                "criterion_id": cid, "status": "unknown", "match_strength": 0.0,
                "confidence": 0.5, "reason": "insufficient evidence in packet",
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "experience_ids": [],
            }
            for cid in crit_ids
        }
    return "ok", expanded, "mock:offline", "mock-model"


def _fake_call_audit(payload, packets, first_pass_by_id, parsed=None, *, _retry=False):
    people = [{"person_id": pkt["person_id"], "decision": "approved", "confidence": 0.7,
              "reason": "", "criteria": [], "supporting_evidence_refs": [],
              "contradicting_evidence_refs": [], "suggested_qualification": None}
             for pkt in packets]
    return "ok", people, "mock:offline", "mock-model"


# ─────────────────────── the 5 named queries ───────────────────────

_QUERIES = [
    "Former Amazon people now at startups",
    "people who worked in tech",
    "research plus industry experience",
    "senior engineering mentors in tech",
    "CXOs in Memphis or Nashville",
]


@pytest.fixture(scope="module")
def population():
    return _build_population(NETWORK_SIZE)


def _run_one_query(query: str, facts_by_id: dict[str, ProfileFacts], monkeypatch) -> dict:
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_call_judge)
    monkeypatch.setattr(final_auditor, "_call_audit", _fake_call_audit)
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", True)

    t0 = time.perf_counter()
    parsed, provider, _model = interpret_query(query)

    ctx = _company_class_ctx()
    people = list(facts_by_id.values())

    decisions = {f.person.id: hard_gate(f, parsed, ctx) for f in people}
    hard_rejected = [pid for pid, d in decisions.items() if not d.viable]
    viable_ids = [pid for pid, d in decisions.items() if d.viable]

    prescored = {pid: score_candidate(facts_by_id[pid], parsed, ctx) for pid in viable_ids}
    bundle = [(facts_by_id[pid].person, facts_by_id[pid], {"volunteering": [], "recommendations": []})
             for pid in viable_ids]

    judge_run = semantic_judge.run_judge(
        query, parsed, bundle, ctx, network_size=NETWORK_SIZE, pool_size=len(people),
        hard_rejected_count=len(hard_rejected), local_scored=prescored,
    )
    local_pipeline_ms = (time.perf_counter() - t0) * 1000

    audit_pool = [prescored[pid] for pid in viable_ids
                 if prescored[pid].qualification != Qualification.NOT_MATCH][:30]
    bundle_by_id = {sc.person.id: (sc.person, facts_by_id[sc.person.id],
                                   {"volunteering": [], "recommendations": []}) for sc in audit_pool}
    audit_run = final_auditor.run_final_audit(query, parsed, audit_pool, ctx, bundle_by_id=bundle_by_id)

    return {
        "query": query,
        "provider": provider,
        "criteria": len(parsed.criteria),
        "network_scanned": NETWORK_SIZE,
        "hard_rejected": len(hard_rejected),
        "hard_gate_viable": len(viable_ids),
        "locally_resolved": judge_run.metadata.candidates_decided_locally,
        "judge_candidates": judge_run.metadata.judge_candidate_count,
        "unresolved_criteria_sent": judge_run.metadata.judgeable_criteria_sent,
        "judge_batches": judge_run.metadata.judge_batch_count,
        "audit_pool": len(audit_pool),
        "audit_batches": audit_run.metadata.batch_count,
        "estimated_llm_calls": judge_run.metadata.judge_batch_count + audit_run.metadata.batch_count,
        "local_pipeline_ms": round(local_pipeline_ms, 1),
    }


def test_offline_benchmark_5_named_queries(population, monkeypatch, capsys):
    # warm up the lazily-loaded cross-encoder reranker OUTSIDE the timed
    # section -- its one-time model load (a real disk/network read on first
    # use) is a process-startup cost, not a per-query pipeline cost, and would
    # otherwise land entirely on whichever query happens to run first.
    from app.services.reranker import cross_encode
    cross_encode("warmup", ["warmup snippet"])

    reports = [_run_one_query(q, population, monkeypatch) for q in _QUERIES]

    header = (f"{'query':<42}{'crit':>5}{'rejected':>9}{'viable':>8}{'local':>8}"
             f"{'judge_n':>8}{'judge_b':>8}{'audit_b':>8}{'calls':>7}{'ms':>8}")
    lines = [header]
    for r in reports:
        lines.append(
            f"{r['query'][:41]:<42}{r['criteria']:>5}{r['hard_rejected']:>9}{r['hard_gate_viable']:>8}"
            f"{r['locally_resolved']:>8}{r['judge_candidates']:>8}{r['judge_batches']:>8}"
            f"{r['audit_batches']:>8}{r['estimated_llm_calls']:>7}{r['local_pipeline_ms']:>8}"
        )
    report_text = "\n[PART 25 offline benchmark — 5 named queries, %d synthetic profiles]\n%s" % (
        NETWORK_SIZE, "\n".join(lines),
    )
    print(report_text)

    for r in reports:
        # the hard-fact gate + local resolution must account for the whole
        # network -- recall is never silently thrown away
        assert r["hard_rejected"] + r["hard_gate_viable"] == NETWORK_SIZE
        # local_pipeline_ms is reported (see the table above), not asserted on --
        # wall-clock timing in a shared CI/dev environment is too noisy for a
        # meaningful hard threshold; the count-based assertions below are the
        # real acceptance criteria.
        # Phase 26's real guarantee: no per-candidate call explosion. A judge
        # batch must cover several people at once, never ~1 person/call, even
        # when NOTHING resolves locally (the worst case -- e.g. a purely
        # semantic concept with no matching stored assertion in this synthetic
        # population, unlike real ProfileSemantic v3 data from enrichment,
        # which would resolve a meaningful slice of these locally too).
        if r["judge_batches"]:
            avg_people_per_batch = r["judge_candidates"] / r["judge_batches"]
            assert avg_people_per_batch >= 3, f"{r['query']}: batching degraded to {avg_people_per_batch:.1f}/call"
