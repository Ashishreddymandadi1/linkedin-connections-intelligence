"""Run the evaluation queries through the REAL search pipeline and capture detail.

``run_query`` calls ``run_connection_search`` unchanged, against a Session bound
to the isolated pilot DB. Judge verdicts and validator downgrades are captured by
wrapping (monkeypatching) the two seams the search service imports — production
code is not modified.

``offline=True`` forces every LLM provider to report "exhausted" so the run is
deterministic and free: query interpretation falls back to the deterministic
parser, the semantic judge and final audit report ``unavailable`` / ``not_used``.
"""
from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

JUDGEABLE_TYPES = (
    "professional_concept", "role_function", "industry_experience",
    "company_category", "skill", "seniority", "career_transition",
)


@dataclass
class JudgeTrace:
    verdict_status_by_type: dict[str, dict[str, int]] = field(default_factory=dict)
    validator_downgrades: int = 0
    validator_dropped: int = 0
    invalid_evidence_refs: int = 0
    people_judged: int = 0
    missing_criterion_verdicts: int = 0

    def _bump(self, ctype: str, status: str) -> None:
        d = self.verdict_status_by_type.setdefault(ctype, {"true": 0, "false": 0, "unknown": 0})
        d[status] = d.get(status, 0) + 1


@dataclass
class QueryRecord:
    query_id: str
    query: str
    group: str
    interpretation: dict[str, Any]
    funnel: dict[str, Any]
    results: list[dict]
    near_matches: list[dict]
    judge_metadata: dict | None
    audit_metadata: dict | None
    judge_trace: dict
    audit_changes: list[dict]
    unknown_required_rate: float | None
    llm_provider: str | None
    llm_model: str | None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@contextlib.contextmanager
def _instrument(trace: JudgeTrace):
    from app.services import search_service as ss

    real_run_judge = ss.run_judge
    real_validate = ss.validate_person

    def wrapped_run_judge(query, parsed, bundle, ctx, **kw):
        run = real_run_judge(query, parsed, bundle, ctx, **kw)
        ctype = {c.id: c.type for c in parsed.criteria}
        trace.people_judged = len(run.verdicts)
        for _pid, crits in run.verdicts.items():
            for cid, raw in crits.items():
                t = ctype.get(cid, "unknown_type")
                status = str(raw.get("status", "unknown")).lower()
                if status not in ("true", "false", "unknown"):
                    status = "unknown"
                trace._bump(t, status)
                if not raw.get("supporting_evidence_refs") and not raw.get("contradicting_evidence_refs") \
                        and status in ("true", "false"):
                    trace.invalid_evidence_refs += 0  # refs presence checked in validator; placeholder
        return run

    def wrapped_validate(person_verdicts, packet, parsed, facts, ctx):
        before = {cid: str(v.get("status", "unknown")).lower() for cid, v in person_verdicts.items()}
        out = real_validate(person_verdicts, packet, parsed, facts, ctx)
        for cid, bstatus in before.items():
            if cid not in out:
                trace.validator_dropped += 1
                continue
            astatus = str(out[cid].get("status", "unknown")).lower()
            if bstatus in ("true", "false") and astatus == "unknown":
                trace.validator_downgrades += 1
        return out

    ss.run_judge = wrapped_run_judge
    ss.validate_person = wrapped_validate
    try:
        yield
    finally:
        ss.run_judge = real_run_judge
        ss.validate_person = real_validate


@contextlib.contextmanager
def _offline():
    """Hard-guarantee zero network LLM calls.

    Three independent locks:
      1. empty the provider chain at its source (``providers.default_chain``)
      2. blank the API keys in ``settings`` (so any stray chain build is empty too)
      3. replace ``generate_structured`` on the router AND on every module that
         imported the name directly, with a function that returns "exhausted".
    """
    from app.config import settings
    from app.services.llm import providers as _providers
    from app.services.llm import router as _router

    def _empty_chain():
        return []

    def _exhausted(*a, **k):  # noqa: ARG001
        return None

    saved_chain = _providers.default_chain
    saved_router_chain = _router.default_chain
    saved_keys = (settings.anthropic_api_key, settings.groq_api_key, settings.openrouter_api_key)
    saved_audit = settings.final_result_audit_enabled
    _providers.default_chain = _empty_chain
    _router.default_chain = _empty_chain
    settings.anthropic_api_key = ""
    settings.groq_api_key = ""
    settings.openrouter_api_key = ""
    # the final audit is an LLM path — meaningless offline; disable so its
    # "not audited -> UNKNOWN" downgrades don't pollute the dry-run report.
    settings.final_result_audit_enabled = False

    patched: list[tuple] = []
    for modname in (
        "app.services.llm.router",
        "app.services.query_interpreter",
        "app.services.semantic_judge",
        "app.services.semantic_llm",
        "app.services.reason_generator",
        "app.services.final_auditor",
    ):
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, "generate_structured"):
            patched.append((mod, mod.generate_structured))
            mod.generate_structured = _exhausted
    try:
        yield
    finally:
        _providers.default_chain = saved_chain
        _router.default_chain = saved_router_chain
        (settings.anthropic_api_key, settings.groq_api_key, settings.openrouter_api_key) = saved_keys
        settings.final_result_audit_enabled = saved_audit
        for mod, fn in patched:
            mod.generate_structured = fn


def _funnel_from_judge(md: dict | None) -> dict:
    md = md or {}
    return {
        "network_size": md.get("network_size"),
        "candidate_pool_size": md.get("candidate_pool_size"),
        "hard_fact_rejected_count": md.get("hard_fact_rejected_count"),
        "viable_candidate_count": md.get("viable_candidate_count"),
        "judge_candidate_count": md.get("judge_candidate_count"),
        "judge_batch_count": md.get("judge_batch_count"),
    }


def _result_row(item: dict) -> dict:
    return {
        "rank": item["rank"],
        "person_id": item["person_id"],
        "name": item.get("name"),
        "qualification": item.get("qualification"),
        "match_score": item.get("match_score"),
        "matched_criteria": item.get("matched_criteria", []),
        "uncertain_criteria": item.get("uncertain_criteria", []),
        "unmet_criteria": item.get("unmet_criteria", []),
        "audit_decision": item.get("audit_decision"),
        "audit_reason": item.get("audit_reason"),
        "audit_issues": item.get("audit_issues", []),
        "llm_verified": item.get("llm_verified", False),
        "evidence_refs": [
            {"type": c["type"], "criterion": c["criterion"], "match_strength": c["match_strength"],
             "evidence": [e.get("text", "")[:160] for e in c.get("evidence", [])]}
            for c in item.get("score_breakdown", [])
        ],
    }


def _unknown_required_rate(interp: dict, results: list[dict]) -> float | None:
    req_semantic = [
        c for c in interp.get("criteria", [])
        if c.get("required") and c.get("type") in JUDGEABLE_TYPES
    ]
    if not req_semantic or not results:
        return None
    total = len(req_semantic) * len(results)
    unknown = sum(len(r.get("uncertain_criteria", [])) for r in results)
    return round(min(1.0, unknown / total), 4) if total else None


def run_query(db: Session, qdef: dict, *, offline: bool = True) -> QueryRecord:
    from app.services.search_service import run_connection_search

    # the pilot DB has exactly one dataset
    from app.models import Dataset
    dataset_id = db.query(Dataset.id).scalar()

    trace = JudgeTrace()
    stack = [_instrument(trace)]
    if offline:
        stack.append(_offline())

    with contextlib.ExitStack() as es:
        for cm in stack:
            es.enter_context(cm)
        resp = run_connection_search(db, dataset_id=dataset_id, query=qdef["query"])
        db.rollback()  # never persist eval searches into the pilot DB

    r = resp.model_dump()
    interp = r["interpreted_query"]
    results = [_result_row(x) for x in r["connections"]["results"]]
    near = [_result_row(x) for x in r["connections"]["near_matches"]]

    audit_changes: list[dict] = []
    for x in r["connections"]["results"]:
        dec = x.get("audit_decision")
        if dec and dec != "approved":
            audit_changes.append({
                "person_id": x["person_id"], "name": x.get("name"),
                "transition": f"{x.get('qualification')}::{dec}",
                "reason": x.get("audit_reason"), "issues": x.get("audit_issues", []),
            })

    return QueryRecord(
        query_id=qdef["id"],
        query=qdef["query"],
        group=qdef["group"],
        interpretation={
            "intent": interp.get("intent"),
            "context": interp.get("context", {}),
            "target_person_context": interp.get("target_person_context", {}),
            "unresolved": interp.get("unresolved", []),
            "interpretation_summary": interp.get("interpretation_summary", ""),
            "interpretation_confidence": interp.get("interpretation_confidence"),
            "criteria": [
                {"id": c["id"], "type": c["type"], "value": c.get("value"),
                 "values": c.get("values", []), "concept": c.get("concept"),
                 "operator": c.get("operator"), "scope": c.get("scope"),
                 "required": c.get("required"), "modality": c.get("modality"),
                 "weight": c.get("weight")}
                for c in interp.get("criteria", [])
            ],
        },
        funnel={
            **_funnel_from_judge(r.get("judge_metadata")),
            "returned": r["connections"]["returned"],
            "exact": r["connections"].get("exact_match_count", 0),
            "possible": r["connections"].get("possible_match_count", 0),
            "near": len(near),
            "audit_pool": (r.get("audit_metadata") or {}).get("audited_candidates"),
        },
        results=results,
        near_matches=near,
        judge_metadata=r.get("judge_metadata"),
        audit_metadata=r.get("audit_metadata"),
        judge_trace=trace.__dict__,
        audit_changes=audit_changes,
        unknown_required_rate=_unknown_required_rate(interp, results),
        llm_provider=r.get("llm_provider"),
        llm_model=r.get("llm_model"),
    )
