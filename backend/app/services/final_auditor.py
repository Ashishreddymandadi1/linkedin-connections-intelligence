"""Grounded final result audit (V4 PART 5).

After the exhaustive judge, fact validation, deterministic rescore, qualification
and cross-encoder rerank have produced a ranked list, ONE batched LLM pass
reviews the candidates about to be shown to the user:

    "Given the ORIGINAL query, the interpreted SearchPlan, every criterion
     verdict, the verified facts and the evidence for this candidate — is this
     person actually an appropriate result?"

This is a correctness BRAKE, not a re-search:
  * it never re-scans the network, never rewrites the SearchPlan
  * it never returns a 0-100 score
  * it can only keep / downgrade / remove a candidate
  * it can NEVER upgrade POSSIBLE -> EXACT (§9)
  * it can NEVER overturn a verified fact, chronology or company classification
    (§10) — the audit validator enforces this
  * one pass only, over TOP_N + BUFFER, batched (§18/§19)

The audit runs BEFORE expensive reason generation and before persistence, so a
removed candidate never costs a reason LLM call (§27).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.constants import AuditDecision, AuditStatus
from app.schemas import FinalAuditBatch, ParsedSearchQuery
from app.services.judge_packet import build_packets
from app.services.llm.router import generate_structured
from app.services.semantic_judge import _make_batches

log = logging.getLogger("app.audit")

_AUDIT_SYSTEM = (
    "You are the FINAL CORRECTNESS AUDITOR for a professional-network search. A first pass has "
    "already scored and ranked these candidates. Your job is a consistency review of the people "
    "about to be shown to the user — NOT a new search and NOT a new score.\n\n"
    "For EACH candidate, reason through:\n"
    "  1. Did they actually satisfy EVERY required criterion?\n"
    "  2. Are any 'true' criteria based on weak evidence or a semantic leap?\n"
    "  3. Are different dimensions being conflated? e.g.\n"
    "       technical role != tech-industry employer\n"
    "       studied at a university != a faculty / professor appointment\n"
    "       healthcare employer != personal HIPAA / compliance expertise\n"
    "       senior engineer != evidence of mentoring or people leadership\n"
    "       used AWS != employed by Amazon\n"
    "       worked with / sold to CXOs != is a CXO\n"
    "       one publication != a professor / career researcher\n"
    "       advises startups != currently employed at a startup\n"
    "  4. For a relational query ('who could mentor a backend engineer moving into "
    "management') — does the person's actual career make sense for that need? "
    "(IC->manager path, explicit mentoring, team leadership.) Relational fit CANNOT rescue a "
    "failed required criterion.\n"
    "  5. Is the current qualification (exact_match / possible_match / not_match) consistent "
    "with the evidence?\n\n"
    "FACTS ARE LOCKED. You cannot overturn a verified employer, past employer, location, "
    "education record, employment date, career chronology, years of experience, a cached "
    "high-confidence company classification, or a validated evidence reference. 'Atlanta is "
    "close enough to Nashville' is NOT allowed.\n\n"
    "For 'AND' / cross-domain queries (cybersecurity AND healthcare, AI AND healthcare, "
    "research AND industry) BOTH dimensions must be individually supported.\n"
    "For a criterion with modality 'possible', its absence must not make the candidate "
    "incorrect — it may simply not be a verified expert.\n\n"
    "DECISION per person:\n"
    "  approved  — the existing qualification is supported by the evidence\n"
    "  downgrade — relevant, but something treated as verified is actually uncertain "
    "(e.g. exact -> possible)\n"
    "  incorrect — at least one REQUIRED condition clearly fails; must not appear in results\n"
    "  unknown   — you cannot reliably decide; be conservative\n\n"
    "You do NOT set the final qualification — the backend does, and it will NEVER promote "
    "possible -> exact from your review. Cite packet evidence references "
    "(exp:/edu:/cert:/skill:/assertion:/company:/pub:/vol:/rec:) for every objection and "
    "approval. Review EACH required criterion in `criteria`. Return JSON only."
)


@dataclass
class AuditMetadata:
    enabled: bool = True
    status: str = AuditStatus.NOT_USED
    requested_candidates: int = 0
    audited_candidates: int = 0
    batch_count: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    oversized_packets: int = 0
    approved: int = 0
    downgraded: int = 0
    incorrect: int = 0
    unknown: int = 0
    #: V4 PART 5.5 §4 — required-criterion reviews the model omitted (total across
    #: the pool) and how many candidates had at least one omitted required review.
    missing_required_reviews: int = 0
    candidates_with_incomplete_reviews: int = 0
    providers: dict[str, int] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "requested_candidates": self.requested_candidates,
            "audited_candidates": self.audited_candidates,
            "batch_count": self.batch_count,
            "successful_batches": self.successful_batches,
            "failed_batches": self.failed_batches,
            "oversized_packets": self.oversized_packets,
            "approved": self.approved,
            "downgraded": self.downgraded,
            "incorrect": self.incorrect,
            "unknown": self.unknown,
            "missing_required_reviews": self.missing_required_reviews,
            "candidates_with_incomplete_reviews": self.candidates_with_incomplete_reviews,
            "providers": self.providers,
            "models": self.models,
        }


@dataclass
class AuditRun:
    #: person_id -> raw decision dict (validated in place by search_service)
    decisions: dict[str, dict]
    packets_by_id: dict[str, dict]
    metadata: AuditMetadata


def run_final_audit(
    query: str,
    parsed: ParsedSearchQuery,
    audit_pool: list,
    ctx,
    *,
    bundle_by_id: dict,
) -> AuditRun:
    """``audit_pool``: the tier-sorted ``ScoredCandidate`` list (TOP_N + BUFFER).
    ``bundle_by_id``: person_id -> (person, ProfileFacts, extras)."""
    meta = AuditMetadata(enabled=settings.final_result_audit_enabled)
    if not settings.final_result_audit_enabled or not audit_pool:
        return AuditRun({}, {}, meta)

    meta.requested_candidates = len(audit_pool)
    bundle = [bundle_by_id[c.person.id] for c in audit_pool if c.person.id in bundle_by_id]
    packets = build_packets(
        bundle, parsed, ctx, query=query,
        max_packet_chars=settings.final_result_audit_max_packet_chars,
    )
    packets_by_id = {p["person_id"]: p for p in packets}
    first_pass_by_id = {c.person.id: _first_pass(c, ctx) for c in audit_pool}
    payload = _audit_payload(query, parsed)

    batches, oversized = _make_batches(
        packets,
        size=settings.final_result_audit_batch_size,
        max_chars=settings.final_result_audit_max_batch_chars,
    )
    meta.batch_count = len(batches)
    meta.oversized_packets = len(oversized)
    if oversized:
        log.warning("audit: %d packet(s) too large — left unaudited: %s", len(oversized), oversized[:10])

    decisions: dict[str, dict] = {}
    for batch in batches:
        res = _audit_batch(payload, batch, first_pass_by_id)
        if res is None:
            meta.failed_batches += 1
            continue
        fab, provider, model = res
        meta.successful_batches += 1
        meta.providers[provider] = meta.providers.get(provider, 0) + 1
        if model and model not in meta.models:
            meta.models.append(model)
        for pd in fab.people:
            decisions[pd.person_id] = pd.model_dump()

    for c in audit_pool:
        if c.person.id not in decisions:
            decisions[c.person.id] = {
                "person_id": c.person.id, "decision": AuditDecision.UNKNOWN, "confidence": 0.0,
                "reason": "", "criteria": [], "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [], "suggested_qualification": None,
                "audit_missing": True,
            }
    meta.audited_candidates = sum(
        1 for c in audit_pool if not decisions[c.person.id].get("audit_missing")
    )

    if meta.successful_batches == 0 and meta.batch_count:
        meta.status = AuditStatus.UNAVAILABLE
    elif meta.failed_batches or meta.oversized_packets \
            or meta.audited_candidates < meta.requested_candidates:
        meta.status = AuditStatus.PARTIAL
    else:
        meta.status = AuditStatus.FULL

    log.info("final audit: pool=%d audited=%d %d/%d batches ok status=%s",
             meta.requested_candidates, meta.audited_candidates,
             meta.successful_batches, meta.batch_count, meta.status)
    return AuditRun(decisions, packets_by_id, meta)


def finalize(run: AuditRun, validated_by_id: dict[str, dict]) -> None:
    """Recompute the decision tally + review-completeness counts from the
    VALIDATED decisions and downgrade FULL -> PARTIAL when any required review
    was omitted (V4 PART 5.5 §4). Called by search_service after validation; the
    tests call it too."""
    m = run.metadata
    m.approved = m.downgraded = m.incorrect = m.unknown = 0
    m.missing_required_reviews = 0
    m.candidates_with_incomplete_reviews = 0
    for v in validated_by_id.values():
        d = v.get("decision", AuditDecision.UNKNOWN)
        if d == AuditDecision.APPROVED:
            m.approved += 1
        elif d == AuditDecision.DOWNGRADE:
            m.downgraded += 1
        elif d == AuditDecision.INCORRECT:
            m.incorrect += 1
        else:
            m.unknown += 1
        mrr = int(v.get("missing_required_reviews", 0) or 0)
        m.missing_required_reviews += mrr
        if mrr:
            m.candidates_with_incomplete_reviews += 1
    if m.status == AuditStatus.FULL and (m.missing_required_reviews or m.candidates_with_incomplete_reviews):
        m.status = AuditStatus.PARTIAL


# ─────────────────────── payload / first-pass ───────────────────────


def _audit_payload(query: str, parsed: ParsedSearchQuery) -> dict:
    return {
        "original_query": query,
        "intent": parsed.intent,
        "context": parsed.context,
        "target_person_context": parsed.target_person_context,
        "unresolved": parsed.unresolved,
        "interpretation_summary": parsed.interpretation_summary,
        "interpretation_confidence": parsed.interpretation_confidence,
        "criteria": [
            {
                "id": c.id, "type": c.type, "concept": c.concept or c.value, "values": c.values,
                "operator": c.operator, "scope": c.scope, "required": c.required,
                "modality": c.modality, "weight": round(c.weight, 1),
            }
            for c in parsed.criteria
        ],
    }


def _first_pass(cand, ctx) -> dict:
    jr = ctx.judge_results.get(cand.person.id, {})
    return {
        "qualification": cand.qualification,
        "match_score": cand.match_score,
        "uncertain_criteria": cand.uncertain_required,
        "unmet_criteria": cand.unmet_required,
        "criteria": [
            {
                "criterion_id": comp.criterion_id,
                "label": comp.criterion,
                "type": comp.type,
                "required": comp.required,
                "match_strength": round(comp.match_strength, 3),
                "judge": {
                    k: jr[comp.criterion_id].get(k)
                    for k in ("status", "confidence", "reason",
                              "supporting_evidence_refs", "contradicting_evidence_refs")
                } if comp.criterion_id in jr else None,
            }
            for comp in cand.components if comp.criterion_id != "relevance"
        ],
    }


def _audit_batch(payload: dict, packets: list[dict], first_pass_by_id: dict):
    """One batched audit request through the central router. Returns
    ``(FinalAuditBatch, provider, model)`` or ``None`` when every provider failed."""
    people = [
        {"packet": pkt, "first_pass": first_pass_by_id.get(pkt["person_id"], {})}
        for pkt in packets
    ]
    user = (
        "SEARCH PLAN (how the query was understood):\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
        + "\n\nCANDIDATES (evidence packet + first-pass result):\n"
        + json.dumps(people, ensure_ascii=False, default=str)
        + '\n\nReturn {"people":[{"person_id":"...","decision":"approved|downgrade|incorrect|unknown",'
        '"confidence":0-1,"reason":"...","criteria":[{"criterion_id":"...",'
        '"status_review":"supported|unsupported|uncertain","reason":"...",'
        '"supporting_evidence_refs":["exp:.."],"contradicting_evidence_refs":[]}],'
        '"supporting_evidence_refs":[],"contradicting_evidence_refs":[],'
        '"suggested_qualification":null}]} — one entry per person_id, and one `criteria` review '
        "per REQUIRED criterion at minimum."
    )
    result = generate_structured(
        _AUDIT_SYSTEM, user, FinalAuditBatch,
        max_tokens=min(4000, 500 + 360 * len(packets)),
        operation="final_result_audit",
    )
    if result is None:
        return None
    fab, provider, model = result
    return fab, provider, model
