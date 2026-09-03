"""Fact-consistency validator for the final result audit (V4 PART 5 §12).

The final audit is LLM output, so it is validated too — with the same evidence-
ref system the exhaustive judge uses. The validator:

  * drops criterion reviews for unknown criterion_ids
  * drops invented evidence references (§11) — an INCORRECT decision grounded
    only by invented refs is downgraded to UNKNOWN
  * runs a DETERMINISTIC-ONLY rescore (no judge verdicts) so verified facts,
    chronology and cached company classifications remain authoritative: an
    auditor cannot APPROVE a candidate the facts alone disqualify (§10/§29),
    and cannot mark INCORRECT a criterion the facts alone verify TRUE
  * computes the APPLIED qualification from the decision + the first-pass
    qualification, and NEVER produces EXACT from a non-EXACT first pass (§9)
  * for UNKNOWN over a first-pass EXACT: downgrade to POSSIBLE unless the
    deterministic facts alone already prove every required criterion (§8)

Output shape (per person):
  {person_id, decision, confidence, reason, criteria[], supporting_evidence_refs,
   contradicting_evidence_refs, applied_qualification, failed_required[],
   audit_issues[], llm_verified, first_pass_qualification, validation{...}}
"""
from __future__ import annotations

from app.constants import AuditDecision, Qualification, TriState
from app.schemas import ParsedSearchQuery
from app.services.judge_packet import packet_refs
from app.services.judge_validator import _deterministic_status
from app.services.scoring import ProfileFacts, ScoringContext, _label, score_candidate


def validate_audit(
    raw: dict,
    packet: dict,
    parsed: ParsedSearchQuery,
    facts: ProfileFacts,
    ctx: ScoringContext,
    *,
    first_pass_qualification: str,
    first_pass_uncertain: list[str] | None = None,
) -> dict:
    crits = {c.id: c for c in parsed.criteria}
    valid_refs = packet_refs(packet)
    notes: list[str] = []
    decision = raw.get("decision", AuditDecision.UNKNOWN)
    conf = float(raw.get("confidence", 0.5) or 0.5)

    if raw.get("audit_missing"):
        decision = AuditDecision.UNKNOWN
        notes.append("candidate not audited (batch failure / oversized packet) — treated as UNKNOWN")

    # ── clean criterion reviews + evidence refs (§11) ────────────────
    reviews: list[dict] = []
    dropped_refs = 0
    for r in raw.get("criteria", []):
        cid = r.get("criterion_id")
        if cid not in crits:
            continue
        sup = [x for x in (r.get("supporting_evidence_refs") or []) if x in valid_refs]
        con = [x for x in (r.get("contradicting_evidence_refs") or []) if x in valid_refs]
        dropped_refs += (len(r.get("supporting_evidence_refs") or []) - len(sup)) + \
                        (len(r.get("contradicting_evidence_refs") or []) - len(con))
        reviews.append({
            "criterion_id": cid,
            "status_review": r.get("status_review", "uncertain"),
            "reason": (r.get("reason") or "")[:200],
            "supporting_evidence_refs": sup,
            "contradicting_evidence_refs": con,
        })
    top_sup = [x for x in (raw.get("supporting_evidence_refs") or []) if x in valid_refs]
    top_con = [x for x in (raw.get("contradicting_evidence_refs") or []) if x in valid_refs]
    dropped_refs += (len(raw.get("supporting_evidence_refs") or []) - len(top_sup)) + \
                    (len(raw.get("contradicting_evidence_refs") or []) - len(top_con))
    if dropped_refs:
        notes.append(f"dropped {dropped_refs} invalid evidence ref(s)")

    # ── deterministic-only rescore — facts stay authoritative (§10/§14) ──
    det = score_candidate(facts, parsed, _fresh_ctx(ctx))

    pid = raw.get("person_id")
    grounded_fail = _grounded_failed_required(reviews, crits, facts, ctx, det, pid)

    # §29 — verified facts already disqualify this candidate: no APPROVAL possible
    if det.qualification == Qualification.NOT_MATCH:
        notes.append("verified facts contradict a required criterion — auditor cannot approve; removed")
        return _out(AuditDecision.INCORRECT, conf, raw, reviews, top_sup, top_con,
                    applied=Qualification.NOT_MATCH, failed=(det.unmet_required or ["a verified fact"]),
                    notes=notes, llm_verified=False, first_pass_qualification=first_pass_qualification)

    # ── ground an INCORRECT decision (§11) ──────────────────────────
    if decision == AuditDecision.INCORRECT and not grounded_fail:
        decision = AuditDecision.UNKNOWN
        notes.append("INCORRECT not grounded (no valid contradicting ref, no deterministic "
                     "contradiction, no first-pass FALSE) — downgraded to UNKNOWN")

    # ── APPROVED cannot stand over a grounded 'unsupported' required review ──
    if decision == AuditDecision.APPROVED and grounded_fail:
        decision = AuditDecision.DOWNGRADE
        notes.append("APPROVED overridden -> DOWNGRADE: a required criterion review is a "
                     "grounded 'unsupported'")

    deterministic_proves_exact = det.qualification == Qualification.EXACT_MATCH
    applied, failed = _apply(decision, first_pass_qualification, deterministic_proves_exact,
                             grounded_fail)

    llm_verified = (decision == AuditDecision.APPROVED
                    and applied == first_pass_qualification
                    and not raw.get("audit_missing"))

    return _out(decision, conf, raw, reviews, top_sup, top_con, applied=applied, failed=failed,
                notes=notes, llm_verified=llm_verified,
                first_pass_qualification=first_pass_qualification)


# ─────────────────────── helpers ───────────────────────


def _fresh_ctx(ctx: ScoringContext) -> ScoringContext:
    return ScoringContext(
        company_class=ctx.company_class,
        company_ids_by_criterion=ctx.company_ids_by_criterion,
    )


def _grounded_failed_required(reviews, crits, facts, ctx, det, pid) -> list[str]:
    """Labels of REQUIRED criteria whose review is a GROUNDED 'unsupported' —
    i.e. backed by a valid contradicting ref, an independent deterministic FALSE,
    a first-pass validated judge FALSE, or a deterministic NOT_MATCH. A criterion
    the deterministic scorer independently VERIFIES TRUE cannot be argued away
    (§10)."""
    jr = ctx.judge_results.get(pid, {}) if pid else {}
    det_unmet = set(det.unmet_required or [])
    out: list[str] = []
    for r in reviews:
        c = crits.get(r["criterion_id"])
        if not c or not c.required or r["status_review"] != "unsupported":
            continue
        has_con = bool(r["contradicting_evidence_refs"])
        det_status = _deterministic_status(c, facts, ctx)
        if det_status == TriState.TRUE and not has_con:
            continue  # verified TRUE — auditor may not overturn it
        judge_false = bool(jr.get(c.id, {}).get("status") == TriState.FALSE)
        label = _label(c)
        if has_con or det_status == TriState.FALSE or judge_false or label in det_unmet:
            out.append(label)
    return out


def _apply(decision: str, fp: str, det_exact: bool, grounded_fail: list[str]):
    """Allowed transitions (§8/§9). NEVER returns EXACT from a non-EXACT first
    pass — the final auditor cannot manufacture confidence."""
    if decision == AuditDecision.INCORRECT:
        return Qualification.NOT_MATCH, (grounded_fail or ["a required criterion"])
    if decision == AuditDecision.DOWNGRADE:
        return (Qualification.POSSIBLE_MATCH if fp == Qualification.EXACT_MATCH else fp), []
    if decision == AuditDecision.UNKNOWN:
        if fp == Qualification.EXACT_MATCH and not det_exact:
            return Qualification.POSSIBLE_MATCH, []
        return fp, []
    # APPROVED — keep the first-pass qualification unchanged
    return fp, []


def _out(decision, conf, raw, reviews, top_sup, top_con, *, applied, failed, notes,
         llm_verified, first_pass_qualification) -> dict:
    unsupported_reasons = [
        r["reason"] for r in reviews if r["status_review"] == "unsupported" and r["reason"]
    ]
    return {
        "person_id": raw.get("person_id"),
        "decision": decision,
        "confidence": round(float(conf), 3),
        "reason": (raw.get("reason") or "")[:400],
        "criteria": reviews,
        "supporting_evidence_refs": top_sup,
        "contradicting_evidence_refs": top_con,
        "applied_qualification": applied,
        "failed_required": failed,
        "audit_issues": (notes + unsupported_reasons)[:6],
        "llm_verified": bool(llm_verified),
        "first_pass_qualification": first_pass_qualification,
        "validation": {"notes": notes},
    }
