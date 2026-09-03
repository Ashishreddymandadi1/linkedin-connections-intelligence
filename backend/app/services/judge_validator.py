"""Fact-consistency validator for LLM judge verdicts (V4 PART 3 §17–§18).

MANDATORY: every judge verdict passes through here before it is allowed into
``ScoringContext.judge_results``. The LLM judges MEANING; the backend stays
authoritative for VERIFIED FACTS.

A verdict is rejected / downgraded when:
  * the criterion is not one the judge may decide
  * a cited evidence reference does not exist in that person's exact packet
  * a TRUE verdict has no surviving supporting reference and the deterministic
    scorer does not independently agree
  * the supporting evidence is out of scope (claims CURRENT from a past-only
    experience, or PAST from a current-only one)
  * a company-category TRUE contradicts a cached HIGH-CONFIDENCE opposite
    classification (locked fact wins)
  * an employment concept ("faculty appointment", "works as …") is supported
    only by an education row (studying somewhere ≠ working there)

Downgraded TRUE -> UNKNOWN (fall back to the deterministic scorer); a locked
contradiction forces FALSE.
"""
from __future__ import annotations

import re

from app.constants import JUDGEABLE_CRITERION_TYPES, CriterionType, Scope, TriState
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.judge_packet import packet_experience_current_map, packet_refs
from app.services.scoring import ProfileFacts, ScoringContext, _score_company_category

_EMPLOYMENT_CONCEPT_RE = re.compile(
    r"\b(faculty|professor|tenure|appointment|employ|works? as|working as|"
    r"position at|role at|on staff|staff (?:member|scientist)|payroll)\b", re.I,
)
_PASSIVE_MENTEE_RE = re.compile(
    r"\b(was mentored|received mentor|being mentored|as a mentee|mentee of|"
    r"participated in (?:a |the )?mentor|mentorship program|was coached|"
    r"benefited from mentor)\b", re.I,
)
_MENTOR_CONCEPT_RE = re.compile(
    r"\b(mentor|coach|advis|guiding others|develop(?:ing)? people|people management)\b", re.I,
)


def validate_person(
    person_verdicts: dict[str, dict],
    packet: dict,
    parsed: ParsedSearchQuery,
    facts: ProfileFacts,
    ctx: ScoringContext,
) -> dict[str, dict]:
    """``person_verdicts``: ``{criterion_id: raw_verdict_dict}``. Returns the same
    shape, cleaned — invalid criteria removed, unsupported TRUEs downgraded."""
    crits: dict[str, SearchCriterion] = {c.id: c for c in parsed.criteria}
    valid_refs = packet_refs(packet)
    cur_map = packet_experience_current_map(packet)
    out: dict[str, dict] = {}

    for cid, raw in person_verdicts.items():
        crit = crits.get(cid)
        if crit is None or crit.type not in JUDGEABLE_CRITERION_TYPES:
            continue  # not the judge's to decide (§15/§16)
        out[cid] = _validate_one(raw, crit, facts, ctx, valid_refs, cur_map)
    return out


def _validate_one(raw: dict, crit: SearchCriterion, facts: ProfileFacts, ctx: ScoringContext,
                  valid_refs: set[str], cur_map: dict[str, bool]) -> dict:
    status = raw.get("status", TriState.UNKNOWN)
    strength = float(raw.get("match_strength", 0.0) or 0.0)
    notes: list[str] = []

    sup = [r for r in (raw.get("supporting_evidence_refs") or []) if r in valid_refs]
    con = [r for r in (raw.get("contradicting_evidence_refs") or []) if r in valid_refs]
    dropped = (len(raw.get("supporting_evidence_refs") or []) - len(sup)) + \
              (len(raw.get("contradicting_evidence_refs") or []) - len(con))
    if dropped:
        notes.append(f"dropped {dropped} invalid evidence ref(s)")
    exp_ids = [r.split("exp:", 1)[1] for r in sup if r.startswith("exp:")]
    raw_exp_ids = [e for e in (raw.get("experience_ids") or []) if f"exp:{e}" in valid_refs]

    downgraded = False

    if raw.get("judge_missing"):
        return _verdict(TriState.UNKNOWN, 0.0, raw, sup, con, raw_exp_ids,
                        notes=["criterion omitted by the model — treated as UNKNOWN"],
                        judge_missing=True)

    if status == TriState.TRUE:
        scope_refs = exp_ids or raw_exp_ids
        if crit.scope in (Scope.CURRENT, Scope.CURRENT_COMPANY) and scope_refs:
            if not any(cur_map.get(e, False) for e in scope_refs):
                status, downgraded = TriState.UNKNOWN, True
                notes.append("supporting experience is not current — cannot prove a current-scoped claim")
        elif crit.scope in (Scope.PAST, Scope.PAST_COMPANY) and scope_refs:
            if not any(cur_map.get(e, True) is False for e in scope_refs):
                status, downgraded = TriState.UNKNOWN, True
                notes.append("supporting experience is current — cannot prove a past-scoped claim")

        concept = (crit.concept or crit.value or "").lower()
        if status == TriState.TRUE and _EMPLOYMENT_CONCEPT_RE.search(concept):
            non_edu = [r for r in sup if not r.startswith("edu:")]
            if sup and not non_edu:
                status, downgraded = TriState.UNKNOWN, True
                notes.append("employment concept supported only by an education row (studying ≠ working)")

        if status == TriState.TRUE and _MENTOR_CONCEPT_RE.search(concept) \
                and _PASSIVE_MENTEE_RE.search(raw.get("reason", "")):
            status, downgraded = TriState.UNKNOWN, True
            notes.append("evidence describes receiving mentorship, not giving it")

        if status == TriState.TRUE and not sup:
            if not _deterministic_true(crit, facts, ctx):
                status, downgraded = TriState.UNKNOWN, True
                notes.append("no valid supporting evidence reference and deterministic scorer does not agree")

    if crit.type == CriterionType.COMPANY_CATEGORY:
        _s, _ev, det_status = _score_company_category(facts, crit, ctx)
        if det_status == TriState.FALSE and status != TriState.FALSE:
            notes.append("cached high-confidence classification is opposite — locked FALSE wins")
            return _verdict(TriState.FALSE, 0.0, raw, sup, con, raw_exp_ids, notes=notes, downgraded=True)

    if downgraded:
        strength = min(strength, 0.45)
    if status == TriState.FALSE:
        strength = min(strength, 0.1)
    return _verdict(status, strength, raw, sup, con, raw_exp_ids, notes=notes, downgraded=downgraded)


def _deterministic_true(crit: SearchCriterion, facts: ProfileFacts, ctx: ScoringContext) -> bool:
    """Would the local scorer independently call this TRUE? Lets a well-grounded
    judgement stand even when the model forgot to cite a ref."""
    from app.services.scoring import _score_one

    try:
        _s, _ev, det_status = _score_one(
            facts,
            SearchCriterion(id=f"_v_{crit.id}", type=crit.type, concept=crit.concept,
                            value=crit.value, values=crit.values, operator=crit.operator,
                            scope=crit.scope, weight=1),
            ScoringContext(company_class=ctx.company_class,
                           company_ids_by_criterion=ctx.company_ids_by_criterion),
        )
        return det_status == TriState.TRUE
    except Exception:  # noqa: BLE001
        return False


def _verdict(status, strength, raw, sup, con, exp_ids, *, notes, downgraded=False, judge_missing=False) -> dict:
    return {
        "status": status,
        "match_strength": round(float(strength), 3),
        "confidence": float(raw.get("confidence", 0.5) or 0.5),
        "reason": (raw.get("reason") or "")[:300],
        "evidence": ([raw["reason"][:200]] if raw.get("reason") else []) + sup[:3],
        "supporting_evidence_refs": sup,
        "contradicting_evidence_refs": con,
        "experience_ids": exp_ids,
        "overall_fit": raw.get("overall_fit"),
        "judge": True,
        "judge_missing": judge_missing,
        "validation": {"downgraded": downgraded, "notes": notes},
    }
