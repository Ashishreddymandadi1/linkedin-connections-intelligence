"""Fact-consistency validator for LLM judge verdicts (V4 PART 3 §17–§18, PART 3.6).

MANDATORY: every judge verdict passes through here before it is allowed into
``ScoringContext.judge_results``. The LLM judges MEANING; the backend stays
authoritative for VERIFIED FACTS and for whether the source data is COMPLETE.

A verdict is downgraded to UNKNOWN when:
  * the criterion is not one the judge may decide
  * a cited evidence reference does not exist in that person's exact packet
  * a TRUE has no surviving supporting reference and the deterministic scorer
    does not independently agree
  * the supporting/contradicting evidence is out of SCOPE (a current-scoped claim
    proved only by a past role, or vice versa) — for exp: AND assertion: refs
  * a FALSE has no scope-appropriate contradicting evidence, no independent
    deterministic contradiction, and no explicit complete-data negative that the
    BACKEND completeness policy independently supports (V4 PART 3.6 §1) — missing
    evidence is UNKNOWN, never FALSE
  * a CAREER / ANY-scoped FALSE is grounded only by one or two experience refs —
    that does not prove a career-wide absence (V4 PART 3.6 §5)
  * an unsupported FALSE would bury a verified deterministic TRUE
  * an employment concept is supported only by an education / publication row

Forced FALSE: a company-category TRUE that contradicts a cached HIGH-CONFIDENCE
opposite classification.
"""
from __future__ import annotations

import math
import re

from app.constants import JUDGEABLE_CRITERION_TYPES, CriterionType, Scope, TriState
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.judge_packet import packet_experience_current_map, packet_refs
from app.services.profile_authority import (
    education_history_authoritative,
    work_history_authoritative,
)
from app.services.scoring import ProfileFacts, ScoringContext, _score_company_category

_EMPLOYMENT_CONCEPT_RE = re.compile(
    r"\b(faculty|professor|tenure|appointment|employ|works? as|working as|"
    r"position at|role at|on staff|staff (?:member|scientist)|payroll)\b", re.I,
)
_PASSIVE_MENTEE_RE = re.compile(
    r"\b(was mentored|received mentor\w*|being mentored|as a mentee|mentee of|"
    r"participated in (?:a |the )?mentor\w*|mentorship program|was coached|"
    r"benefited from mentor\w*)\b", re.I,
)
#: \w* (not a trailing \b) so this matches inflections too — "mentoring",
#: "mentors", "coaching", "advising", "advisor" — not just the bare stem.
_MENTOR_CONCEPT_RE = re.compile(
    r"\b(mentor\w*|coach\w*|advis\w*|guiding others|develop(?:ing)? people|people management)",
    re.I,
)
_EDUCATION_CONCEPT_RE = re.compile(
    r"\b(degree|studied|alumni|alumnus|graduat|university education|college education)\b", re.I,
)
#: reason text that CLAIMS the source data was complete — a clue only; it must be
#: ANDed with the backend completeness policy (V4 PART 3.6 §1).
_COMPLETE_DATA_NEGATIVE_RE = re.compile(
    r"\b(entire career|whole career|every (?:role|position|job)|all (?:roles|positions|jobs)|"
    r"throughout (?:their|his|her) career|full (?:work )?history|complete (?:work )?history|"
    r"reviewed (?:the |all )?(?:roles|positions|history)|none of (?:the|their) (?:roles|positions)|"
    r"no such (?:role|experience))\b", re.I,
)

_CURRENT_SCOPES = (Scope.CURRENT, Scope.CURRENT_COMPANY)
_PAST_SCOPES = (Scope.PAST, Scope.PAST_COMPANY)

#: criterion types where the deterministic / code-authoritative scorer is the
#: authority — an LLM "this is FALSE / unsupported" cannot overturn a verified
#: TRUE, and a verified FALSE stands on its own (V4 PART 3.6 §16 / PART 5.5 §9).
_FACT_AUTHORITATIVE_TYPES = {
    CriterionType.LOCATION, CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY,
    CriterionType.EDUCATION, CriterionType.CERTIFICATION, CriterionType.LANGUAGE,
    CriterionType.CAREER_TRANSITION, CriterionType.YEARS_EXPERIENCE,
    CriterionType.COMPANY_CATEGORY,
}


# ─────────────────────── evidence-ref scope helpers (§6) ───────────────────────


def _assertion_scope(idx: int, packet: dict, cur_map: dict[str, bool]) -> str:
    """current | past | career | unknown for an assertion: ref, derived from its
    linked experience_ids where possible, else its own declared scope."""
    try:
        a = packet.get("semantic_assertions", [])[idx]
    except (IndexError, TypeError):
        return "unknown"
    linked = [cur_map[e] for e in (a.get("experience_ids") or []) if e in cur_map]
    if linked:
        return "current" if any(linked) else "past"
    declared = (a.get("scope") or "").lower()
    if declared in _CURRENT_SCOPES:
        return "current"
    if declared in _PAST_SCOPES:
        return "past"
    if declared in (Scope.CAREER, Scope.ANY_EXPERIENCE):
        return "career"
    return "unknown"


def _ref_scope(ref: str, packet: dict, cur_map: dict[str, bool]) -> str:
    if ref.startswith("exp:"):
        eid = ref.split("exp:", 1)[1]
        if eid in cur_map:
            return "current" if cur_map[eid] else "past"
        return "unknown"
    if ref.startswith("assertion:"):
        try:
            return _assertion_scope(int(ref.split("assertion:", 1)[1]), packet, cur_map)
        except ValueError:
            return "unknown"
    return "not_scoped"  # edu:/pub:/vol:/rec:/cert:/skill:/company: carry no role scope


def _scope_ok_for(crit_scope: str | None, refs: list[str], packet: dict, cur_map: dict[str, bool]) -> bool:
    """Do the given scope-bearing refs prove something in the criterion's scope?
    A current-scoped criterion needs a CURRENT role/assertion; past needs PAST.
    career / any / unscoped: any relevant experience is fine."""
    scoped = [_ref_scope(r, packet, cur_map) for r in refs if r.startswith(("exp:", "assertion:"))]
    if not scoped:
        return True  # no scope-bearing ref to contradict the claim
    if crit_scope in _CURRENT_SCOPES:
        return any(s == "current" for s in scoped)
    if crit_scope in _PAST_SCOPES:
        return any(s in ("past", "career") for s in scoped)
    return True


def validate_negative_grounding(
    crit: SearchCriterion,
    facts: ProfileFacts,
    ctx: ScoringContext,
    packet: dict,
    contradicting_refs: list[str],
    reason: str,
) -> tuple[bool, str]:
    """ONE place that decides whether "this criterion is FALSE / not supported
    for this person" is a GROUNDED conclusion. Shared by the first-pass judge
    (a semantic FALSE verdict) and the final auditor (an 'unsupported' review of
    a required criterion) — V4 PART 3.6 §1/§4/§5, PART 5.5 §5–§9/§16.

    Grounded iff ONE of:
      * scope-appropriate contradicting exp:/assertion: ref(s) — and for a
        CAREER / ANY-scoped criterion, broad career coverage (same threshold as
        the judge, no second percentage)
      * the deterministic scorer independently returns FALSE
      * an explicit complete-data-negative phrase in ``reason`` that the BACKEND
        completeness policy independently supports

    NEVER grounded when the deterministic / code-authoritative scorer verifies
    the criterion TRUE and there is no scope-valid contradiction (§9/§16). For a
    structured / code-authoritative fact type the deterministic verdict is the
    sole authority.

    Returns ``(grounded, note)``.
    """
    cur_map = packet_experience_current_map(packet)
    valid = packet_refs(packet)
    con = [r for r in (contradicting_refs or []) if r in valid]
    det_status = _deterministic_status(crit, facts, ctx)

    if crit.type in _FACT_AUTHORITATIVE_TYPES:
        if det_status == TriState.TRUE:
            return False, ("deterministic / code-authoritative scorer verifies this fact — "
                           "the LLM objection is ignored")
        if det_status == TriState.FALSE:
            return True, "deterministic / code-authoritative scorer confirms this fact fails"

    con_scope_ok = _scope_ok_for(crit.scope, con, packet, cur_map)
    con_exp = [r for r in con if r.startswith(("exp:", "assertion:")) and con_scope_ok]
    has_scoped_contradiction = bool(con_exp)
    note = ""

    # a criterion EXPLICITLY scoped to the whole career needs broad coverage for
    # an absence claim — one or two roles do not prove a career-wide gap (§5).
    # An unscoped criterion is point-in-time enough that a scope-valid
    # contradiction suffices.
    if has_scoped_contradiction and crit.scope in (Scope.CAREER, Scope.ANY_EXPERIENCE):
        if not _career_coverage_ok(con_exp, cur_map):
            has_scoped_contradiction = False
            note = ("one or two experience refs do not prove a career-wide absence "
                    "(V4 PART 3.6 §5) — need broad coverage, a deterministic FALSE, or "
                    "authoritative complete-data")

    det_agrees = det_status == TriState.FALSE
    explicit_complete = (bool(_COMPLETE_DATA_NEGATIVE_RE.search(reason or ""))
                         and _backend_absence_authoritative(crit, facts))

    grounded = has_scoped_contradiction or det_agrees or explicit_complete
    if grounded and det_status == TriState.TRUE and not has_scoped_contradiction:
        return False, "deterministic scorer independently verifies TRUE — unsupported negative ignored"
    if grounded:
        return True, note or "grounded"
    if not note:
        if con and not con_scope_ok:
            note = f"contradicting evidence is not in the criterion's '{crit.scope}' scope"
        else:
            note = ("not grounded — no scope-valid contradiction, no deterministic contradiction, "
                    "no backend-authoritative complete-data negative")
    return False, note


# ─────────────────────── entry ───────────────────────


def validate_person(
    person_verdicts: dict[str, dict],
    packet: dict,
    parsed: ParsedSearchQuery,
    facts: ProfileFacts,
    ctx: ScoringContext,
) -> dict[str, dict]:
    """``person_verdicts``: ``{criterion_id: raw_verdict_dict}``. Returns the same
    shape, cleaned — invalid criteria removed, ungrounded TRUE/FALSE downgraded."""
    crits: dict[str, SearchCriterion] = {c.id: c for c in parsed.criteria}
    valid_refs = packet_refs(packet)
    cur_map = packet_experience_current_map(packet)
    out: dict[str, dict] = {}

    for cid, raw in person_verdicts.items():
        crit = crits.get(cid)
        if crit is None or crit.type not in JUDGEABLE_CRITERION_TYPES:
            continue  # not the judge's to decide (§15/§16)
        out[cid] = _validate_one(raw, crit, facts, ctx, packet, valid_refs, cur_map)
    return out


def _validate_one(raw: dict, crit: SearchCriterion, facts: ProfileFacts, ctx: ScoringContext,
                  packet: dict, valid_refs: set[str], cur_map: dict[str, bool]) -> dict:
    status = raw.get("status", TriState.UNKNOWN)
    strength = float(raw.get("match_strength", 0.0) or 0.0)
    reason = raw.get("reason", "") or ""
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

    # ── TRUE must be grounded and in scope ────────────────────────────
    if status == TriState.TRUE:
        sup_scoped = sup + [f"exp:{e}" for e in raw_exp_ids]
        if not _scope_ok_for(crit.scope, sup_scoped, packet, cur_map):
            status, downgraded = TriState.UNKNOWN, True
            _need = "not current" if crit.scope in _CURRENT_SCOPES else "not past"
            notes.append(f"supporting experience is {_need} — cannot prove a {crit.scope}-scoped claim")

        concept = (crit.concept or crit.value or "").lower()
        if status == TriState.TRUE and _EMPLOYMENT_CONCEPT_RE.search(concept):
            non_employment = [r for r in sup if not r.startswith(("edu:", "pub:"))]
            if sup and not non_employment:
                status, downgraded = TriState.UNKNOWN, True
                notes.append("employment concept supported only by an education / publication row "
                             "(studying / publishing ≠ working there)")

        if status == TriState.TRUE and _MENTOR_CONCEPT_RE.search(concept) \
                and _PASSIVE_MENTEE_RE.search(reason):
            status, downgraded = TriState.UNKNOWN, True
            notes.append("evidence describes receiving mentorship, not giving it")

        if status == TriState.TRUE and not sup:
            if _deterministic_status(crit, facts, ctx) != TriState.TRUE:
                status, downgraded = TriState.UNKNOWN, True
                notes.append("no valid supporting evidence reference and deterministic scorer does not agree")

    # ── FALSE must be GROUNDED (shared rule — V4 PART 3.6 §1 / PART 5.5 §5) ───
    if status == TriState.FALSE:
        grounded, gnote = validate_negative_grounding(crit, facts, ctx, packet, con, reason)
        if not grounded:
            status, downgraded = TriState.UNKNOWN, True
            notes.append(gnote)
            notes.append("missing evidence is not FALSE — downgraded to UNKNOWN")

    # ── locked company classification wins ───────────────────────────
    if crit.type == CriterionType.COMPANY_CATEGORY:
        _s, _ev, det_cc = _score_company_category(facts, crit, ctx)
        if det_cc == TriState.FALSE and status != TriState.FALSE:
            notes.append("cached high-confidence classification is opposite — locked FALSE wins")
            return _verdict(TriState.FALSE, 0.0, raw, sup, con, raw_exp_ids, notes=notes, downgraded=True)

    if downgraded:
        strength = min(strength, 0.45)
    if status == TriState.FALSE:
        strength = min(strength, 0.1)
    return _verdict(status, strength, raw, sup, con, raw_exp_ids, notes=notes, downgraded=downgraded)


# ─────────────────────── helpers ───────────────────────


def _career_coverage_ok(con_refs: list[str], cur_map: dict[str, bool]) -> bool:
    """Enough contradicting refs to argue a CAREER-wide absence: >= 2 and >= ~2/3
    of the experiences the packet actually shows (§5). With few experiences,
    citing both is enough; with many, one is never enough."""
    n_exps = len(cur_map)
    distinct = {r for r in con_refs if r.startswith("exp:")}
    if len(distinct) < 2:
        return False
    if n_exps <= 0:
        return len(distinct) >= 2
    return len(distinct) >= max(2, math.ceil(0.66 * n_exps))


def _backend_absence_authoritative(crit: SearchCriterion, facts: ProfileFacts) -> bool:
    """Does the BACKEND completeness policy support an absence-based negative for
    this criterion? The LLM's "I reviewed everything" is worthless without this."""
    concept = (crit.concept or crit.value or "").lower()
    if _EDUCATION_CONCEPT_RE.search(concept) and not _EMPLOYMENT_CONCEPT_RE.search(concept):
        return education_history_authoritative(facts)
    return work_history_authoritative(facts)


def _deterministic_status(crit: SearchCriterion, facts: ProfileFacts, ctx: ScoringContext) -> str:
    """The local scorer's independent tri-state for this criterion. Lets a
    well-grounded judgement stand when the model forgot a ref, and stops an
    unsupported FALSE / 'unsupported' review from burying a verified fact.

    Semantic criteria already return a tri-state. Structured facts (company /
    location / education / cert / language) return only a strength — a strong
    in-scope match is TRUE, a clear miss is FALSE, anything else UNKNOWN."""
    from app.services.scoring import _EXACT_MIN, _REQUIRED_MIN, _score_one

    try:
        s, _ev, det_status = _score_one(
            facts,
            SearchCriterion(id=f"_v_{crit.id}", type=crit.type, concept=crit.concept,
                            value=crit.value, values=crit.values, operator=crit.operator,
                            scope=crit.scope, weight=1),
            ScoringContext(company_class=ctx.company_class,
                           company_ids_by_criterion=ctx.company_ids_by_criterion),
        )
        if det_status is not None:
            return det_status
        if s >= _EXACT_MIN:
            return TriState.TRUE
        if s < _REQUIRED_MIN:
            return TriState.FALSE
        return TriState.UNKNOWN
    except Exception:  # noqa: BLE001
        return TriState.UNKNOWN


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
