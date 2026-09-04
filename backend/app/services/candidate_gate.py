"""Hard-fact viability gate (V4 PART 3 §4–§7).

Between the full local network scan and the exhaustive LLM judge sits ONE
filter: reject a candidate before spending judge tokens ONLY when a VERIFIED
fact contradicts a required criterion so plainly that no reasonable LLM reading
of the profile could overturn it.

REJECTS (only these):
  * required location known and different
  * required current company known and different
  * required past company absent from an otherwise-populated work history
  * required NOT-current-company that the person verifiably IS at now
  * required school absent from an otherwise-populated education history
  * required company_category with a cached HIGH-CONFIDENCE opposite classification
  * required career_transition whose ordering deterministic chronology contradicts
  * required years_experience the real dates fall short of (dates complete)

NEVER rejects for (these are exactly what the judge is for, §6):
  low embedding / cross-encoder score, missing / stale semantic enrichment,
  UNKNOWN company category, non-obvious mentor evidence, HIPAA / research not
  literally mentioned, a local semantic scorer returning FALSE or a weak fuzzy
  strength, a ``modality=possible`` criterion.

Company classification is CACHE-ONLY here (V4 §8 / PART 1) — the gate reads
``ScoringContext.company_class`` and never launches a classification job.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.constants import CriterionType, Modality, Operator, TriState
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services import career_chronology as _career
from app.services.profile_authority import (
    current_employer_known,
    education_history_authoritative,
    location_known,
    work_history_authoritative,
)
from app.services.scoring import (
    ProfileFacts,
    ScoringContext,
    _score_company,
    _score_company_category,
    _score_education,
    _score_location,
    _want_current_for,
)

# V4 PART 3.6 §2 — completeness / authority policy is centralised in
# ``profile_authority``; the gate hard-rejects a MISSING required past employer
# / school ONLY when that module says the section is authoritative.


@dataclass
class ViabilityDecision:
    person_id: str
    viable: bool
    #: criterion_id -> TriState, for the hard-fact criteria the gate evaluated
    hard_fact_statuses: dict[str, str] = field(default_factory=dict)
    #: set when viable is False — the single criterion that killed viability
    rejection_reason: str | None = None
    rejected_criterion_id: str | None = None
    #: verified facts to hand the judge / validator as immovable context
    #: (criterion_id -> {"status", "detail"})
    locked_facts: dict[str, dict] = field(default_factory=dict)


def hard_gate(facts: ProfileFacts, parsed: ParsedSearchQuery, ctx: ScoringContext) -> ViabilityDecision:
    pid = facts.person.id
    statuses: dict[str, str] = {}
    locked: dict[str, dict] = {}

    for crit in parsed.criteria:
        if not crit.required or crit.modality == Modality.POSSIBLE:
            continue

        # ── required location ──────────────────────────────────
        if crit.type == CriterionType.LOCATION and crit.operator != Operator.NOT:
            if _location_strength(facts, crit) > 0:
                statuses[crit.id] = TriState.TRUE
                locked[crit.id] = {"status": TriState.TRUE, "detail": "verified location match"}
            elif location_known(facts):
                return _reject(pid, crit,
                               f"verified location {facts.person.location_text!r} does not match "
                               f"required {crit.values or [crit.value]}", statuses, locked)
            else:
                statuses[crit.id] = TriState.UNKNOWN  # no location data — leave to the judge/scorer
            continue

        # ── required current company ───────────────────────────
        if crit.type == CriterionType.CURRENT_COMPANY:
            want = _want_current_for(crit)
            matched = _company_strength(facts, crit, want_current=want, ctx=ctx)
            if crit.operator == Operator.NOT:
                if matched > 0:
                    return _reject(pid, crit,
                                   f"verifiably still at excluded company {crit.values or [crit.value]}",
                                   statuses, locked)
                # absence of DATA is not proof of NOT (V4 PART 3.5 §3)
                if current_employer_known(facts):
                    statuses[crit.id] = TriState.TRUE
                    locked[crit.id] = {"status": TriState.TRUE,
                                       "detail": "current employer verified, not the excluded one"}
                else:
                    statuses[crit.id] = TriState.UNKNOWN
                continue
            if matched > 0:
                statuses[crit.id] = TriState.TRUE
                locked[crit.id] = {"status": TriState.TRUE, "detail": "verified current employer"}
            elif want is True and current_employer_known(facts):
                return _reject(pid, crit,
                               f"current employer {facts.person.current_company!r} is not "
                               f"{crit.values or [crit.value]}", statuses, locked)
            else:
                statuses[crit.id] = TriState.UNKNOWN
            continue

        # ── required past company ──────────────────────────────
        if crit.type == CriterionType.PAST_COMPANY and crit.operator != Operator.NOT:
            want = _want_current_for(crit)
            if _company_strength(facts, crit, want_current=want, ctx=ctx) > 0:
                statuses[crit.id] = TriState.TRUE
                locked[crit.id] = {"status": TriState.TRUE, "detail": "verified in work history"}
            elif work_history_authoritative(facts):
                return _reject(pid, crit,
                               f"no {crit.values or [crit.value]} role in a strongly-complete, "
                               f"fully-dated work history", statuses, locked)
            else:
                statuses[crit.id] = TriState.UNKNOWN  # history not complete enough to trust absence
            continue

        # ── required school ────────────────────────────────────
        if crit.type == CriterionType.EDUCATION and crit.operator != Operator.NOT:
            if _education_strength(facts, crit) > 0:
                statuses[crit.id] = TriState.TRUE
            elif education_history_authoritative(facts):
                return _reject(pid, crit,
                               f"{crit.values or [crit.value]} absent from a strongly-complete "
                               f"education history", statuses, locked)
            else:
                statuses[crit.id] = TriState.UNKNOWN
            continue

        # ── required company category (cache-only, high-confidence FALSE) ──
        if crit.type == CriterionType.COMPANY_CATEGORY and crit.operator != Operator.NOT:
            _s, _ev, status = _score_company_category(facts, crit, ctx)
            statuses[crit.id] = status
            if status == TriState.FALSE:
                return _reject(pid, crit,
                               f"employer has a high-confidence classification opposite to "
                               f"{crit.concept!r}", statuses, locked)
            if status == TriState.TRUE:
                locked[crit.id] = {"status": TriState.TRUE, "detail": "verified company classification"}
            continue

        # ── required career transition (chronology-authoritative) ──
        if crit.type == CriterionType.CAREER_TRANSITION and crit.operator != Operator.NOT:
            _s, _ev, status = _career.score_transition(facts, crit, ctx)
            statuses[crit.id] = status
            if status == TriState.FALSE:
                return _reject(pid, crit,
                               f"career chronology contradicts the transition {crit.concept!r}",
                               statuses, locked)
            locked[crit.id] = {"status": status, "detail": "deterministic chronology"}
            continue

        # ── required minimum years (dates-complete shortfall) ──
        if crit.type == CriterionType.YEARS_EXPERIENCE and crit.operator != Operator.NOT:
            _s, _ev, status = _career.score_years_experience(facts, crit)
            statuses[crit.id] = status
            if status == TriState.FALSE:
                return _reject(pid, crit,
                               f"verified dates fall short of {crit.value or crit.concept!r}",
                               statuses, locked)
            locked[crit.id] = {"status": status, "detail": "deterministic duration"}
            continue

        # everything else (seniority, skill, title, semantic concepts, …) is NOT
        # a hard-fact rejection — the judge / final rescore decides those.

    return ViabilityDecision(person_id=pid, viable=True, hard_fact_statuses=statuses, locked_facts=locked)


# ─────────────────────── deterministic-scorer wrappers ───────────────────────


def _reject(pid: str, crit: SearchCriterion, reason: str, statuses: dict, locked: dict) -> ViabilityDecision:
    return ViabilityDecision(
        person_id=pid, viable=False, hard_fact_statuses=statuses,
        rejection_reason=reason, rejected_criterion_id=crit.id, locked_facts=locked,
    )


def _location_strength(facts: ProfileFacts, crit: SearchCriterion) -> float:
    return max(
        (_score_location(facts, v)[0] for v in (crit.values or ([crit.value] if crit.value else []))),
        default=0.0,
    )


def _company_strength(facts: ProfileFacts, crit: SearchCriterion, *, want_current, ctx: ScoringContext) -> float:
    rids = ctx.company_ids_by_criterion.get(crit.id)
    vals = crit.values or ([crit.value] if crit.value else [])
    if crit.operator == Operator.ALL_OF:
        strengths = [_score_company(facts, v, want_current=want_current, resolved_ids=rids)[0] for v in vals]
        return min(strengths) if strengths and all(s > 0 for s in strengths) else 0.0
    return max(
        (_score_company(facts, v, want_current=want_current, resolved_ids=rids)[0] for v in vals),
        default=0.0,
    )


def _education_strength(facts: ProfileFacts, crit: SearchCriterion) -> float:
    return max(
        (_score_education(facts, v)[0] for v in (crit.values or ([crit.value] if crit.value else []))),
        default=0.0,
    )
