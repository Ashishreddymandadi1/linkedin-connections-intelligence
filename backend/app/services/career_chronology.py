"""Ordered career reasoning (V4 §18-21, V4 §10 conservative dates).

Work history is a SEQUENCE, not a bag of concepts.

Missing dates are NOT invented. `_start` / `_end` return ``None`` when a year is
genuinely unknown; transition / years verdicts are UNKNOWN when the sequence or
threshold cannot be checked from real dates, and FALSE only when real chronology
contradicts the request.
"""
from __future__ import annotations

import datetime as _dt
import re

from app.constants import TriState
from app.schemas import EvidenceItem, SearchCriterion
from app.services.matching import category_field, company_matches, concept_overlap, norm

_NOW = (_dt.date.today().year, _dt.date.today().month)


def _start(e) -> tuple[int, int] | None:
    y = getattr(e, "start_year", None)
    if not y:
        return None
    return (int(y), int(getattr(e, "start_month", None) or 1))


def _end(e) -> tuple[int, int] | None:
    if getattr(e, "is_current", False):
        return _NOW
    y = getattr(e, "end_year", None)
    if not y:
        return None  # completed role, unknown end — do NOT assume "now"
    return (int(y), int(getattr(e, "end_month", None) or 12))


def _months(t: tuple[int, int]) -> int:
    return t[0] * 12 + t[1]


def ordered_experiences(experiences: list) -> list:
    """Oldest first. Dated rows sort by start date; undated rows keep their
    order_index order and are placed after all dated rows (their true position
    is unknown, so nothing may be inferred from where they land)."""
    dated = sorted((e for e in experiences if _start(e)), key=lambda e: _months(_start(e)))
    undated = sorted((e for e in experiences if not _start(e)),
                     key=lambda e: int(getattr(e, "order_index", 0) or 0), reverse=True)
    return dated + undated


def total_years_matching(experiences: list, predicate) -> tuple[float, bool]:
    """(years, complete). Union of the date intervals of matching experiences.
    `complete` is False when a matching experience has an unknown start or end
    (so `years` is only a lower bound)."""
    intervals: list[tuple[int, int]] = []
    complete = True
    for e in experiences:
        if not predicate(e):
            continue
        s, en = _start(e), _end(e)
        if not s or not en:
            complete = False
            continue
        intervals.append((_months(s), _months(en)))
    if not intervals:
        return 0.0, complete
    intervals.sort()
    merged = [list(intervals[0])]
    for s, en in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([s, en])
    return round(sum(en - s for s, en in merged) / 12.0, 1), complete


def _exp_text(e) -> str:
    return norm(f"{getattr(e, 'position', '')} {getattr(e, 'company_name', '')} "
               f"{getattr(e, 'description', '') or ''}")


def exp_semantics_by_id(sem: dict | None) -> dict:
    if not isinstance(sem, dict):
        return {}
    cached = sem.get("experience_semantics_by_id")
    if isinstance(cached, dict):
        return cached
    return {es.get("experience_id"): es for es in sem.get("experience_semantics", [])
            if isinstance(es, dict) and es.get("experience_id")}


def _matches_concept(e, concept: str, sem_by_exp: dict | None = None) -> float:
    concept = concept.strip().lower()
    best = concept_overlap(_exp_text(e), concept)
    es = (sem_by_exp or {}).get(getattr(e, "id", None)) or {}
    for fld in ("role_function", "professional_domain"):
        if es.get(fld):
            best = max(best, concept_overlap(str(es[fld]), concept))
    for lst in ("employer_industries", "employer_categories", "role_domains"):
        for v in es.get(lst, []) or []:
            best = max(best, concept_overlap(str(v), concept))
    return best


_FROM_TO_RE = re.compile(r"from\s+(.+?)\s+to\s+(.+)$", re.I)
_THRESH = 0.5


def _endpoint_strength(e, concept: str, ctx, sem_by_exp: dict) -> float:
    """How strongly experience ``e`` matches a transition ENDPOINT (a company
    name, a company category like "startup"/"big tech", or a role/industry
    phrase).

    Reuses the SAME authoritative matchers the sibling past_company /
    company_category criteria already use for the identical fact, instead of
    re-deriving an independent (weaker) verdict from free-text overlap alone —
    otherwise a candidate with a verified past employer AND a verified current
    company category could still be denied the transition itself, because its
    own matcher couldn't see what the other two criteria already established
    (hardening pass: generic career-transition dedup, no company special-casing).
    """
    best = _matches_concept(e, concept, sem_by_exp)
    company_name = getattr(e, "company_name", None)
    if company_name and company_matches(company_name, concept):
        best = max(best, 1.0)
    field_name = category_field(concept)
    if field_name and ctx is not None:
        from app.services.company_intel import company_key

        row = ctx.company_class.get(company_key(getattr(e, "company_id", None), company_name))
        if row and row.get(field_name) is True and float(row.get("confidence") or 0.0) >= 0.6:
            best = max(best, 1.0)
    return best


def score_transition(facts, crit: SearchCriterion, ctx=None) -> tuple[float, list[EvidenceItem], str]:
    m = _FROM_TO_RE.search(crit.concept or "")
    if not m:
        return 0.0, [], TriState.UNKNOWN
    frm, to = m.group(1).strip(), m.group(2).strip()
    sem_by_exp = exp_semantics_by_id(facts.semantic)
    exps = facts.experiences
    from_hits = [e for e in exps if _endpoint_strength(e, frm, ctx, sem_by_exp) >= _THRESH]
    to_hits = [e for e in exps if _endpoint_strength(e, to, ctx, sem_by_exp) >= _THRESH]
    if not from_hits or not to_hits:
        return 0.0, [], TriState.UNKNOWN

    contradicted = False
    for fe in from_hits:
        for te in to_hits:
            if fe is te:
                continue
            fe_end, te_start = _end(fe), _start(te)
            if fe_end and te_start:
                if _months(fe_end) <= _months(te_start):
                    return 0.9, [EvidenceItem(
                        type="experience",
                        text=f"{getattr(fe, 'position', '?')} at {getattr(fe, 'company_name', '?')} "
                             f"-> {getattr(te, 'position', '?')} at {getattr(te, 'company_name', '?')}",
                        detail={"transition": f"{frm} -> {to}"},
                    )], TriState.TRUE
                fe_start, te_end = _start(fe), _end(te)
                if fe_start and te_end and _months(te_end) <= _months(fe_start):
                    contradicted = True  # this to-role is entirely before this from-role

    # dates present and every ordering evidence says to-before-from -> FALSE;
    # otherwise the sequence just cannot be verified -> UNKNOWN (V4 §10)
    if contradicted and not any(_end(fe) and _start(te) for fe in from_hits for te in to_hits
                                if fe is not te and _months(_end(fe)) <= _months(_start(te))):
        return 0.0, [], TriState.FALSE
    return 0.3, [], TriState.UNKNOWN


def score_years_experience(facts, crit: SearchCriterion) -> tuple[float, list[EvidenceItem], str]:
    try:
        min_years = int(re.search(r"\d+", crit.value or crit.concept or "0").group())
    except (AttributeError, ValueError):
        return 0.0, [], TriState.UNKNOWN
    dom_m = re.search(r"\bin\s+(.+)$", crit.concept or "")
    domain = dom_m.group(1).strip() if dom_m else ""
    sem_by_exp = exp_semantics_by_id(facts.semantic)

    if domain:
        def pred(e):
            return _matches_concept(e, domain, sem_by_exp) >= 0.45
    else:
        def pred(e):
            return True

    matching = [e for e in facts.experiences if pred(e)]
    if not matching:
        return 0.0, [], TriState.UNKNOWN
    years, complete = total_years_matching(matching, lambda e: True)

    if years >= min_years:
        return min(1.0, 0.6 + (years - min_years) * 0.05), [EvidenceItem(
            type="experience",
            text=f"~{years} years{' in ' + domain if domain else ''} (needs {min_years}+)",
            detail={"years": years, "minimum": min_years, "dates_complete": complete},
        )], TriState.TRUE
    if not complete:
        return round(years / max(min_years, 1) * 0.4, 3), [], TriState.UNKNOWN  # can't verify
    return round(years / max(min_years, 1) * 0.4, 3), [], TriState.FALSE
