"""Ordered career reasoning (V4 §18–§21).

Work history is a SEQUENCE, not a bag of concepts. This module provides:

  ordered_experiences()      oldest -> newest, conservative with missing dates
  total_years_matching()     merge overlapping intervals, sum (no double-count)
  score_transition()         from-concept must END before to-concept STARTS
  score_years_experience()   minimum years in a domain, from relevant roles only

Transition / years criteria carry their structure in ``crit.concept``:
  career_transition  -> "from <A> to <B>"
  years_experience   -> crit.value = "<n>",  crit.concept = "at least n years in <domain>"
"""
from __future__ import annotations

import datetime as _dt
import re

from app.constants import TriState
from app.schemas import EvidenceItem, SearchCriterion
from app.services.matching import concept_overlap, norm

_NOW_YEAR = _dt.date.today().year


def _start(e) -> tuple[int, int]:
    y = getattr(e, "start_year", None)
    m = getattr(e, "start_month", None) or 1
    return (y if y else 9999 - int(getattr(e, "order_index", 0) or 0), m)


def _end(e) -> tuple[int, int]:
    if getattr(e, "is_current", False):
        return (_NOW_YEAR, 12)
    y = getattr(e, "end_year", None)
    m = getattr(e, "end_month", None) or 12
    return (y if y else _NOW_YEAR, m)


def ordered_experiences(experiences: list) -> list:
    """Oldest first. Rows with no dates fall back to reversed order_index."""
    return sorted(experiences, key=_start)


def total_years_matching(experiences: list, predicate) -> float:
    """Union of the date intervals of experiences that satisfy ``predicate``,
    in years. Concurrent roles are counted once (V4 §21)."""
    spans = sorted(
        (_start(e)[0] * 12 + _start(e)[1], _end(e)[0] * 12 + _end(e)[1])
        for e in experiences if predicate(e)
    )
    if not spans:
        return 0.0
    merged: list[list[int]] = [list(spans[0])]
    for s, en in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([s, en])
    return round(sum(en - s for s, en in merged) / 12.0, 1)


def _exp_text(e) -> str:
    return norm(f"{getattr(e, 'position', '')} {getattr(e, 'company_name', '')} "
               f"{getattr(e, 'description', '') or ''}")


def exp_semantics_by_id(sem: dict | None) -> dict:
    """Map experience_id -> its ExperienceSemantic dict, from stored profile data."""
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


def score_transition(facts, crit: SearchCriterion) -> tuple[float, list[EvidenceItem], str]:
    m = _FROM_TO_RE.search(crit.concept or "")
    if not m:
        return 0.0, [], TriState.UNKNOWN
    frm, to = m.group(1).strip(), m.group(2).strip()
    sem_by_exp = exp_semantics_by_id(facts.semantic)
    exps = ordered_experiences(facts.experiences)
    if len(exps) < 2:
        return 0.0, [], TriState.UNKNOWN

    THRESH = 0.5
    from_hits = [(i, e) for i, e in enumerate(exps) if _matches_concept(e, frm, sem_by_exp) >= THRESH]
    to_hits = [(i, e) for i, e in enumerate(exps) if _matches_concept(e, to, sem_by_exp) >= THRESH]
    if not from_hits or not to_hits:
        return 0.0, [], TriState.UNKNOWN

    for fi, fe in from_hits:
        for ti, te in to_hits:
            if ti > fi and _end(fe) <= _start(te):
                ev = [EvidenceItem(
                    type="experience",
                    text=f"{getattr(fe, 'position', '?')} at {getattr(fe, 'company_name', '?')} "
                         f"-> {getattr(te, 'position', '?')} at {getattr(te, 'company_name', '?')}",
                    detail={"transition": f"{frm} -> {to}"},
                )]
                return 0.9, ev, TriState.TRUE

    # both concepts present but not in order (V4 §20 — B must not match)
    return 0.0, [], TriState.FALSE


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
    years = total_years_matching(facts.experiences, pred)

    if years >= min_years:
        return min(1.0, 0.6 + (years - min_years) * 0.05), [EvidenceItem(
            type="experience",
            text=f"~{years} years{' in ' + domain if domain else ''} (needs {min_years}+)",
            detail={"years": years, "minimum": min_years},
        )], TriState.TRUE
    if years == 0.0:
        return 0.0, [], TriState.UNKNOWN
    return round(years / max(min_years, 1) * 0.4, 3), [], TriState.FALSE
