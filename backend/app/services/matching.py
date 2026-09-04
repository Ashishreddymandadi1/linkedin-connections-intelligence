"""Deterministic fuzzy matchers shared by candidate retrieval and scoring."""
from __future__ import annotations

import re

_COMPANY_NOISE = re.compile(
    r"\b(inc|inc\.|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|gmbh|group|holdings|the)\b",
    re.I,
)
_ALIASES = {
    "amazon web services": "aws",
    "amazon web services aws": "aws",
    "aws amazon": "aws",
    "google llc": "google",
    "alphabet": "google",
    "meta platforms": "meta",
    "facebook": "meta",
    "microsoft corporation": "microsoft",
}


def norm(text: str | None) -> str:
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s+.#-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def norm_company(name: str | None) -> str:
    t = norm(name)
    t = _COMPANY_NOISE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return _ALIASES.get(t, t)


def company_matches(candidate: str | None, target: str | None) -> bool:
    a, b = norm_company(candidate), norm_company(target)
    if not a or not b:
        return False
    if a == b:
        return True
    at, bt = set(a.split()), set(b.split())
    if not at or not bt:
        return False
    # one side's tokens fully contained in the other (e.g. "aws" ⊂ "aws professional services")
    return at <= bt or bt <= at


def phrase_matches(haystack: str | None, needle: str | None) -> bool:
    h, n = norm(haystack), norm(needle)
    if not h or not n:
        return False
    if n in h:
        return True
    nt = [w for w in n.split() if len(w) > 2]
    if not nt:
        return n in h
    hits = sum(1 for w in nt if w in h)
    return hits / len(nt) >= 0.75


def token_overlap(a: str | None, b: str | None) -> float:
    ta = {w for w in norm(a).split() if len(w) > 2}
    tb = {w for w in norm(b).split() if len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


#: concept text -> the CompanySemantic boolean field it maps to. Shared by
#: scoring._score_company_category and career_chronology's transition endpoint
#: matcher so "startup" / "big tech" resolve identically everywhere (V4 §7,
#: hardening PART: career-transition dedup).
CATEGORY_FIELD_MAP = {
    "startup": "is_startup", "early stage": "is_startup", "early-stage": "is_startup",
    "big tech": "is_big_tech", "big technology": "is_big_tech", "faang": "is_big_tech",
    "major technology": "is_big_tech", "large tech": "is_big_tech",
    "tech": "is_technology_company", "technology": "is_technology_company",
    "software": "is_technology_company", "tech company": "is_technology_company",
}


def category_field(concept: str) -> str | None:
    c = norm(concept)
    for key, fld in sorted(CATEGORY_FIELD_MAP.items(), key=lambda kv: -len(kv[0])):
        if key in c:
            return fld
    return None


# ─────────────────── recency / duration weighting (v2) ───────────────────

_RECENCY_FLOOR = 0.6


def _current_year() -> int:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).year


def recency_factor(end_year: int | None, is_current: bool) -> float:
    """How much to trust a signal given how long ago the role ended."""
    if is_current or end_year is None:
        return 1.0
    age = _current_year() - end_year
    if age <= 3:
        return 1.0
    if age <= 6:
        return 0.9
    if age <= 10:
        return 0.78
    return 0.65


def duration_factor(start_year: int | None, end_year: int | None, is_current: bool) -> float:
    """Longer tenure => stronger signal."""
    if start_year is None:
        return 1.0
    end = _current_year() if (is_current or end_year is None) else end_year
    years = max(0, end - start_year)
    if years >= 3:
        return 1.0
    if years >= 1.5:
        return 0.92
    return 0.82


def experience_weight(exp, *, enabled: bool = True) -> float:
    """Combined recency×duration multiplier for a matched experience, floored so a
    real match never collapses to noise."""
    if not enabled:
        return 1.0
    f = recency_factor(getattr(exp, "end_year", None), getattr(exp, "is_current", False))
    f *= duration_factor(
        getattr(exp, "start_year", None), getattr(exp, "end_year", None), getattr(exp, "is_current", False)
    )
    return max(_RECENCY_FLOOR, f)


_SENIORITY_RANK = {
    "intern": 0,
    "entry": 1,
    "junior": 1,
    "mid": 2,
    "senior": 3,
    "staff": 4,
    "lead": 4,
    "principal": 5,
    "manager": 4,
    "director": 6,
    "head": 6,
    "vp": 7,
    "vice president": 7,
    "cxo": 8,
    "chief": 8,
    "founder": 8,
    "owner": 8,
    "partner": 7,
    # explicit C-suite abbreviations — a title of literally "CEO" has no
    # "chief" substring, so it was previously unranked (spec §22)
    "ceo": 8,
    "cto": 8,
    "cfo": 8,
    "coo": 8,
    "cmo": 8,
    "cpo": 8,
    "ciso": 8,
    "cio": 8,
}


def seniority_rank(text: str | None) -> int | None:
    t = norm(text)
    if not t:
        return None
    for kw, rank in sorted(_SENIORITY_RANK.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(kw)}\b", t):
            return rank
    return None


def is_cxo_title(title: str | None) -> bool:
    """True for actual C-suite titles (CEO/CTO/'Chief ... Officer'/founder-as-CXO)
    — NOT for a title/description that merely mentions working with executives
    ('sold to CXO customers' must not qualify, spec §22/§35)."""
    t = norm(title)
    if not t:
        return False
    return bool(re.search(r"\b(ceo|cto|cfo|coo|cmo|cpo|ciso|cio)\b", t)) or "chief" in t


def concept_overlap(a: str | None, b: str | None) -> float:
    """Loose similarity between two short concept phrases — token overlap with
    a floor for substring containment. Used to match e.g. semantic-assertion
    concepts / industries / job_families against a criterion's concept text."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    if na == nb or na in nb or nb in na:
        return 1.0
    return token_overlap(a, b)
