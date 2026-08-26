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
}


def seniority_rank(text: str | None) -> int | None:
    t = norm(text)
    if not t:
        return None
    for kw, rank in sorted(_SENIORITY_RANK.items(), key=lambda kv: -len(kv[0])):
        if kw in t:
            return rank
    return None
