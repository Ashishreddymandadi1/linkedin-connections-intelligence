"""Backend-authoritative profile data-completeness policy (V4 PART 3.6 §2).

ONE place decides whether a section of a scraped LinkedIn profile is complete
enough that the ABSENCE of something in it may be treated as a real negative.

The LLM never makes this call. Its "I reviewed the whole work history" is only
an explanatory clue that must be ANDed with a decision made here — the model
cannot know whether the source profile itself is complete.

Heuristics (tuned for HarvestAPI-scraped LinkedIn profiles):

  work history is AUTHORITATIVE  iff
      profile_completeness >= 70   AND
      >= 3 work experiences        AND
      every work experience has a start year (no undated gap)

  education history is AUTHORITATIVE  iff
      profile_completeness >= 70   AND
      at least one education row

  current employer is KNOWN  iff  a current_company or a current experience row exists
  location is KNOWN          iff  any of location_text / city / state is set

Rationale: a modern, actively-maintained profile that lists 3+ fully-dated roles
and scores 70+ on completeness is one where "no Amazon role" or "no mentoring in
any role" is a reliable read. Anything sparser is left UNKNOWN for the judge —
never a false negative.

This module imports nothing from the service layer, so ``scoring``,
``candidate_gate`` and ``judge_validator`` can all share it without a cycle.
"""
from __future__ import annotations

#: profile_completeness threshold for treating an absence as authoritative
STRONG_COMPLETENESS = 70
#: minimum number of (dated) work experiences for a reliable work-history absence
MIN_DATED_ROLES = 3


def _completeness(facts) -> int:
    return int(getattr(facts.person, "profile_completeness", 0) or 0)


def work_history_authoritative(facts) -> bool:
    """True only when a MISSING employer / a career-wide missing concept can be
    treated as a real negative (V4 PART 3.6 §2)."""
    exps = list(getattr(facts, "experiences", []) or [])
    if len(exps) < MIN_DATED_ROLES:
        return False
    if _completeness(facts) < STRONG_COMPLETENESS:
        return False
    return all(getattr(e, "start_year", None) for e in exps)


def education_history_authoritative(facts) -> bool:
    return bool(getattr(facts, "education", None)) and _completeness(facts) >= STRONG_COMPLETENESS


def current_employer_known(facts) -> bool:
    return bool(getattr(facts.person, "current_company", None)) or any(
        getattr(e, "is_current", False) for e in getattr(facts, "experiences", []) or []
    )


def current_employer_from(person, experiences) -> str | None:
    """The ONE authoritative current-employer name (hardening PART 13).

    A normalized experience row flagged ``is_current`` is the verified, backend-
    maintained fact — it wins over the denormalized ``Person.current_company``
    scrape field, which can go stale relative to enrichment/re-scrapes. Every
    caller that shows or reasons about "current employer" (scoring, the judge/
    audit packet, the API response, reason generation) MUST go through this (or
    ``current_employer``) so they can never disagree with each other."""
    for e in experiences or []:
        if getattr(e, "is_current", False) and getattr(e, "company_name", None):
            return e.company_name
    return getattr(person, "current_company", None)


def current_employer(facts) -> str | None:
    """``current_employer_from`` for a ``ProfileFacts``-shaped object."""
    return current_employer_from(facts.person, getattr(facts, "experiences", None))


def location_known(facts) -> bool:
    p = facts.person
    return bool(
        getattr(p, "location_text", None) or getattr(p, "city", None) or getattr(p, "state", None)
    )
