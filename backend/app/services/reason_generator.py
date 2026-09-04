"""Turn deterministic evidence into a short explanation (spec §37, §61).

The LLM may only rephrase the evidence it is given — it must not introduce new
claims. If every free provider is down we fall back to a deterministic template,
which is still specific because it is built from the same evidence rows.

``generate_reasons_batch`` is the normal entry point for a search's top results
(hardening PART 10): ONE structured request covers every candidate that gets an
LLM-written reason, instead of one request per candidate. Reason text is
DISPLAY-ONLY — it can never affect ranking, qualification, or score, and any
candidate missing or invalid in the batch response falls back to the same
deterministic template used when the LLM path is off or unavailable.
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from app.config import settings
from app.services.llm.router import generate_structured
from app.services.profile_authority import current_employer_from
from app.services.scoring import ScoredCandidate

log = logging.getLogger("app.reason")


class _Reason(BaseModel):
    reason: str = Field(min_length=10, max_length=400)


class _ReasonItem(BaseModel):
    person_id: str = ""
    reason: str = ""


class _ReasonBatch(BaseModel):
    reasons: list[_ReasonItem] = []


_SYSTEM = (
    "You write one or two plain sentences explaining why a person matches a search. "
    "Use ONLY the evidence provided. Do not add employers, skills, dates or claims that "
    "are not in the evidence. If a 'current_employer' field is given, that is the ONLY "
    "correct name for where they currently work — use it verbatim if you mention their "
    "current employer, never a different company name. No generic praise ('great engineer', "
    "'strong background'). Name the concrete facts. Output JSON: {\"reason\": \"...\"}."
)

_BATCH_SYSTEM = (
    "You write one or two plain sentences PER PERSON explaining why they match a search. "
    "Use ONLY that person's own evidence — never borrow a fact from a different person_id. "
    "Each person's 'current_employer' field (when present) is the ONLY correct name for "
    "where THEY currently work — use it verbatim if you mention their current employer, "
    "never a different company name and never another person's employer. Do not add "
    "employers, skills, dates or claims that are not in their evidence. No generic "
    "praise ('great engineer', 'strong background'). Name the concrete facts. "
    "Return one entry per person_id: {\"reasons\":[{\"person_id\":\"...\",\"reason\":\"...\"}]}."
)


def generate_reason(
    candidate: ScoredCandidate, query: str, *, allow_llm: bool = True, facts=None,
) -> str:
    """Single-candidate path — used for near-matches and any candidate outside
    the batched top-N (see ``generate_reasons_batch`` for the normal case).
    ``facts`` (optional ``ProfileFacts``) — same bundle scoring/judge/audit used,
    so the current-employer name given to the LLM can never disagree with theirs
    (hardening PART 13)."""
    name = candidate.person.full_name or "This connection"
    ev_lines = [f"- {e.text}" for e in candidate.evidence[:8]]
    matched = ", ".join(candidate.matched_criteria) or "general profile relevance"
    current_employer = current_employer_from(
        candidate.person, facts.experiences if facts is not None else None,
    )

    if allow_llm and settings.llm_reason_generation and ev_lines:
        user = (
            f"Search: {query!r}\nPerson: {name}\n"
            + (f"current_employer: {current_employer}\n" if current_employer else "")
            + f"Matched: {matched}\nEvidence:\n"
            + "\n".join(ev_lines)
            + "\n\nWrite the explanation."
        )
        result = generate_structured(
            _SYSTEM, user, _Reason, max_tokens=300, operation="reason_generation",
        )
        if result is not None:
            return result[0].reason.strip()
        log.info("reason generator: using deterministic template")

    return _template(name, candidate)


def generate_reasons_batch(
    candidates: list[ScoredCandidate], query: str, *, facts_by_id: dict | None = None,
) -> dict[str, str]:
    """ONE structured LLM request covering every candidate in ``candidates``
    (hardening PART 10 — replaces up to ``llm_reason_top_n`` individual calls
    with one). Returns ``{person_id: reason}`` for EVERY candidate passed in —
    a candidate the model omitted, or the whole LLM path being off/unavailable,
    always resolves to the deterministic per-candidate template, never a gap.
    ``facts_by_id`` (optional) — the SAME bundle scoring/judge/audit used, so the
    current-employer name given to the LLM per person can never disagree with
    theirs (hardening PART 13)."""
    if not candidates:
        return {}

    out: dict[str, str] = {}
    eligible = [c for c in candidates if c.evidence]
    if settings.llm_reason_generation and eligible:
        people = [
            {
                "person_id": c.person.id,
                "name": c.person.full_name or "This connection",
                "current_employer": current_employer_from(
                    c.person,
                    (facts_by_id or {}).get(c.person.id).experiences
                    if (facts_by_id or {}).get(c.person.id) is not None else None,
                ),
                "matched": ", ".join(c.matched_criteria) or "general profile relevance",
                "evidence": [e.text for e in c.evidence[:8]],
            }
            for c in eligible
        ]
        user = (
            f"Search: {query!r}\nPEOPLE:\n" + json.dumps(people, ensure_ascii=False)
            + "\n\nWrite one explanation per person_id, using ONLY that person's own evidence."
        )
        result = generate_structured(
            _BATCH_SYSTEM, user, _ReasonBatch,
            max_tokens=min(4000, 200 + 220 * len(people)),
            operation="reason_generation",
        )
        if result is not None:
            batch = result[0]
            for item in batch.reasons:
                if item.person_id and len(item.reason.strip()) >= 10:
                    out[item.person_id] = item.reason.strip()
        else:
            log.info("batched reason generator: LLM path exhausted — using deterministic templates")

    for c in candidates:
        if c.person.id not in out:
            out[c.person.id] = _template(c.person.full_name or "This connection", c)
    return out


def _template(name: str, candidate: ScoredCandidate) -> str:
    if not candidate.evidence:
        return f"{name} came up as a broad match for this query but without a specific evidence hit."
    facts = [e.text for e in candidate.evidence[:4]]
    if len(facts) == 1:
        return f"{name} matches because {facts[0]}."
    joined = "; ".join(facts[:-1]) + f"; and {facts[-1]}"
    return f"{name} matches on several points: {joined}."
