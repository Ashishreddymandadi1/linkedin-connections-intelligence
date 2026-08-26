"""Turn deterministic evidence into a short explanation (spec §37, §61).

The LLM may only rephrase the evidence it is given — it must not introduce new
claims. If every free provider is down we fall back to a deterministic template,
which is still specific because it is built from the same evidence rows.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.config import settings
from app.services.llm.router import generate_structured
from app.services.scoring import ScoredCandidate

log = logging.getLogger("app.reason")


class _Reason(BaseModel):
    reason: str = Field(min_length=10, max_length=400)


_SYSTEM = (
    "You write one or two plain sentences explaining why a person matches a search. "
    "Use ONLY the evidence provided. Do not add employers, skills, dates or claims that "
    "are not in the evidence. No generic praise ('great engineer', 'strong background'). "
    "Name the concrete facts. Output JSON: {\"reason\": \"...\"}."
)


def generate_reason(candidate: ScoredCandidate, query: str, *, allow_llm: bool = True) -> str:
    name = candidate.person.full_name or "This connection"
    ev_lines = [f"- {e.text}" for e in candidate.evidence[:8]]
    matched = ", ".join(candidate.matched_criteria) or "general profile relevance"

    if allow_llm and settings.llm_reason_generation and ev_lines:
        user = (
            f"Search: {query!r}\nPerson: {name}\nMatched: {matched}\nEvidence:\n"
            + "\n".join(ev_lines)
            + "\n\nWrite the explanation."
        )
        result = generate_structured(_SYSTEM, user, _Reason, max_tokens=300)
        if result is not None:
            return result[0].reason.strip()
        log.info("reason generator: using deterministic template")

    return _template(name, candidate)


def _template(name: str, candidate: ScoredCandidate) -> str:
    if not candidate.evidence:
        return f"{name} came up as a broad match for this query but without a specific evidence hit."
    facts = [e.text for e in candidate.evidence[:4]]
    if len(facts) == 1:
        return f"{name} matches because {facts[0]}."
    joined = "; ".join(facts[:-1]) + f"; and {facts[-1]}"
    return f"{name} matches on several points: {joined}."
