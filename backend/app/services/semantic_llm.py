"""Derive ProfileSemanticData for one person via the free-LLM chain (spec §26–§27)."""
from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from app.models import Person
from app.schemas import ProfileSemanticData
from app.services.compact_profile import build_compact_profile
from app.services.llm.router import generate_structured

log = logging.getLogger("app.semantic_llm")

_SYSTEM = (
    "You are a precise professional-profile analyst. You extract and lightly interpret "
    "structured facts from a LinkedIn profile. You MUST NOT invent employers, job titles, "
    "employment dates, schools, degrees, certifications, projects, publications or awards. "
    "If something is not supported by the profile, use null or an empty list. "
    "Inferred skills are allowed ONLY with direct textual evidence from the profile, and "
    "each must include the exact evidence phrase and a confidence between 0 and 1. "
    "Return a single JSON object, nothing else."
)

_INSTRUCTIONS = """From the profile JSON below, produce:
- seniority_level: one of "intern","entry","mid","senior","staff","principal","director","vp","cxo","founder" or null
- job_families: e.g. ["backend engineering","data science"]
- technical_domains: e.g. ["distributed systems","cloud infrastructure","fraud detection"]
- industries: inferred from employers, e.g. ["fintech","big tech"]
- explicit_skills: skills clearly stated on the profile (copy them)
- inferred_skills: [{skill, confidence, evidence}] — evidence must be a phrase copied from the profile
- leadership_experience: short phrases describing team/tech leadership, [] if none
- domain_expertise: areas of deep expertise backed by multiple roles/description
- career_summary: 1-2 sentence factual summary (no praise, no adjectives like "excellent")
- years_of_experience: rough number from earliest professional role to now
- current_role_summary: 1 sentence about the current role, or null
- past_company_names / current_company_names: copied exactly from experience
- education_keywords: schools + fields of study
- role_keywords: normalized role words ("backend","ml","platform","security"...)
- searchable_keywords: 10-25 lowercase keywords someone might search to find this person

PROFILE:
"""


def derive_semantics(db: Session, person: Person) -> tuple[dict, str, str] | None:
    compact = build_compact_profile(db, person)
    user = _INSTRUCTIONS + json.dumps(compact, ensure_ascii=False, indent=2)

    result = generate_structured(_SYSTEM, user, ProfileSemanticData, max_tokens=1600)
    if result is None:
        return None

    data, provider, model = result
    payload = _ground(data, compact)
    return payload.model_dump(), provider, model


def _ground(data: ProfileSemanticData, compact: dict) -> ProfileSemanticData:
    """Defensive pass: drop hallucinated company names, clamp obvious drift."""
    real_companies = {
        (e.get("company") or "").strip().lower()
        for e in compact.get("experience", [])
        if e.get("company")
    }
    if real_companies:
        data.past_company_names = [c for c in data.past_company_names if c.strip().lower() in real_companies]
        data.current_company_names = [
            c for c in data.current_company_names if c.strip().lower() in real_companies
        ]
    # inferred skills must carry evidence
    data.inferred_skills = [s for s in data.inferred_skills if s.evidence and s.evidence.strip()]
    return data
