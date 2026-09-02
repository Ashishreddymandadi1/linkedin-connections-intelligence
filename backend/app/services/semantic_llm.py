"""Derive ProfileSemanticData for one person via the free-LLM chain (spec §26–§27, §7)."""
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
    "Inferred skills and semantic assertions are allowed ONLY with direct textual evidence "
    "copied from the profile, with a confidence between 0 and 1. "
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
- experience_semantics: [ONE object per meaningful experience, keyed by the exact
  experience_id from the PROFILE input]:
    {"experience_id": "<copy exactly>",
     "role_function": what the PERSON DID ("software engineering", "accounting",
        "product management", "sales") — this is the FUNCTION, NOT the employer's industry,
     "professional_domain": broader domain ("software systems", "finance", "marketing"),
     "role_domains": specific areas worked on,
     "role_seniority": one of intern/entry/mid/senior/staff/principal/director/vp/cxo/founder,
     "employer_industries": the industry the EMPLOYER operates in ("technology",
        "financial services", "healthcare") — inferred from the role context, kept
        SEPARATE from role_function,
     "employer_categories": rough size/type ("big tech","large bank","startup") — advisory only,
     "leadership_signals": phrases showing this role involved leading people/teams, [] if none,
     "mentoring_signals": phrases showing the person MENTORED others (not just "mentor program"),
     "founder_signals": phrases showing they founded/co-founded something,
     "confidence": 0..1}
  CRITICAL — role_function is NOT employer_industry:
    "Accountant at Google" -> role_function "accounting", employer_industries ["technology"].
      This person does NOT have a software/technical role.
    "Software Engineer at JPMorgan" -> role_function "software engineering",
      employer_industries ["financial services"]. This does NOT make JPMorgan a tech company.
    A consultant who "led a technology transformation" has role_function "consulting",
      NOT technology-industry employment.
- semantic_assertions: [{concept, category, scope, confidence, evidence, experience_ids}] —
  professional concepts the career supports. experience_ids MUST be a list of exact
  experience_id values from the PROFILE input that support the concept. e.g.
    {"concept": "technology industry experience", "category": "industry_experience",
     "scope": "past", "confidence": 0.95, "experience_ids": ["<id>"],
     "evidence": ["Software Engineer at Google"]}
  Only assert a concept when evidence genuinely supports it:
    "technology transformation" in a non-tech role is NOT technology-industry experience.
    "worked with Amazon API" is NOT employment at Amazon.
    "participated in a mentor program" does NOT make someone a mentor.
    "sold to CXOs" does NOT make someone a CXO.
  category is one of: industry_experience, role_function, leadership, mentorship,
  company_category, domain_expertise. scope is one of: current, past, any_experience, career.
  Do NOT classify well-known companies from general knowledge here — that is done
  separately. Describe the ROLE and cite the experience_id.

PROFILE:
"""


def derive_semantics(db: Session, person: Person) -> tuple[dict, str, str] | None:
    compact = build_compact_profile(db, person)
    user = _INSTRUCTIONS + json.dumps(compact, ensure_ascii=False, indent=2)

    result = generate_structured(
        _SYSTEM, user, ProfileSemanticData, max_tokens=2200,
        operation="profile_semantic_enrichment",
    )
    if result is None:
        return None

    data, provider, model = result
    payload = _ground(data, compact)
    return payload.model_dump(), provider, model


def _ground(data: ProfileSemanticData, compact: dict) -> ProfileSemanticData:
    """Defensive pass (V4 §14): drop hallucinated company names, clamp drift, drop
    assertions with no evidence, and REMOVE every source id the LLM invented."""
    exps = compact.get("experience", [])
    real_companies = {(e.get("company") or "").strip().lower() for e in exps if e.get("company")}
    valid_exp = {e["experience_id"] for e in exps if e.get("experience_id")}
    valid_edu = {e["education_id"] for e in compact.get("education", []) if e.get("education_id")}
    valid_cert = {c["certification_id"] for c in compact.get("certifications", []) if isinstance(c, dict) and c.get("certification_id")}

    if real_companies:
        data.past_company_names = [c for c in data.past_company_names if c.strip().lower() in real_companies]
        data.current_company_names = [c for c in data.current_company_names if c.strip().lower() in real_companies]

    data.inferred_skills = [s for s in data.inferred_skills if s.evidence and s.evidence.strip()]

    # experience_semantics: keep only rows that point at a real experience_id
    data.experience_semantics = [
        es for es in data.experience_semantics if es.experience_id in valid_exp
    ]

    # semantic_assertions: drop invalid ids; keep the assertion only if it still
    # has SOME grounding (an evidence phrase OR at least one valid source id)
    kept = []
    for a in data.semantic_assertions:
        a.experience_ids = [i for i in a.experience_ids if i in valid_exp]
        a.education_ids = [i for i in a.education_ids if i in valid_edu]
        a.certification_ids = [i for i in a.certification_ids if i in valid_cert]
        has_evidence = bool(a.evidence and any(e.strip() for e in a.evidence))
        has_ids = bool(a.experience_ids or a.education_ids or a.certification_ids)
        if has_evidence or has_ids:
            kept.append(a)
    data.semantic_assertions = kept
    return data
