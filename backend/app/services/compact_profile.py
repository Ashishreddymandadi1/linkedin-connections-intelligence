"""Build the compact, text-only profile payload for the LLM (spec §8, §11–§12).

Strips every image/logo/cover URL, tracking param and repeated company-metadata
blob. HarvestAPI already gave us the structured facts — the LLM only needs the
searchable text. Keeping this tight is what stays inside free token quotas.

Reads from the NORMALIZED tables (not raw_json) for certifications/languages/
publications so this stays consistent with the v2 sub-tables, and includes
volunteering + recommendations — evidence for concepts like "mentor" or
"leadership" often lives there, not in a skill named "mentor" (spec §8/§23).
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app import repositories as repo
from app.models import Person


def build_compact_profile(db: Session, person: Person) -> dict:
    exps = repo.get_experiences(db, person.id)
    edus = repo.get_education(db, person.id)
    skills = repo.get_skills(db, person.id)
    certs = repo.get_certifications(db, person.id)
    langs = repo.get_languages(db, person.id)
    pubs = repo.get_publications(db, person.id)
    volunteering = repo.get_extra_section(db, person.id, "volunteering")
    recommendations = repo.get_extra_section(db, person.id, "recommendations")

    return {
        "name": person.full_name,
        "headline": person.headline,
        "location": person.location_text,
        "about": (person.about or "")[:2200] or None,
        "current": {
            "title": person.current_title,
            "company": person.current_company,
        },
        # experience_id is the stable normalized-row id — semantic output points
        # back to it (V4 §9), never to duplicable evidence text.
        "experience": [
            {
                "experience_id": e.id,
                "title": e.position,
                "company": e.company_name,
                "start_year": e.start_year,
                "end_year": e.end_year if not e.is_current else None,
                "is_current": e.is_current,
                "employment_type": e.employment_type,
                "location": e.location,
                "description": (e.description or "")[:1200] or None,
                "listed_skills": e.skills_json or [],
            }
            for e in exps[:14]
        ],
        "education": [
            {
                "education_id": ed.id,
                "school": ed.school_name,
                "degree": ed.degree,
                "field_of_study": ed.field_of_study,
                "start_year": ed.start_year,
                "end_year": ed.end_year,
            }
            for ed in edus[:6]
        ],
        "explicit_skills": [s.skill_name for s in skills][:40],
        "certifications": [
            {"certification_id": c.id, "name": c.name} for c in certs if c.name
        ][:15],
        "languages": [lang.name for lang in langs if lang.name][:10],
        "publications": [p.title for p in pubs if p.title][:8],
        # evidence sources for leadership/mentorship/advising concepts (spec §8/§23)
        "volunteering": [
            {"role": v.role, "organization": v.organization, "description": (v.description or "")[:300]}
            for v in volunteering[:8]
        ],
        "recommendations_received": [
            (r.text or "")[:400] for r in recommendations[:5] if r.text
        ],
    }
