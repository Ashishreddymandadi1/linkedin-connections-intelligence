"""Build the compact, text-only profile payload for the LLM (spec §11–§12).

Strips every image/logo/cover URL, tracking param and repeated company-metadata
blob. HarvestAPI already gave us the structured facts — the LLM only needs the
searchable text. Keeping this tight is what stays inside free token quotas.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app import repositories as repo
from app.models import Person


def build_compact_profile(db: Session, person: Person) -> dict:
    exps = repo.get_experiences(db, person.id)
    edus = repo.get_education(db, person.id)
    skills = repo.get_skills(db, person.id)
    raw = repo.latest_raw_profile(db, person.id)
    raw_json = raw.raw_json if raw else {}

    return {
        "name": person.full_name,
        "headline": person.headline,
        "location": person.location_text,
        "about": (person.about or "")[:1500] or None,
        "current": {
            "title": person.current_title,
            "company": person.current_company,
        },
        "experience": [
            {
                "title": e.position,
                "company": e.company_name,
                "start_year": e.start_year,
                "end_year": e.end_year if not e.is_current else None,
                "is_current": e.is_current,
                "employment_type": e.employment_type,
                "location": e.location,
                "description": (e.description or "")[:800] or None,
                "listed_skills": e.skills_json or [],
            }
            for e in exps[:10]
        ],
        "education": [
            {
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
            _s(c.get("title")) for c in _list(raw_json.get("certifications")) if isinstance(c, dict)
        ][:15],
        "languages": [
            _s(x.get("name")) if isinstance(x, dict) else _s(x)
            for x in _list(raw_json.get("languages"))
        ][:10],
        "publications": [
            _s(p.get("title")) for p in _list(raw_json.get("publications")) if isinstance(p, dict)
        ][:8],
    }


def _list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _s(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None
