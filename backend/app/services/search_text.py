"""Build the compact searchable paragraph per profile (spec §28).

Text only — no media URLs, no logos. Used for embeddings and keyword search.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app import repositories as repo
from app.models import Person


def build_search_text(db: Session, person: Person) -> str:
    parts: list[str] = []
    name = person.full_name or "This person"
    if person.current_title and person.current_company:
        parts.append(f"{name} is a {person.current_title} at {person.current_company}.")
    elif person.headline:
        parts.append(f"{name}: {person.headline}.")
    else:
        parts.append(f"{name}.")

    if person.location_text:
        parts.append(f"Based in {person.location_text}.")

    exps = repo.get_experiences(db, person.id)
    for e in exps[:6]:
        seg = " ".join(
            x
            for x in [
                f"{e.position or 'Worked'}",
                f"at {e.company_name}" if e.company_name else "",
                f"({e.start_year}-{e.end_year or 'present'})" if e.start_year else "",
            ]
            if x
        )
        if seg:
            parts.append(seg + ".")
        if e.description:
            parts.append(e.description[:400])

    edus = repo.get_education(db, person.id)
    for ed in edus[:3]:
        seg = " ".join(
            x
            for x in [
                ed.degree or "Studied",
                f"in {ed.field_of_study}" if ed.field_of_study else "",
                f"at {ed.school_name}" if ed.school_name else "",
            ]
            if x
        )
        if seg:
            parts.append(seg + ".")

    skills = [s.skill_name for s in repo.get_skills(db, person.id)]
    if skills:
        parts.append("Skills: " + ", ".join(skills[:25]) + ".")

    sem = repo.get_semantic(db, person.id)
    if sem and sem.data:
        kws = sem.data.get("searchable_keywords") or []
        doms = sem.data.get("technical_domains") or []
        extra = list(dict.fromkeys([*doms, *kws]))[:20]
        if extra:
            parts.append("Also: " + ", ".join(extra) + ".")
        if sem.data.get("career_summary"):
            parts.append(sem.data["career_summary"][:400])

    if person.about:
        parts.append(person.about[:600])

    return "\n".join(p.strip() for p in parts if p and p.strip())
