"""Build the compact searchable paragraph per profile (spec §9, §28).

Text only — no media URLs, no logos. Used for embeddings and cross-encoder
reranking. Describes professional MEANING (industries, job families,
leadership, company categories, career summary) — not just a keyword dump —
so semantic similarity can find "big tech engineering leader now at a
startup" without those exact words ever appearing on the profile.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app import repositories as repo
from app.models import Person
from app.services.company_intel import company_key, get_or_classify


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

    # company classifications — cache-only lookup, never triggers a new LLM
    # call from the search/enrichment hot path (classification happens in its
    # own backfill step).
    companies = [(e.company_id, e.company_name, e.company_linkedin_url) for e in exps[:6] if e.company_name]
    if companies:
        keys = {company_key(cid, nm) for cid, nm, _ in companies}
        classified = repo.get_company_semantics(db, list(keys))
        cat_phrases = []
        for cid, nm, _url in companies:
            row = classified.get(company_key(cid, nm))
            if not row:
                continue
            bits = []
            if row.is_big_tech:
                bits.append("a major technology company")
            elif row.is_startup:
                bits.append("an early-stage/independent company")
            elif row.is_technology_company:
                bits.append("a technology company")
            if row.industries:
                bits.append("in " + ", ".join(row.industries[:3]))
            if bits:
                cat_phrases.append(f"{nm} is " + " ".join(bits) + ".")
        parts.extend(cat_phrases[:4])

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
        d = sem.data
        if d.get("seniority_level"):
            parts.append(f"Seniority: {d['seniority_level']}.")
        if d.get("current_role_summary"):
            parts.append(d["current_role_summary"])
        if d.get("career_summary"):
            parts.append(d["career_summary"][:400])
        if d.get("industries"):
            parts.append("Industry experience: " + ", ".join(d["industries"][:6]) + ".")
        if d.get("job_families"):
            parts.append("Career includes: " + ", ".join(d["job_families"][:6]) + ".")
        if d.get("leadership_experience"):
            parts.append("Leadership: " + "; ".join(d["leadership_experience"][:4]) + ".")
        if d.get("domain_expertise"):
            parts.append("Domain expertise: " + ", ".join(d["domain_expertise"][:6]) + ".")
        for assertion in (d.get("semantic_assertions") or [])[:6]:
            concept = assertion.get("concept") if isinstance(assertion, dict) else None
            if concept:
                parts.append(concept + ".")
        # experience-level meaning (V4 §17) — role functions kept distinct from
        # employer industries in the embedding document too
        roles, inds = [], []
        for es in (d.get("experience_semantics") or [])[:14]:
            if not isinstance(es, dict):
                continue
            if es.get("role_function"):
                roles.append(es["role_function"])
            inds.extend(es.get("employer_industries") or [])
        if roles:
            parts.append("Roles held: " + ", ".join(dict.fromkeys(roles)) + ".")
        if inds:
            parts.append("Employer industries: " + ", ".join(dict.fromkeys(inds)) + ".")
        extra = list(dict.fromkeys([*(d.get("technical_domains") or []), *(d.get("searchable_keywords") or [])]))[:20]
        if extra:
            parts.append("Also: " + ", ".join(extra) + ".")

    if person.about:
        parts.append(person.about[:600])

    return "\n".join(p.strip() for p in parts if p and p.strip())
