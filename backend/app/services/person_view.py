"""Assemble the full ``PersonOut`` view model from normalized rows."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app import repositories as repo
from app.models import Person
from app.schemas import EducationOut, ExperienceOut, PersonOut, SkillOut


def experience_to_out(e) -> ExperienceOut:
    return ExperienceOut(
        position=e.position,
        company_name=e.company_name,
        company_linkedin_url=e.company_linkedin_url,
        start_year=e.start_year,
        end_year=e.end_year,
        is_current=e.is_current,
        duration_text=e.duration_text,
        description=e.description,
        location=e.location,
    )


def education_to_out(e) -> EducationOut:
    return EducationOut(
        school_name=e.school_name,
        degree=e.degree,
        field_of_study=e.field_of_study,
        start_year=e.start_year,
        end_year=e.end_year,
    )


def skill_to_out(s) -> SkillOut:
    return SkillOut(
        skill_name=s.skill_name,
        source=s.source,
        is_inferred=s.is_inferred,
        confidence=s.confidence,
        evidence=s.evidence,
    )


def person_to_out(db: Session, p: Person) -> PersonOut:
    sem = repo.get_semantic(db, p.id)
    return PersonOut(
        person_id=p.id,
        is_connection=p.is_connection,
        linkedin_url=p.linkedin_url,
        public_identifier=p.public_identifier,
        full_name=p.full_name,
        headline=p.headline,
        about=p.about,
        location_text=p.location_text,
        current_title=p.current_title,
        current_company=p.current_company,
        profile_picture_url=p.profile_picture_url,
        connections_count=p.connections_count,
        followers_count=p.followers_count,
        profile_completeness=p.profile_completeness,
        enrichment_state=p.enrichment_state,
        last_scraped_at=p.last_scraped_at,
        experiences=[experience_to_out(e) for e in repo.get_experiences(db, p.id)],
        education=[education_to_out(e) for e in repo.get_education(db, p.id)],
        skills=[skill_to_out(s) for s in repo.get_skills(db, p.id)],
        semantics=sem.data if sem else None,
    )
