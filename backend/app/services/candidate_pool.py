"""Retrieve the candidate connection pool for scoring (spec §10, §31, §33–§34).

A perfect scorer cannot recover a candidate that retrieval discarded — so for a
network up to ``FULL_SCAN_MAX_CONNECTIONS`` (default 5000) we score EVERYONE and
skip the prefilter entirely. ~1000 connections is small; correctness beats
shaving milliseconds.

Above that threshold we build a UNION of retrieval channels — structured
factual matches, embedding nearest-neighbours, semantic-assertion / company-
category matches, geographic matches — dedupe, and keep the top pool. An
initial literal SQL match never gets to decide who is eligible.
"""
from __future__ import annotations

import logging

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.constants import CriterionType, EnrichmentState
from app.models import (
    Certification, Education, Experience, Language, Person, ProfileSemantic, Publication, Skill,
)
from app.schemas import ParsedSearchQuery
from app.services.geo import expand_values
from app.services.matching import norm

log = logging.getLogger("app.candidates")

_SCORABLE = (
    EnrichmentState.READY,
    EnrichmentState.PARTIAL,
    EnrichmentState.WAITING_FOR_FREE_LLM,
)


def get_candidates(
    db: Session, dataset_id: str, parsed: ParsedSearchQuery, query_embedding: bytes | None
) -> tuple[list[Person], int]:
    all_people = list(
        db.scalars(
            select(Person)
            .where(Person.dataset_id == dataset_id)
            .where(Person.is_connection.is_(True))
            .where(Person.enrichment_state.in_(_SCORABLE))
        )
    )
    total = len(all_people)
    if total <= settings.full_scan_max_connections:
        return all_people, total

    by_id = {p.id: p for p in all_people}
    keep_ids: set[str] = set()

    # channel 1 — structured factual matches
    keep_ids |= _sql_matches(db, dataset_id, parsed)
    # channel 2 — semantic assertion / industry / company-category text signals
    keep_ids |= _semantic_matches(db, dataset_id, parsed)
    # channel 3 — geographic
    keep_ids |= _geo_matches(db, dataset_id, parsed)
    # channel 4 — embedding nearest-neighbours (always, so a purely-semantic query
    # that no SQL channel caught still gets a real candidate set)
    if query_embedding is not None:
        from app import repositories as repo
        from app.services.embeddings import cosine_scores

        ranked = cosine_scores(query_embedding, repo.all_embeddings(db, dataset_id))
        for pid, _score in ranked[: settings.candidate_pool_size]:
            keep_ids.add(pid)

    keep = [by_id[pid] for pid in keep_ids if pid in by_id]

    # if the union is still thin, top up with more embedding neighbours
    if query_embedding is not None and len(keep) < settings.candidate_pool_size:
        from app import repositories as repo
        from app.services.embeddings import cosine_scores

        for pid, _s in cosine_scores(query_embedding, repo.all_embeddings(db, dataset_id)):
            if pid not in keep_ids and pid in by_id:
                keep.append(by_id[pid])
                keep_ids.add(pid)
            if len(keep) >= settings.candidate_pool_size:
                break

    log.info("candidate pool (union): %d of %d connections", len(keep), total)
    return keep[: settings.candidate_pool_size], total


def _q(db: Session, stmt) -> set[str]:
    return set(db.scalars(stmt))


def _sql_matches(db: Session, dataset_id: str, parsed: ParsedSearchQuery) -> set[str]:
    ids: set[str] = set()
    for crit in parsed.criteria:
        for value in (crit.values or ([crit.value] if crit.value else [])):
            v = f"%{norm(value)}%"
            if crit.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY, CriterionType.COMPANY_CATEGORY):
                ids |= _q(db, select(Experience.person_id).join(Person).where(Person.dataset_id == dataset_id, Experience.company_name.ilike(v)))
                ids |= _q(db, select(Person.id).where(Person.dataset_id == dataset_id, Person.current_company.ilike(v)))
            elif crit.type == CriterionType.SKILL:
                ids |= _q(db, select(Skill.person_id).join(Person).where(Person.dataset_id == dataset_id, Skill.skill_name_norm.ilike(v)))
            elif crit.type == CriterionType.EDUCATION:
                ids |= _q(db, select(Education.person_id).join(Person).where(Person.dataset_id == dataset_id, or_(Education.school_name.ilike(v), Education.field_of_study.ilike(v))))
            elif crit.type == CriterionType.TITLE:
                ids |= _q(db, select(Experience.person_id).join(Person).where(Person.dataset_id == dataset_id, Experience.position.ilike(v)))
                ids |= _q(db, select(Person.id).where(Person.dataset_id == dataset_id, Person.current_title.ilike(v)))
            elif crit.type == CriterionType.CERTIFICATION:
                ids |= _q(db, select(Certification.person_id).join(Person).where(Person.dataset_id == dataset_id, or_(Certification.name.ilike(v), Certification.issuer.ilike(v))))
            elif crit.type == CriterionType.LANGUAGE:
                for tok in norm(value).split():
                    if len(tok) > 1:
                        ids |= _q(db, select(Language.person_id).join(Person).where(Person.dataset_id == dataset_id, Language.name_norm.ilike(f"%{tok}%")))
            elif crit.type == CriterionType.PUBLICATION:
                ids |= _q(db, select(Publication.person_id).join(Person).where(Person.dataset_id == dataset_id, or_(Publication.title.ilike(v), Publication.description.ilike(v))))
    return ids


def _semantic_matches(db: Session, dataset_id: str, parsed: ParsedSearchQuery) -> set[str]:
    """Match a semantic_concept / company_category against the JSON semantic
    blob (industries / job_families / assertion concepts / searchable_keywords).
    A recall channel only — the scorer decides the real strength."""
    concepts = [
        (c.concept or c.value or "")
        for c in parsed.criteria
        if c.type in (CriterionType.SEMANTIC_CONCEPT, CriterionType.COMPANY_CATEGORY, CriterionType.DOMAIN)
    ]
    if not concepts:
        return set()
    ids: set[str] = set()
    rows = db.execute(
        select(ProfileSemantic.person_id, ProfileSemantic.data)
        .join(Person, Person.id == ProfileSemantic.person_id)
        .where(Person.dataset_id == dataset_id)
    ).all()
    for pid, data in rows:
        if not data:
            continue
        blob = " ".join(
            str(x) for x in (
                data.get("industries", []) + data.get("job_families", []) + data.get("domain_expertise", [])
                + data.get("leadership_experience", []) + data.get("role_keywords", [])
                + data.get("searchable_keywords", [])
                + [a.get("concept", "") for a in data.get("semantic_assertions", []) if isinstance(a, dict)]
            )
        ).lower()
        if any(any(tok in blob for tok in norm(cc).split() if len(tok) > 3) for cc in concepts):
            ids.add(pid)
    return ids


def _geo_matches(db: Session, dataset_id: str, parsed: ParsedSearchQuery) -> set[str]:
    ids: set[str] = set()
    for crit in parsed.criteria:
        if crit.type != CriterionType.LOCATION:
            continue
        for value in expand_values(crit.values or [crit.value]):
            v = f"%{norm(value)}%"
            ids |= _q(db, select(Person.id).where(
                Person.dataset_id == dataset_id,
                or_(Person.location_text.ilike(v), Person.city.ilike(v), Person.state.ilike(v)),
            ))
    return ids
