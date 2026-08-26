"""Retrieve the candidate connection pool for scoring (spec §33–§34).

Ordinary searches NEVER call Apify — the data is already local. For datasets up
to a few hundred connections we score everyone (scoring is deterministic and
cheap). For larger datasets we pre-rank with SQL signals + embedding similarity
and keep the top ``CANDIDATE_POOL_SIZE``.
"""
from __future__ import annotations

import logging

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.constants import CriterionType, EnrichmentState
from app.models import Education, Experience, Person, Skill
from app.schemas import ParsedSearchQuery
from app.services.matching import norm

log = logging.getLogger("app.candidates")

# WAITING_FOR_FREE_LLM profiles are fully scraped + normalized + embedded — only
# the optional semantic keywords are missing, so they are still searchable.
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
    if total <= settings.candidate_pool_size * 2:
        return all_people, total

    # large dataset: keep a superset via SQL signal, then embedding top-up
    ids = _sql_prefilter(db, dataset_id, parsed)
    keep = [p for p in all_people if p.id in ids]

    if query_embedding is not None and len(keep) < settings.candidate_pool_size:
        keep_ids = {p.id for p in keep}
        from app import repositories as repo
        from app.services.embeddings import cosine_scores

        ranked = cosine_scores(query_embedding, repo.all_embeddings(db, dataset_id))
        by_id = {p.id: p for p in all_people}
        for pid, _score in ranked:
            if pid not in keep_ids and pid in by_id:
                keep.append(by_id[pid])
                keep_ids.add(pid)
            if len(keep) >= settings.candidate_pool_size:
                break

    log.info("candidate pool: %d of %d connections", len(keep), total)
    return keep[: settings.candidate_pool_size], total


def _sql_prefilter(db: Session, dataset_id: str, parsed: ParsedSearchQuery) -> set[str]:
    ids: set[str] = set()
    for crit in parsed.criteria:
        v = f"%{norm(crit.value)}%"
        if crit.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY):
            ids |= _q(db, select(Experience.person_id).where(Experience.company_name.ilike(v)))
            ids |= _q(
                db,
                select(Person.id).where(Person.dataset_id == dataset_id).where(Person.current_company.ilike(v)),
            )
        elif crit.type == CriterionType.SKILL:
            ids |= _q(db, select(Skill.person_id).where(Skill.skill_name_norm.ilike(v)))
        elif crit.type == CriterionType.EDUCATION:
            ids |= _q(
                db,
                select(Education.person_id).where(
                    or_(Education.school_name.ilike(v), Education.field_of_study.ilike(v))
                ),
            )
        elif crit.type == CriterionType.TITLE:
            ids |= _q(db, select(Experience.person_id).where(Experience.position.ilike(v)))
            ids |= _q(
                db,
                select(Person.id).where(Person.dataset_id == dataset_id).where(Person.current_title.ilike(v)),
            )
        else:
            ids |= _q(
                db,
                select(Person.id)
                .where(Person.dataset_id == dataset_id)
                .where(or_(Person.headline.ilike(v), Person.about.ilike(v))),
            )
    return ids


def _q(db: Session, stmt) -> set[str]:
    return set(db.scalars(stmt))
