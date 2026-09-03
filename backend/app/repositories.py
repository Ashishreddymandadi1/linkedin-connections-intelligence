"""Data-access layer — the only place that builds queries against the ORM.

Services and routers call these functions; they never touch ``Session.query``
directly. This keeps the Postgres migration surface to one file.
"""
from __future__ import annotations

from collections import Counter

from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.orm import Session

from app.constants import EnrichmentState
from app.models import (
    Certification,
    CompanySemantic,
    Connection,
    Dataset,
    Education,
    EnrichmentJob,
    Experience,
    Language,
    Patent,
    Person,
    ProfileEmbedding,
    ProfileSemantic,
    Publication,
    RawProfile,
    Recommendation,
    SearchQuery,
    SearchResult,
    SearchRunState,
    Skill,
    Volunteering,
)

#: extra profile sections broken out in v2 — model class keyed by the dict key
#: that ``normalize_profile`` returns them under.
EXTRA_SECTION_MODELS = {
    "certifications": Certification,
    "publications": Publication,
    "patents": Patent,
    "languages": Language,
    "volunteering": Volunteering,
    "recommendations": Recommendation,
}

# ─────────────────────────── datasets ───────────────────────────


def create_dataset(db: Session, name: str) -> Dataset:
    ds = Dataset(name=name or "My LinkedIn Network")
    db.add(ds)
    db.flush()
    return ds


def get_dataset(db: Session, dataset_id: str) -> Dataset | None:
    return db.get(Dataset, dataset_id)


def list_datasets(db: Session) -> list[Dataset]:
    return list(db.scalars(select(Dataset).order_by(Dataset.created_at.desc())))


def delete_dataset(db: Session, dataset_id: str) -> bool:
    """Explicit cascade (spec §59) — also works if FK pragma is off."""
    ds = db.get(Dataset, dataset_id)
    if not ds:
        return False
    person_ids = list(db.scalars(select(Person.id).where(Person.dataset_id == dataset_id)))
    search_ids = list(db.scalars(select(SearchQuery.id).where(SearchQuery.dataset_id == dataset_id)))
    if search_ids:
        db.execute(delete(SearchRunState).where(SearchRunState.search_id.in_(search_ids)))
        db.execute(delete(SearchResult).where(SearchResult.search_id.in_(search_ids)))
    db.execute(delete(SearchQuery).where(SearchQuery.dataset_id == dataset_id))
    if person_ids:
        for model in (ProfileEmbedding, ProfileSemantic, Skill, Education, Experience, RawProfile):
            db.execute(delete(model).where(model.person_id.in_(person_ids)))
    db.execute(delete(Connection).where(Connection.dataset_id == dataset_id))
    db.execute(delete(EnrichmentJob).where(EnrichmentJob.dataset_id == dataset_id))
    db.execute(delete(Person).where(Person.dataset_id == dataset_id))
    db.delete(ds)
    db.flush()
    return True


def touch_dataset(db: Session, dataset_id: str, *, status: str | None = None) -> None:
    ds = db.get(Dataset, dataset_id)
    if ds and status:
        ds.status = status
    db.flush()


# ─────────────────────────── people ───────────────────────────


def add_person(db: Session, **kwargs) -> Person:
    p = Person(**kwargs)
    db.add(p)
    db.flush()
    return p


def get_person(db: Session, person_id: str) -> Person | None:
    return db.get(Person, person_id)


def list_people(db: Session, dataset_id: str, *, is_connection: bool | None = True) -> list[Person]:
    stmt = select(Person).where(Person.dataset_id == dataset_id)
    if is_connection is not None:
        stmt = stmt.where(Person.is_connection == is_connection)
    return list(db.scalars(stmt.order_by(Person.full_name)))


def people_needing_enrichment(db: Session, dataset_id: str, *, defer_waiting: bool = True) -> list[Person]:
    """Non-terminal people. WAITING_FOR_FREE_LLM is sorted LAST so a rate-limited
    semantic step never blocks the rest of the pipeline (spec §25, §54)."""
    stmt = (
        select(Person)
        .where(Person.dataset_id == dataset_id)
        .where(Person.enrichment_state.notin_(list(_TERMINAL)))
    )
    if defer_waiting:
        waiting_last = case((Person.enrichment_state == EnrichmentState.WAITING_FOR_FREE_LLM, 1), else_=0)
        stmt = stmt.order_by(waiting_last, Person.created_at)
    else:
        stmt = stmt.order_by(Person.created_at)
    return list(db.scalars(stmt))


_TERMINAL = {EnrichmentState.READY, EnrichmentState.PARTIAL, EnrichmentState.FAILED}


def people_missing_semantics(db: Session, dataset_id: str, *, current_version: int | None = None) -> list[Person]:
    """READY/PARTIAL people whose semantic pass never completed, OR (when
    ``current_version`` is given) whose semantics predate a version bump —
    picked up by a resume / backfill. No Apify re-scrape needed either way."""
    stmt = (
        select(Person)
        .where(Person.dataset_id == dataset_id)
        .where(Person.enrichment_state.in_([EnrichmentState.READY, EnrichmentState.PARTIAL]))
    )
    if current_version is None:
        stmt = stmt.where(Person.semantic_version.is_(None))
    else:
        stmt = stmt.where(
            or_(Person.semantic_version.is_(None), Person.semantic_version < current_version)
        )
    return list(db.scalars(stmt.order_by(Person.created_at)))


def enrichment_state_counts(db: Session, dataset_id: str) -> dict[str, int]:
    rows = db.execute(
        select(Person.enrichment_state, func.count())
        .where(Person.dataset_id == dataset_id)
        .group_by(Person.enrichment_state)
    ).all()
    counts = Counter()
    for state, n in rows:
        counts[state] = n
    return dict(counts)


def find_person_by_public_id(db: Session, dataset_id: str, public_id: str) -> Person | None:
    return db.scalars(
        select(Person)
        .where(Person.dataset_id == dataset_id)
        .where(Person.public_identifier == public_id)
    ).first()


def company_name_index(db: Session, dataset_id: str) -> dict[str, set[str]]:
    """``{normalized company name -> {company_id, …}}`` across the dataset's
    experiences — used to resolve a company criterion to a LinkedIn company id."""
    rows = db.execute(
        select(Experience.company_name, Experience.company_id)
        .join(Person, Person.id == Experience.person_id)
        .where(Person.dataset_id == dataset_id)
        .where(Experience.company_id.is_not(None))
        .where(Experience.company_name.is_not(None))
    ).all()
    from app.services.matching import norm_company

    idx: dict[str, set[str]] = {}
    for name, cid in rows:
        idx.setdefault(norm_company(name), set()).add(cid)
    return idx


# ─────────────────── profile parts ───────────────────


def replace_experiences(db: Session, person_id: str, rows: list[dict]) -> None:
    db.execute(delete(Experience).where(Experience.person_id == person_id))
    for i, r in enumerate(rows):
        db.add(Experience(person_id=person_id, order_index=i, **r))
    db.flush()


def replace_education(db: Session, person_id: str, rows: list[dict]) -> None:
    db.execute(delete(Education).where(Education.person_id == person_id))
    for i, r in enumerate(rows):
        db.add(Education(person_id=person_id, order_index=i, **r))
    db.flush()


def replace_skills(db: Session, person_id: str, rows: list[dict]) -> None:
    db.execute(delete(Skill).where(Skill.person_id == person_id))
    for r in rows:
        db.add(Skill(person_id=person_id, **r))
    db.flush()


def get_experiences(db: Session, person_id: str) -> list[Experience]:
    return list(
        db.scalars(
            select(Experience).where(Experience.person_id == person_id).order_by(Experience.order_index)
        )
    )


def get_education(db: Session, person_id: str) -> list[Education]:
    return list(
        db.scalars(
            select(Education).where(Education.person_id == person_id).order_by(Education.order_index)
        )
    )


def get_skills(db: Session, person_id: str) -> list[Skill]:
    return list(db.scalars(select(Skill).where(Skill.person_id == person_id)))


def replace_extra_sections(db: Session, person_id: str, normalized: dict) -> None:
    """Replace all v2 sub-section rows (certifications/publications/…) for a person."""
    for key, model in EXTRA_SECTION_MODELS.items():
        db.execute(delete(model).where(model.person_id == person_id))
        for i, row in enumerate(normalized.get(key, []) or []):
            db.add(model(person_id=person_id, order_index=i, **row))
    db.flush()


def get_extra_section(db: Session, person_id: str, key: str) -> list:
    model = EXTRA_SECTION_MODELS[key]
    stmt = select(model).where(model.person_id == person_id)
    if hasattr(model, "order_index"):
        stmt = stmt.order_by(model.order_index)
    return list(db.scalars(stmt))


def get_certifications(db: Session, person_id: str) -> list[Certification]:
    return get_extra_section(db, person_id, "certifications")


def get_languages(db: Session, person_id: str) -> list[Language]:
    return get_extra_section(db, person_id, "languages")


def get_publications(db: Session, person_id: str) -> list[Publication]:
    return get_extra_section(db, person_id, "publications")


# ─────────────────── raw / semantic / embedding ───────────────────


def add_raw_profile(db: Session, **kwargs) -> RawProfile:
    rp = RawProfile(**kwargs)
    db.add(rp)
    db.flush()
    return rp


def latest_raw_profile(db: Session, person_id: str) -> RawProfile | None:
    return db.scalars(
        select(RawProfile).where(RawProfile.person_id == person_id).order_by(RawProfile.scraped_at.desc())
    ).first()


def upsert_semantic(db: Session, person_id: str, data: dict, *, version: int, provider: str, model: str) -> ProfileSemantic:
    existing = db.scalars(select(ProfileSemantic).where(ProfileSemantic.person_id == person_id)).first()
    if existing:
        existing.data = data
        existing.version = version
        existing.llm_provider = provider
        existing.llm_model = model
        db.flush()
        return existing
    row = ProfileSemantic(
        person_id=person_id, data=data, version=version, llm_provider=provider, llm_model=model
    )
    db.add(row)
    db.flush()
    return row


def get_semantic(db: Session, person_id: str) -> ProfileSemantic | None:
    return db.scalars(select(ProfileSemantic).where(ProfileSemantic.person_id == person_id)).first()


def upsert_embedding(db: Session, person_id: str, *, model: str, dim: int, vector: bytes, search_text: str) -> None:
    existing = db.scalars(select(ProfileEmbedding).where(ProfileEmbedding.person_id == person_id)).first()
    if existing:
        existing.model = model
        existing.dim = dim
        existing.vector = vector
        existing.search_text = search_text
    else:
        db.add(
            ProfileEmbedding(
                person_id=person_id, model=model, dim=dim, vector=vector, search_text=search_text
            )
        )
    db.flush()


def all_embeddings(db: Session, dataset_id: str) -> list[tuple[str, bytes]]:
    rows = db.execute(
        select(ProfileEmbedding.person_id, ProfileEmbedding.vector)
        .join(Person, Person.id == ProfileEmbedding.person_id)
        .where(Person.dataset_id == dataset_id)
    ).all()
    return [(pid, vec) for pid, vec in rows]


# ─────────────────── jobs ───────────────────


def create_job(db: Session, dataset_id: str, *, actor_id: str, requested: int) -> EnrichmentJob:
    job = EnrichmentJob(dataset_id=dataset_id, actor_id=actor_id, requested_profiles=requested)
    db.add(job)
    db.flush()
    return job


def latest_job(db: Session, dataset_id: str) -> EnrichmentJob | None:
    return db.scalars(
        select(EnrichmentJob).where(EnrichmentJob.dataset_id == dataset_id).order_by(EnrichmentJob.created_at.desc())
    ).first()


# ─────────────────── search history ───────────────────


def create_search_query(db: Session, **kwargs) -> SearchQuery:
    sq = SearchQuery(**kwargs)
    db.add(sq)
    db.flush()
    return sq


def get_search_query(db: Session, search_id: str) -> SearchQuery | None:
    return db.get(SearchQuery, search_id)


def list_search_queries(db: Session, dataset_id: str, limit: int = 25) -> list[SearchQuery]:
    return list(
        db.scalars(
            select(SearchQuery)
            .where(SearchQuery.dataset_id == dataset_id)
            .order_by(SearchQuery.created_at.desc())
            .limit(limit)
        )
    )


def add_search_result(db: Session, **kwargs) -> None:
    db.add(SearchResult(**kwargs))
    db.flush()


def get_search_results(db: Session, search_id: str) -> list[SearchResult]:
    return list(
        db.scalars(
            select(SearchResult).where(SearchResult.search_id == search_id).order_by(SearchResult.rank)
        )
    )


def upsert_search_run_state(db: Session, search_id: str, **fields) -> SearchRunState:
    """One row per search — the FINAL validated response-level snapshot (V4 PART 7)."""
    existing = db.scalars(
        select(SearchRunState).where(SearchRunState.search_id == search_id)
    ).first()
    if existing:
        for k, v in fields.items():
            setattr(existing, k, v)
        db.flush()
        return existing
    row = SearchRunState(search_id=search_id, **fields)
    db.add(row)
    db.flush()
    return row


def get_search_run_state(db: Session, search_id: str) -> SearchRunState | None:
    return db.scalars(
        select(SearchRunState).where(SearchRunState.search_id == search_id)
    ).first()


# ─────────────────── company intelligence (spec §4) ───────────────────


def get_company_semantics(db: Session, keys: list[str]) -> dict[str, CompanySemantic]:
    """Bulk lookup by ``company_key`` (see company_intel.company_key)."""
    if not keys:
        return {}
    rows = db.scalars(select(CompanySemantic).where(CompanySemantic.company_key.in_(keys)))
    return {r.company_key: r for r in rows}


def upsert_company_semantic(db: Session, company_key: str, **fields) -> CompanySemantic:
    existing = db.scalars(
        select(CompanySemantic).where(CompanySemantic.company_key == company_key)
    ).first()
    if existing:
        for k, v in fields.items():
            setattr(existing, k, v)
        db.flush()
        return existing
    row = CompanySemantic(company_key=company_key, **fields)
    db.add(row)
    db.flush()
    return row


def distinct_companies(db: Session, dataset_id: str) -> list[tuple[str | None, str, str | None]]:
    """``[(company_id, company_name, company_linkedin_url)]`` for every distinct
    employer appearing in this dataset's experiences (for classification backfill)."""
    rows = db.execute(
        select(Experience.company_id, Experience.company_name, Experience.company_linkedin_url)
        .join(Person, Person.id == Experience.person_id)
        .where(Person.dataset_id == dataset_id)
        .where(Experience.company_name.is_not(None))
        .distinct()
    ).all()
    return list(rows)


# ─────────────────── bulk fact loading (avoids N+1 over a candidate pool) ───────────────────


def bulk_experiences(db: Session, person_ids: list[str]) -> dict[str, list[Experience]]:
    return _bulk_by_person(db, Experience, person_ids, order_col=Experience.order_index)


def bulk_education(db: Session, person_ids: list[str]) -> dict[str, list[Education]]:
    return _bulk_by_person(db, Education, person_ids, order_col=Education.order_index)


def bulk_skills(db: Session, person_ids: list[str]) -> dict[str, list[Skill]]:
    return _bulk_by_person(db, Skill, person_ids)


def bulk_certifications(db: Session, person_ids: list[str]) -> dict[str, list[Certification]]:
    return _bulk_by_person(db, Certification, person_ids, order_col=Certification.order_index)


def bulk_languages(db: Session, person_ids: list[str]) -> dict[str, list[Language]]:
    return _bulk_by_person(db, Language, person_ids, order_col=Language.order_index)


def bulk_publications(db: Session, person_ids: list[str]) -> dict[str, list[Publication]]:
    return _bulk_by_person(db, Publication, person_ids, order_col=Publication.order_index)


def bulk_semantics(db: Session, person_ids: list[str]) -> dict[str, ProfileSemantic]:
    if not person_ids:
        return {}
    rows = db.scalars(select(ProfileSemantic).where(ProfileSemantic.person_id.in_(person_ids)))
    return {r.person_id: r for r in rows}


def bulk_embeddings_by_person(db: Session, person_ids: list[str]) -> dict[str, bytes]:
    if not person_ids:
        return {}
    rows = db.execute(
        select(ProfileEmbedding.person_id, ProfileEmbedding.vector).where(
            ProfileEmbedding.person_id.in_(person_ids)
        )
    ).all()
    return {pid: vec for pid, vec in rows}


def bulk_volunteering(db: Session, person_ids: list[str]) -> dict[str, list[Volunteering]]:
    return _bulk_by_person(db, Volunteering, person_ids, order_col=Volunteering.order_index)


def bulk_recommendations(db: Session, person_ids: list[str]) -> dict[str, list[Recommendation]]:
    return _bulk_by_person(db, Recommendation, person_ids, order_col=Recommendation.order_index)


def _bulk_by_person(db: Session, model, person_ids: list[str], *, order_col=None) -> dict[str, list]:
    if not person_ids:
        return {}
    stmt = select(model).where(model.person_id.in_(person_ids))
    if order_col is not None:
        stmt = stmt.order_by(model.person_id, order_col)
    out: dict[str, list] = {pid: [] for pid in person_ids}
    for row in db.scalars(stmt):
        out.setdefault(row.person_id, []).append(row)
    return out
