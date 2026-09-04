"""Read-only inventory of the production database (V4 PART 10 §1).

Opens ``data/app.db`` with ``mode=ro``. Mutates nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import (
    Certification,
    CompanySemantic,
    Education,
    Experience,
    Language,
    Patent,
    Person,
    ProfileEmbedding,
    ProfileSemantic,
    Publication,
    Recommendation,
    Skill,
    Volunteering,
)

_COMPLETENESS_BANDS = [(0, 29), (30, 49), (50, 69), (70, 84), (85, 100)]
_PERSON_FACT_TABLES = {
    "experiences": Experience, "education": Education, "skills": Skill,
    "certifications": Certification, "publications": Publication, "languages": Language,
    "volunteering": Volunteering, "recommendations": Recommendation, "patents": Patent,
}


@dataclass
class DatasetInventory:
    dataset_id: str
    name: str
    connection_count: int
    people: int
    enrichment_states: dict[str, int]
    semantic_version_dist: dict[str, int]
    at_target_semantic_version: int
    missing_target_semantic_version: int
    target_semantic_version: int
    embeddings: int
    profile_semantics_rows: int
    company_semantics_referenced: int
    fact_counts: dict[str, int] = field(default_factory=dict)
    completeness_bands: dict[str, int] = field(default_factory=dict)


def _ro_url() -> str:
    url = settings.database_url
    if url.startswith("sqlite:///") and "mode=ro" not in url:
        return url.replace("sqlite:///", "sqlite:///file:", 1) + "?mode=ro&uri=true"
    return url


def inventory(dataset_id: str | None = None) -> dict:
    engine = create_engine(_ro_url())
    S = sessionmaker(bind=engine, future=True)
    out: dict = {"database_url": settings.database_url, "datasets": []}
    with S() as db:
        from app.models import Dataset

        datasets = db.execute(select(Dataset)).scalars().all()
        out["all_dataset_ids"] = [d.id for d in datasets]
        targets = [d for d in datasets if dataset_id is None or d.id == dataset_id]
        tgt_ver = settings.semantic_profile_version

        for d in targets:
            pids = [r for r in db.execute(select(Person.id).where(Person.dataset_id == d.id)).scalars()]
            pset = set(pids)

            states: dict[str, int] = {}
            for st, n in db.execute(
                select(Person.enrichment_state, func.count()).where(Person.dataset_id == d.id).group_by(Person.enrichment_state)
            ):
                states[st] = n

            sv: dict[str, int] = {}
            for ver, n in db.execute(
                select(Person.semantic_version, func.count()).where(Person.dataset_id == d.id).group_by(Person.semantic_version)
            ):
                sv[str(ver)] = n
            at_target = sv.get(str(tgt_ver), 0)

            fc: dict[str, int] = {}
            for label, model in _PERSON_FACT_TABLES.items():
                fc[label] = db.execute(
                    select(func.count()).select_from(model).where(model.person_id.in_(pset))
                ).scalar_one() if pset else 0

            emb = db.execute(
                select(func.count()).select_from(ProfileEmbedding).where(ProfileEmbedding.person_id.in_(pset))
            ).scalar_one() if pset else 0
            psem = db.execute(
                select(func.count()).select_from(ProfileSemantic).where(ProfileSemantic.person_id.in_(pset))
            ).scalar_one() if pset else 0

            company_keys = set(
                db.execute(
                    select(Experience.company_id, Experience.company_name).where(Experience.person_id.in_(pset))
                ).all()
            ) if pset else set()
            from app.services.company_intel import company_key
            ck = {company_key(cid, nm) for cid, nm in company_keys if nm}
            cs_ref = db.execute(
                select(func.count()).select_from(CompanySemantic).where(CompanySemantic.company_key.in_(ck))
            ).scalar_one() if ck else 0

            bands: dict[str, int] = {}
            for lo, hi in _COMPLETENESS_BANDS:
                bands[f"{lo}-{hi}"] = db.execute(
                    select(func.count()).select_from(Person)
                    .where(Person.dataset_id == d.id)
                    .where(Person.profile_completeness >= lo)
                    .where(Person.profile_completeness <= hi)
                ).scalar_one()

            inv = DatasetInventory(
                dataset_id=d.id, name=d.name, connection_count=d.connection_count,
                people=len(pids), enrichment_states=states, semantic_version_dist=sv,
                at_target_semantic_version=at_target,
                missing_target_semantic_version=len(pids) - at_target,
                target_semantic_version=tgt_ver,
                embeddings=emb, profile_semantics_rows=psem,
                company_semantics_referenced=cs_ref, fact_counts=fc, completeness_bands=bands,
            )
            out["datasets"].append(inv.__dict__)

        out["company_semantics_total"] = db.execute(select(func.count()).select_from(CompanySemantic)).scalar_one()
    engine.dispose()
    return out
