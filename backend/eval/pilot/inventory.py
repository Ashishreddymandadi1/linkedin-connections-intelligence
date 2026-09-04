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


def pilot_semantic_status(pilot_db_path: str) -> dict:
    """DB-only (NO network) semantic-enrichment status of the isolated pilot DB.

    Safe to run after a future live ``enrich`` to see what actually landed
    (V4 PART 10.1 §10).
    """
    from pathlib import Path

    from app.constants import EnrichmentState
    from app.models import Dataset, Person, ProfileSemantic

    p = Path(pilot_db_path)
    if not p.exists():
        return {"error": f"pilot DB not found: {p}"}

    engine = create_engine(f"sqlite:///{p.as_posix()}", future=True)
    S = sessionmaker(bind=engine, future=True)
    tgt = settings.semantic_profile_version
    with S() as db:
        ds_id = db.execute(select(Dataset.id)).scalar_one_or_none()
        people = db.execute(select(Person)).scalars().all()
        pids = [x.id for x in people]
        sems = db.execute(
            select(ProfileSemantic).where(ProfileSemantic.person_id.in_(pids))
        ).scalars().all() if pids else []

        by_person = {s.person_id: s for s in sems}
        at_target = [x for x in people if x.semantic_version == tgt]
        missing = [x for x in people if x.semantic_version != tgt]

        provider_counts: dict[str, int] = {}
        model_counts: dict[str, int] = {}
        for s in sems:
            if s.version == tgt:
                provider_counts[s.llm_provider or "unknown"] = provider_counts.get(s.llm_provider or "unknown", 0) + 1
                model_counts[s.llm_model or "unknown"] = model_counts.get(s.llm_model or "unknown", 0) + 1

        failures = [
            {"person_id": x.id, "state": x.enrichment_state,
             "error": (x.enrichment_error or "")[:200]}
            for x in people
            if x.enrichment_state in (EnrichmentState.FAILED, EnrichmentState.WAITING_FOR_FREE_LLM)
            or x.enrichment_error
        ]
        stale = [x.id for x in people
                 if x.id in by_person and by_person[x.id].version != tgt]
        never_enriched = [x.id for x in missing if x.id not in by_person]
    engine.dispose()

    return {
        "network": "none — DB inspection only",
        "pilot_db": str(p),
        "dataset_id": ds_id,
        "pilot_profiles": len(people),
        "target_semantic_version": tgt,
        "at_target_semantic_version": len(at_target),
        "missing_target_semantic_version": len(missing),
        "semantic_rows_present": len(sems),
        "stale_semantic_version": len(stale),        # has a row, but older version
        "never_semantically_enriched": len(never_enriched),
        "provider_counts_at_target": provider_counts,
        "model_counts_at_target": model_counts,
        "failures": failures,
    }
