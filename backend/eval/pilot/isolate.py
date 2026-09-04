"""Copy the pilot sample into an ISOLATED SQLite database.

The production ``data/app.db`` is opened READ-ONLY. A fresh ``pilot.db`` is built
with the schema from ``Base.metadata`` and populated with:

* the one Dataset row
* the selected Person rows (same ids as production, so labels/reports line up)
* every normalized fact row for those people
* the CompanySemantic rows their experiences reference

Nothing is written back to production. ``pilot.db`` matches ``*.db`` in
.gitignore, so real profile data never reaches git.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import create_engine, insert, select
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models import (
    Certification,
    CompanySemantic,
    Connection,
    Dataset,
    Education,
    Experience,
    Language,
    Patent,
    Person,
    ProfileEmbedding,
    ProfileSemantic,
    Publication,
    RawProfile,
    Recommendation,
    Skill,
    Volunteering,
)
from app.services.company_intel import company_key

PILOT_DIR = Path(__file__).resolve().parent
DEFAULT_PILOT_DB = PILOT_DIR / "pilot.db"

# person-keyed fact tables copied verbatim for the sample
_PERSON_TABLES = [
    Experience, Education, Skill, Certification, Publication, Patent, Language,
    Volunteering, Recommendation, ProfileSemantic, ProfileEmbedding, RawProfile,
]


@dataclass
class IsolationReport:
    pilot_db: str
    dataset_id: str
    people: int
    row_counts: dict[str, int] = field(default_factory=dict)
    people_missing_semantic: list[str] = field(default_factory=list)
    people_missing_embedding: list[str] = field(default_factory=list)
    company_semantics_copied: int = 0


def _rows_as_dicts(src: Session, model, whereclause) -> list[dict]:
    cols = [c.name for c in model.__table__.columns]
    result = src.execute(select(model).where(whereclause)).scalars().all()
    return [{c: getattr(obj, c) for c in cols} for obj in result]


def build_pilot_db(
    prod_db_url: str,
    dataset_id: str,
    person_ids: list[str],
    *,
    pilot_db_path: Path | str = DEFAULT_PILOT_DB,
    overwrite: bool = True,
) -> IsolationReport:
    pilot_path = Path(pilot_db_path)
    if pilot_path.exists() and overwrite:
        pilot_path.unlink()
    pilot_path.parent.mkdir(parents=True, exist_ok=True)

    # production — READ ONLY
    ro_url = prod_db_url
    if ro_url.startswith("sqlite:///") and "mode=ro" not in ro_url:
        ro_url = ro_url.replace("sqlite:///", "sqlite:///file:", 1) + "?mode=ro&uri=true"
    src_engine = create_engine(ro_url)
    dst_engine = create_engine(f"sqlite:///{pilot_path.as_posix()}")
    Base.metadata.create_all(dst_engine)

    SrcS = sessionmaker(bind=src_engine, future=True)
    DstS = sessionmaker(bind=dst_engine, future=True)
    pid_set = set(person_ids)
    report = IsolationReport(pilot_db=str(pilot_path), dataset_id=dataset_id, people=len(pid_set))

    with SrcS() as src, DstS() as dst:
        # Dataset
        ds = src.get(Dataset, dataset_id)
        if ds is None:
            raise ValueError(f"dataset {dataset_id} not found in production DB")
        ds_cols = [c.name for c in Dataset.__table__.columns]
        dst.execute(insert(Dataset.__table__), [{c: getattr(ds, c) for c in ds_cols}])

        # People
        people_rows = _rows_as_dicts(src, Person, Person.id.in_(pid_set))
        dst.execute(insert(Person.__table__), people_rows)
        report.row_counts["people"] = len(people_rows)

        # Connection rows (person + dataset keyed)
        conn_rows = _rows_as_dicts(src, Connection, Connection.person_id.in_(pid_set))
        if conn_rows:
            dst.execute(insert(Connection.__table__), conn_rows)
        report.row_counts["connections"] = len(conn_rows)

        # person-keyed fact tables
        have_semantic: set[str] = set()
        have_embedding: set[str] = set()
        company_keys: set[str] = set()
        for model in _PERSON_TABLES:
            rows = _rows_as_dicts(src, model, model.person_id.in_(pid_set))
            if rows:
                dst.execute(insert(model.__table__), rows)
            report.row_counts[model.__tablename__] = len(rows)
            if model is ProfileSemantic:
                have_semantic = {r["person_id"] for r in rows}
            elif model is ProfileEmbedding:
                have_embedding = {r["person_id"] for r in rows}
            elif model is Experience:
                for r in rows:
                    if r.get("company_name"):
                        company_keys.add(company_key(r.get("company_id"), r["company_name"]))

        # CompanySemantic — only the keys the sample's experiences reference
        if company_keys:
            cs_rows = _rows_as_dicts(
                src, CompanySemantic, CompanySemantic.company_key.in_(company_keys)
            )
            if cs_rows:
                dst.execute(insert(CompanySemantic.__table__), cs_rows)
            report.company_semantics_copied = len(cs_rows)
            report.row_counts["company_semantics"] = len(cs_rows)

        dst.commit()

        report.people_missing_semantic = sorted(pid_set - have_semantic)
        report.people_missing_embedding = sorted(pid_set - have_embedding)

    src_engine.dispose()
    dst_engine.dispose()
    return report


def pilot_sessionmaker(pilot_db_path: Path | str = DEFAULT_PILOT_DB):
    engine = create_engine(f"sqlite:///{Path(pilot_db_path).as_posix()}", future=True)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True), engine
