"""Reset a dataset's people to PENDING and drop all derived data.

Keeps the dataset + people + connections rows (and the CSV fields); deletes
raw_profiles / experiences / education / skills / embeddings / semantics so a
fresh enrichment run starts clean. Use after a bad run (e.g. fixture data
scraped by mistake).

    python -m scripts.reset_dataset <dataset_id>
"""
from __future__ import annotations

import sys

from sqlalchemy import delete, select, update

from app.constants import EnrichmentState
from app.database import SessionLocal
from app.models import (
    Education,
    Experience,
    Person,
    ProfileEmbedding,
    ProfileSemantic,
    RawProfile,
    Skill,
)


def reset(dataset_id: str) -> None:
    db = SessionLocal()
    try:
        pids = list(db.scalars(select(Person.id).where(Person.dataset_id == dataset_id)))
        if not pids:
            print(f"no people for dataset {dataset_id}")
            return
        for model in (RawProfile, Experience, Education, Skill, ProfileEmbedding, ProfileSemantic):
            db.execute(delete(model).where(model.person_id.in_(pids)))
        db.execute(
            update(Person)
            .where(Person.dataset_id == dataset_id)
            .values(
                enrichment_state=EnrichmentState.PENDING,
                enrichment_error=None,
                apify_attempts=0,
                semantic_version=None,
                raw_hash=None,
                last_scraped_at=None,
                profile_completeness=0,
                completeness_detail=None,
            )
        )
        db.commit()
        print(f"reset {len(pids)} people in dataset {dataset_id} -> PENDING")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    reset(sys.argv[1])
