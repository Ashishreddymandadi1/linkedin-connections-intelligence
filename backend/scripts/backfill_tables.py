"""Backfill the v2 sub-section tables from stored raw profiles — no re-scrape.

Re-runs the pure ``normalize_profile()`` over each ``raw_profiles.raw_json`` for a
dataset and writes ONLY the extra tables (certifications / publications / patents /
languages / volunteering / recommendations). Everything else is left untouched.

    python -m scripts.backfill_tables <dataset_id>
"""
from __future__ import annotations

import sys

from app import repositories as repo
from app.database import SessionLocal, init_db
from app.services.normalize import normalize_profile


def backfill(dataset_id: str) -> None:
    init_db()  # create the new tables if they don't exist yet
    db = SessionLocal()
    try:
        people = repo.list_people(db, dataset_id, is_connection=None)
        if not people:
            print(f"no people for dataset {dataset_id}")
            return
        done = counts = 0
        totals = {k: 0 for k in repo.EXTRA_SECTION_MODELS}
        for p in people:
            raw = repo.latest_raw_profile(db, p.id)
            if not raw:
                continue
            normalized = normalize_profile(raw.raw_json)
            repo.replace_extra_sections(db, p.id, normalized)
            done += 1
            for k in totals:
                totals[k] += len(normalized.get(k, []) or [])
            counts += 1
            if counts % 25 == 0:
                db.commit()
        db.commit()
        print(f"backfilled {done} profiles in dataset {dataset_id}")
        for k, n in totals.items():
            print(f"  {k:16} {n}")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    backfill(sys.argv[1])
