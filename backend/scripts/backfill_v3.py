"""One-shot v3 backfill for an existing dataset — NO Apify re-scrape (spec §37).

    python -m scripts.backfill_v3 <dataset_id>

1. classify every distinct employer (company_semantics) — cached forever
2. (re)run the semantic pass for anyone missing it or predating the
   semantic_profile_version bump — adds semantic_assertions
3. re-embed (search_text now describes professional meaning + company category)

Safe to run repeatedly; already-done work is skipped.
"""
from __future__ import annotations

import sys

from app.config import settings
from app.database import SessionLocal
from app import repositories as repo
from app.services.enrichment_runner import backfill_semantics, classify_companies


def main(dataset_id: str) -> None:
    db = SessionLocal()
    try:
        ds = repo.get_dataset(db, dataset_id)
        if not ds:
            print(f"no dataset {dataset_id}")
            return
        n_companies = len(repo.distinct_companies(db, dataset_id))
        n_missing = len(repo.people_missing_semantics(db, dataset_id, current_version=settings.semantic_profile_version))
        print(f"{ds.name}: {n_companies} distinct employers, {n_missing} profiles need (re)semantic")
    finally:
        db.close()

    print("→ classifying companies …")
    classify_companies(dataset_id)
    print("→ semantic pass + re-embed …")
    backfill_semantics(dataset_id)
    print("done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
