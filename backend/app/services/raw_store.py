"""Persist the verbatim Apify item before any transformation (spec §9)."""
from __future__ import annotations

import hashlib
import json

from sqlalchemy.orm import Session

from app import repositories as repo
from app.constants import RawSource


def raw_hash(raw: dict) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()


def store_raw_profile(
    db: Session,
    *,
    person_id: str,
    dataset_id: str,
    raw: dict,
    actor_id: str | None,
    apify_run_id: str | None = None,
    apify_dataset_id: str | None = None,
) -> str:
    h = raw_hash(raw)
    repo.add_raw_profile(
        db,
        person_id=person_id,
        dataset_id=dataset_id,
        source=RawSource.APIFY_HARVESTAPI,
        actor_id=actor_id,
        apify_run_id=apify_run_id,
        apify_dataset_id=apify_dataset_id,
        raw_hash=h,
        raw_json=raw,
    )
    return h
