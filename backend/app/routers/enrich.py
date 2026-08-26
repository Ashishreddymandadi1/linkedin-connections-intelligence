from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app import repositories as repo
from app.database import get_db

log = logging.getLogger("app.routers.enrich")
router = APIRouter(prefix="/datasets", tags=["enrichment"])


@router.post("/{dataset_id}/enrich")
def enrich_dataset(
    dataset_id: str,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict:
    ds = repo.get_dataset(db, dataset_id)
    if not ds:
        raise HTTPException(404, "dataset not found")

    pending = repo.people_needing_enrichment(db, dataset_id)
    if not pending:
        return {"started": False, "message": "all profiles already enriched", "pending": 0}

    from app.config import settings
    from app.services.enrichment_runner import start_enrichment

    job = repo.create_job(db, dataset_id, actor_id=settings.apify_actor_id, requested=len(pending))
    db.commit()

    background.add_task(start_enrichment, dataset_id, job.id)
    return {"started": True, "job_id": job.id, "pending": len(pending)}
