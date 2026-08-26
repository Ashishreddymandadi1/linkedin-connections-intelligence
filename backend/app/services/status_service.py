"""Build the dashboard/enrichment status report (spec §8, §56)."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app import repositories as repo
from app.constants import EnrichmentState
from app.models import Dataset
from app.schemas import DatasetStatusReport

_DONE_STATES = {EnrichmentState.READY, EnrichmentState.PARTIAL, EnrichmentState.FAILED}


def build_status_report(db: Session, ds: Dataset) -> DatasetStatusReport:
    counts = repo.enrichment_state_counts(db, ds.id)
    total = sum(counts.values())
    done = sum(n for s, n in counts.items() if s in _DONE_STATES)
    ready = counts.get(EnrichmentState.READY, 0)
    partial = counts.get(EnrichmentState.PARTIAL, 0)
    failed = counts.get(EnrichmentState.FAILED, 0)
    pending = counts.get(EnrichmentState.PENDING, 0)
    waiting = counts.get(EnrichmentState.WAITING_FOR_FREE_LLM, 0)

    job = repo.latest_job(db, ds.id)
    job_dict = None
    if job:
        job_dict = {
            "job_id": job.id,
            "status": job.status,
            "requested_profiles": job.requested_profiles,
            "completed_profiles": job.completed_profiles,
            "failed_profiles": job.failed_profiles,
            "apify_run_id": job.apify_run_id,
            "error": job.error,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }

    return DatasetStatusReport(
        dataset_id=ds.id,
        name=ds.name,
        status=ds.status,
        connections=ds.connection_count,
        counts=counts,
        ready=ready,
        partial=partial,
        failed=failed,
        pending=pending,
        waiting_for_llm=waiting,
        progress_done=done,
        progress_total=total,
        progress_pct=round(done / total * 100) if total else 0,
        last_updated=ds.updated_at,
        job=job_dict,
    )
