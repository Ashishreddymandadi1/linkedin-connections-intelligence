from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app import repositories as repo
from app.database import get_db
from app.schemas import DatasetStatusReport, DatasetSummary, UploadReport
from app.services.csv_parser import CSVParseError
from app.services.dataset_service import create_dataset_from_csv
from app.services.status_service import build_status_report

log = logging.getLogger("app.routers.datasets")
router = APIRouter(prefix="/datasets", tags=["datasets"])

_MAX_BYTES = 15 * 1024 * 1024


@router.post("", response_model=UploadReport, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> UploadReport:
    content = await file.read()
    if not content:
        raise HTTPException(400, "uploaded file is empty")
    if len(content) > _MAX_BYTES:
        raise HTTPException(413, "file too large (max 15 MB)")
    try:
        return create_dataset_from_csv(db, content=content, name=name or (file.filename or "").rsplit(".", 1)[0])
    except CSVParseError as e:
        raise HTTPException(422, f"could not parse CSV: {e}") from e


@router.get("", response_model=list[DatasetSummary])
def list_all(db: Session = Depends(get_db)) -> list[DatasetSummary]:
    return [
        DatasetSummary(
            dataset_id=d.id,
            name=d.name,
            connection_count=d.connection_count,
            status=d.status,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        for d in repo.list_datasets(db)
    ]


@router.get("/{dataset_id}", response_model=DatasetSummary)
def get_one(dataset_id: str, db: Session = Depends(get_db)) -> DatasetSummary:
    d = repo.get_dataset(db, dataset_id)
    if not d:
        raise HTTPException(404, "dataset not found")
    return DatasetSummary(
        dataset_id=d.id,
        name=d.name,
        connection_count=d.connection_count,
        status=d.status,
        created_at=d.created_at,
        updated_at=d.updated_at,
    )


@router.get("/{dataset_id}/status", response_model=DatasetStatusReport)
def dataset_status(dataset_id: str, db: Session = Depends(get_db)) -> DatasetStatusReport:
    d = repo.get_dataset(db, dataset_id)
    if not d:
        raise HTTPException(404, "dataset not found")
    return build_status_report(db, d)


@router.delete("/{dataset_id}", status_code=204)
def delete_one(dataset_id: str, db: Session = Depends(get_db)) -> None:
    ok = repo.delete_dataset(db, dataset_id)
    db.commit()
    if not ok:
        raise HTTPException(404, "dataset not found")
