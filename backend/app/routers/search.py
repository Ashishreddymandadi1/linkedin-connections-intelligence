from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import repositories as repo
from app.database import get_db
from app.schemas import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def run_search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    if not repo.get_dataset(db, payload.dataset_id):
        raise HTTPException(404, "dataset not found")
    from app.services.search_service import run_connection_search

    resp = run_connection_search(db, dataset_id=payload.dataset_id, query=payload.query)
    db.commit()
    return resp


@router.get("/search/{search_id}", response_model=SearchResponse)
def get_search(search_id: str, db: Session = Depends(get_db)) -> SearchResponse:
    from app.services.search_service import load_search

    resp = load_search(db, search_id)
    if not resp:
        raise HTTPException(404, "search not found")
    return resp
