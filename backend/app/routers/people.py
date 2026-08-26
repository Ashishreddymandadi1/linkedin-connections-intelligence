from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import repositories as repo
from app.database import get_db
from app.schemas import PersonListItem, PersonOut
from app.services.person_view import person_to_out

router = APIRouter(tags=["people"])


@router.get("/datasets/{dataset_id}/people", response_model=list[PersonListItem])
def list_people(
    dataset_id: str,
    connections_only: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> list[PersonListItem]:
    if not repo.get_dataset(db, dataset_id):
        raise HTTPException(404, "dataset not found")
    people = repo.list_people(db, dataset_id, is_connection=True if connections_only else None)
    return [
        PersonListItem(
            person_id=p.id,
            full_name=p.full_name,
            headline=p.headline,
            current_title=p.current_title,
            current_company=p.current_company,
            linkedin_url=p.linkedin_url,
            profile_completeness=p.profile_completeness,
            enrichment_state=p.enrichment_state,
        )
        for p in people
    ]


@router.get("/people/{person_id}", response_model=PersonOut)
def get_person(person_id: str, db: Session = Depends(get_db)) -> PersonOut:
    p = repo.get_person(db, person_id)
    if not p:
        raise HTTPException(404, "person not found")
    return person_to_out(db, p)


@router.post("/people/{person_id}/refresh")
def refresh_person(
    person_id: str,
    force: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> dict:
    from app.services.enrichment_runner import refresh_single_person

    p = repo.get_person(db, person_id)
    if not p:
        raise HTTPException(404, "person not found")
    result = refresh_single_person(db, person_id, force=force)
    db.commit()
    return result
