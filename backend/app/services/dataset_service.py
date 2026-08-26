"""Upload orchestration (spec §4–§6): parse CSV → normalize URLs → dedupe →
create dataset + people + connection rows.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app import repositories as repo
from app.constants import DatasetStatus, EnrichmentState
from app.schemas import DatasetSummary, UploadReport
from app.services.csv_parser import parse_connections_csv
from app.services.dedup import dedupe_rows

log = logging.getLogger("app.dataset")


def _summary(ds) -> DatasetSummary:
    return DatasetSummary(
        dataset_id=ds.id,
        name=ds.name,
        connection_count=ds.connection_count,
        status=ds.status,
        created_at=ds.created_at,
        updated_at=ds.updated_at,
    )


def create_dataset_from_csv(db: Session, *, content: bytes, name: str | None) -> UploadReport:
    parsed = parse_connections_csv(content)
    usable = parsed.usable
    dd = dedupe_rows(usable)

    ds = repo.create_dataset(db, name or "My LinkedIn Network")
    log.info("dataset %s created: %d rows, %d usable, %d unique", ds.id, parsed.total_data_rows, len(usable), len(dd.unique))

    for row in dd.unique:
        full_name = " ".join(p for p in [row.first_name, row.last_name] if p) or None
        person = repo.add_person(
            db,
            dataset_id=ds.id,
            is_connection=True,
            linkedin_url=row.linkedin_url,
            public_identifier=row.public_identifier,
            first_name=row.first_name,
            last_name=row.last_name,
            full_name=full_name,
            csv_company=row.company,
            csv_position=row.position,
            current_company=row.company,
            current_title=row.position,
            connected_on=row.connected_on,
            enrichment_state=EnrichmentState.PENDING,
        )
        db.add(
            _connection_row(ds.id, person.id, row)
        )

    ds.connection_count = len(dd.unique)
    ds.status = DatasetStatus.READY_FOR_ENRICHMENT
    db.flush()
    db.commit()

    return UploadReport(
        dataset=_summary(ds),
        total_rows=parsed.total_data_rows,
        imported=len(dd.unique),
        duplicates_removed=len(dd.duplicates),
        skipped_no_url=len(parsed.skipped),
        skipped=parsed.skipped[:50],
        duplicates=dd.duplicates[:50],
    )


def _connection_row(dataset_id: str, person_id: str, row):
    from app.models import Connection

    return Connection(
        dataset_id=dataset_id,
        person_id=person_id,
        csv_first_name=row.first_name,
        csv_last_name=row.last_name,
        csv_email=row.email,
        csv_company=row.company,
        csv_position=row.position,
        connected_on=row.connected_on,
    )
