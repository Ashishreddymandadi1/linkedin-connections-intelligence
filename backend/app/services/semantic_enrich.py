"""LLM semantic enrichment of one normalized profile (spec §26–§27).

Caching: a person is only re-run when the raw profile changed materially or the
``semantic_profile_version`` bumped (spec §66).
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app import repositories as repo
from app.config import settings
from app.constants import EnrichmentState
from app.models import Person

log = logging.getLogger("app.semantic")


def _store_empty(db: Session, person: Person, reason: str) -> None:
    repo.upsert_semantic(
        db, person.id, {}, version=settings.semantic_profile_version, provider="none", model=reason
    )
    person.semantic_version = settings.semantic_profile_version
    person.enrichment_state = EnrichmentState.LLM_COMPLETE


def enrich_person_semantics(db: Session, person: Person, *, force: bool = False) -> None:
    if not settings.semantic_enabled:
        _store_empty(db, person, "disabled")
        return

    existing = repo.get_semantic(db, person.id)
    if (
        not force
        and existing
        and existing.data
        and existing.version == settings.semantic_profile_version
        and person.semantic_version == settings.semantic_profile_version
    ):
        person.enrichment_state = EnrichmentState.LLM_COMPLETE
        return

    from app.services.semantic_llm import derive_semantics

    try:
        result = derive_semantics(db, person)
    except Exception:  # noqa: BLE001
        log.exception("semantic derivation crashed for %s — queuing", person.id)
        person.enrichment_state = EnrichmentState.WAITING_FOR_FREE_LLM
        return

    if result is None:
        # every free provider exhausted — do not fail the dataset (spec §25)
        person.enrichment_state = EnrichmentState.WAITING_FOR_FREE_LLM
        return

    data, provider, model = result
    repo.upsert_semantic(
        db, person.id, data, version=settings.semantic_profile_version, provider=provider, model=model
    )
    person.semantic_version = settings.semantic_profile_version
    person.enrichment_state = EnrichmentState.LLM_COMPLETE
    log.info("semantics for %s via %s", person.id, provider)
