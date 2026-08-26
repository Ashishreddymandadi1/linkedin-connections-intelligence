"""LLM semantic enrichment of one normalized profile (spec §26–§27).

M4 fills in the real Groq-chain call. For now this advances the state machine
and stores an empty semantic record so the pipeline is end-to-end runnable.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app import repositories as repo
from app.config import settings
from app.constants import EnrichmentState
from app.models import Person

log = logging.getLogger("app.semantic")

_ENABLED = True  # set False to skip semantic step entirely


def enrich_person_semantics(db: Session, person: Person) -> None:
    """Populate ``profile_semantics`` for ``person`` and move state forward."""
    try:
        from app.services.semantic_llm import derive_semantics  # optional, added in M4
    except ImportError:
        derive_semantics = None

    if not _ENABLED or derive_semantics is None:
        repo.upsert_semantic(
            db, person.id, {}, version=settings.semantic_profile_version, provider="none", model="none"
        )
        person.semantic_version = settings.semantic_profile_version
        person.enrichment_state = EnrichmentState.LLM_COMPLETE
        return

    result = derive_semantics(db, person)
    if result is None:
        # all free providers exhausted — do not fail the dataset (spec §25)
        person.enrichment_state = EnrichmentState.WAITING_FOR_FREE_LLM
        return

    data, provider, model = result
    repo.upsert_semantic(
        db, person.id, data, version=settings.semantic_profile_version, provider=provider, model=model
    )
    person.semantic_version = settings.semantic_profile_version
    person.enrichment_state = EnrichmentState.LLM_COMPLETE
