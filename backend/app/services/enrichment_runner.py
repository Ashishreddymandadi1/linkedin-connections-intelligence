"""Resumable enrichment worker (spec §54).

Per-person state machine, persisted after every step. If the process dies at
137/300, restarting resumes at 138 — profiles already past ``APIFY_COMPLETE``
reuse their stored raw JSON and are never re-scraped.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app import repositories as repo
from app.config import settings
from app.constants import DatasetStatus, EnrichmentState, JobStatus
from app.database import SessionLocal
from app.models import EnrichmentJob, Person
from app.models_base import utcnow
from app.services.apify_client import ApifyError, run_metadata, scrape_profiles
from app.services.completeness import compute_completeness
from app.services.normalize import normalize_profile
from app.services.raw_store import store_raw_profile
from app.services.url_normalize import extract_public_identifier

log = logging.getLogger("app.enrichment")

_NEEDS_APIFY = {EnrichmentState.PENDING, EnrichmentState.APIFY_RUNNING}


# ───────────────────────── public entrypoints ─────────────────────────


def start_enrichment(dataset_id: str, job_id: str) -> None:
    """Background task. Owns its own DB session."""
    db = SessionLocal()
    try:
        _run(db, dataset_id, job_id)
    except Exception:  # noqa: BLE001
        log.exception("enrichment run crashed for dataset %s", dataset_id)
        job = db.get(EnrichmentJob, job_id)
        if job and job.status == JobStatus.RUNNING:
            job.status = JobStatus.FAILED
            job.error = "worker crashed — see server logs"
            db.commit()
    finally:
        db.close()


def classify_companies(dataset_id: str, job_id: str | None = None) -> None:
    """Background task (spec §4/§37): classify every distinct employer in the
    dataset once, cached forever. No Apify — reads stored experiences."""
    db = SessionLocal()
    try:
        from app.services.company_intel import get_or_classify

        companies = repo.distinct_companies(db, dataset_id)
        log.info("company classification: %d distinct employers for dataset %s", len(companies), dataset_id)
        # get_or_classify batches internally and skips already-cached companies
        get_or_classify(db, companies)
        db.commit()
        job = db.get(EnrichmentJob, job_id) if job_id else None
        if job:
            job.status = JobStatus.COMPLETED
            job.completed_profiles = len(companies)
            job.completed_at = utcnow()
            db.commit()
    except Exception:  # noqa: BLE001
        log.exception("company classification crashed for %s", dataset_id)
    finally:
        db.close()


def backfill_semantics(dataset_id: str, job_id: str | None = None) -> None:
    """Background task: run/refresh the semantic pass for READY people that were
    deferred during a rate-limited run OR predate a semantic_profile_version
    bump (spec §37 — no Apify re-scrape). Re-embeds afterwards."""
    db = SessionLocal()
    try:
        from app.services.semantic_enrich import enrich_person_semantics

        ctx = _RunContext()
        people = repo.people_missing_semantics(
            db, dataset_id, current_version=settings.semantic_profile_version
        )
        log.info("semantic backfill: %d profiles for dataset %s", len(people), dataset_id)
        done = 0
        for p in people:
            if ctx.semantic_disabled:
                break
            prev_state = p.enrichment_state
            p.enrichment_state = EnrichmentState.NORMALIZED  # let the semantic step run
            enrich_person_semantics(db, p)
            if p.enrichment_state == EnrichmentState.WAITING_FOR_FREE_LLM:
                ctx.note_llm_failure()
                p.enrichment_state = prev_state  # leave as-is, still searchable
            else:
                ctx.note_llm_success()
                _embedding_step(db, p)
                _mark_ready_or_partial(db, p)
                done += 1
            db.commit()
        job = db.get(EnrichmentJob, job_id) if job_id else None
        if job:
            job.status = JobStatus.COMPLETED if not ctx.semantic_disabled else JobStatus.PARTIAL
            job.completed_profiles = done
            job.completed_at = utcnow()
            db.commit()
        log.info("semantic backfill done: %d completed, breaker=%s", done, ctx.semantic_disabled)
    except Exception:  # noqa: BLE001
        log.exception("semantic backfill crashed for %s", dataset_id)
    finally:
        db.close()


def refresh_single_person(db: Session, person_id: str, *, force: bool = False) -> dict:
    person = repo.get_person(db, person_id)
    if not person:
        return {"refreshed": False, "reason": "not found"}

    if not force and person.last_scraped_at:
        age_days = (utcnow() - _aware(person.last_scraped_at)).days
        if age_days < settings.profile_ttl_days:
            return {"refreshed": False, "reason": f"profile is fresh ({age_days}d < TTL {settings.profile_ttl_days}d)"}

    person.enrichment_state = EnrichmentState.PENDING
    person.apify_attempts = 0
    person.enrichment_error = None
    db.flush()
    _scrape_batch(db, person.dataset_id, None, [person])
    _drive_to_ready(db, person)
    return {"refreshed": True, "state": person.enrichment_state}


# ───────────────────────── run loop ─────────────────────────


def _run(db: Session, dataset_id: str, job_id: str) -> None:
    job = db.get(EnrichmentJob, job_id)
    ds = repo.get_dataset(db, dataset_id)
    if not job or not ds:
        return

    job.status = JobStatus.RUNNING
    job.started_at = job.started_at or utcnow()
    job.actor_id = run_metadata()["actor_id"]
    ds.status = DatasetStatus.ENRICHING
    db.commit()

    batch_size = max(1, settings.effective_batch_size)
    # Run-scoped semantic circuit breaker: after this many consecutive
    # free-LLM failures, stop attempting the semantic step for the rest of the
    # run (profiles still get scraped + normalized + embedded + marked READY;
    # their semantic_version stays NULL for a later backfill / resume).
    ctx = _RunContext()
    guard = 0
    while True:
        guard += 1
        if guard > 20_000:
            log.error("enrichment loop guard tripped for %s", dataset_id)
            break

        people = repo.people_needing_enrichment(db, dataset_id)
        if not people:
            break
        batch = people[:batch_size]

        need_apify = [p for p in batch if p.enrichment_state in _NEEDS_APIFY]
        if need_apify:
            _scrape_batch(db, dataset_id, job, need_apify)
            db.commit()

        progressed = False
        for p in batch:
            before = p.enrichment_state
            _drive_to_ready(db, p, ctx)
            db.commit()
            progressed = progressed or (p.enrichment_state != before)
            if p.enrichment_state in (EnrichmentState.READY, EnrichmentState.PARTIAL):
                job.completed_profiles += 1
            elif p.enrichment_state == EnrichmentState.FAILED:
                job.failed_profiles += 1
        db.commit()

        if not progressed:
            remaining = repo.enrichment_state_counts(db, dataset_id)
            waiting = remaining.get(EnrichmentState.WAITING_FOR_FREE_LLM, 0)
            log.warning(
                "no progress this iteration — stopping (waiting_for_llm=%d, semantic_breaker=%s)",
                waiting,
                ctx.semantic_disabled,
            )
            break

    _finalize(db, job, dataset_id)


class _RunContext:
    """Mutable state shared across one enrichment run."""

    _MAX_CONSECUTIVE_LLM_FAILS = 3

    def __init__(self) -> None:
        self.consecutive_llm_fails = 0
        self.semantic_disabled = False

    def note_llm_failure(self) -> None:
        self.consecutive_llm_fails += 1
        if self.consecutive_llm_fails >= self._MAX_CONSECUTIVE_LLM_FAILS:
            if not self.semantic_disabled:
                log.warning(
                    "free LLM unavailable %d times in a row — skipping the semantic step "
                    "for the rest of this run; profiles will still be scraped, normalized, "
                    "embedded and marked READY. Resume later to backfill semantics.",
                    self.consecutive_llm_fails,
                )
            self.semantic_disabled = True

    def note_llm_success(self) -> None:
        self.consecutive_llm_fails = 0


def _finalize(db: Session, job: EnrichmentJob, dataset_id: str) -> None:
    counts = repo.enrichment_state_counts(db, dataset_id)
    failed = counts.get(EnrichmentState.FAILED, 0)
    waiting = counts.get(EnrichmentState.WAITING_FOR_FREE_LLM, 0)
    pending = sum(
        counts.get(s, 0)
        for s in (EnrichmentState.PENDING, EnrichmentState.APIFY_COMPLETE, EnrichmentState.NORMALIZED, EnrichmentState.LLM_COMPLETE)
    )
    if waiting or pending:
        job.status = JobStatus.PARTIAL
    elif failed:
        job.status = JobStatus.PARTIAL if failed < job.requested_profiles else JobStatus.FAILED
    else:
        job.status = JobStatus.COMPLETED
    job.completed_at = utcnow()

    ds = repo.get_dataset(db, dataset_id)
    if ds:
        ds.status = DatasetStatus.READY
    db.commit()
    log.info("enrichment finalized dataset=%s status=%s counts=%s", dataset_id, job.status, counts)


# ───────────────────────── steps ─────────────────────────


def _scrape_batch(db: Session, dataset_id: str, job: EnrichmentJob | None, people: list[Person]) -> None:
    meta = run_metadata()
    by_key: dict[str, Person] = {}
    for p in people:
        p.enrichment_state = EnrichmentState.APIFY_RUNNING
        p.apify_attempts += 1
        slug = p.public_identifier or extract_public_identifier(p.linkedin_url)
        if slug:
            by_key[slug] = p
        by_key[p.linkedin_url.rstrip("/").lower()] = p
    db.flush()

    urls = [p.linkedin_url for p in people]
    hints = {
        (p.public_identifier or extract_public_identifier(p.linkedin_url) or ""): {
            "company": p.csv_company,
            "position": p.csv_position,
        }
        for p in people
    }

    try:
        items = scrape_profiles(urls, hints=hints)
    except ApifyError as e:
        log.warning("apify batch failed: %s", e)
        _fail_or_retry(db, people, str(e))
        if job:
            job.error = str(e)[:500]
        return

    matched: set[str] = set()
    for item in items:
        p = _match_person(item, by_key)
        if not p:
            continue
        matched.add(p.id)
        h = store_raw_profile(
            db,
            person_id=p.id,
            dataset_id=dataset_id,
            raw=item,
            actor_id=meta["actor_id"],
            apify_run_id=(job.apify_run_id if job else None),
        )
        p.raw_hash = h
        p.enrichment_state = EnrichmentState.APIFY_COMPLETE
        p.enrichment_error = None
    db.flush()

    missing = [p for p in people if p.id not in matched]
    if missing:
        _fail_or_retry(db, missing, "no profile returned by scraper")


def _match_person(item: dict, by_key: dict[str, Person]) -> Person | None:
    for key in (
        (item.get("publicIdentifier") or "").strip().lower(),
        (item.get("linkedinUrl") or "").rstrip("/").lower(),
        extract_public_identifier(item.get("linkedinUrl") or "") or "",
    ):
        if key and key in by_key:
            return by_key[key]
    return None


def _fail_or_retry(db: Session, people: list[Person], reason: str) -> None:
    for p in people:
        if p.apify_attempts >= max(1, settings.max_apify_retries):
            p.enrichment_state = EnrichmentState.FAILED
            p.enrichment_error = reason
        else:
            p.enrichment_state = EnrichmentState.PENDING
            p.enrichment_error = f"retrying: {reason}"
    db.flush()


def _drive_to_ready(db: Session, p: Person, ctx: "_RunContext | None" = None) -> None:
    ctx = ctx or _RunContext()
    if p.enrichment_state == EnrichmentState.APIFY_COMPLETE:
        _normalize_step(db, p)
        db.flush()

    # Embed as soon as we have normalized rows — independent of the LLM step, so a
    # rate-limited semantic pass never leaves a profile unsearchable.
    if p.enrichment_state in (EnrichmentState.NORMALIZED, EnrichmentState.WAITING_FOR_FREE_LLM):
        _embedding_step(db, p)

        if ctx.semantic_disabled:
            p.enrichment_state = EnrichmentState.LLM_COMPLETE  # skip; semantic_version stays NULL
        else:
            from app.services.semantic_enrich import enrich_person_semantics

            enrich_person_semantics(db, p)
            if p.enrichment_state == EnrichmentState.WAITING_FOR_FREE_LLM:
                ctx.note_llm_failure()
                if ctx.semantic_disabled:
                    p.enrichment_state = EnrichmentState.LLM_COMPLETE
            elif p.enrichment_state == EnrichmentState.LLM_COMPLETE:
                ctx.note_llm_success()
        db.flush()

    if p.enrichment_state == EnrichmentState.LLM_COMPLETE:
        _embedding_step(db, p)  # re-embed if semantics just landed (adds keywords)
        _mark_ready_or_partial(db, p)
        db.flush()


def _normalize_step(db: Session, p: Person) -> None:
    raw = repo.latest_raw_profile(db, p.id)
    if not raw:
        p.enrichment_state = EnrichmentState.FAILED
        p.enrichment_error = "normalize: no raw profile"
        return

    normalized = normalize_profile(raw.raw_json)
    person_fields = normalized["person"]

    for k, v in person_fields.items():
        if v is None:
            continue
        if k in {"public_identifier"} and getattr(p, k):
            continue
        setattr(p, k, v)
    if not p.full_name:
        p.full_name = " ".join(x for x in [p.first_name, p.last_name] if x) or None
    # keep CSV role as a fallback if the scrape had none
    if not p.current_company:
        p.current_company = p.csv_company
    if not p.current_title:
        p.current_title = p.csv_position

    repo.replace_experiences(db, p.id, normalized["experiences"])
    repo.replace_education(db, p.id, normalized["education"])
    repo.replace_skills(db, p.id, normalized["skills"])
    repo.replace_extra_sections(db, p.id, normalized)

    score, detail = compute_completeness(normalized, raw.raw_json)
    p.profile_completeness = score
    p.completeness_detail = detail
    p.last_scraped_at = utcnow()
    p.enrichment_state = EnrichmentState.NORMALIZED


def _embedding_step(db: Session, p: Person) -> None:
    try:
        from app.services.embeddings import embed_text
        from app.services.search_text import build_search_text

        text = build_search_text(db, p)
        blob = embed_text(text)
        repo.upsert_embedding(
            db, p.id, model=settings.embedding_model, dim=settings.embedding_dim, vector=blob, search_text=text
        )
    except Exception:  # noqa: BLE001
        log.exception("embedding step failed for %s (non-fatal)", p.id)


def _mark_ready_or_partial(db: Session, p: Person) -> None:
    exp = repo.get_experiences(db, p.id)
    edu = repo.get_education(db, p.id)
    skills = repo.get_skills(db, p.id)
    if not exp and not edu and not skills:
        p.enrichment_state = EnrichmentState.PARTIAL
        p.enrichment_error = p.enrichment_error or "scrape returned no experience, education or skills"
    else:
        p.enrichment_state = EnrichmentState.READY


def _aware(dt):
    from datetime import timezone

    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
