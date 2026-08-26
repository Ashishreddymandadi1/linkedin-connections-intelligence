"""Orchestrate a connection search (spec §33, §38–§41, §50–§51)."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app import repositories as repo
from app.config import settings
from app.constants import CriterionType
from app.models import SearchQuery
from app.schemas import (
    ConnectionBucket,
    EducationOut,
    ExternalBucket,
    ParsedSearchQuery,
    SearchResponse,
    SearchResultItem,
)
from app.services.candidate_pool import get_candidates
from app.services.person_view import education_to_out, experience_to_out, skill_to_out
from app.services.query_interpreter import interpret_query
from app.services.reason_generator import generate_reason
from app.services.scoring import ScoredCandidate, load_facts, score_candidate

log = logging.getLogger("app.search")


def run_connection_search(db: Session, *, dataset_id: str, query: str) -> SearchResponse:
    parsed, provider, model = interpret_query(query)
    log.info("query %r -> %d criteria via %s", query, len(parsed.criteria), provider)
    # Per-result LLM reasons only when the fast provider answered. If we already
    # had to fall back (or go deterministic), the primary model is congested —
    # use the evidence-based templates instead of N slow retry cycles.
    llm_available = provider in ("groq:primary", "openrouter:free")

    query_embedding = _maybe_embed(query)
    candidates, total = get_candidates(db, dataset_id, parsed, query_embedding)

    scored: list[ScoredCandidate] = []
    for person in candidates:
        facts = load_facts(db, person)
        result = score_candidate(facts, parsed)
        if result.excluded_reason:
            continue
        if result.match_score < settings.min_match_score:
            continue
        scored.append(result)

    scored.sort(key=lambda s: s.match_score, reverse=True)
    top = scored[: settings.top_connections]

    sq = repo.create_search_query(
        db,
        dataset_id=dataset_id,
        query_text=query,
        interpreted_query_json=parsed.model_dump(),
        llm_provider=provider,
        llm_model=model,
        total_candidates=len(scored),
    )

    results: list[SearchResultItem] = []
    for rank, cand in enumerate(top, start=1):
        item = _to_result_item(
            db, rank, cand, parsed, query,
            use_llm_reason=llm_available and rank <= settings.llm_reason_top_n,
        )
        results.append(item)
        repo.add_search_result(
            db,
            search_id=sq.id,
            person_id=cand.person.id,
            bucket="connection",
            rank=rank,
            match_score=item.match_score,
            data_confidence=item.data_confidence,
            reason=item.reason,
            payload=item.model_dump(),
        )

    return SearchResponse(
        search_id=sq.id,
        query=query,
        interpreted_query=parsed.model_dump(),
        connections=ConnectionBucket(total_candidates=len(scored), returned=len(results), results=results),
        external=ExternalBucket(searched=False),
        llm_provider=provider,
        llm_model=model,
    )


def load_search(db: Session, search_id: str) -> SearchResponse | None:
    sq: SearchQuery | None = repo.get_search_query(db, search_id)
    if not sq:
        return None
    rows = repo.get_search_results(db, search_id)
    conn_rows = [SearchResultItem(**r.payload) for r in rows if r.bucket == "connection"]
    return SearchResponse(
        search_id=sq.id,
        query=sq.query_text,
        interpreted_query=sq.interpreted_query_json or {},
        connections=ConnectionBucket(
            total_candidates=sq.total_candidates, returned=len(conn_rows), results=conn_rows
        ),
        external=ExternalBucket(searched=bool(sq.external_searched)),
        llm_provider=sq.llm_provider,
        llm_model=sq.llm_model,
    )


# ─────────────────────── helpers ───────────────────────


def _maybe_embed(query: str) -> bytes | None:
    try:
        from app.services.embeddings import embed_text

        return embed_text(query)
    except Exception:  # noqa: BLE001
        log.warning("query embedding failed — continuing without semantic prefilter", exc_info=False)
        return None


def _to_result_item(
    db: Session,
    rank: int,
    cand: ScoredCandidate,
    parsed: ParsedSearchQuery,
    query: str,
    *,
    use_llm_reason: bool = True,
) -> SearchResultItem:
    p = cand.person
    reason = generate_reason(cand, query, allow_llm=use_llm_reason)

    skill_terms = {
        c.value.lower() for c in parsed.criteria if c.type in (CriterionType.SKILL, CriterionType.DOMAIN)
    }
    company_terms = {
        c.value.lower()
        for c in parsed.criteria
        if c.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY)
    }
    edu_terms = {c.value.lower() for c in parsed.criteria if c.type == CriterionType.EDUCATION}

    exps = repo.get_experiences(db, p.id)
    rel_exp = [
        experience_to_out(e)
        for e in exps
        if e.is_current
        or any(t in (e.company_name or "").lower() or t in (e.position or "").lower() for t in company_terms)
    ][:5]

    rel_skills = [
        skill_to_out(s)
        for s in repo.get_skills(db, p.id)
        if not skill_terms or any(t in s.skill_name_norm or s.skill_name_norm in t for t in skill_terms)
    ][:12]

    edus = repo.get_education(db, p.id)
    rel_edu = [
        education_to_out(e)
        for e in edus
        if not edu_terms
        or any(t in (e.school_name or "").lower() or t in (e.field_of_study or "").lower() for t in edu_terms)
    ]
    if not rel_edu and edus:
        rel_edu = [education_to_out(edus[0])]

    return SearchResultItem(
        rank=rank,
        person_id=p.id,
        name=p.full_name,
        linkedin_url=p.linkedin_url,
        profile_picture_url=p.profile_picture_url,
        current_title=p.current_title,
        current_company=p.current_company,
        location=p.location_text,
        is_connection=True,
        match_score=cand.match_score,
        data_confidence=p.profile_completeness,
        reason=reason,
        matched_criteria=cand.matched_criteria,
        score_breakdown=cand.components,
        evidence=cand.evidence,
        relevant_experience=rel_exp or [experience_to_out(e) for e in exps[:2]],
        relevant_skills=rel_skills,
        relevant_education=rel_edu,
    )
