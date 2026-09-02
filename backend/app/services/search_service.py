"""Orchestrate a connection search (spec §33, §38–§41, §50–§51)."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app import repositories as repo
from app.config import settings
from app.constants import _QUALIFICATION_RANK, CriterionType, Qualification
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
from app.services.matching import company_matches, norm_company
from app.services.person_view import education_to_out, experience_to_out, skill_to_out
from app.services.query_interpreter import interpret_query
from app.services.reason_generator import generate_reason
from app.services.scoring import ScoredCandidate, ScoringContext, load_facts, score_candidate

log = logging.getLogger("app.search")


def _tier_key(s):
    """Rank by qualification tier FIRST, then match score (V4 §25)."""
    return (_QUALIFICATION_RANK.get(s.qualification, 1), -s.match_score)


def run_connection_search(db: Session, *, dataset_id: str, query: str) -> SearchResponse:
    parsed, provider, model = interpret_query(query)
    log.info("query %r -> %d criteria via %s", query, len(parsed.criteria), provider)
    # any real LLM answered (not the deterministic fallback) -> downstream LLM
    # steps (reasons, judge, rerank) are worth attempting (spec §29)
    llm_available = provider != "deterministic"

    query_embedding = _maybe_embed(query)
    candidates, total = get_candidates(db, dataset_id, parsed, query_embedding)
    pids = [p.id for p in candidates]

    # bulk-load every fact once for the whole pool (spec §31 — no per-candidate N+1)
    facts_cache = {
        "experiences": repo.bulk_experiences(db, pids),
        "education": repo.bulk_education(db, pids),
        "skills": repo.bulk_skills(db, pids),
        "certifications": repo.bulk_certifications(db, pids),
        "languages": repo.bulk_languages(db, pids),
        "publications": repo.bulk_publications(db, pids),
        "semantics": repo.bulk_semantics(db, pids),
        "embeddings": repo.bulk_embeddings_by_person(db, pids),
    }

    ctx = ScoringContext(
        query_embedding=query_embedding,
        company_ids_by_criterion=_resolve_company_ids(db, dataset_id, parsed),
        company_class=_pool_company_class(db, parsed, facts_cache["experiences"]),
    )

    facts_by_id: dict = {p.id: load_facts(db, p, facts_cache) for p in candidates}

    # pass 1 — deterministic facts + assertion/company-class concepts + embedding relevance
    scored: list[ScoredCandidate] = []
    near_pool: list[ScoredCandidate] = []
    for person in candidates:
        result = score_candidate(facts_by_id[person.id], parsed, ctx)
        if result.qualification == Qualification.NOT_MATCH:
            # a candidate that misses exactly ONE required criterion is a near-match
            if len(result.unmet_required) == 1:
                near_pool.append(result)
            continue
        if result.match_score < settings.min_match_score:
            continue
        scored.append(result)
    scored.sort(key=_tier_key)

    # semantic judge — bounded, batched LLM pass for ambiguous concept criteria
    scored = _maybe_judge(db, query, parsed, scored, facts_by_id, candidates, ctx, llm_available)
    scored.sort(key=_tier_key)

    # rerank pool — cross-encoder over the top RERANK_POOL, then re-score
    pool = scored[: settings.rerank_pool]
    if settings.reranker_enabled and pool:
        from app.services.reranker import cross_encode

        texts = [_candidate_text(db, c, facts_by_id) for c in pool]
        for c, ce in zip(pool, cross_encode(query, texts)):
            ctx.reranker_scores[c.person.id] = ce
        rescored = [score_candidate(facts_by_id[c.person.id], parsed, ctx) for c in pool]
        rescored = [r for r in rescored
                    if r.qualification != Qualification.NOT_MATCH and r.match_score >= settings.min_match_score]
        rescored.sort(key=_tier_key)
        scored = rescored + scored[settings.rerank_pool :]
        scored.sort(key=_tier_key)  # cross-encoder must NOT reorder across tiers (V4 §25)

    total_scored = len(scored)
    exact_n = sum(1 for s in scored if s.qualification == Qualification.EXACT_MATCH)
    possible_n = sum(1 for s in scored if s.qualification == Qualification.POSSIBLE_MATCH)
    top = scored[: settings.top_connections]
    top = _maybe_llm_rerank(db, query, top, facts_by_id, llm_available)

    sq = repo.create_search_query(
        db,
        dataset_id=dataset_id,
        query_text=query,
        interpreted_query_json=parsed.model_dump(),
        llm_provider=provider,
        llm_model=model,
        total_candidates=total_scored,
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

    near_pool.sort(key=lambda s: s.match_score, reverse=True)
    near_items = [
        _to_result_item(db, i, cand, parsed, query, use_llm_reason=False)
        for i, cand in enumerate(near_pool[:5], start=1)
    ]

    return SearchResponse(
        search_id=sq.id,
        query=query,
        interpreted_query=parsed.model_dump(),
        connections=ConnectionBucket(
            total_candidates=total_scored, returned=len(results), results=results,
            exact_match_count=exact_n, possible_match_count=possible_n, near_matches=near_items,
        ),
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


def _resolve_company_ids(db: Session, dataset_id: str, parsed: ParsedSearchQuery) -> dict[str, set[str]]:
    """Map each company criterion to the LinkedIn company_ids its value(s)
    resolve to within this dataset (fuzzy name match on the index keys)."""
    company_crits = [
        c for c in parsed.criteria
        if c.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY)
    ]
    if not company_crits or not settings.company_id_matching:
        return {}
    index = repo.company_name_index(db, dataset_id)
    out: dict[str, set[str]] = {}
    for c in company_crits:
        ids: set[str] = set()
        for value in (c.values or [c.value]):
            target = norm_company(value)
            for name_key, cids in index.items():
                if name_key == target or company_matches(name_key, value):
                    ids |= cids
        if ids:
            out[c.id] = ids
    return out


def _pool_company_class(db: Session, parsed: ParsedSearchQuery, exp_by_person: dict) -> dict:
    """Classify (once, cached) the distinct employers that appear in the
    candidate pool — only when the query actually asks about company category /
    industry, and bounded to the pool's companies. Every later search reuses the
    cache (spec §4/§36 — batched, cached, never per-search-per-company)."""
    wants_company_semantics = any(
        c.type in (CriterionType.COMPANY_CATEGORY, CriterionType.SEMANTIC_CONCEPT) for c in parsed.criteria
    )
    seen: dict[tuple, tuple] = {}
    for exps in exp_by_person.values():
        for e in exps:
            if e.company_name:
                seen.setdefault((e.company_id, e.company_name), (e.company_id, e.company_name, e.company_linkedin_url))
    companies = list(seen.values())
    if not companies:
        return {}
    if not wants_company_semantics or not settings.company_classification_enabled:
        # cache-only — don't spend LLM calls if the query didn't ask
        from app.services.company_intel import company_key

        keys = [company_key(cid, nm) for cid, nm, _ in companies]
        rows = repo.get_company_semantics(db, keys)
        from app.services.company_intel import to_dict

        return {k: to_dict(r) for k, r in rows.items()}
    from app.services.company_intel import get_or_classify

    result = get_or_classify(db, companies)
    db.commit()
    return result


def _maybe_judge(db, query, parsed, scored, facts_by_id, candidates, ctx, llm_available):
    """Batched LLM judge for ambiguous semantic-concept criteria (spec §16-18)."""
    if not settings.semantic_judge_enabled or not llm_available:
        return scored
    sem_crits = [c for c in parsed.criteria if c.type in (CriterionType.SEMANTIC_CONCEPT, CriterionType.COMPANY_CATEGORY)]
    if not sem_crits:
        return scored

    lo, hi = settings.semantic_judge_low, settings.semantic_judge_high
    by_person = {c.person.id: c for c in scored}
    ambiguous_ids = set()
    for cand in scored[: settings.semantic_judge_pool]:
        for comp in cand.components:
            if comp.type in ("semantic_concept", "company_category") and lo <= comp.match_strength <= hi:
                ambiguous_ids.add(cand.person.id)
    if not ambiguous_ids:
        return scored

    from app.services.semantic_judge import judge

    to_judge = [
        (next(p for p in candidates if p.id == pid), facts_by_id[pid])
        for pid in ambiguous_ids
        if pid in facts_by_id
    ][: settings.semantic_judge_pool]
    verdicts = judge(db, query, sem_crits, to_judge, ctx)
    if not verdicts:
        return scored

    ctx.judge_results.update(verdicts)
    rescored = []
    for cand in scored:
        if cand.person.id in verdicts:
            r = score_candidate(facts_by_id[cand.person.id], parsed, ctx)
            if r.excluded_reason or r.match_score < settings.min_match_score:
                continue
            rescored.append(r)
        else:
            rescored.append(cand)
    return rescored


def _candidate_text(db: Session, cand: ScoredCandidate, facts_by_id: dict) -> str:
    """Compact text for the cross-encoder — prefer the stored embedding text."""
    from app.models import ProfileEmbedding

    row = db.query(ProfileEmbedding.search_text).filter(
        ProfileEmbedding.person_id == cand.person.id
    ).first()
    if row and row[0]:
        return row[0][:1200]
    p = cand.person
    return " · ".join(
        filter(None, [p.full_name, p.headline, p.current_title, p.current_company, p.location_text])
    )


def _maybe_llm_rerank(db: Session, query: str, top: list, facts_by_id: dict, llm_available: bool) -> list:
    if not settings.llm_rerank_enabled or not llm_available or len(top) < 3:
        return top
    from app.services.reranker import llm_rerank

    cands = []
    for c in top:
        p = c.person
        line = " · ".join(filter(None, [
            p.full_name, p.current_title, p.current_company,
            "matched: " + ", ".join(c.matched_criteria) if c.matched_criteria else None,
        ]))
        cands.append({"person_id": p.id, "line": line[:220]})
    res = llm_rerank(query, cands)
    if not res:
        return top
    by_id = {c.person.id: c for c in top}
    reordered = [by_id[pid] for pid in res["order"] if pid in by_id and pid not in res["drop"]]
    # append any the LLM omitted, preserving their prior order
    seen = {c.person.id for c in reordered}
    reordered += [c for c in top if c.person.id not in seen and c.person.id not in res["drop"]]
    return reordered


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

    def _terms(*types: str) -> set[str]:
        return {
            v.lower()
            for c in parsed.criteria if c.type in types
            for v in (c.values or [c.value] or [c.concept or ""]) if v
        }

    skill_terms = _terms(CriterionType.SKILL, CriterionType.DOMAIN, CriterionType.SEMANTIC_CONCEPT)
    company_terms = _terms(
        CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY, CriterionType.COMPANY_CATEGORY
    )
    edu_terms = _terms(CriterionType.EDUCATION)

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
        qualification=cand.qualification,
        uncertain_criteria=cand.uncertain_required,
        unmet_criteria=cand.unmet_required,
        matched_criteria=cand.matched_criteria,
        score_breakdown=cand.components,
        evidence=cand.evidence,
        relevant_experience=rel_exp or [experience_to_out(e) for e in exps[:2]],
        relevant_skills=rel_skills,
        relevant_education=rel_edu,
    )
