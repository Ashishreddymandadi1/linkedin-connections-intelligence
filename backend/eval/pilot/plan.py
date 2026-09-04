"""Live-pilot cost / call plan (V4 PART 10 §5, §15, §17).

Pure estimation from configuration + the offline dry-run funnels. No dollar
figures are invented — only call counts and configured routing.
"""
from __future__ import annotations

import math

from app.config import settings
from app.services.llm.providers import default_chain


def provider_routing() -> dict:
    chain = default_chain()
    return {
        "chain": [{"name": p.name, "model": getattr(p, "model", None)} for p in chain],
        "anthropic_configured": bool(settings.anthropic_api_key),
        "anthropic_first": bool(settings.anthropic_api_key) and (not chain or chain[0].name.startswith("anthropic")),
        "anthropic_model": settings.anthropic_model if settings.anthropic_api_key else None,
        "groq_configured": bool(settings.groq_api_key),
        "openrouter_configured": bool(settings.openrouter_api_key),
        "anthropic_would_incur_paid_usage": bool(settings.anthropic_api_key),
    }


def semantic_v3_plan(missing_ids: list[str], present_ids: list[str]) -> dict:
    """1 structured LLM request per profile that lacks the target semantic version."""
    n = len(missing_ids)
    return {
        "target_semantic_version": settings.semantic_profile_version,
        "pilot_profiles": n + len(present_ids),
        "already_at_target": len(present_ids),
        "needing_enrichment": n,
        "estimated_llm_requests": n,   # derive_semantics == 1 structured call each
        "max_tokens_per_request": 2200,
        "batching": "one profile per request (not batched)",
        "source_data": "normalized DB rows via build_compact_profile — NO Apify, NO re-scrape",
        "provider_routing": provider_routing(),
    }


def _judge_batches(viable: int | None) -> int:
    if not viable:
        return 0
    return math.ceil(viable / settings.semantic_judge_batch_size)


def _audit_batches(returned: int | None, exact: int | None, possible: int | None) -> int:
    pool = min(
        settings.top_connections + settings.final_result_audit_buffer,
        (exact or 0) + (possible or 0) if (exact is not None) else (returned or 0),
    )
    if pool <= 0:
        return 0
    return math.ceil(pool / settings.final_result_audit_batch_size)


def search_call_estimate(records: list[dict]) -> dict:
    """Per-query call estimate for a LIVE run, derived from the offline funnels."""
    per_query = []
    totals = {"interpretation": 0, "judge_batches": 0, "audit_batches": 0, "reason": 0}
    for rec in records:
        f = rec.get("funnel", {})
        viable = f.get("viable_candidate_count") or f.get("candidate_pool_size")
        jb = _judge_batches(viable)
        ab = _audit_batches(f.get("returned"), f.get("exact"), f.get("possible")) if settings.final_result_audit_enabled else 0
        reason = min(f.get("returned") or 0, settings.llm_reason_top_n) if settings.llm_reason_generation else 0
        interp = 1 if settings.llm_query_interpretation else 0
        per_query.append({
            "query_id": rec["query_id"],
            "interpretation_calls": interp,
            "viable_candidates_offline": viable,
            "expected_judge_batches": jb,
            "expected_audit_batches": ab,
            "expected_reason_calls": reason,
        })
        totals["interpretation"] += interp
        totals["judge_batches"] += jb
        totals["audit_batches"] += ab
        totals["reason"] += reason
    totals["all_llm_requests"] = sum(totals.values())
    return {"per_query": per_query, "totals": totals,
            "note": "offline funnels understate viable counts when the deterministic "
                    "parser produces fewer criteria than the LLM would; treat as a floor"}


def live_commands(dataset_id: str) -> list[str]:
    return [
        "# 1. (optional) semantic v3 backfill for the pilot sample only — STORED DATA ONLY:",
        "python -m eval.pilot.run_pilot enrich --live --i-understand-costs",
        "# 2. live evaluation run against the isolated pilot DB:",
        "python -m eval.pilot.run_pilot run --live --i-understand-costs",
        f"#    (both operate ONLY on eval/pilot/pilot.db built from {dataset_id}; production untouched)",
    ]
