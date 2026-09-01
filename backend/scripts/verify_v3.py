"""PHASE F — real verification of the 5 mandatory queries (spec §42).

Runs a BOUNDED semantic backfill first (Groq free-tier is rate limited, a full
987-profile backfill takes hours), then runs each mandatory query in-process and
prints the top people + their evidence so a human can eyeball correctness.

    python -m scripts.verify_v3 <dataset_id> [n_backfill]
"""
from __future__ import annotations

import sys

from app.database import SessionLocal
from app import repositories as repo
from app.models import EnrichmentState
from app.services.semantic_enrich import enrich_person_semantics
from app.services.enrichment_runner import _embedding_step, _mark_ready_or_partial

QUERIES = [
    "people who worked in tech",
    "Former Amazon people now at startups",
    "Who should I invite to a CXO networking event in Memphis or Nashville?",
    "people working in big tech in Bay Area",
    "senior engineering mentors in tech",
]


def bounded_backfill(dataset_id: str, n: int) -> None:
    db = SessionLocal()
    try:
        people = repo.people_missing_semantics(db, dataset_id, current_version=2)[:n]
        print(f"bounded backfill: {len(people)} profiles")
        for i, p in enumerate(people, 1):
            prev = p.enrichment_state
            p.enrichment_state = EnrichmentState.NORMALIZED
            try:
                enrich_person_semantics(db, p)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}] {p.full_name}: FAIL {e}")
                p.enrichment_state = prev
                db.commit()
                continue
            if p.enrichment_state == EnrichmentState.WAITING_FOR_FREE_LLM:
                print(f"  [{i}] {p.full_name}: rate-limited, stopping backfill")
                p.enrichment_state = prev
                db.commit()
                break
            _embedding_step(db, p)
            _mark_ready_or_partial(db, p)
            db.commit()
            print(f"  [{i}] {p.full_name}: ok")
    finally:
        db.close()


def run_queries(dataset_id: str) -> None:
    from app.services.search_service import run_connection_search

    db = SessionLocal()
    try:
        for q in QUERIES:
            print("\n" + "=" * 78)
            print(f"QUERY: {q}")
            print("=" * 78)
            resp = run_connection_search(db, dataset_id=dataset_id, query=q)
            db.commit()
            iq = resp.interpreted_query or {}
            crits = iq.get("criteria", []) if isinstance(iq, dict) else getattr(iq, "criteria", [])

            def _g(c, k):
                return c.get(k) if isinstance(c, dict) else getattr(c, k, None)

            print("PLAN:", [
                {"type": _g(c, "type"), "val": _g(c, "concept") or _g(c, "values") or _g(c, "value"),
                 "op": _g(c, "operator"), "scope": _g(c, "scope"),
                 "req": _g(c, "required"), "w": round(_g(c, "weight") or 0)}
                for c in crits
            ])
            print(f"provider={resp.llm_provider}  "
                  f"candidates={resp.connections.total_candidates}  returned={resp.connections.returned}")
            for it in resp.connections.results[:5]:
                print(f"\n  #{it.rank} {it.name}  —  {it.current_title} @ {it.current_company}  [{it.location}]")
                print(f"      score={it.match_score:.0f}  conf={it.data_confidence:.0f}")
                print(f"      reason: {it.reason}")
                for e in it.evidence[:4]:
                    print(f"        · [{e.type}] {e.text}")
    finally:
        db.close()


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "dataset_0ba27cae09d4"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    if n:
        bounded_backfill(ds, n)
    run_queries(ds)
