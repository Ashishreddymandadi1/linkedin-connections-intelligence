"""PART O — real-data verification of the V4 semantic core (B.5 + C + D + E).

Deterministic path (no LLM spend): every plan comes from the deterministic fact
layer + merge + validator. Prints, per query: plan, interpretation summary +
confidence, connections evaluated, EXACT / POSSIBLE counts, top results with the
qualification tier, and the near-matches with their failed required criterion.

    python -m scripts.verify_v4 [dataset_id]
"""
from __future__ import annotations

import sys

from app.database import SessionLocal
from app.services.search_service import run_connection_search

QUERIES = [
    "people who worked in tech",
    "people who worked at tech companies",
    "people with technical backgrounds",
    "Former Amazon people now at startups",
    "Who should I invite to a CXO networking event in Memphis or Nashville?",
    "people working in big tech in Bay Area",
    "people who moved from consulting to tech",
    "software engineers in financial services",
]


def main(dataset_id: str) -> None:
    db = SessionLocal()
    try:
        for q in QUERIES:
            print("\n" + "=" * 78 + f"\nQUERY: {q}\n" + "=" * 78)
            r = run_connection_search(db, dataset_id=dataset_id, query=q)
            db.commit()
            iq = r.interpreted_query
            crits = iq.get("criteria", [])
            print("PLAN:", [
                {"type": c["type"], "val": c.get("concept") or c.get("values") or c.get("value"),
                 "op": c.get("operator"), "scope": c.get("scope"), "req": c.get("required")}
                for c in crits
            ])
            print("summary   :", iq.get("interpretation_summary", "")[:200])
            print("confidence:", iq.get("interpretation_confidence"))
            print(f"provider={r.llm_provider}  evaluated={r.connections.total_candidates}  "
                  f"EXACT={r.connections.exact_match_count}  POSSIBLE={r.connections.possible_match_count}  "
                  f"returned={r.connections.returned}")
            for it in r.connections.results[:5]:
                print(f"  #{it.rank} [{it.qualification}] {it.name} - {it.current_title} @ {it.current_company}"
                      f" [{it.location}]  score={it.match_score:.0f}")
                if it.uncertain_criteria:
                    print(f"       uncertain: {it.uncertain_criteria}")
                for e in it.evidence[:2]:
                    print(f"       . [{e.type}] {e.text[:120]}")
            for nm in r.connections.near_matches[:3]:
                print(f"  ~near {nm.name} - {nm.current_title} @ {nm.current_company} [{nm.location}]"
                      f"  FAILS: {nm.unmet_criteria}")
    finally:
        db.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset_0ba27cae09d4")
