"""Search-quality evaluation harness.

    python -m eval.run_eval                       # run, print table
    python -m eval.run_eval --baseline            # run + save as the baseline
    python -m eval.run_eval --compare eval/results/<file>.json   # run + show deltas

Metrics per query: precision@5, precision@10, recall@20, MRR (rank of first hit),
plus mean match_score of relevant vs non-relevant results. Aggregates at the end,
split by query `kind` (precision / recall / new).

Runs each query in-process through ``run_connection_search`` against the DB named
in ``queries.json``. Requires that dataset to be enriched.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Windows consoles default to cp1252 and choke on unicode in query text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif isinstance(sys.stdout, io.TextIOWrapper):  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_EVAL_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _EVAL_DIR / "results"


def _load_queries() -> dict:
    return json.loads((_EVAL_DIR / "queries.json").read_text(encoding="utf-8"))


def _resolve_relevant(db, dataset_id: str, public_ids: list[str]) -> set[str]:
    from app import repositories as repo

    out: set[str] = set()
    for pid in public_ids:
        p = repo.find_person_by_public_id(db, dataset_id, pid)
        if p:
            out.add(p.id)
    return out


def _metrics(ranked_ids: list[str], relevant: set[str], scores: dict[str, float]) -> dict:
    if not relevant:
        return {"skipped": True}
    hits = [1 if pid in relevant else 0 for pid in ranked_ids]
    p5 = sum(hits[:5]) / 5
    p10 = sum(hits[:10]) / 10
    recall20 = sum(hits[:20]) / min(len(relevant), 20)
    mrr = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            mrr = 1 / i
            break
    rel_scores = [scores[p] for p in ranked_ids if p in relevant]
    non_scores = [scores[p] for p in ranked_ids if p not in relevant]
    return {
        "skipped": False,
        "p@5": round(p5, 3),
        "p@10": round(p10, 3),
        "recall@20": round(recall20, 3),
        "mrr": round(mrr, 3),
        "found": sum(hits),
        "n_relevant": len(relevant),
        "mean_rel_score": round(statistics.mean(rel_scores), 1) if rel_scores else None,
        "mean_non_score": round(statistics.mean(non_scores), 1) if non_scores else None,
    }


def run() -> dict:
    spec = _load_queries()
    dataset_id = spec["dataset_id"]

    from app.config import settings
    from app.database import SessionLocal
    from app.services.search_service import run_connection_search

    # Reasons are cosmetic and don't affect ranking — skip the LLM calls for them
    # so the eval is faster and less rate-limit-flaky. Query interpretation stays on.
    settings.llm_reason_generation = False

    db = SessionLocal()
    per_query: list[dict] = []
    try:
        for q in spec["queries"]:
            t0 = time.time()
            resp = run_connection_search(db, dataset_id=dataset_id, query=q["query"])
            db.rollback()  # don't persist eval searches
            elapsed = time.time() - t0

            ranked = [r.person_id for r in resp.connections.results]
            scores = {r.person_id: r.match_score for r in resp.connections.results}
            relevant = _resolve_relevant(db, dataset_id, q.get("relevant", []))
            m = _metrics(ranked, relevant, scores)
            m.update(query=q["query"], kind=q.get("kind", "?"), returned=len(ranked), seconds=round(elapsed, 2))
            per_query.append(m)
            _print_row(m)
    finally:
        db.close()

    agg = _aggregate(per_query)
    _print_aggregate(agg)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "per_query": per_query,
        "aggregate": agg,
    }


def _aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if not r.get("skipped")]
    by_kind: dict[str, list[dict]] = {}
    for r in scored:
        by_kind.setdefault(r["kind"], []).append(r)

    def avg(items, key):
        vals = [i[key] for i in items if i.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    out = {"overall": {k: avg(scored, k) for k in ("p@5", "p@10", "recall@20", "mrr")}}
    out["overall"]["n_queries"] = len(scored)
    for kind, items in by_kind.items():
        out[kind] = {k: avg(items, k) for k in ("p@5", "p@10", "recall@20", "mrr")}
        out[kind]["n_queries"] = len(items)
    return out


def _print_row(m: dict) -> None:
    if m.get("skipped"):
        print(f"  -  {m['query'][:60]:60}  (no labels)")
        return
    print(
        f"  {m['query'][:52]:52} [{m['kind']:9}] "
        f"P@5={m['p@5']:.2f} P@10={m['p@10']:.2f} R@20={m['recall@20']:.2f} "
        f"MRR={m['mrr']:.2f}  {m['found']}/{m['n_relevant']}  {m['seconds']}s"
    )


def _print_aggregate(agg: dict) -> None:
    print("\n" + "=" * 70)
    for kind, v in agg.items():
        if kind == "overall":
            continue
        print(f"  {kind:10} (n={v['n_queries']:2})  P@5={v['p@5']}  P@10={v['p@10']}  R@20={v['recall@20']}  MRR={v['mrr']}")
    o = agg["overall"]
    print("-" * 70)
    print(f"  {'OVERALL':10} (n={o['n_queries']:2})  P@5={o['p@5']}  P@10={o['p@10']}  R@20={o['recall@20']}  MRR={o['mrr']}")
    print("=" * 70)


def _save(result: dict, *, baseline: bool) -> Path:
    _RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = _RESULTS_DIR / (f"baseline_{ts}.json" if baseline else f"run_{ts}.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if baseline:
        (_RESULTS_DIR / "baseline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved → {path}")
    return path


def _compare(current: dict, baseline_path: str) -> None:
    base = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    print("\n" + "=" * 70 + "\n  DELTA vs baseline\n" + "=" * 70)
    for kind in current["aggregate"]:
        cur = current["aggregate"][kind]
        old = base["aggregate"].get(kind, {})
        parts = []
        for k in ("p@5", "p@10", "recall@20", "mrr"):
            if cur.get(k) is not None and old.get(k) is not None:
                d = cur[k] - old[k]
                parts.append(f"{k} {d:+.3f}")
        print(f"  {kind:10}  " + "  ".join(parts))
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="save this run as the baseline")
    ap.add_argument("--compare", metavar="FILE", help="compare this run against a saved results file")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    result = run()
    if not args.no_save:
        _save(result, baseline=args.baseline)
    if args.compare:
        _compare(result, args.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
