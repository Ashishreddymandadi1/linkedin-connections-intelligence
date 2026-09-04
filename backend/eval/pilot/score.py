"""Offline (re)scoring of a saved pilot run — V4 PART 10.1 §1.

Workflow this enables:

    run --live  (once)  ->  results/<run>.json
    human edits labels/labels.json
    score --result results/<run>.json --labels labels/labels.json   (ZERO LLM calls)

``rescore`` recomputes every metric from the already-recorded results, judge
trace and audit transitions in the run file. It NEVER imports ``run_connection_
search``, any LLM provider, embeddings, or ``pilot.db``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eval.pilot.labels import load_labels
from eval.pilot.metrics import compute
from eval.pilot.report import write_reports


def _q_metrics(rec: dict, labels_map) -> dict:
    results = rec.get("results", [])
    ranked = [r["person_id"] for r in results]
    quals = {r["person_id"]: r["qualification"] for r in results}
    violations = {
        r["person_id"]: bool(r.get("uncertain_criteria") or r.get("unmet_criteria"))
        for r in results
    }
    m = compute(
        rec["query_id"], ranked,
        labels=labels_map.get(rec["query_id"]),
        qualifications=quals,
        required_violations=violations,
        audit_transitions=rec.get("audit_transitions") or [],
    )
    return m.as_dict()


def rescore(result_path: str | Path, labels_path: str | Path,
            out_dir: str | Path | None = None) -> tuple[Path, Path]:
    result_path = Path(result_path)
    doc = json.loads(result_path.read_text(encoding="utf-8"))
    records: list[dict] = doc.get("queries", [])
    labels_map = load_labels(labels_path)

    metrics = [_q_metrics(rec, labels_map) for rec in records]
    n_labeled = sum(1 for m in metrics if m.get("labeled"))

    out_dir = Path(out_dir) if out_dir else result_path.parent
    jp, mp = write_reports(
        mode=f"rescored (from {result_path.name})",
        sample_summary=doc.get("sample", {}),
        records=records,
        metrics=metrics,
        interp_flags=doc.get("interpretation_flags", {}),
        live_plan=doc.get("live_plan"),
        out_dir=out_dir,
        extra={
            "rescored_at": datetime.now(timezone.utc).isoformat(),
            "source_run": str(result_path),
            "labels_file": str(labels_path),
            "queries_scored": len(records),
            "queries_with_labels": n_labeled,
        },
    )
    return jp, mp
