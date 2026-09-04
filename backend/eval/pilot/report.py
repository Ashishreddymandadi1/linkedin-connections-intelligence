"""Render pilot evaluation runs to JSON + Markdown (V4 PART 10 §9).

Output goes to ``eval/pilot/results/<timestamp>.{json,md}``. That directory is
gitignored — real profile names never reach git.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PILOT_DIR / "results"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _fmt_metric(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        return ", ".join(f"P@{k}={val:.2f}" for k, val in v.items())
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_reports(
    *,
    mode: str,
    sample_summary: dict,
    records: list[dict],
    metrics: list[dict],
    interp_flags: dict[str, list[dict]],
    live_plan: dict | None = None,
    out_dir: Path | str = RESULTS_DIR,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    json_path = out_dir / f"{stamp}.json"
    md_path = out_dir / f"{stamp}.md"

    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "sample": sample_summary,
        "live_plan": live_plan,
        "queries": records,
        "metrics": metrics,
        "interpretation_flags": interp_flags,
    }
    json_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    metrics_by_id = {m["query_id"]: m for m in metrics}
    lines: list[str] = []
    lines.append(f"# Pilot evaluation — {mode} — {doc['generated_at']}")
    lines.append("")
    lines.append(f"Sample: {sample_summary.get('selected')} profiles from "
                 f"`{sample_summary.get('dataset_id')}` (seed `{sample_summary.get('seed')}`)")
    lines.append(f"Completeness tiers: {sample_summary.get('tiers')}")
    lines.append("")

    for rec in records:
        lines.append(f"## {rec['query_id']} — “{rec['query']}”")
        lines.append("")
        interp = rec["interpretation"]
        lines.append("**INTERPRETATION**")
        lines.append("")
        lines.append(f"- intent: `{interp.get('intent')}`  ·  confidence: "
                     f"{_fmt_metric(interp.get('interpretation_confidence'))}")
        if interp.get("interpretation_summary"):
            lines.append(f"- summary: {interp['interpretation_summary']}")
        if interp.get("context"):
            lines.append(f"- context: `{interp['context']}`")
        if interp.get("target_person_context"):
            lines.append(f"- target_person_context: `{interp['target_person_context']}`")
        if interp.get("unresolved"):
            lines.append(f"- unresolved: `{interp['unresolved']}`")
        for c in interp.get("criteria", []):
            lines.append(f"  - `{c['type']}` {c.get('concept') or c.get('value') or c.get('values')} "
                         f"| op={c.get('operator')} scope={c.get('scope')} "
                         f"required={c.get('required')} modality={c.get('modality')} w={c.get('weight')}")
        flags = interp_flags.get(rec["query_id"], [])
        if flags:
            lines.append("")
            lines.append("**PLAN FLAGS**")
            for f in flags:
                lines.append(f"- ⚠️ `{f['code']}` — {f['detail']}")
        lines.append("")

        f = rec["funnel"]
        lines.append(f"**FUNNEL** network={f.get('network_size')} · viable={f.get('viable_candidate_count')} "
                     f"· judged={f.get('judge_candidate_count')} ({f.get('judge_batch_count')} batch) "
                     f"· audit_pool={f.get('audit_pool')} · returned={f.get('returned')} "
                     f"(exact {f.get('exact')} / possible {f.get('possible')} / near {f.get('near')})")
        lines.append("")

        exact = [r for r in rec["results"] if r["qualification"] == "exact_match"]
        possible = [r for r in rec["results"] if r["qualification"] == "possible_match"]
        lines.append(f"**EXACT MATCHES ({len(exact)})**")
        for r in exact:
            lines.append(f"- #{r['rank']} {r['name']} — score {r['match_score']:.0f} — "
                         f"matched: {r['matched_criteria']}")
        lines.append(f"**POSSIBLE MATCHES ({len(possible)})**")
        for r in possible:
            lines.append(f"- #{r['rank']} {r['name']} — score {r['match_score']:.0f} — "
                         f"uncertain: {r['uncertain_criteria']}")
        lines.append(f"**NEAR MATCHES ({len(rec['near_matches'])})**")
        for r in rec["near_matches"]:
            lines.append(f"- {r['name']} — missing: {r['unmet_criteria']}")
        lines.append("")

        if rec["audit_changes"]:
            lines.append("**FINAL AUDIT CHANGES**")
            for a in rec["audit_changes"]:
                lines.append(f"- {a['name']}: {a['transition']} — {a.get('reason')} {a.get('issues')}")
            lines.append("")

        jt = rec["judge_trace"]
        lines.append(f"**JUDGE** people judged={jt.get('people_judged')} · "
                     f"validator downgrades={jt.get('validator_downgrades')} · "
                     f"dropped={jt.get('validator_dropped')}")
        if jt.get("verdict_status_by_type"):
            for ctype, counts in jt["verdict_status_by_type"].items():
                lines.append(f"  - {ctype}: {counts}")
        lines.append(f"**UNKNOWN required rate:** {_fmt_metric(rec.get('unknown_required_rate'))}")
        lines.append("")

        m = metrics_by_id.get(rec["query_id"], {})
        lines.append("**METRICS**")
        if not m.get("labeled"):
            lines.append("- not available — ground truth not labeled")
        else:
            lines.append(f"- {_fmt_metric(m.get('precision_at'))}")
            lines.append(f"- recall@20={_fmt_metric(m.get('recall_at_20'))} · "
                         f"MRR={_fmt_metric(m.get('mrr'))} · nDCG@20={_fmt_metric(m.get('ndcg_at_20'))}")
            lines.append(f"- required-violation rate={_fmt_metric(m.get('required_violation_rate'))} · "
                         f"exact precision={_fmt_metric(m.get('exact_precision'))} · "
                         f"possible precision={_fmt_metric(m.get('possible_precision'))}")
        lines.append("")
        lines.append("**POTENTIAL QUALITY ISSUES** _(reviewer fills in)_")
        lines.append("")
        lines.append("---")
        lines.append("")

    if live_plan:
        lines.append("## LIVE PILOT PLAN (not executed)")
        lines.append("```json")
        lines.append(json.dumps(live_plan, indent=2))
        lines.append("```")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
