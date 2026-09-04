"""Render pilot evaluation runs to JSON + Markdown (V4 PART 10 §9, PART 10.1).

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


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        return ", ".join(f"P@{k}={val:.2f}" if isinstance(val, float) else f"{k}={val}"
                         for k, val in v.items())
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
    extra: dict | None = None,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    suffix = "_scored" if mode.startswith("rescored") else ""
    json_path = out_dir / f"{stamp}{suffix}.json"
    md_path = out_dir / f"{stamp}{suffix}.md"

    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "extra": extra or {},
        "sample": sample_summary,
        "live_plan": live_plan,
        "queries": records,
        "metrics": metrics,
        "interpretation_flags": interp_flags,
    }
    json_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    metrics_by_id = {m["query_id"]: m for m in metrics}
    lines: list[str] = [f"# Pilot evaluation — {mode} — {doc['generated_at']}", ""]
    if extra:
        for k, v in extra.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append(f"Sample: {sample_summary.get('selected')} profiles from "
                 f"`{sample_summary.get('dataset_id')}` (seed `{sample_summary.get('seed')}`)")
    lines.append(f"Completeness tiers: {sample_summary.get('tiers')}")
    lines.append("")

    for rec in records:
        rec.setdefault("results", [])
        rec.setdefault("near_matches", [])
        rec.setdefault("judge_trace", {})
        rec.setdefault("interpretation", {"criteria": []})
        rec.setdefault("funnel", {})
        lines.append(f"## {rec['query_id']} — “{rec.get('query', '')}”")
        lines.append("")
        interp = rec["interpretation"]
        lines.append("**INTERPRETATION**  ·  "
                     f"reason_generation_enabled={rec.get('reason_generation_enabled')}")
        lines.append("")
        lines.append(f"- intent: `{interp.get('intent')}`  ·  confidence: "
                     f"{_fmt(interp.get('interpretation_confidence'))}")
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
            for fl in flags:
                lines.append(f"- ⚠️ `{fl['code']}` — {fl['detail']}")
        lines.append("")

        f = rec["funnel"]
        lines.append(f"**FUNNEL** network={f.get('network_size')} · viable={f.get('viable_candidate_count')} "
                     f"· judged={f.get('judge_candidate_count')} ({f.get('judge_batch_count')} batch) "
                     f"· audit_pool={f.get('audit_pool')} · returned={f.get('returned')} "
                     f"(exact {f.get('exact')} / possible {f.get('possible')} / near {f.get('near')})")
        lines.append("")

        def _sc(r) -> str:
            v = r.get("match_score")
            return f"{v:.0f}" if isinstance(v, (int, float)) else "—"

        exact = [r for r in rec["results"] if r.get("qualification") == "exact_match"]
        possible = [r for r in rec["results"] if r.get("qualification") == "possible_match"]
        lines.append(f"**EXACT MATCHES ({len(exact)})**")
        for r in exact:
            lines.append(f"- #{r.get('rank', '?')} {r.get('name')} — score {_sc(r)} — "
                         f"matched: {r.get('matched_criteria', [])}")
        lines.append(f"**POSSIBLE MATCHES ({len(possible)})**")
        for r in possible:
            lines.append(f"- #{r.get('rank', '?')} {r.get('name')} — score {_sc(r)} — "
                         f"uncertain: {r.get('uncertain_criteria', [])}")
        lines.append(f"**NEAR MATCHES ({len(rec.get('near_matches', []))})**")
        for r in rec.get("near_matches", []):
            lines.append(f"- {r.get('name')} — missing: {r.get('unmet_criteria', [])}")
        lines.append("")

        tally = rec.get("audit_transition_tally") or {}
        lines.append("**FINAL AUDIT TRANSITIONS** " + " · ".join(
            f"{k}={v}" for k, v in tally.items() if v))
        for t in rec.get("audit_transitions", []):
            if t["bucket"] in ("exact_to_exact", "possible_to_possible"):
                continue
            lines.append(f"- {t['person_id']}: {t['first_pass_qualification']} → "
                         f"{t['final_qualification']} ({t['audit_decision']}) — {t.get('reason')}")
        lines.append("")

        jt = rec["judge_trace"]
        lines.append(f"**JUDGE** people judged={jt.get('people_judged')} · "
                     f"validator downgrades={jt.get('validator_downgrades')} "
                     f"(grounding {jt.get('grounding_downgrades')}) · dropped verdicts={jt.get('validator_dropped')}")
        lines.append(f"  - invalid evidence refs stripped: {jt.get('invalid_evidence_refs')} · "
                     f"wrong-scope verdicts: {jt.get('wrong_scope_refs')}")
        lines.append(f"  - missing judgeable verdicts: {jt.get('missing_criterion_verdicts')} "
                     f"/ {jt.get('judgeable_criteria_expected')} expected")
        lines.append(f"  - REQUIRED semantic verdicts: true={jt.get('required_semantic_true')} "
                     f"false={jt.get('required_semantic_false')} unknown={jt.get('required_semantic_unknown')} "
                     f"→ UNKNOWN rate {_fmt(jt.get('required_semantic_unknown_rate'))}")
        if jt.get("verdict_status_by_type"):
            for ctype, counts in jt["verdict_status_by_type"].items():
                lines.append(f"  - {ctype}: {counts}")
        lines.append(f"**Final-result uncertainty rate:** {_fmt(rec.get('final_uncertainty_rate'))}")
        lines.append("")

        m = metrics_by_id.get(rec["query_id"], {})
        lines.append("**METRICS**")
        if not m.get("labeled"):
            lines.append("- not available — ground truth not labeled")
        else:
            lines.append(f"- {_fmt(m.get('precision_at'))}")
            lines.append(f"- recall@20={_fmt(m.get('recall_at_20'))} · MRR={_fmt(m.get('mrr'))} · "
                         f"nDCG@20={_fmt(m.get('ndcg_at_20'))}")
            lines.append(f"- required-violation rate={_fmt(m.get('required_violation_rate'))} · "
                         f"exact precision={_fmt(m.get('exact_precision'))} · "
                         f"possible precision={_fmt(m.get('possible_precision'))}")
            ag = m.get("audit_grading") or {}
            lines.append(f"- audit correction rate={_fmt(m.get('audit_correction_rate'))} · "
                         f"false-removal rate={_fmt(m.get('audit_false_removal_rate'))} "
                         f"(graded {len(ag.get('graded', []))}, questionable {len(ag.get('questionable', []))}, "
                         f"ungraded {ag.get('ungraded', 0)})")
        lines.append("")
        lines.append("**POTENTIAL QUALITY ISSUES** _(reviewer fills in)_")
        lines.append("")
        lines.append("---")
        lines.append("")

    if live_plan:
        lines.append("## LIVE PILOT PLAN / CALL ESTIMATE")
        lines.append("```json")
        lines.append(json.dumps(live_plan, indent=2))
        lines.append("```")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
