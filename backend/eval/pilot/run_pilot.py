"""Pilot / evaluation orchestrator CLI (V4 PART 10 / 10.1).

    python -m eval.pilot.run_pilot inventory        # read-only prod DB inventory
    python -m eval.pilot.run_pilot sample           # deterministic ~40-profile pick
    python -m eval.pilot.run_pilot isolate          # build eval/pilot/pilot.db
    python -m eval.pilot.run_pilot labels           # write the label template
    python -m eval.pilot.run_pilot plan             # live-pilot plan + call estimates
    python -m eval.pilot.run_pilot preflight        # provider config check (ZERO network)
    python -m eval.pilot.run_pilot run              # OFFLINE eval run + report
    python -m eval.pilot.run_pilot score --result R.json --labels L.json   # OFFLINE (re)score
    python -m eval.pilot.run_pilot postenrich       # DB-only semantic status of pilot.db

Staged live workflow (needs BOTH --live and --i-understand-costs):
    python -m eval.pilot.run_pilot preflight --live --i-understand-costs    # ONE tiny call
    python -m eval.pilot.run_pilot enrich   --live --i-understand-costs     # v3 backfill, pilot.db only
    python -m eval.pilot.run_pilot run --live --i-understand-costs --only q03_cxo_event_memphis_nashville,q09_ex_amazon_now_startup,q10_cyber_and_healthcare,q11_academia_to_industry,q12_backend_to_management_mentor

Nothing here scrapes or touches the production dataset.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings

PILOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PILOT_DIR / "results"
MAIN_DATASET_ID = "dataset_0ba27cae09d4"  # suraj_1000_connections — the real network


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# ─────────────────────── read-only helpers ───────────────────────


def _load_sample(args):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from eval.pilot.sample import select_pilot

    url = settings.database_url
    ro = url.replace("sqlite:///", "sqlite:///file:", 1) + "?mode=ro&uri=true" if url.startswith("sqlite:///") else url
    eng = create_engine(ro)
    S = sessionmaker(bind=eng, future=True)
    with S() as db:
        sample = select_pilot(db, args.dataset or MAIN_DATASET_ID, target=args.target, seed=args.seed)
    eng.dispose()
    return sample


def _sample_summary(sample) -> dict:
    from collections import Counter

    return {
        "dataset_id": sample.dataset_id,
        "seed": sample.seed,
        "target": sample.target,
        "selected": len(sample.people),
        "tiers": dict(Counter(p.completeness_tier for p in sample.people)),
        "facet_coverage": dict(Counter(f for p in sample.people for f in p.facets)),
        "semantic_at_target": sum(1 for p in sample.people
                                  if p.semantic_version == settings.semantic_profile_version),
        "semantic_any": sum(1 for p in sample.people if p.has_semantic),
        "people": [
            {"person_id": p.person_id, "public_identifier": p.public_identifier,
             "completeness": p.completeness, "tier": p.completeness_tier,
             "semantic_version": p.semantic_version, "has_embedding": p.has_embedding,
             "facets": p.facets, "selected_via": p.selected_via}
            for p in sample.people
        ],
    }


def _latest_offline_run() -> dict | None:
    files = sorted(glob.glob(str(RESULTS_DIR / "*.json")))
    for f in reversed(files):
        try:
            doc = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if doc.get("queries"):
            doc["_path"] = f
            return doc
    return None


# ─────────────────────── commands ───────────────────────


def cmd_inventory(args) -> None:
    from eval.pilot.inventory import inventory

    _print(inventory(args.dataset or None))


def cmd_sample(args) -> None:
    _print(_sample_summary(_load_sample(args)))


def cmd_isolate(args) -> None:
    from eval.pilot.isolate import build_pilot_db

    sample = _load_sample(args)
    report = build_pilot_db(settings.database_url, sample.dataset_id, sample.person_ids,
                            pilot_db_path=args.pilot_db)
    _print(report.__dict__)


def cmd_labels(args) -> None:
    from eval.pilot.labels import write_template
    from eval.pilot.queries import ALL_QUERIES

    sample = _load_sample(args)
    directory = [
        {"person_id": p.person_id, "public_identifier": p.public_identifier,
         "name": p.full_name, "facets": p.facets}
        for p in sample.people
    ]
    print(f"wrote label template: {write_template(ALL_QUERIES, directory, args.labels)}")


def cmd_preflight(args) -> None:
    from eval.pilot.preflight import inspect, probe_live

    if not (args.live and args.i_understand_costs):
        _print({"mode": "inspect (zero network)", **inspect()})
        print("\n(for ONE tiny live probe: preflight --live --i-understand-costs)")
        return
    print("running ONE tiny structured LLM request through the normal router...")
    _print({"mode": "live probe", **probe_live()})


def cmd_plan(args) -> None:
    from eval.pilot.plan import full_estimate, live_commands, provider_routing, semantic_v3_plan

    sample = _load_sample(args)
    at_target = [p.person_id for p in sample.people
                 if p.semantic_version == settings.semantic_profile_version]
    missing = [p.person_id for p in sample.people
               if p.semantic_version != settings.semantic_profile_version]

    out = {
        "provider_routing": provider_routing(),
        "semantic_v3_plan": semantic_v3_plan(missing, at_target),
        "live_commands": live_commands(sample.dataset_id),
        "search_config": {
            "top_connections": settings.top_connections,
            "final_result_audit_enabled": settings.final_result_audit_enabled,
            "final_result_audit_buffer": settings.final_result_audit_buffer,
            "semantic_judge_batch_size": settings.semantic_judge_batch_size,
            "final_result_audit_batch_size": settings.final_result_audit_batch_size,
            "llm_query_interpretation": settings.llm_query_interpretation,
            "llm_reason_generation": settings.llm_reason_generation,
            "llm_reason_top_n": settings.llm_reason_top_n,
        },
    }
    prior = _latest_offline_run()
    if prior:
        out["call_estimate"] = full_estimate(prior["queries"])
        out["call_estimate"]["_derived_from"] = prior["_path"]
    else:
        out["call_estimate"] = "run `run_pilot run` (offline) first to derive funnels"
    _print(out)


def cmd_postenrich(args) -> None:
    from eval.pilot.inventory import pilot_semantic_status

    _print(pilot_semantic_status(args.pilot_db))


def _require_live_ack(args) -> bool:
    if args.live and args.i_understand_costs:
        return True
    print("REFUSING to make live LLM calls.")
    print("This would run against configured providers (Anthropic first — PAID).")
    print("Re-run with:  --live --i-understand-costs   after reviewing `run_pilot plan`.")
    return False


def cmd_run(args) -> None:
    from eval.pilot.interp_eval import flag_plan
    from eval.pilot.isolate import pilot_sessionmaker
    from eval.pilot.labels import load_labels
    from eval.pilot.metrics import compute
    from eval.pilot.plan import full_estimate
    from eval.pilot.queries import ALL_QUERIES
    from eval.pilot.recorder import run_query
    from eval.pilot.report import write_reports

    offline = not args.live
    if not offline and not _require_live_ack(args):
        return

    pilot_db = Path(args.pilot_db)
    if not pilot_db.exists():
        print(f"pilot DB missing: {pilot_db}. Run `run_pilot isolate` first.")
        return

    only = {x for x in args.only.split(",") if x} if args.only else set()
    queries = [q for q in ALL_QUERIES if not only or q["id"] in only]
    if only:
        print(f"staged run — {len(queries)} of {len(ALL_QUERIES)} queries: {sorted(q['id'] for q in queries)}")

    Session, engine = pilot_sessionmaker(pilot_db)
    labels = load_labels(args.labels)
    records: list[dict] = []
    interp_flags: dict[str, list[dict]] = {}
    metrics: list[dict] = []
    reasons_enabled = not args.no_reasons

    with Session() as db:
        for q in queries:
            rec = run_query(db, q, offline=offline, reasons_enabled=reasons_enabled)
            d = rec.as_dict()
            records.append(d)
            interp_flags[q["id"]] = flag_plan(q["query"], rec.interpretation)
            ranked = [r["person_id"] for r in rec.results]
            quals = {r["person_id"]: r["qualification"] for r in rec.results}
            violations = {r["person_id"]: bool(r["uncertain_criteria"] or r["unmet_criteria"])
                          for r in rec.results}
            m = compute(q["id"], ranked, labels=labels.get(q["id"]),
                        qualifications=quals, required_violations=violations,
                        audit_transitions=rec.audit_transitions)
            metrics.append(m.as_dict())
            print(f"  {q['id']:34s} exact={rec.funnel.get('exact')} possible={rec.funnel.get('possible')} "
                  f"near={rec.funnel.get('near')} flags={len(interp_flags[q['id']])} "
                  f"audit_moves={sum(v for k, v in rec.audit_transition_tally.items() if k not in ('exact_to_exact','possible_to_possible'))}")
    engine.dispose()

    summary = _sample_summary(_load_sample(args))
    jp, mp = write_reports(
        mode=("offline-dry-run" if offline else "live") + ("" if reasons_enabled else " (no-reasons)"),
        sample_summary=summary, records=records, metrics=metrics, interp_flags=interp_flags,
        live_plan=full_estimate(records),
        extra={"reason_generation_enabled": reasons_enabled,
               "queries_run": [q["id"] for q in queries],
               "offline": offline},
    )
    print(f"\nwrote {jp}\nwrote {mp}")


def cmd_score(args) -> None:
    from eval.pilot.score import rescore

    if not args.result:
        print("--result <run>.json is required")
        return
    jp, mp = rescore(args.result, args.labels)
    print(f"rescored (zero LLM calls)\nwrote {jp}\nwrote {mp}")


def cmd_enrich(args) -> None:
    from eval.pilot.isolate import pilot_sessionmaker

    if not _require_live_ack(args):
        return
    pilot_db = Path(args.pilot_db)
    if not pilot_db.exists():
        print(f"pilot DB missing: {pilot_db}. Run `run_pilot isolate` first.")
        return

    from app import repositories as repo
    from app.models import Dataset
    from app.services.semantic_enrich import enrich_person_semantics

    Session, engine = pilot_sessionmaker(pilot_db)
    done = failed = skipped = 0
    with Session() as db:
        ds_id = db.query(Dataset.id).scalar()
        for p in repo.list_people(db, ds_id, is_connection=True):
            if p.semantic_version == settings.semantic_profile_version:
                skipped += 1
                continue
            try:
                enrich_person_semantics(db, p, force=True)
                db.commit()
                done += 1
                print(f"  enriched {p.full_name or p.id}")
            except Exception as e:  # noqa: BLE001
                db.rollback()
                failed += 1
                print(f"  FAILED {p.id}: {e}")
    engine.dispose()
    print(f"\nsemantic v3 backfill (pilot DB only): done={done} failed={failed} skipped={skipped}")


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default=None)
    common.add_argument("--target", type=int, default=40)
    common.add_argument("--seed", default="v4-part10-pilot")
    common.add_argument("--pilot-db", default=str(PILOT_DIR / "pilot.db"))
    common.add_argument("--labels", default=str(PILOT_DIR / "labels" / "labels.json"))
    common.add_argument("--result", default="", help="score: path to a saved run JSON")
    common.add_argument("--live", action="store_true")
    common.add_argument("--i-understand-costs", dest="i_understand_costs", action="store_true")
    common.add_argument("--only", default="", help="comma-separated query ids")
    common.add_argument("--no-reasons", dest="no_reasons", action="store_true",
                        help="disable LLM display-reason generation for this run only (no ranking effect)")

    ap = argparse.ArgumentParser(description="V4 PART 10 / 10.1 pilot harness", parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inventory", "sample", "isolate", "labels", "plan", "preflight",
                 "run", "score", "postenrich", "enrich"):
        sub.add_parser(name, parents=[common])

    args = ap.parse_args(argv)
    {
        "inventory": cmd_inventory, "sample": cmd_sample, "isolate": cmd_isolate,
        "labels": cmd_labels, "plan": cmd_plan, "preflight": cmd_preflight,
        "run": cmd_run, "score": cmd_score, "postenrich": cmd_postenrich, "enrich": cmd_enrich,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
