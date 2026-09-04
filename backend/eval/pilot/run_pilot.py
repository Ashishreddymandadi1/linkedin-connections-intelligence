"""Pilot / evaluation orchestrator CLI (V4 PART 10).

    python -m eval.pilot.run_pilot inventory                 # read-only prod DB inventory
    python -m eval.pilot.run_pilot sample                    # deterministic ~40-profile pick
    python -m eval.pilot.run_pilot isolate                   # build eval/pilot/pilot.db
    python -m eval.pilot.run_pilot labels                    # write the label template
    python -m eval.pilot.run_pilot run                       # OFFLINE eval run + report
    python -m eval.pilot.run_pilot plan                      # print the live-pilot plan
    python -m eval.pilot.run_pilot run --live --i-understand-costs    # LIVE (paid) eval run
    python -m eval.pilot.run_pilot enrich --live --i-understand-costs # LIVE (paid) v3 backfill

Nothing here scrapes or touches the production dataset. ``run``/``enrich`` refuse
to make live LLM calls without BOTH ``--live`` and ``--i-understand-costs``.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings

PILOT_DIR = Path(__file__).resolve().parent
MAIN_DATASET_ID = "dataset_0ba27cae09d4"  # suraj_1000_connections — the real network


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_inventory(args) -> None:
    from eval.pilot.inventory import inventory

    _print(inventory(args.dataset or None))


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

    tiers = Counter(p.completeness_tier for p in sample.people)
    facets = Counter(f for p in sample.people for f in p.facets)
    return {
        "dataset_id": sample.dataset_id,
        "seed": sample.seed,
        "target": sample.target,
        "selected": len(sample.people),
        "tiers": dict(tiers),
        "facet_coverage": dict(facets),
        "semantic_at_target": sum(1 for p in sample.people if p.semantic_version == settings.semantic_profile_version),
        "semantic_any": sum(1 for p in sample.people if p.has_semantic),
        "people": [
            {"person_id": p.person_id, "public_identifier": p.public_identifier,
             "completeness": p.completeness, "tier": p.completeness_tier,
             "semantic_version": p.semantic_version, "has_embedding": p.has_embedding,
             "facets": p.facets, "selected_via": p.selected_via}
            for p in sample.people
        ],
    }


def cmd_sample(args) -> None:
    _print(_sample_summary(_load_sample(args)))


def cmd_isolate(args) -> None:
    from eval.pilot.isolate import build_pilot_db

    sample = _load_sample(args)
    report = build_pilot_db(
        settings.database_url, sample.dataset_id, sample.person_ids,
        pilot_db_path=args.pilot_db,
    )
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
    path = write_template(ALL_QUERIES, directory, args.labels)
    print(f"wrote label template: {path}")


def cmd_plan(args) -> None:
    from eval.pilot.plan import live_commands, provider_routing, semantic_v3_plan

    sample = _load_sample(args)
    at_target = [p.person_id for p in sample.people
                 if p.semantic_version == settings.semantic_profile_version]
    missing = [p.person_id for p in sample.people
               if p.semantic_version != settings.semantic_profile_version]
    _print({
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
    })


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
    from eval.pilot.metrics import compute
    from eval.pilot.labels import load_labels
    from eval.pilot.plan import search_call_estimate
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

    Session, engine = pilot_sessionmaker(pilot_db)
    labels = load_labels(args.labels)
    records: list[dict] = []
    interp_flags: dict[str, list[dict]] = {}
    metrics: list[dict] = []

    queries = [q for q in ALL_QUERIES if not args.only or q["id"] in set(args.only.split(","))]
    with Session() as db:
        for q in queries:
            rec = run_query(db, q, offline=offline)
            d = rec.as_dict()
            records.append(d)
            interp_flags[q["id"]] = flag_plan(q["query"], rec.interpretation)
            ranked = [r["person_id"] for r in rec.results]
            quals = {r["person_id"]: r["qualification"] for r in rec.results}
            violations = {
                r["person_id"]: bool(r["uncertain_criteria"] or r["unmet_criteria"])
                for r in rec.results
            }
            m = compute(q["id"], ranked, labels=labels.get(q["id"]),
                        qualifications=quals, required_violations=violations,
                        audit_changes=None)
            metrics.append(m.as_dict())
            print(f"  {q['id']:34s} exact={rec.funnel.get('exact')} "
                  f"possible={rec.funnel.get('possible')} near={rec.funnel.get('near')} "
                  f"flags={len(interp_flags[q['id']])}")
    engine.dispose()

    sample = _load_sample(args)
    summary = _sample_summary(sample)
    live_plan = search_call_estimate(records) if offline else None
    jp, mp = write_reports(
        mode="offline-dry-run" if offline else "live",
        sample_summary=summary, records=records, metrics=metrics,
        interp_flags=interp_flags, live_plan=live_plan,
    )
    print(f"\nwrote {jp}\nwrote {mp}")


def cmd_enrich(args) -> None:
    from eval.pilot.isolate import pilot_sessionmaker

    if not _require_live_ack(args):
        return
    pilot_db = Path(args.pilot_db)
    if not pilot_db.exists():
        print(f"pilot DB missing: {pilot_db}. Run `run_pilot isolate` first.")
        return

    from app import repositories as repo
    from app.services.semantic_enrich import enrich_person_semantics

    Session, engine = pilot_sessionmaker(pilot_db)
    done = failed = skipped = 0
    with Session() as db:
        from app.models import Person, Dataset

        ds_id = db.query(Dataset.id).scalar()
        people = repo.list_people(db, ds_id, is_connection=True)
        for p in people:
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
    common.add_argument("--dataset", default=None, help="dataset id (default: the main network)")
    common.add_argument("--target", type=int, default=40)
    common.add_argument("--seed", default="v4-part10-pilot")
    common.add_argument("--pilot-db", default=str(PILOT_DIR / "pilot.db"))
    common.add_argument("--labels", default=str(PILOT_DIR / "labels" / "labels.json"))
    common.add_argument("--live", action="store_true")
    common.add_argument("--i-understand-costs", dest="i_understand_costs", action="store_true")
    common.add_argument("--only", default="", help="comma-separated query ids")

    ap = argparse.ArgumentParser(description="V4 PART 10 pilot / evaluation harness", parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inventory", "sample", "isolate", "labels", "plan", "run", "enrich"):
        sub.add_parser(name, parents=[common])

    args = ap.parse_args(argv)
    {
        "inventory": cmd_inventory, "sample": cmd_sample, "isolate": cmd_isolate,
        "labels": cmd_labels, "plan": cmd_plan, "run": cmd_run, "enrich": cmd_enrich,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
