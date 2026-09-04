"""V4 PART 10 — offline tests for the controlled real-data evaluation harness.

Everything here runs against the throwaway test DB with LLM disabled. No network,
no Apify, no production DB.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import repositories as repo
from app.config import settings
from app.database import SessionLocal

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


@pytest.fixture()
def enriched_dataset(client) -> str:
    ds_id = client.post(
        "/datasets", files={"file": ("c.csv", FIXTURE_CSV.read_bytes(), "text/csv")}
    ).json()["dataset"]["dataset_id"]
    client.post(f"/datasets/{ds_id}/enrich")
    return ds_id


# ─────────────────────── sample selection ───────────────────────


def test_sample_selection_is_deterministic_and_stratified(enriched_dataset):
    from eval.pilot.sample import select_pilot

    db = SessionLocal()
    try:
        a = select_pilot(db, enriched_dataset, target=4, seed="unit")
        b = select_pilot(db, enriched_dataset, target=4, seed="unit")
        c = select_pilot(db, enriched_dataset, target=4, seed="different")
    finally:
        db.close()

    assert a.person_ids == b.person_ids            # reproducible
    assert len(a.person_ids) == len(set(a.person_ids))  # no dupes
    assert a.person_ids != c.person_ids or len(a.person_ids) <= 1  # seed matters
    for p in a.people:
        assert p.completeness_tier in ("strong", "medium", "sparse")
        assert p.selected_via  # every pick is explained


def test_sample_never_uses_name_order(enriched_dataset):
    from eval.pilot.sample import build_people, select_pilot

    db = SessionLocal()
    try:
        tagged = build_people(db, enriched_dataset)
        sample = select_pilot(db, enriched_dataset, target=3, seed="unit")
    finally:
        db.close()
    by_name = sorted((p.full_name or "") for p in tagged)[:3]
    picked_names = sorted((p.full_name or "") for p in sample.people)
    # extremely unlikely to coincide with alphabetical order for a real hash seed
    assert picked_names != by_name or len(tagged) <= 3


# ─────────────────────── isolation ───────────────────────


def test_isolate_builds_pilot_db_and_leaves_prod_untouched(enriched_dataset, tmp_path):
    from eval.pilot.isolate import build_pilot_db, pilot_sessionmaker

    db = SessionLocal()
    try:
        before = len(repo.list_people(db, enriched_dataset, is_connection=True))
        from eval.pilot.sample import select_pilot
        sample = select_pilot(db, enriched_dataset, target=3, seed="unit")
    finally:
        db.close()

    pilot_db = tmp_path / "pilot.db"
    report = build_pilot_db(settings.database_url, enriched_dataset, sample.person_ids,
                            pilot_db_path=pilot_db)
    assert pilot_db.exists()
    assert report.people == len(sample.person_ids)
    assert report.row_counts["people"] == len(sample.person_ids)

    # production DB unchanged
    db = SessionLocal()
    try:
        after = len(repo.list_people(db, enriched_dataset, is_connection=True))
    finally:
        db.close()
    assert after == before

    # pilot DB has exactly the sample
    PS, engine = pilot_sessionmaker(pilot_db)
    with PS() as pdb:
        from app.models import Person
        pilot_people = {p.id for p in pdb.query(Person).all()}
    engine.dispose()
    assert pilot_people == set(sample.person_ids)


# ─────────────────────── recorder: zero live calls ───────────────────────


def test_offline_recorder_makes_no_llm_calls(enriched_dataset, tmp_path, monkeypatch):
    from eval.pilot.isolate import build_pilot_db, pilot_sessionmaker
    from eval.pilot.recorder import run_query
    from eval.pilot.sample import select_pilot

    monkeypatch.setattr(settings, "reranker_enabled", False)

    db = SessionLocal()
    try:
        sample = select_pilot(db, enriched_dataset, target=4, seed="unit")
    finally:
        db.close()
    pilot_db = tmp_path / "pilot.db"
    build_pilot_db(settings.database_url, enriched_dataset, sample.person_ids, pilot_db_path=pilot_db)

    # any LLM call now explodes
    def boom(*a, **k):  # noqa: ARG001
        raise AssertionError("offline recorder must not call an LLM provider")

    for modname in ("app.services.llm.router", "app.services.query_interpreter",
                    "app.services.semantic_judge", "app.services.reason_generator",
                    "app.services.final_auditor"):
        import importlib
        mod = importlib.import_module(modname)
        if hasattr(mod, "generate_structured"):
            monkeypatch.setattr(mod, "generate_structured", boom)

    PS, engine = pilot_sessionmaker(pilot_db)
    try:
        with PS() as pdb:
            rec = run_query(pdb, {"id": "t1", "query": "People who work at Amazon", "group": "core"},
                            offline=True)
    finally:
        engine.dispose()

    assert rec.query_id == "t1"
    assert "criteria" in rec.interpretation
    assert isinstance(rec.results, list)
    # offline audit is disabled -> no audit metadata pollution
    assert rec.audit_metadata is None or rec.audit_metadata.get("enabled") is False


# ─────────────────────── metrics ───────────────────────


def test_metrics_unavailable_without_labels():
    from eval.pilot.metrics import compute

    m = compute("q", ["a", "b"], labels=None)
    assert m.labeled is False
    assert m.as_dict()["note"].startswith("not available")


def test_metrics_computed_with_labels():
    from eval.pilot.labels import QueryLabels
    from eval.pilot.metrics import compute

    labels = QueryLabels("q", must_match={"a", "c"}, should_match=set(), must_not_match={"z"})
    ranked = ["a", "x", "c", "z", "y"]
    m = compute("q", ranked, labels=labels,
                qualifications={"a": "exact_match", "x": "exact_match", "c": "possible_match", "z": "possible_match"})
    d = m.as_dict()
    assert d["precision_at"][5] == pytest.approx(2 / 5)
    assert d["recall_at_20"] == pytest.approx(1.0)
    assert d["mrr"] == pytest.approx(1.0)
    assert d["exact_precision"] == pytest.approx(0.5)   # x is not relevant
    assert d["possible_precision"] == pytest.approx(0.5)  # z is must_not_match


# ─────────────────────── labels + interp flags + plan ───────────────────────


def test_label_template_roundtrips(tmp_path):
    from eval.pilot.labels import load_labels, write_template

    p = tmp_path / "labels.json"
    write_template([{"id": "q1", "query": "x"}], [{"person_id": "p1", "name": "N"}], p)
    loaded = load_labels(p)
    assert "q1" in loaded and not loaded["q1"].any_labeled


def test_interp_flags_fire_on_bad_plans():
    from eval.pilot.interp_eval import flag_plan

    flags = flag_plan(
        "Who should I invite to a CXO networking event in Memphis?",
        {"criteria": [{"type": "professional_concept", "concept": "networking event",
                       "required": False, "operator": "ANY_OF"}],
         "context": {}},
    )
    codes = {f["code"] for f in flags}
    assert "context_leak" in codes or "context_missing" in codes
    assert "no_required" in codes

    ok = flag_plan(
        "Former Amazon people now at startups",
        {"criteria": [
            {"type": "past_company", "value": "Amazon", "required": True, "scope": "past_company", "operator": "ANY_OF"},
            {"type": "company_category", "value": "startup", "required": True, "scope": "current_company", "operator": "ANY_OF"},
        ], "context": {}},
    )
    assert "scope_past_missing" not in {f["code"] for f in ok}


def test_plan_shapes():
    from eval.pilot.plan import provider_routing, search_call_estimate, semantic_v3_plan

    pr = provider_routing()
    assert "anthropic_configured" in pr and "chain" in pr

    sp = semantic_v3_plan(["a", "b", "c"], ["d"])
    assert sp["needing_enrichment"] == 3
    assert sp["estimated_llm_requests"] == 3
    assert "NO Apify" in sp["source_data"]

    est = search_call_estimate([
        {"query_id": "q1", "funnel": {"viable_candidate_count": 25, "returned": 20, "exact": 5, "possible": 5}},
    ])
    assert est["totals"]["judge_batches"] >= 1
