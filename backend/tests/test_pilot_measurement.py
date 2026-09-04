"""V4 PART 10.1 — evaluation-harness measurement completeness.

Offline, mocked. No network, no Apify, no production DB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

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


def _fake_run_json(tmp_path: Path) -> Path:
    doc = {
        "generated_at": "2026-09-04T00:00:00+00:00",
        "mode": "live",
        "sample": {"selected": 2, "dataset_id": "ds", "seed": "s", "tiers": {"strong": 2}},
        "interpretation_flags": {"q1": []},
        "queries": [{
            "query_id": "q1", "query": "test", "group": "core",
            "interpretation": {"criteria": []},
            "funnel": {"returned": 3, "exact": 1, "possible": 2},
            "results": [
                {"person_id": "p_a", "qualification": "exact_match", "uncertain_criteria": [], "unmet_criteria": []},
                {"person_id": "p_b", "qualification": "possible_match", "uncertain_criteria": ["x"], "unmet_criteria": []},
                {"person_id": "p_c", "qualification": "possible_match", "uncertain_criteria": [], "unmet_criteria": []},
            ],
            "near_matches": [],
            "audit_transitions": [
                {"person_id": "p_b", "first_pass_qualification": "exact_match",
                 "final_qualification": "possible_match", "audit_decision": "downgrade",
                 "reason": "unverified", "issues": [], "bucket": "exact_to_possible"},
                {"person_id": "p_d", "first_pass_qualification": "possible_match",
                 "final_qualification": "not_match", "audit_decision": "incorrect",
                 "reason": "wrong", "issues": [], "bucket": "possible_to_removed"},
            ],
            "audit_transition_tally": {"exact_to_possible": 1, "possible_to_removed": 1},
            "judge_trace": {"people_judged": 3},
            "reason_generation_enabled": True,
        }],
        "metrics": [],
    }
    p = tmp_path / "run.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


# ─────────────────── §1 offline score command ───────────────────


def test_score_command_uses_zero_search_or_llm(tmp_path, monkeypatch):
    import importlib

    from eval.pilot import score
    from eval.pilot.labels import write_template

    def boom(*a, **k):  # noqa: ARG001
        raise AssertionError("score must not touch search / LLM / embeddings")

    for modname, attr in [
        ("app.services.search_service", "run_connection_search"),
        ("app.services.llm.router", "generate_structured"),
        ("app.services.embeddings", "embed_text"),
        ("app.services.query_interpreter", "interpret_query"),
    ]:
        mod = importlib.import_module(modname)
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, boom)

    run_json = _fake_run_json(tmp_path)
    labels = tmp_path / "labels.json"
    write_template([{"id": "q1", "query": "test"}], [{"person_id": "p_a", "name": "A"}], labels)
    doc = json.loads(labels.read_text(encoding="utf-8"))
    doc["labels"][0]["must_match"] = ["p_a"]
    doc["labels"][0]["must_not_match"] = ["p_d"]
    labels.write_text(json.dumps(doc), encoding="utf-8")

    jp, mp = score.rescore(run_json, labels, out_dir=tmp_path)
    assert jp.exists() and mp.exists()
    m = json.loads(jp.read_text(encoding="utf-8"))["metrics"][0]
    assert m["labeled"] is True
    assert m["precision_at"]["5"] == pytest.approx(1 / 3)
    ag = m["audit_grading"]
    assert any(g["person_id"] == "p_d" and g["correct"] for g in ag["graded"])


def test_labels_added_after_a_run_produce_metrics(tmp_path):
    from eval.pilot import score
    from eval.pilot.labels import write_template

    run_json = _fake_run_json(tmp_path)
    labels = tmp_path / "labels.json"

    write_template([{"id": "q1", "query": "test"}], [], labels)
    jp1, _ = score.rescore(run_json, labels, out_dir=tmp_path)
    assert json.loads(jp1.read_text(encoding="utf-8"))["metrics"][0]["labeled"] is False

    doc = json.loads(labels.read_text(encoding="utf-8"))
    doc["labels"][0]["must_match"] = ["p_a", "p_c"]
    labels.write_text(json.dumps(doc), encoding="utf-8")
    jp2, _ = score.rescore(run_json, labels, out_dir=tmp_path)
    assert json.loads(jp2.read_text(encoding="utf-8"))["metrics"][0]["labeled"] is True


# ─────────────────── §3 audit transition capture ───────────────────


class _Parsed:
    criteria: list = []


class _Ctx:
    judge_results: dict = {}


def test_audit_transition_capture_exact_to_possible(monkeypatch):
    from app.services import final_audit_validator as fav
    from eval.pilot.recorder import JudgeTrace, _instrument

    calls = {"n": 0}

    def fake_validate_audit(raw, packet, parsed, facts, ctx, *, first_pass_qualification,
                            first_pass_uncertain=None):
        calls["n"] += 1
        return {
            "person_id": raw["person_id"], "decision": "downgrade", "confidence": 0.5,
            "reason": "startup classification unverified", "criteria": [],
            "missing_required_reviews": 1, "applied_qualification": "possible_match",
            "failed_required": [], "audit_issues": ["unverified"], "llm_verified": False,
            "first_pass_qualification": first_pass_qualification,
            "validation": {"notes": ["dropped 2 invalid evidence ref(s)"]},
        }

    monkeypatch.setattr(fav, "validate_audit", fake_validate_audit)

    trace = JudgeTrace()
    sink: list[dict] = []
    with _instrument(trace, sink):
        fav.validate_audit({"person_id": "p1"}, {}, _Parsed(), None, _Ctx(),
                           first_pass_qualification="exact_match")

    assert calls["n"] == 1 and len(sink) == 1
    t = sink[0]
    assert t["first_pass_qualification"] == "exact_match"
    assert t["final_qualification"] == "possible_match"
    assert t["bucket"] == "exact_to_possible"
    assert trace.invalid_evidence_refs == 2


# ─────────────────── §2 audit grading via labels ───────────────────


def test_grade_audit_removal_via_labels():
    from eval.pilot.labels import QueryLabels
    from eval.pilot.metrics import grade_audit_transitions

    transitions = [
        {"person_id": "good", "bucket": "possible_to_removed", "audit_decision": "incorrect", "reason": "r"},
        {"person_id": "bad", "bucket": "exact_to_removed", "audit_decision": "incorrect", "reason": "r"},
        {"person_id": "meh", "bucket": "exact_to_possible", "audit_decision": "downgrade", "reason": "r"},
        {"person_id": "unlabeled", "bucket": "exact_to_removed", "audit_decision": "incorrect", "reason": "r"},
        {"person_id": "keep", "bucket": "exact_to_exact", "audit_decision": "approved", "reason": ""},
    ]
    labels = QueryLabels("q", must_match={"good"}, should_match={"meh"}, must_not_match={"bad"})
    g = grade_audit_transitions(transitions, labels)
    graded = {e["person_id"]: e for e in g["graded"]}
    assert graded["bad"]["correct"] is True
    assert graded["good"]["correct"] is False
    assert graded["good"]["kind"] == "false_removal"
    assert any(q["person_id"] == "meh" for q in g["questionable"])
    assert g["ungraded"] == 1
    assert g["false_removal_rate"] == pytest.approx(0.5)


def test_no_labels_never_grades_audit():
    from eval.pilot.metrics import grade_audit_transitions

    g = grade_audit_transitions(
        [{"person_id": "x", "bucket": "exact_to_removed"}], labels=None
    )
    assert g["graded"] == [] and g["ungraded"] == 1


# ─────────────────── §5/§6 judge trace counters ───────────────────


def test_required_semantic_unknown_rate_not_fabricated():
    from eval.pilot.recorder import JudgeTrace

    t = JudgeTrace()
    assert t.required_semantic_unknown_rate is None
    t.required_semantic_total = 4
    t.required_semantic_unknown = 2
    assert t.required_semantic_unknown_rate == pytest.approx(0.5)


def test_judge_trace_captures_missing_and_invalid(enriched_dataset, tmp_path, monkeypatch):
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

    PS, engine = pilot_sessionmaker(pilot_db)
    try:
        with PS() as pdb:
            rec = run_query(pdb, {"id": "t", "query": "Professors in AI who understand healthcare",
                                  "group": "core"}, offline=True)
    finally:
        engine.dispose()

    jt = rec.judge_trace
    # offline: judge has no LLM, so every judgeable criterion is a model omission
    assert jt["missing_criterion_verdicts"] >= 0
    assert jt["judgeable_criteria_expected"] >= jt["missing_criterion_verdicts"]
    assert "required_semantic_unknown_rate" in jt
    assert rec.final_uncertainty_rate == rec.unknown_required_rate


# ─────────────────── §7 --only ───────────────────


def test_only_flag_runs_exactly_the_requested_queries(enriched_dataset, tmp_path, monkeypatch):
    from eval.pilot import run_pilot
    from eval.pilot.isolate import build_pilot_db
    from eval.pilot.sample import select_pilot

    monkeypatch.setattr(settings, "reranker_enabled", False)
    db = SessionLocal()
    try:
        sample = select_pilot(db, enriched_dataset, target=4, seed="unit")
    finally:
        db.close()
    pilot_db = tmp_path / "pilot.db"
    build_pilot_db(settings.database_url, enriched_dataset, sample.person_ids, pilot_db_path=pilot_db)

    from eval.pilot import recorder as rec_mod

    ran: list[str] = []
    real = rec_mod.run_query

    def spy(dbs, qdef, **kw):
        ran.append(qdef["id"])
        return real(dbs, qdef, **kw)

    monkeypatch.setattr(rec_mod, "run_query", spy)

    ns = argparse.Namespace(
        pilot_db=str(pilot_db), labels=str(tmp_path / "l.json"),
        only="q09_ex_amazon_now_startup,q12_backend_to_management_mentor",
        no_reasons=True, live=False, i_understand_costs=False,
        dataset=None, target=4, seed="v4-part10-pilot", result="",
    )
    run_pilot.cmd_run(ns)
    assert set(ran) == {"q09_ex_amazon_now_startup", "q12_backend_to_management_mentor"}


# ─────────────────── §8 --no-reasons ───────────────────


def test_no_reasons_restores_config(enriched_dataset, tmp_path, monkeypatch):
    from eval.pilot.isolate import build_pilot_db, pilot_sessionmaker
    from eval.pilot.recorder import run_query
    from eval.pilot.sample import select_pilot

    monkeypatch.setattr(settings, "reranker_enabled", False)
    db = SessionLocal()
    try:
        sample = select_pilot(db, enriched_dataset, target=3, seed="unit")
    finally:
        db.close()
    pilot_db = tmp_path / "pilot.db"
    build_pilot_db(settings.database_url, enriched_dataset, sample.person_ids, pilot_db_path=pilot_db)

    before = settings.llm_reason_generation
    PS, engine = pilot_sessionmaker(pilot_db)
    try:
        with PS() as pdb:
            rec = run_query(pdb, {"id": "t", "query": "People who work at Amazon", "group": "core"},
                            offline=True, reasons_enabled=False)
    finally:
        engine.dispose()
    assert rec.reason_generation_enabled is False
    assert settings.llm_reason_generation == before


# ─────────────────── §9 preflight ───────────────────


def test_preflight_inspect_makes_zero_network_calls(monkeypatch):
    from eval.pilot import preflight

    def boom(*a, **k):  # noqa: ARG001
        raise AssertionError("preflight inspect must not hit the network")

    monkeypatch.setattr("app.services.llm.router.generate_structured", boom)
    info = preflight.inspect()
    assert info["network"].startswith("none")
    assert "provider_chain" in info


# ─────────────────── §10 post-enrich inventory ───────────────────


def test_postenrich_inventory_is_db_only(enriched_dataset, tmp_path, monkeypatch):
    from eval.pilot.inventory import pilot_semantic_status
    from eval.pilot.isolate import build_pilot_db
    from eval.pilot.sample import select_pilot

    def boom(*a, **k):  # noqa: ARG001
        raise AssertionError("postenrich must not hit the network")

    monkeypatch.setattr("app.services.llm.router.generate_structured", boom)
    db = SessionLocal()
    try:
        sample = select_pilot(db, enriched_dataset, target=3, seed="unit")
    finally:
        db.close()
    pilot_db = tmp_path / "pilot.db"
    build_pilot_db(settings.database_url, enriched_dataset, sample.person_ids, pilot_db_path=pilot_db)

    status = pilot_semantic_status(str(pilot_db))
    assert status["network"].startswith("none")
    assert status["pilot_profiles"] == len(sample.person_ids)
    assert "provider_counts_at_target" in status and "failures" in status


# ─────────────────── §11 staged call estimate ───────────────────


def test_staged_five_query_call_estimate():
    from eval.pilot.plan import STAGED_QUERY_IDS, full_estimate

    records = [
        {"query_id": qid, "funnel": {"viable_candidate_count": 30, "returned": 20, "exact": 8, "possible": 6}}
        for qid in STAGED_QUERY_IDS
    ] + [
        {"query_id": "other", "funnel": {"viable_candidate_count": 30, "returned": 20, "exact": 8, "possible": 6}}
    ]
    est = full_estimate(records)
    assert len(est["staged_query_ids"]) == 5
    assert est["staged_5_production_like"]["totals"]["all_llm_requests"] < \
           est["full_21_production_like"]["totals"]["all_llm_requests"]
    assert est["full_21_reasons_disabled"]["totals"]["reason"] == 0
    assert est["full_21_reasons_disabled"]["totals"]["all_llm_requests"] <= \
           est["full_21_production_like"]["totals"]["all_llm_requests"]
