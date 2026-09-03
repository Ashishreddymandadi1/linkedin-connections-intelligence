"""V4 PART 7-9 — a completed search is a historical snapshot.

Reloading a saved search rebuilds the response it first returned — same order,
qualifications, uncertainty, audit fields, tier counts, near matches, and
judge / audit metadata — WITHOUT re-running query interpretation, embeddings,
the semantic judge, the final auditor, reason generation, or Apify.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import repositories as repo
from app.constants import Qualification
from app.database import SessionLocal
from app.models import Person, SearchQuery, SearchResult, SearchRunState
from app.schemas import SearchResultItem
from app.services import search_service

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


def _enriched(client) -> str:
    ds_id = client.post(
        "/datasets", files={"file": ("c.csv", FIXTURE_CSV.read_bytes(), "text/csv")}
    ).json()["dataset"]["dataset_id"]
    client.post(f"/datasets/{ds_id}/enrich")
    return ds_id


# ─────────────────── full round-trip via the real deterministic pipeline ───────────────────


def test_reload_equals_original_response(client):
    ds_id = _enriched(client)
    original = client.post(
        "/search", json={"dataset_id": ds_id, "query": "People who previously worked at Amazon and know AWS"}
    ).json()
    sid = original["search_id"]

    reloaded = client.get(f"/search/{sid}").json()

    assert reloaded["query"] == original["query"]
    assert reloaded["interpreted_query"] == original["interpreted_query"]
    assert reloaded["llm_provider"] == original["llm_provider"]
    assert reloaded["llm_model"] == original["llm_model"]
    assert reloaded["judge_metadata"] == original["judge_metadata"]
    assert reloaded["audit_metadata"] == original["audit_metadata"]

    oc, rc = original["connections"], reloaded["connections"]
    assert rc["total_candidates"] == oc["total_candidates"]
    assert rc["returned"] == oc["returned"]
    assert rc["exact_match_count"] == oc["exact_match_count"]
    assert rc["possible_match_count"] == oc["possible_match_count"]

    assert [r["person_id"] for r in rc["results"]] == [r["person_id"] for r in oc["results"]]
    assert [r["qualification"] for r in rc["results"]] == [r["qualification"] for r in oc["results"]]
    assert [r["uncertain_criteria"] for r in rc["results"]] == [r["uncertain_criteria"] for r in oc["results"]]
    assert [r["match_score"] for r in rc["results"]] == [r["match_score"] for r in oc["results"]]
    assert [r["person_id"] for r in rc["near_matches"]] == [r["person_id"] for r in oc["near_matches"]]


def test_reload_runs_no_llm_or_search_logic(client, monkeypatch):
    ds_id = _enriched(client)
    sid = client.post(
        "/search", json={"dataset_id": ds_id, "query": "Senior engineers at Microsoft who know Java"}
    ).json()["search_id"]

    def boom(*a, **k):  # noqa: ARG001
        raise AssertionError("load_search must not re-run any search / LLM step")

    monkeypatch.setattr("app.services.search_service.interpret_query", boom)
    monkeypatch.setattr("app.services.search_service.run_judge", boom)
    monkeypatch.setattr("app.services.search_service._run_final_audit", boom)
    monkeypatch.setattr("app.services.search_service.generate_reason", boom)
    monkeypatch.setattr("app.services.search_service.get_candidates", boom)
    monkeypatch.setattr("app.services.reason_generator.generate_reason", boom)
    monkeypatch.setattr("app.services.embeddings.embed_text", boom)

    reloaded = client.get(f"/search/{sid}")
    assert reloaded.status_code == 200
    assert reloaded.json()["query"] == "Senior engineers at Microsoft who know Java"


def test_near_matches_persisted_in_own_bucket(client):
    ds_id = _enriched(client)
    body = client.post(
        "/search",
        json={"dataset_id": ds_id, "query": "People currently at Amazon who studied at Georgia Tech and know Rust"},
    ).json()
    sid = body["search_id"]

    db = SessionLocal()
    try:
        rows = repo.get_search_results(db, sid)
        by_bucket: dict[str, list[str]] = {}
        for r in rows:
            by_bucket.setdefault(r.bucket, []).append(r.person_id)
        # near rows, when present, are never in the "connection" bucket
        main_ids = set(by_bucket.get("connection", []))
        near_ids = set(by_bucket.get("connection_near", []))
        assert main_ids.isdisjoint(near_ids)

        state = repo.get_search_run_state(db, sid)
        assert state is not None
        assert state.response_version == search_service.RESPONSE_VERSION
        assert state.near_match_count == len(by_bucket.get("connection_near", []))
    finally:
        db.close()

    reloaded = client.get(f"/search/{sid}").json()
    reloaded_near = {r["person_id"] for r in reloaded["connections"]["near_matches"]}
    reloaded_main = {r["person_id"] for r in reloaded["connections"]["results"]}
    assert reloaded_near.isdisjoint(reloaded_main)
    assert {r["person_id"] for r in body["connections"]["near_matches"]} == reloaded_near


# ─────────────────── seeded-snapshot fidelity (items 21 / 23 / 24) ───────────────────


def _seed_person(db, ds_id: str, name: str) -> str:
    p = repo.add_person(
        db,
        dataset_id=ds_id,
        is_connection=True,
        linkedin_url=f"https://www.linkedin.com/in/{name.lower().replace(' ', '-')}",
        full_name=name,
        current_title="Engineer",
        current_company="Acme",
    )
    return p.id


def _item(pid: str, rank: int, **over) -> dict:
    base = dict(
        rank=rank,
        person_id=pid,
        name=f"P{rank}",
        linkedin_url="https://www.linkedin.com/in/x",
        current_title="Engineer",
        current_company="Acme",
        is_connection=True,
        match_score=90.0 - rank,
        data_confidence=80,
        reason="seeded",
        qualification=Qualification.EXACT_MATCH,
        uncertain_criteria=[],
        unmet_criteria=[],
        matched_criteria=[],
    )
    base.update(over)
    return SearchResultItem(**base).model_dump()


def test_seeded_snapshot_reloads_exactly(client):
    ds_id = _enriched(client)
    db = SessionLocal()
    try:
        ex1 = _seed_person(db, ds_id, "Exact One")
        ex2 = _seed_person(db, ds_id, "Exact Two")
        po1 = _seed_person(db, ds_id, "Possible One")
        nr1 = _seed_person(db, ds_id, "Near One")
        nr2 = _seed_person(db, ds_id, "Near Two")

        sq = repo.create_search_query(
            db,
            dataset_id=ds_id,
            query_text="seeded historical search",
            interpreted_query_json={"intent": "professional_recommendation", "criteria": [], "interpretation_summary": "s"},
            llm_provider="anthropic",
            llm_model="claude-x",
            total_candidates=3,
        )

        payloads = [
            _item(ex1, 1, name="Exact One", audit_decision="approved", audit_confidence=0.93, llm_verified=True),
            _item(ex2, 2, name="Exact Two", audit_decision="approved", audit_confidence=0.88, llm_verified=True),
            _item(
                po1, 3, name="Possible One",
                qualification=Qualification.POSSIBLE_MATCH,
                uncertain_criteria=["startup classification", "mentoring experience"],
                audit_decision="downgrade", audit_confidence=0.55,
                audit_reason="healthcare experience is not fully verified",
                audit_issues=["healthcare unverified"], llm_verified=False,
            ),
        ]
        for pl in payloads:
            repo.add_search_result(
                db, search_id=sq.id, person_id=pl["person_id"], bucket="connection",
                rank=pl["rank"], match_score=pl["match_score"], data_confidence=pl["data_confidence"],
                reason=pl["reason"], payload=pl,
            )
        near_payloads = [
            _item(nr1, 1, name="Near One", qualification=Qualification.NOT_MATCH,
                  unmet_criteria=["CXO-level seniority"]),
            _item(nr2, 2, name="Near Two", qualification=Qualification.NOT_MATCH,
                  unmet_criteria=["healthcare industry experience"]),
        ]
        for pl in near_payloads:
            repo.add_search_result(
                db, search_id=sq.id, person_id=pl["person_id"], bucket="connection_near",
                rank=pl["rank"], match_score=pl["match_score"], data_confidence=pl["data_confidence"],
                reason=pl["reason"], payload=pl,
            )

        judge_md = {"mode": "all_viable", "status": "full", "judge_candidate_count": 5}
        audit_md = {"enabled": True, "status": "full", "approved": 2, "downgraded": 1}
        repo.upsert_search_run_state(
            db, sq.id,
            response_version=1, exact_match_count=2, possible_match_count=1,
            returned_count=3, near_match_count=2, total_candidates=3, external_searched=False,
            judge_metadata=judge_md, audit_metadata=audit_md,
        )
        db.commit()
        sid = sq.id
    finally:
        db.close()

    r = client.get(f"/search/{sid}").json()
    c = r["connections"]

    assert r["query"] == "seeded historical search"
    assert r["interpreted_query"]["interpretation_summary"] == "s"
    assert r["judge_metadata"] == judge_md
    assert r["audit_metadata"] == audit_md
    assert c["exact_match_count"] == 2
    assert c["possible_match_count"] == 1
    assert [x["name"] for x in c["results"]] == ["Exact One", "Exact Two", "Possible One"]
    assert [x["qualification"] for x in c["results"]] == ["exact_match", "exact_match", "possible_match"]

    possible = c["results"][2]
    assert possible["uncertain_criteria"] == ["startup classification", "mentoring experience"]
    assert possible["audit_decision"] == "downgrade"
    assert possible["audit_confidence"] == 0.55
    assert possible["audit_reason"] == "healthcare experience is not fully verified"
    assert possible["llm_verified"] is False

    assert c["results"][0]["llm_verified"] is True
    assert c["results"][0]["audit_confidence"] == 0.93

    assert [x["name"] for x in c["near_matches"]] == ["Near One", "Near Two"]
    assert c["near_matches"][0]["unmet_criteria"] == ["CXO-level seniority"]
    assert c["near_matches"][0]["qualification"] == "not_match"
    # near matches never leak into the main results
    assert {x["person_id"] for x in c["near_matches"]}.isdisjoint({x["person_id"] for x in c["results"]})


# ─────────────────── backward compatibility (item 22) ───────────────────


def test_old_search_without_run_state_still_loads(client):
    ds_id = _enriched(client)
    db = SessionLocal()
    try:
        pid = _seed_person(db, ds_id, "Legacy Person")
        sq = repo.create_search_query(
            db, dataset_id=ds_id, query_text="legacy query",
            interpreted_query_json={"criteria": []}, llm_provider="deterministic",
            llm_model=None, total_candidates=1,
        )
        # a pre-PART-7 payload: NO qualification / audit / uncertainty keys
        legacy_payload = {
            "rank": 1, "person_id": pid, "name": "Legacy Person",
            "linkedin_url": "https://www.linkedin.com/in/legacy",
            "is_connection": True, "match_score": 71.0, "data_confidence": 60,
            "reason": "legacy reason", "matched_criteria": ["Amazon"],
        }
        repo.add_search_result(
            db, search_id=sq.id, person_id=pid, bucket="connection", rank=1,
            match_score=71.0, data_confidence=60, reason="legacy reason", payload=legacy_payload,
        )
        db.commit()
        sid = sq.id
        assert repo.get_search_run_state(db, sid) is None
    finally:
        db.close()

    r = client.get(f"/search/{sid}")
    assert r.status_code == 200
    body = r.json()
    c = body["connections"]
    assert body["judge_metadata"] is None
    assert body["audit_metadata"] is None
    assert c["near_matches"] == []
    assert c["results"][0]["qualification"] == "possible_match"  # schema default
    assert c["results"][0]["uncertain_criteria"] == []
    assert c["results"][0]["llm_verified"] is False
    # counts derived from payload qualifications when the snapshot row is absent
    assert c["exact_match_count"] == 0
    assert c["possible_match_count"] == 1


def test_delete_dataset_cascades_run_state(client):
    ds_id = _enriched(client)
    sid = client.post("/search", json={"dataset_id": ds_id, "query": "Amazon AWS"}).json()["search_id"]
    db = SessionLocal()
    try:
        assert repo.get_search_run_state(db, sid) is not None
    finally:
        db.close()

    client.delete(f"/datasets/{ds_id}")

    db = SessionLocal()
    try:
        assert db.get(SearchRunState, sid) is None
        assert repo.get_search_run_state(db, sid) is None
    finally:
        db.close()
