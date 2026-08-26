from __future__ import annotations

from pathlib import Path

from app import repositories as repo
from app.constants import EnrichmentState
from app.database import SessionLocal

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


def _upload(client) -> str:
    r = client.post(
        "/datasets",
        files={"file": ("Connections.csv", FIXTURE_CSV.read_bytes(), "text/csv")},
    )
    assert r.status_code == 201, r.text
    return r.json()["dataset"]["dataset_id"]


def test_full_enrichment_flow_on_fixtures(client):
    ds_id = _upload(client)
    r = client.post(f"/datasets/{ds_id}/enrich")
    assert r.status_code == 200, r.text

    status = client.get(f"/datasets/{ds_id}/status").json()
    assert status["progress_total"] == status["progress_done"] > 0
    assert status["ready"] >= 5  # 5 hand-written fixtures + synthesized
    assert status["job"]["status"] in {"COMPLETED", "PARTIAL"}

    people = client.get(f"/datasets/{ds_id}/people").json()
    jane = next(p for p in people if p["full_name"] == "Jane Smith")
    detail = client.get(f"/people/{jane['person_id']}").json()
    assert detail["current_company"] == "Google"
    assert any(e["company_name"] == "Amazon" for e in detail["experiences"])
    assert detail["profile_completeness"] > 60
    assert any(s["skill_name"].lower().startswith("amazon web") or s["skill_name"] == "AWS" for s in detail["skills"])


def test_raw_json_is_stored_before_transform(client):
    ds_id = _upload(client)
    client.post(f"/datasets/{ds_id}/enrich")

    db = SessionLocal()
    try:
        people = repo.list_people(db, ds_id)
        jane = next(p for p in people if p.full_name == "Jane Smith")
        raw = repo.latest_raw_profile(db, jane.id)
        assert raw is not None
        assert raw.source == "apify_harvestapi"
        assert raw.raw_json["publicIdentifier"] == "jane-smith"
        assert "experience" in raw.raw_json  # untouched original
    finally:
        db.close()


def test_enrichment_is_resumable(client):
    ds_id = _upload(client)

    db = SessionLocal()
    try:
        people = repo.list_people(db, ds_id)
        # simulate a crash: mark half the people already READY with a raw profile
        from app.services.raw_store import store_raw_profile

        done = people[:4]
        for p in done:
            store_raw_profile(
                db, person_id=p.id, dataset_id=ds_id, raw={"publicIdentifier": p.public_identifier}, actor_id="fixtures"
            )
            p.enrichment_state = EnrichmentState.READY
        db.commit()
        raw_before = sum(1 for _ in [r for pp in done for r in [repo.latest_raw_profile(db, pp.id)]])
    finally:
        db.close()

    client.post(f"/datasets/{ds_id}/enrich")

    db = SessionLocal()
    try:
        for p in done:
            # the 4 pre-marked people keep exactly one raw profile — never re-scraped
            all_raw = [r for r in [repo.latest_raw_profile(db, p.id)]]
            assert len(all_raw) == 1
        status = repo.enrichment_state_counts(db, ds_id)
        assert status.get(EnrichmentState.READY, 0) >= 4
    finally:
        db.close()
    assert raw_before == 4


def test_partial_when_scrape_has_no_sections(client, monkeypatch):
    from app.services import apify_client

    def fake_scrape(urls, hints=None):  # noqa: ARG001
        return [{"publicIdentifier": apify_client.extract_public_identifier(u), "linkedinUrl": u} for u in urls]

    monkeypatch.setattr(apify_client, "scrape_profiles", fake_scrape)
    monkeypatch.setattr("app.services.enrichment_runner.scrape_profiles", fake_scrape)

    ds_id = _upload(client)
    client.post(f"/datasets/{ds_id}/enrich")
    status = client.get(f"/datasets/{ds_id}/status").json()
    assert status["partial"] >= 5
    assert status["ready"] == 0
