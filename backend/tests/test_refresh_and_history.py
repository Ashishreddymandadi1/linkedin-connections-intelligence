from __future__ import annotations

from pathlib import Path

from app import repositories as repo
from app.database import SessionLocal
from app.models_base import utcnow

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


def _enriched(client) -> str:
    ds_id = client.post("/datasets", files={"file": ("c.csv", FIXTURE_CSV.read_bytes(), "text/csv")}).json()[
        "dataset"
    ]["dataset_id"]
    client.post(f"/datasets/{ds_id}/enrich")
    return ds_id


def test_search_history_endpoint(client):
    ds_id = _enriched(client)
    client.post("/search", json={"dataset_id": ds_id, "query": "worked at Amazon"})
    client.post("/search", json={"dataset_id": ds_id, "query": "studied at Georgia Tech"})

    hist = client.get(f"/datasets/{ds_id}/searches").json()
    assert len(hist) == 2
    assert hist[0]["query"] == "studied at Georgia Tech"  # newest first
    assert "search_id" in hist[0]


def test_refresh_respects_ttl(client):
    ds_id = _enriched(client)
    db = SessionLocal()
    try:
        person = repo.list_people(db, ds_id)[0]
        person.last_scraped_at = utcnow()
        db.commit()
        pid = person.id
    finally:
        db.close()

    r = client.post(f"/people/{pid}/refresh")
    assert r.status_code == 200
    assert r.json()["refreshed"] is False
    assert "fresh" in r.json()["reason"]

    forced = client.post(f"/people/{pid}/refresh?force=true")
    assert forced.status_code == 200
    assert forced.json()["refreshed"] is True


def test_delete_cascades_search_history(client):
    ds_id = _enriched(client)
    client.post("/search", json={"dataset_id": ds_id, "query": "Amazon"})
    client.delete(f"/datasets/{ds_id}")
    assert client.get(f"/datasets/{ds_id}/searches").status_code == 404
