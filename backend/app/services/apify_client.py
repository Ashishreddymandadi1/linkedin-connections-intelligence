"""Apify HarvestAPI profile scraper wrapper (spec §7, §65).

Two modes:
* ``USE_FIXTURES=true`` (dev): read hand-written JSON from ``fixtures/profiles/``.
  Unknown slugs get a deterministic synthetic profile so a full CSV still runs
  end-to-end at zero cost.
* production: call actor ``LpVuK3Zozwuipa5bp`` in ``Profile details no email``
  mode with the profile URLs, poll the run, return the dataset items.

Never enables email search. Never switches to a higher-cost mode.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from app.config import settings
from app.services.url_normalize import extract_public_identifier

log = logging.getLogger("app.apify")

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "profiles"


class ApifyError(RuntimeError):
    pass


def _load_fixture(slug: str) -> dict | None:
    fp = _FIXTURE_DIR / f"{slug}.json"
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    return None


def _synthesize(url: str, slug: str, hint: dict | None) -> dict:
    """Deterministic minimal profile for slugs without a fixture."""
    seed = int(hashlib.sha1(slug.encode()).hexdigest(), 16)
    parts = [p.capitalize() for p in slug.replace(".", "-").split("-") if p][:2] or ["Unknown"]
    first = parts[0]
    last = parts[1] if len(parts) > 1 else ""
    company = (hint or {}).get("company") or ""
    position = (hint or {}).get("position") or "Professional"
    start_year = 2015 + seed % 8
    return {
        "id": f"SYN{seed % 10_000_000}",
        "publicIdentifier": slug,
        "linkedinUrl": url,
        "firstName": first,
        "lastName": last,
        "headline": f"{position}{f' at {company}' if company else ''}",
        "about": None,
        "photo": None,
        "location": {"linkedinText": None, "parsed": {}},
        "connectionsCount": 200 + seed % 800,
        "followerCount": None,
        "topSkills": "",
        "currentPosition": [{"companyName": company}] if company else [],
        "experience": (
            [
                {
                    "position": position,
                    "companyName": company,
                    "duration": f"{start_year} - Present",
                    "description": None,
                    "skills": [],
                    "startDate": {"year": start_year},
                    "endDate": {"year": None, "text": "Present"},
                }
            ]
            if company
            else []
        ),
        "education": [],
        "skills": [],
        "_synthetic": True,
    }


def _scrape_fixtures(urls: list[str], hints: dict[str, dict] | None) -> list[dict]:
    hints = hints or {}
    out: list[dict] = []
    for url in urls:
        slug = extract_public_identifier(url) or url
        item = _load_fixture(slug) or _synthesize(url, slug, hints.get(slug))
        out.append(item)
    return out


def _scrape_apify(urls: list[str]) -> list[dict]:
    if not settings.apify_api_token:
        raise ApifyError("APIFY_API_TOKEN is not set")
    from apify_client import ApifyClient

    client = ApifyClient(settings.apify_api_token)
    run_input = {
        "profileScraperMode": settings.apify_profile_scraper_mode,
        "queries": urls,
    }
    log.info("apify run: actor=%s profiles=%d mode=%s", settings.apify_actor_id, len(urls), settings.apify_profile_scraper_mode)
    run = client.actor(settings.apify_actor_id).call(run_input=run_input, timeout_secs=600)
    if not run or run.get("status") != "SUCCEEDED":
        raise ApifyError(f"apify run did not succeed: {run.get('status') if run else 'no run'}")
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    log.info("apify run %s returned %d items", run.get("id"), len(items))
    return items


def scrape_profiles(urls: list[str], *, hints: dict[str, dict] | None = None) -> list[dict]:
    """Return one raw profile dict per input URL (order preserved best-effort)."""
    if not urls:
        return []
    if settings.use_fixtures:
        return _scrape_fixtures(urls, hints)
    return _scrape_apify(urls)


def run_metadata() -> dict:
    return {
        "actor_id": settings.apify_actor_id if not settings.use_fixtures else "fixtures",
        "mode": settings.apify_profile_scraper_mode,
    }
