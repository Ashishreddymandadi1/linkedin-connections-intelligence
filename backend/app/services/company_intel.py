"""Company-level semantic classification (spec §4–§5, §15).

Classify a company ONCE, cache it, reuse for every person who worked there.
Never re-classified per person. Tri-state: TRUE / FALSE / UNKNOWN — missing
evidence is UNKNOWN, never FALSE. Uses the model's general knowledge of
well-known companies (legitimate — this is public-knowledge classification,
not invented company-specific facts); never asked for funding stage, employee
count, or founding year, and told explicitly not to guess those.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app import repositories as repo
from app.config import settings
from app.constants import CompanyClassProvenance
from app.schemas import CompanyClassificationBatch
from app.services.llm.router import generate_structured
from app.services.matching import norm_company

log = logging.getLogger("app.company_intel")

_BATCH_SIZE = 20

_SYSTEM = (
    "You classify companies for a professional search tool using your general knowledge "
    "of well-known companies and organizations. For each company, decide: "
    "is_technology_company (their core business is software/hardware/internet technology), "
    "is_big_tech (a major, well-known large technology company — e.g. Google, Meta, Amazon, "
    "Microsoft, Apple, Netflix, Nvidia and similar-scale peers), "
    "is_startup (a small, early-stage, independent company — NOT a large established "
    "corporation, NOT a division of a big company). "
    "Use true/false only when you clearly recognize the company; use null when you do not "
    "recognize it or are not confident. "
    "Do NOT invent funding stage, employee count, revenue, or founding year — you were not "
    "given that data. Also list 1-4 short industries (e.g. \"fintech\", \"healthcare\", "
    "\"e-commerce\", \"consulting\", \"cloud infrastructure\") if recognizable, else []. "
    "Return JSON only."
)


def company_key(company_id: str | None, company_name: str | None) -> str:
    if company_id:
        return f"id:{company_id}"
    return f"name:{norm_company(company_name)}"


def get_or_classify(
    db: Session, companies: list[tuple[str | None, str | None, str | None]]
) -> dict[str, dict]:
    """``companies``: [(company_id, company_name, company_linkedin_url)].
    Returns ``{company_key: classification_dict}`` for every requested company
    (unknown/unclassifiable ones come back as an UNKNOWN stub, never omitted).
    """
    keyed = [(company_key(cid, name), cid, name) for cid, name, _url in companies if name]
    keys = [k for k, _, _ in keyed]
    cached = repo.get_company_semantics(db, keys)

    missing = [(k, cid, name) for k, cid, name in keyed if k not in cached]
    if missing and settings.company_classification_enabled:
        _classify_batch(db, missing)
        cached = repo.get_company_semantics(db, keys)  # re-fetch, now includes new rows

    out: dict[str, dict] = {}
    for k, _cid, name in keyed:
        row = cached.get(k)
        out[k] = to_dict(row) if row else _unknown_stub(name)
    return out


def _unknown_stub(name: str | None) -> dict:
    return {
        "display_name": name,
        "industries": [],
        "categories": [],
        "is_technology_company": None,
        "is_startup": None,
        "is_big_tech": None,
        "confidence": 0.0,
        "reason": "not yet classified",
        "provenance": CompanyClassProvenance.UNKNOWN,
    }


def to_dict(row) -> dict:  # public — used by search_service._pool_company_class
    return {
        "display_name": row.display_name,
        "industries": row.industries or [],
        "categories": row.categories or [],
        "is_technology_company": row.is_technology_company,
        "is_startup": row.is_startup,
        "is_big_tech": row.is_big_tech,
        "confidence": row.confidence,
        "reason": row.reason,
        "provenance": row.provenance,
    }


def _classify_batch(db: Session, missing: list[tuple[str, str | None, str | None]]) -> None:
    for i in range(0, len(missing), _BATCH_SIZE):
        chunk = missing[i : i + _BATCH_SIZE]
        listing = "\n".join(f'- key="{k}" name="{name}"' for k, _cid, name in chunk)
        user = (
            "Classify each of these companies:\n"
            + listing
            + '\n\nReturn {"companies": [{"key": "...", "industries": [...], "categories": [...], '
            '"is_technology_company": bool|null, "is_big_tech": bool|null, "is_startup": bool|null, '
            '"confidence": 0-1, "reason": "..."}]} — one entry per key, echoing the key exactly.'
        )
        result = generate_structured(
            _SYSTEM, user, CompanyClassificationBatch, max_tokens=2000,
            operation="company_classification",
        )
        by_key = {c[0]: c for c in chunk}
        if result is None:
            log.info("company classification unavailable for %d companies — left UNKNOWN", len(chunk))
            continue
        batch, provider, _model = result
        seen = set()
        for item in batch.companies:
            k = item.key
            if k not in by_key:
                continue
            seen.add(k)
            _, cid, name = by_key[k]
            repo.upsert_company_semantic(
                db,
                k,
                company_id=cid,
                display_name=name,
                industries=item.industries,
                categories=item.categories,
                is_technology_company=item.is_technology_company,
                is_startup=item.is_startup,
                is_big_tech=item.is_big_tech,
                confidence=item.confidence,
                reason=item.reason[:300] if item.reason else None,
                provenance=CompanyClassProvenance.LLM_INFERENCE,
                llm_provider=provider,
            )
        for k in by_key:
            if k not in seen:
                _, cid, name = by_key[k]
                repo.upsert_company_semantic(
                    db, k, company_id=cid, display_name=name, provenance=CompanyClassProvenance.UNKNOWN
                )
        db.commit()
