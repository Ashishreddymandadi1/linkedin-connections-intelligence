"""Deduplicate parsed rows by canonical public identifier (spec §5).

First occurrence wins; later duplicates are reported, not scraped again.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.csv_parser import ParsedRow


@dataclass
class DedupResult:
    unique: list[ParsedRow]
    duplicates: list[dict]  # {public_identifier, name, kept_name}


def dedupe_rows(rows: list[ParsedRow]) -> DedupResult:
    seen: dict[str, ParsedRow] = {}
    unique: list[ParsedRow] = []
    duplicates: list[dict] = []

    for row in rows:
        if not row.public_identifier:
            # unusable rows are handled by the caller; keep them out of dedup
            continue
        key = row.public_identifier
        if key in seen:
            kept = seen[key]
            duplicates.append(
                {
                    "public_identifier": key,
                    "name": f"{row.first_name or ''} {row.last_name or ''}".strip(),
                    "kept_name": f"{kept.first_name or ''} {kept.last_name or ''}".strip(),
                }
            )
            continue
        seen[key] = row
        unique.append(row)

    return DedupResult(unique=unique, duplicates=duplicates)
