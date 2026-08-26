"""Parse a LinkedIn Connections export CSV (spec §4).

The real export has a 2–3 line "Notes:" preamble before the header row:

    Notes:
    "When exporting your connection data, ..."
    <blank>
    First Name,Last Name,URL,Email Address,Company,Position,Connected On
    Jane,Smith,https://www.linkedin.com/in/jane-smith,,Amazon,SWE,01 Jan 2025

We tolerate: missing preamble, missing Email/Company/Position columns, reordered
columns, BOM, and semicolon delimiters. Email is not required. A row is only
usable if it yields a LinkedIn profile URL.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

from app.services.url_normalize import try_normalize

_HEADER_TOKENS = {"first name", "last name", "url", "connected on", "position", "company"}

_ALIASES = {
    "first name": "first_name",
    "firstname": "first_name",
    "last name": "last_name",
    "lastname": "last_name",
    "url": "url",
    "profile url": "url",
    "linkedin url": "url",
    "email address": "email",
    "email": "email",
    "company": "company",
    "current company": "company",
    "position": "position",
    "title": "position",
    "connected on": "connected_on",
}


@dataclass
class ParsedRow:
    first_name: str | None
    last_name: str | None
    email: str | None
    company: str | None
    position: str | None
    connected_on: str | None
    raw_url: str | None
    linkedin_url: str | None
    public_identifier: str | None


@dataclass
class ParseResult:
    rows: list[ParsedRow] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)  # {row_number, reason, data}
    total_data_rows: int = 0

    @property
    def usable(self) -> list[ParsedRow]:
        return [r for r in self.rows if r.linkedin_url]


class CSVParseError(ValueError):
    pass


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def _find_header_index(lines: list[str], delimiter: str) -> int:
    for i, line in enumerate(lines[:15]):
        cells = {c.strip().strip('"').lower() for c in line.split(delimiter)}
        if len(cells & _HEADER_TOKENS) >= 2:
            return i
    return 0


def parse_connections_csv(content: bytes | str) -> ParseResult:
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise CSVParseError("file is empty")

    lines = [ln for ln in text.split("\n")]
    delimiter = _sniff_delimiter("\n".join(lines[:20]))
    header_idx = _find_header_index(lines, delimiter)

    body = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(body), delimiter=delimiter)
    if not reader.fieldnames:
        raise CSVParseError("no header row found")

    colmap: dict[str, str] = {}
    for name in reader.fieldnames:
        key = (name or "").strip().strip('"').lower()
        if key in _ALIASES:
            colmap[name] = _ALIASES[key]

    if "url" not in colmap.values():
        raise CSVParseError(
            "CSV has no recognizable LinkedIn URL column "
            f"(headers: {reader.fieldnames})"
        )

    result = ParseResult()
    for n, rowdict in enumerate(reader, start=header_idx + 2):
        mapped = {v: (rowdict.get(k) or "").strip() for k, v in colmap.items()}
        if not any(mapped.values()):
            continue
        result.total_data_rows += 1

        raw_url = mapped.get("url") or None
        canonical, public_id = try_normalize(raw_url) if raw_url else (None, None)

        parsed = ParsedRow(
            first_name=mapped.get("first_name") or None,
            last_name=mapped.get("last_name") or None,
            email=mapped.get("email") or None,
            company=mapped.get("company") or None,
            position=mapped.get("position") or None,
            connected_on=mapped.get("connected_on") or None,
            raw_url=raw_url,
            linkedin_url=canonical,
            public_identifier=public_id,
        )
        result.rows.append(parsed)
        if not canonical:
            result.skipped.append(
                {
                    "row_number": n,
                    "reason": "missing or invalid LinkedIn URL",
                    "name": f"{parsed.first_name or ''} {parsed.last_name or ''}".strip(),
                }
            )

    if result.total_data_rows == 0:
        raise CSVParseError("no data rows after the header")

    return result
