"""Canonicalize LinkedIn profile URLs (spec §5).

    https://linkedin.com/in/john-smith/
    https://www.linkedin.com/in/john-smith?utm_source=x
    https://www.linkedin.com/in/john-smith/?trk=abc
        -> https://www.linkedin.com/in/john-smith

Rules: force ``https://www.linkedin.com`` host, keep only the ``/in/<slug>``
path, drop query + fragment + trailing slash, lower-case the slug, URL-decode it.
The public identifier (the ``<slug>``) is the dedup key.
"""
from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

_CANONICAL_HOST = "https://www.linkedin.com"
# slug after decode + lower-case: no whitespace / path / query / fragment chars
_SLUG_BAD = re.compile(r"[\s/?#\\]")


class InvalidLinkedInURL(ValueError):
    pass


def extract_public_identifier(url: str) -> str | None:
    """Return the ``/in/<slug>`` slug (lower-cased, decoded) or ``None``."""
    if not url or not isinstance(url, str):
        return None
    raw = url.strip()
    if not raw:
        return None
    if raw.startswith("//"):
        raw = "https:" + raw
    if not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw

    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if "linkedin.com" not in host:
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0].lower() in {"in", "pub"}:
        slug = unquote(parts[1]).strip().lower().rstrip("/")
        if slug and not _SLUG_BAD.search(slug):
            return slug
    return None


def normalize_linkedin_url(url: str) -> str:
    """Return the canonical URL. Raises ``InvalidLinkedInURL`` if unusable."""
    slug = extract_public_identifier(url)
    if not slug:
        raise InvalidLinkedInURL(f"not a usable LinkedIn profile URL: {url!r}")
    return f"{_CANONICAL_HOST}/in/{slug}"


def try_normalize(url: str) -> tuple[str | None, str | None]:
    """Non-raising variant: ``(canonical_url, public_id)`` or ``(None, None)``."""
    slug = extract_public_identifier(url)
    if not slug:
        return None, None
    return f"{_CANONICAL_HOST}/in/{slug}", slug
