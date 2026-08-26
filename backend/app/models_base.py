"""Tiny helpers shared by ORM models."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
