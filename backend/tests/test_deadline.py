"""hardening PART 14 — the per-search wall-clock budget primitive."""
from __future__ import annotations

from app.services.deadline import Deadline


def test_unlimited_when_seconds_is_zero_or_none():
    for seconds in (0, -1, None):
        d = Deadline(seconds)
        assert d.remaining() is None
        assert d.expired() is False


def test_expires_once_seconds_elapse():
    d = Deadline(0.01)
    assert d.expired() is False
    import time
    time.sleep(0.02)
    assert d.expired() is True
    assert d.remaining() < 0


def test_as_dict_reports_seconds_elapsed_and_reached():
    d = Deadline(100)
    out = d.as_dict()
    assert out["seconds"] == 100
    assert out["reached"] is False
    assert isinstance(out["elapsed_ms"], int) and out["elapsed_ms"] >= 0
