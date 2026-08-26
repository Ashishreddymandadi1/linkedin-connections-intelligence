"""Deterministic profile completeness / data-confidence score (spec §19).

This is NOT the match score. It answers "how much of this person's profile did
we actually get?" and is shown separately in the UI.
"""
from __future__ import annotations

#: field -> weight (sum = 100)
_WEIGHTS = {
    "name": 10,
    "headline": 12,
    "about": 14,
    "location": 8,
    "current_position": 14,
    "experience": 16,
    "education": 12,
    "skills": 10,
    "extras": 4,  # certifications / projects / publications / languages
}


def compute_completeness(normalized: dict, raw: dict | None = None) -> tuple[int, dict]:
    person = normalized.get("person", {})
    exp = normalized.get("experiences", [])
    edu = normalized.get("education", [])
    skills = normalized.get("skills", [])
    raw = raw or {}

    checks = {
        "name": bool(person.get("full_name")),
        "headline": bool(person.get("headline")),
        "about": bool(person.get("about")),
        "location": bool(person.get("location_text")),
        "current_position": bool(person.get("current_title") or person.get("current_company")),
        "experience": len(exp) > 0,
        "education": len(edu) > 0,
        "skills": len(skills) > 0,
        "extras": any(
            raw.get(k)
            for k in ("certifications", "projects", "publications", "languages", "honorsAndAwards", "courses")
        ),
    }
    score = sum(_WEIGHTS[k] for k, ok in checks.items() if ok)
    detail = {**checks, "profile_completeness": score}
    return score, detail
