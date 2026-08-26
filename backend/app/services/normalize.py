"""Deterministic HarvestAPI JSON → normalized rows (spec §11, §14–§17).

Pure Python. No LLM. Every field access is guarded — HarvestAPI omits or nulls
fields freely, and ``topSkills`` / ``skills`` come back as strings *or* lists
*or* lists of objects depending on the profile.
"""
from __future__ import annotations

import re
from typing import Any

from app.constants import SkillSource

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}


def _s(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return str(v)


def _month(v: Any) -> int | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        n = int(v)
        return n if 1 <= n <= 12 else None
    key = str(v).strip().lower()[:3]
    if key in _MONTHS:
        return _MONTHS[key]
    if key.isdigit():
        n = int(key)
        return n if 1 <= n <= 12 else None
    return None


def _year(v: Any) -> int | None:
    if v is None or v == "":
        return None
    m = re.search(r"(19|20)\d{2}", str(v))
    return int(m.group(0)) if m else None


def _date_parts(node: Any) -> tuple[int | None, int | None, str | None]:
    if not isinstance(node, dict):
        text = _s(node)
        return None, _year(text), text
    return _month(node.get("month")), _year(node.get("year")), _s(node.get("text"))


def _list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _split_skills_string(v: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;|·•\n]+", v) if p.strip()]


def normalize_skill_name(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"\s*\((?:programming language|the|aws|framework|library)\)\s*", " ", n)
    n = n.replace("amazon web services", "aws").replace("microsoft azure", "azure")
    n = re.sub(r"[^a-z0-9+.# ]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


# ─────────────────────────── sections ───────────────────────────


def _experience_rows(raw: dict) -> list[dict]:
    rows: list[dict] = []
    for e in _list(raw.get("experience")):
        if not isinstance(e, dict):
            continue
        sm, sy, st = _date_parts(e.get("startDate") or e.get("start"))
        em, ey, et = _date_parts(e.get("endDate") or e.get("end"))
        end_text = (et or "").lower()
        is_current = (
            e.get("isCurrent") is True
            or (ey is None and (end_text in {"present", ""} or "present" in end_text) and sy is not None)
        )
        rows.append(
            {
                "position": _s(e.get("position") or e.get("title")),
                "company_name": _s(e.get("companyName") or e.get("company")),
                "company_linkedin_url": _s(e.get("companyLinkedinUrl") or e.get("companyLink")),
                "company_id": _s(e.get("companyId")),
                "company_universal_name": _s(e.get("companyUniversalName")),
                "employment_type": _s(e.get("employmentType")),
                "workplace_type": _s(e.get("workplaceType")),
                "location": _s(e.get("location")),
                "start_month": sm,
                "start_year": sy,
                "start_text": st,
                "end_month": em,
                "end_year": ey,
                "end_text": et,
                "is_current": bool(is_current),
                "duration_text": _s(e.get("duration")),
                "description": _s(e.get("description")),
                "skills_json": [_s(x) for x in _list(e.get("skills")) if _s(x)] or None,
            }
        )
    return rows


def _education_rows(raw: dict) -> list[dict]:
    rows: list[dict] = []
    for e in _list(raw.get("education")):
        if not isinstance(e, dict):
            continue
        _, sy, _ = _date_parts(e.get("startDate") or e.get("start"))
        _, ey, _ = _date_parts(e.get("endDate") or e.get("end"))
        period = _s(e.get("period"))
        if not sy and period:
            sy = _year(period.split("-")[0]) if "-" in period else _year(period)
        if not ey and period and "-" in period:
            ey = _year(period.split("-")[-1])
        rows.append(
            {
                "school_name": _s(e.get("schoolName") or e.get("school")),
                "school_linkedin_url": _s(e.get("schoolLinkedinUrl") or e.get("schoolLink")),
                "school_id": _s(e.get("schoolId")),
                "degree": _s(e.get("degree")),
                "field_of_study": _s(e.get("fieldOfStudy")),
                "start_year": sy,
                "end_year": ey,
                "period": period,
                "description": _s(e.get("description")),
                "skills_json": [_s(x) for x in _list(e.get("skills")) if _s(x)] or None,
            }
        )
    return rows


def _skill_rows(raw: dict, experience_rows: list[dict], education_rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}

    def add(name: str, source: str, evidence: str | None = None):
        clean = _s(name)
        if not clean:
            return
        norm = normalize_skill_name(clean)
        if not norm or norm in seen:
            return
        seen[norm] = {
            "skill_name": clean,
            "skill_name_norm": norm,
            "source": source,
            "is_inferred": False,
            "confidence": 1.0,
            "evidence": evidence,
        }

    for sk in _list(raw.get("skills")):
        if isinstance(sk, dict):
            add(sk.get("name") or sk.get("skill"), SkillSource.PROFILE)
        else:
            add(sk, SkillSource.PROFILE)

    ts = raw.get("topSkills")
    if isinstance(ts, str):
        for name in _split_skills_string(ts):
            add(name, SkillSource.PROFILE)
    else:
        for name in _list(ts):
            add(name, SkillSource.PROFILE)

    for exp in experience_rows:
        for name in exp.get("skills_json") or []:
            add(name, SkillSource.EXPERIENCE, evidence=f"listed under experience at {exp.get('company_name') or 'a company'}")
    for edu in education_rows:
        for name in edu.get("skills_json") or []:
            add(name, SkillSource.EDUCATION, evidence=f"listed under education at {edu.get('school_name') or 'a school'}")

    return list(seen.values())


def _location(raw: dict) -> dict:
    loc = raw.get("location") or {}
    if isinstance(loc, str):
        return {"location_text": _s(loc)}
    parsed = loc.get("parsed") or {}
    return {
        "location_text": _s(loc.get("linkedinText") or loc.get("text") or parsed.get("text")),
        "city": _s(parsed.get("city")),
        "state": _s(parsed.get("state")),
        "country": _s(parsed.get("country")),
        "country_code": _s(parsed.get("countryCode") or loc.get("countryCode")),
    }


def _current_from_experience(experience_rows: list[dict]) -> dict:
    for e in experience_rows:
        if e["is_current"]:
            return {
                "current_title": e["position"],
                "current_company": e["company_name"],
                "current_company_linkedin_url": e["company_linkedin_url"],
                "current_start_month": e["start_month"],
                "current_start_year": e["start_year"],
            }
    return {}


# ─────────────────────────── entry point ───────────────────────────


def normalize_profile(raw: dict) -> dict:
    """Return ``{person: {...}, experiences: [...], education: [...], skills: [...]}``."""
    raw = raw or {}
    exp = _experience_rows(raw)
    edu = _education_rows(raw)
    skills = _skill_rows(raw, exp, edu)

    first = _s(raw.get("firstName"))
    last = _s(raw.get("lastName"))
    full = _s(raw.get("name")) or " ".join(p for p in [first, last] if p) or None

    person = {
        "linkedin_id": _s(raw.get("id")),
        "public_identifier": _s(raw.get("publicIdentifier")),
        "first_name": first,
        "last_name": last,
        "full_name": full,
        "headline": _s(raw.get("headline")),
        "about": _s(raw.get("about") or raw.get("summary")),
        "profile_picture_url": _s(raw.get("photo") or raw.get("profilePicture")),
        "connections_count": _int(raw.get("connectionsCount")),
        "followers_count": _int(raw.get("followerCount") or raw.get("followersCount")),
        "open_to_work": _bool(raw.get("openToWork")),
        "hiring": _bool(raw.get("hiring")),
        "premium": _bool(raw.get("premium")),
        "verified": _bool(raw.get("verified")),
        "influencer": _bool(raw.get("influencer")),
        "creator": _bool(raw.get("creator")),
        "registered_at": _s(raw.get("registeredAt")),
    }
    person.update(_location(raw))

    current = _current_from_experience(exp)
    if not current:
        cp = _list(raw.get("currentPosition"))
        if cp and isinstance(cp[0], dict):
            person["current_company"] = _s(cp[0].get("companyName"))
    person.update(current)

    return {"person": person, "experiences": exp, "education": edu, "skills": skills}


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return bool(v)
