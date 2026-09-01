"""Batched LLM semantic judge for a bounded shortlist (spec §16–§18).

NOT called 1000× per query. For the candidates whose semantic-concept strength
lands in an ambiguous band after deterministic + assertion scoring, we send a
COMPACT evidence packet (current role, relevant past roles, company
classifications, semantic industries/job_families, career summary, relevant
skills, location — not raw JSON) for ~10 candidates at a time and ask the model
to judge ONLY the semantic criteria, with validated TRUE/FALSE/UNKNOWN output.
The model may interpret; it may NOT invent companies/roles/dates/skills.
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.constants import TriState
from app.services.company_intel import company_key
from app.services.llm.router import generate_structured

log = logging.getLogger("app.judge")


class _CriterionVerdict(BaseModel):
    criterion_id: str
    status: str = TriState.UNKNOWN
    match_strength: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: str = ""
    evidence: list[str] = []

    @field_validator("status", mode="before")
    @classmethod
    def _norm_status(cls, v):
        v = str(v).strip().lower()
        return v if v in (TriState.TRUE, TriState.FALSE, TriState.UNKNOWN) else TriState.UNKNOWN


class _PersonVerdict(BaseModel):
    person_id: str
    criteria: list[_CriterionVerdict] = []


class JudgeBatch(BaseModel):
    people: list[_PersonVerdict] = []


_SYSTEM = (
    "You judge whether each person satisfies specific PROFESSIONAL CONCEPTS, using only the "
    "evidence packet provided. You may interpret (a Software Engineer at Google IS technology-"
    "industry experience even if the profile never says 'tech'; a Founding Engineer at a company "
    "classified as a startup DOES satisfy 'now at a startup'). You may NOT invent employers, "
    "roles, dates, skills, education, or company facts not in the packet. "
    "For each concept return status: 'true' (evidence clearly supports it), 'false' (evidence "
    "clearly contradicts it), or 'unknown' (packet is insufficient). match_strength 0-1, "
    "confidence 0-1, a one-line reason, and up to 3 evidence phrases copied from the packet. "
    "Return JSON only."
)


def judge(
    db, query: str, criteria: list, candidates: list, ctx
) -> dict[str, dict[str, dict]]:
    """``criteria``: SearchCriterion list (only semantic ones are judged).
    ``candidates``: [(person, ProfileFacts)]. Returns
    ``{person_id: {criterion_id: verdict_dict}}``."""
    sem_crits = [
        c for c in criteria
        if c.type in ("semantic_concept", "company_category")
    ]
    if not sem_crits or not candidates or not settings.semantic_judge_enabled:
        return {}

    crit_desc = "\n".join(
        f'  - id={c.id} concept="{c.concept or c.value}"'
        + (f' scope={c.scope}' if c.scope else "")
        for c in sem_crits
    )

    out: dict[str, dict[str, dict]] = {}
    batch = settings.semantic_judge_batch_size
    for i in range(0, len(candidates), batch):
        chunk = candidates[i : i + batch]
        packets = [_packet(p, f, ctx) for p, f in chunk]
        user = (
            f"Search query: {query!r}\n\nConcepts to judge (per person):\n{crit_desc}\n\n"
            f"People:\n{json.dumps(packets, ensure_ascii=False)}\n\n"
            'Return {"people":[{"person_id":"...","criteria":[{"criterion_id":"...","status":"true|false|unknown",'
            '"match_strength":0-1,"confidence":0-1,"reason":"...","evidence":["..."]}]}]}'
        )
        result = generate_structured(_SYSTEM, user, JudgeBatch, max_tokens=2600)
        if result is None:
            log.info("semantic judge unavailable — %d candidates left to deterministic scoring", len(chunk))
            break
        verdicts, _prov, _model = result
        for pv in verdicts.people:
            out.setdefault(pv.person_id, {})
            for cv in pv.criteria:
                out[pv.person_id][cv.criterion_id] = cv.model_dump()
    return out


def _packet(person, facts, ctx) -> dict:
    exps = facts.experiences[:8]
    company_cats = {}
    for e in exps:
        row = ctx.company_class.get(company_key(getattr(e, "company_id", None), e.company_name))
        if row and (row.get("is_startup") is not None or row.get("is_big_tech") is not None or row.get("industries")):
            company_cats[e.company_name] = {
                k: row.get(k) for k in ("is_startup", "is_big_tech", "is_technology_company", "industries")
            }
    sem = facts.semantic
    return {
        "person_id": person.id,
        "name": person.full_name,
        "location": person.location_text,
        "current": {"title": person.current_title, "company": person.current_company},
        "experience": [
            {"title": e.position, "company": e.company_name, "is_current": e.is_current,
             "years": f"{e.start_year or '?'}-{e.end_year or ('present' if e.is_current else '?')}",
             "description": (e.description or "")[:220] or None}
            for e in exps
        ],
        "company_classifications": company_cats,
        "industries": sem.get("industries", []),
        "job_families": sem.get("job_families", []),
        "leadership": sem.get("leadership_experience", []),
        "career_summary": sem.get("career_summary"),
        "semantic_assertions": [
            {"concept": a.get("concept"), "evidence": a.get("evidence", [])[:2]}
            for a in sem.get("semantic_assertions", []) if isinstance(a, dict)
        ][:8],
        "skills": [s.skill_name for s in facts.skills][:15],
    }
