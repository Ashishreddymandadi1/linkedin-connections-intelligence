"""Deterministic, evidence-backed match scoring (spec §35–§37).

No LLM decides a score. For every criterion we compute ``match_strength ∈ [0,1]``
in code, multiply by the criterion weight, and attach a structured evidence list
pointing at real rows. ``match_score = min(100, Σ criterion_score)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import repositories as repo
from app.constants import CriterionType, SkillSource
from app.models import Person
from app.schemas import EvidenceItem, ParsedSearchQuery, ScoreComponent, SearchCriterion
from app.services.matching import (
    company_matches,
    norm,
    norm_company,
    phrase_matches,
    seniority_rank,
    token_overlap,
)

_REQUIRED_MIN = 0.15
_MATCHED_MIN = 0.2


@dataclass
class ProfileFacts:
    person: Person
    experiences: list
    education: list
    skills: list
    semantic: dict
    embedding: bytes | None
    certifications: list = field(default_factory=list)
    languages: list = field(default_factory=list)
    publications: list = field(default_factory=list)

    @property
    def skill_norms(self) -> dict[str, object]:
        return {s.skill_name_norm: s for s in self.skills}


@dataclass
class ScoredCandidate:
    person: Person
    match_score: float
    components: list[ScoreComponent]
    evidence: list[EvidenceItem]
    matched_criteria: list[str] = field(default_factory=list)
    excluded_reason: str | None = None


def load_facts(db: Session, person: Person) -> ProfileFacts:
    sem = repo.get_semantic(db, person.id)
    emb = None
    from app.models import ProfileEmbedding  # local import to avoid cycle at module load

    row = db.query(ProfileEmbedding).filter(ProfileEmbedding.person_id == person.id).first()
    if row:
        emb = row.vector
    return ProfileFacts(
        person=person,
        experiences=repo.get_experiences(db, person.id),
        education=repo.get_education(db, person.id),
        skills=repo.get_skills(db, person.id),
        semantic=(sem.data if sem and sem.data else {}),
        embedding=emb,
        certifications=repo.get_certifications(db, person.id),
        languages=repo.get_languages(db, person.id),
        publications=repo.get_publications(db, person.id),
    )


# ─────────────────────── per-criterion strategies ───────────────────────


def _score_company(facts: ProfileFacts, value: str, *, want_current: bool | None) -> tuple[float, list[EvidenceItem]]:
    best = 0.0
    ev: list[EvidenceItem] = []
    for e in facts.experiences:
        if not company_matches(e.company_name, value):
            continue
        item = EvidenceItem(
            type="experience",
            text=f"{e.position or 'Role'} at {e.company_name}"
            + (f" ({e.start_year}–{e.end_year or 'present'})" if e.start_year else ""),
            detail={
                "company": e.company_name,
                "title": e.position,
                "start_year": e.start_year,
                "end_year": e.end_year,
                "is_current": e.is_current,
            },
        )
        if want_current is True:
            strength = 1.0 if e.is_current else 0.35
        elif want_current is False:
            strength = 1.0 if not e.is_current else 0.55
        else:
            strength = 1.0
        if strength > best:
            best = strength
            ev = [item]
    return best, ev


def _score_skill(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    target = norm(value)
    # 1. explicit LinkedIn skill
    for s in facts.skills:
        if s.skill_name_norm == target or phrase_matches(s.skill_name_norm, target):
            if not s.is_inferred and s.source in (
                SkillSource.PROFILE,
                SkillSource.EXPERIENCE,
                SkillSource.EDUCATION,
            ):
                return 1.0, [EvidenceItem(type="skill", text=f"{s.skill_name} — listed on LinkedIn", detail={"source": s.source})]
    # 2. semantic explicit / inferred
    sem = facts.semantic
    for sk in sem.get("explicit_skills", []):
        if phrase_matches(sk, value):
            return 0.9, [EvidenceItem(type="skill", text=f"{sk} — stated on profile", detail={"source": "semantic_explicit"})]
    for isk in sem.get("inferred_skills", []):
        name = isk.get("skill") if isinstance(isk, dict) else None
        if name and phrase_matches(name, value):
            conf = float(isk.get("confidence", 0.7))
            return min(0.9, 0.5 + conf * 0.4), [
                EvidenceItem(
                    type="semantic",
                    text=f"{name} — inferred: {isk.get('evidence', '')[:160]}",
                    detail={"confidence": conf, "inferred": True},
                )
            ]
    # 3. mentioned in experience descriptions / listed experience skills
    for e in facts.experiences:
        exp_skill_text = " ".join(e.skills_json) if e.skills_json else ""
        if phrase_matches(e.description, value) or phrase_matches(exp_skill_text, value):
            return 0.6, [EvidenceItem(type="experience", text=f"referenced in the {e.company_name or 'role'} description", detail={})]
    for dom in sem.get("technical_domains", []) + sem.get("domain_expertise", []):
        if phrase_matches(dom, value):
            return 0.6, [EvidenceItem(type="semantic", text=f"domain: {dom}", detail={"inferred": True})]
    hay = " ".join(filter(None, [facts.person.headline, facts.person.about]))
    if phrase_matches(hay, value):
        return 0.5, [EvidenceItem(type="headline", text="mentioned in headline / about", detail={})]
    return 0.0, []


def _score_domain(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    sem = facts.semantic
    for dom in sem.get("technical_domains", []) + sem.get("domain_expertise", []) + sem.get("job_families", []):
        if phrase_matches(dom, value):
            return 0.9, [EvidenceItem(type="semantic", text=f"domain expertise: {dom}", detail={"inferred": True})]
    for e in facts.experiences:
        if phrase_matches(e.description, value):
            return 0.75, [EvidenceItem(type="experience", text=f"{e.company_name or 'a role'}: {(e.description or '')[:160]}", detail={})]
    hay = " ".join(filter(None, [facts.person.headline, facts.person.about, *(sem.get("searchable_keywords") or [])]))
    if phrase_matches(hay, value):
        return 0.6, [EvidenceItem(type="headline", text="referenced in profile text", detail={})]
    # fall back to skill logic
    return _score_skill(facts, value)


def _title_strength(title: str | None, value: str) -> float:
    if not title:
        return 0.0
    if phrase_matches(title, value):
        return 1.0
    ov = token_overlap(title, value)
    return 0.6 if ov >= 0.34 else 0.0


def _score_title(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    p = facts.person
    cur = _title_strength(p.current_title, value)
    if cur > 0:
        return cur, [EvidenceItem(type="experience", text=f"current title: {p.current_title}", detail={"current": True})]
    best = 0.0
    ev: list[EvidenceItem] = []
    for e in facts.experiences:
        s = _title_strength(e.position, value) * (1.0 if e.is_current else 0.9)
        if s > best:
            best, ev = s, [EvidenceItem(type="experience", text=f"{e.position} at {e.company_name}", detail={})]
    return best, ev


def _score_education(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    v = norm(value)
    for ed in facts.education:
        if ed.school_name and phrase_matches(ed.school_name, value):
            return 1.0, [EvidenceItem(type="education", text=f"{ed.degree or 'Studied'} at {ed.school_name}", detail={"school": ed.school_name})]
    for ed in facts.education:
        if ed.field_of_study and phrase_matches(ed.field_of_study, value):
            return 0.9, [EvidenceItem(type="education", text=f"{ed.field_of_study} at {ed.school_name or 'a school'}", detail={})]
        if ed.degree and (v in norm(ed.degree) or norm(ed.degree) in v):
            return 0.85, [EvidenceItem(type="education", text=f"{ed.degree} — {ed.school_name or ''}".strip(" —"), detail={})]
    for kw in facts.semantic.get("education_keywords", []):
        if phrase_matches(kw, value):
            return 0.7, [EvidenceItem(type="semantic", text=f"education: {kw}", detail={"inferred": True})]
    return 0.0, []


def _score_location(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    p = facts.person
    hay = " ".join(filter(None, [p.location_text, p.city, p.state, p.country]))
    if phrase_matches(hay, value):
        return 1.0, [EvidenceItem(type="headline", text=f"located in {p.location_text}", detail={})]
    return 0.0, []


def _score_seniority(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    want = seniority_rank(value)
    have = seniority_rank(facts.semantic.get("seniority_level")) or seniority_rank(facts.person.current_title)
    if want is None or have is None:
        return 0.0, []
    gap = abs(want - have)
    strength = 1.0 if gap == 0 else 0.6 if gap == 1 else 0.0
    if strength:
        return strength, [
            EvidenceItem(type="semantic", text=f"seniority: {facts.semantic.get('seniority_level') or facts.person.current_title}", detail={})
        ]
    return 0.0, []


def _score_certification(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    for c in facts.certifications:
        hay = " ".join(filter(None, [c.name, c.issuer]))
        if phrase_matches(hay, value) or phrase_matches(value, c.name or ""):
            return 1.0, [
                EvidenceItem(
                    type="certification",
                    text=f"{c.name}" + (f" — {c.issuer}" if c.issuer else "") + " (certification)",
                    detail={"issuer": c.issuer, "issued_at": c.issued_at},
                )
            ]
    # a bare cert-provider query ("AWS certification") also matches issuer text
    for c in facts.certifications:
        toks = [t for t in norm(value).split() if t not in {"certification", "certified", "cert", "certificate"}]
        if toks and all(t in norm(f"{c.name} {c.issuer}") for t in toks):
            return 0.9, [EvidenceItem(type="certification", text=f"{c.name} (certification)", detail={})]
    return 0.0, []


def _score_language(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    wanted = {t for t in norm(value).replace(" or ", " ").replace(" and ", " ").split() if len(t) > 1}
    wanted -= {"speaks", "speak", "language", "languages", "fluent", "native"}
    for lang in facts.languages:
        if lang.name_norm in wanted or any(w in lang.name_norm for w in wanted):
            return 1.0, [
                EvidenceItem(
                    type="language",
                    text=f"speaks {lang.name}" + (f" ({lang.proficiency})" if lang.proficiency else ""),
                    detail={"proficiency": lang.proficiency},
                )
            ]
    return 0.0, []


def _score_publication(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    if not facts.publications:
        return 0.0, []
    generic = {"published", "publication", "publications", "research", "paper", "papers", "author", "wrote"}
    topical = [t for t in norm(value).split() if t not in generic and len(t) > 2]
    if not topical:
        p = facts.publications[0]
        return 0.85, [EvidenceItem(type="publication", text=f'published "{(p.title or "")[:90]}"', detail={"count": len(facts.publications)})]
    for p in facts.publications:
        hay = norm(f"{p.title} {p.description or ''}")
        if sum(1 for t in topical if t in hay) / len(topical) >= 0.5:
            return 1.0, [EvidenceItem(type="publication", text=f'published "{(p.title or "")[:90]}"', detail={})]
    return 0.4, [EvidenceItem(type="publication", text=f"has {len(facts.publications)} publication(s)", detail={})]


def _score_keyword(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    parts = [
        facts.person.headline,
        facts.person.about,
        " ".join(e.description or "" for e in facts.experiences),
        " ".join(s.skill_name for s in facts.skills),
        " ".join(facts.semantic.get("searchable_keywords", [])),
    ]
    hay = " ".join(filter(None, parts))
    if phrase_matches(hay, value):
        return 0.8, [EvidenceItem(type="headline", text=f"profile mentions '{value}'", detail={})]
    return 0.0, []


_STRATEGIES = {
    CriterionType.CURRENT_COMPANY: lambda f, v: _score_company(f, v, want_current=True),
    CriterionType.PAST_COMPANY: lambda f, v: _score_company(f, v, want_current=False),
    CriterionType.SKILL: _score_skill,
    CriterionType.DOMAIN: _score_domain,
    CriterionType.TITLE: _score_title,
    CriterionType.EDUCATION: _score_education,
    CriterionType.LOCATION: _score_location,
    CriterionType.SENIORITY: _score_seniority,
    CriterionType.KEYWORD: _score_keyword,
    CriterionType.CERTIFICATION: _score_certification,
    CriterionType.LANGUAGE: _score_language,
    CriterionType.PUBLICATION: _score_publication,
}


def _score_one(facts: ProfileFacts, crit: SearchCriterion) -> tuple[float, list[EvidenceItem]]:
    strat = _STRATEGIES.get(crit.type, _score_keyword)
    try:
        return strat(facts, crit.value)
    except Exception:  # noqa: BLE001 — a matcher bug must not kill a search
        return 0.0, []


def score_candidate(facts: ProfileFacts, parsed: ParsedSearchQuery) -> ScoredCandidate:
    components: list[ScoreComponent] = []
    all_evidence: list[EvidenceItem] = []
    matched: list[str] = []
    total = 0.0

    for crit in parsed.criteria:
        strength, ev = _score_one(facts, crit)
        score = round(crit.weight * strength, 2)
        components.append(
            ScoreComponent(
                criterion=_label(crit),
                criterion_id=crit.id,
                type=crit.type,
                weight=round(crit.weight, 2),
                match_strength=round(strength, 3),
                score=score,
                required=crit.required,
                evidence=ev,
            )
        )
        if crit.required and strength < _REQUIRED_MIN:
            return ScoredCandidate(
                person=facts.person,
                match_score=0.0,
                components=components,
                evidence=[],
                excluded_reason=f"required criterion not met: {crit.value}",
            )
        total += score
        if strength >= _MATCHED_MIN:
            matched.append(crit.value)
            all_evidence.extend(ev)

    return ScoredCandidate(
        person=facts.person,
        match_score=round(min(100.0, total), 1),
        components=components,
        evidence=_dedupe_evidence(all_evidence),
        matched_criteria=matched,
    )


def _label(c: SearchCriterion) -> str:
    pretty = {
        CriterionType.CURRENT_COMPANY: f"Currently at {c.value}",
        CriterionType.PAST_COMPANY: f"Previously at {c.value}",
        CriterionType.SKILL: c.value,
        CriterionType.DOMAIN: c.value,
        CriterionType.TITLE: f"{c.value} role",
        CriterionType.EDUCATION: f"Studied / attended {c.value}",
        CriterionType.LOCATION: f"Located in {c.value}",
        CriterionType.SENIORITY: f"{c.value.title()} level",
        CriterionType.KEYWORD: c.value,
        CriterionType.CERTIFICATION: f"{c.value} certification",
        CriterionType.LANGUAGE: f"Speaks {c.value}",
        CriterionType.PUBLICATION: f"Published on {c.value}" if c.value else "Has publications",
    }
    return pretty.get(c.type, c.value)


def _dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen = set()
    out = []
    for it in items:
        key = (it.type, it.text)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out
