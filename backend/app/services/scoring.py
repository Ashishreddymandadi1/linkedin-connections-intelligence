"""Deterministic + semantic evidence-backed match scoring (spec §14–§20, §27–§28).

Hybrid (spec: "Use deterministic matching for facts. Use semantic reasoning for
concepts."):

* EXACT FACTS (company / school / location / certification / language / title /
  skill) — computed in code against normalized rows. ``match_strength ∈ [0,1]``,
  attached evidence points at a real row. A ``required`` fact that isn't found is
  a hard exclude (we have the person's complete LinkedIn experience list, so
  "not found" is a real negative).

* SEMANTIC CONCEPTS (semantic_concept / company_category) — evaluated strongest→
  weakest: profile semantic assertion → company classification against a real
  experience row → semantic fields (industries / job_families / …) → concept-vs-
  career cross-encoder. Returns a TRUE / FALSE / UNKNOWN status. A ``required``
  semantic concept excludes ONLY on a confident FALSE — never on UNKNOWN
  (spec §15/§28: missing data ≠ verified absence).

No LLM decides a numeric score here. (An optional batched LLM *judge* runs
upstream in search_service for ambiguous semantic concepts and feeds its
verdict back in via ``ScoringContext.judge_results``.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import repositories as repo
from app.config import settings
from app.constants import CriterionType, Operator, Scope, SkillSource, TriState
from app.models import Person
from app.schemas import EvidenceItem, ParsedSearchQuery, ScoreComponent, SearchCriterion
from app.services import career_chronology as _career
from app.services.company_intel import company_key
from app.services.geo import expand_values, location_matches
from app.services.matching import (
    company_matches,
    concept_overlap,
    experience_weight,
    is_cxo_title,
    norm,
    phrase_matches,
    seniority_rank,
    token_overlap,
)

_REQUIRED_MIN = 0.15
_MATCHED_MIN = 0.2
#: types evaluated as meaning (tri-state), never phrase matches (V4 §6/§29)
_SEMANTIC_TYPES = {
    CriterionType.SEMANTIC_CONCEPT, CriterionType.COMPANY_CATEGORY,
    CriterionType.PROFESSIONAL_CONCEPT, CriterionType.INDUSTRY_EXPERIENCE,
    CriterionType.ROLE_FUNCTION, CriterionType.CAREER_TRANSITION,
    CriterionType.YEARS_EXPERIENCE,
}


@dataclass
class ScoringContext:
    """Per-search signals shared across candidates."""

    query_embedding: bytes | None = None
    reranker_scores: dict[str, float] = field(default_factory=dict)
    company_ids_by_criterion: dict[str, set[str]] = field(default_factory=dict)
    #: company_key -> classification dict (from company_intel.get_or_classify)
    company_class: dict[str, dict] = field(default_factory=dict)
    #: person_id -> {criterion_id: {"status","match_strength","confidence","reason","evidence"}}
    judge_results: dict[str, dict[str, dict]] = field(default_factory=dict)


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


def load_facts(db: Session, person: Person, facts_cache: dict | None = None) -> ProfileFacts:
    """Build a ProfileFacts. ``facts_cache`` (from repositories.bulk_*) avoids
    the per-candidate N+1 (spec §31) — pass ``{"experiences": {...}, ...}``."""
    fc = facts_cache or {}
    pid = person.id

    if fc:
        sem_row = fc.get("semantics", {}).get(pid)
        return ProfileFacts(
            person=person,
            experiences=fc.get("experiences", {}).get(pid, []),
            education=fc.get("education", {}).get(pid, []),
            skills=fc.get("skills", {}).get(pid, []),
            semantic=(sem_row.data if sem_row and sem_row.data else {}),
            embedding=fc.get("embeddings", {}).get(pid),
            certifications=fc.get("certifications", {}).get(pid, []),
            languages=fc.get("languages", {}).get(pid, []),
            publications=fc.get("publications", {}).get(pid, []),
        )

    sem = repo.get_semantic(db, pid)
    from app.models import ProfileEmbedding

    row = db.query(ProfileEmbedding).filter(ProfileEmbedding.person_id == pid).first()
    return ProfileFacts(
        person=person,
        experiences=repo.get_experiences(db, pid),
        education=repo.get_education(db, pid),
        skills=repo.get_skills(db, pid),
        semantic=(sem.data if sem and sem.data else {}),
        embedding=row.vector if row else None,
        certifications=repo.get_certifications(db, pid),
        languages=repo.get_languages(db, pid),
        publications=repo.get_publications(db, pid),
    )


# ─────────────────────── exact-fact strategies ───────────────────────


def _experiences_in_scope(experiences: list, scope: str | None) -> list:
    if scope in (Scope.CURRENT, Scope.CURRENT_COMPANY):
        return [e for e in experiences if e.is_current]
    if scope in (Scope.PAST, Scope.PAST_COMPANY):
        return [e for e in experiences if not e.is_current]
    return list(experiences)


def _score_company(
    facts: ProfileFacts, value: str, *, want_current: bool | None, resolved_ids: set[str] | None = None
) -> tuple[float, list[EvidenceItem]]:
    best = 0.0
    ev: list[EvidenceItem] = []
    recency_on = settings.recency_weighting_enabled
    for e in facts.experiences:
        by_id = bool(resolved_ids) and bool(e.company_id) and e.company_id in resolved_ids
        if not by_id and not company_matches(e.company_name, value):
            continue
        item = EvidenceItem(
            type="experience",
            text=f"{e.position or 'Role'} at {e.company_name}"
            + (f" ({e.start_year}–{e.end_year or 'present'})" if e.start_year else "")
            + (" — verified company" if by_id else ""),
            detail={
                "company": e.company_name, "title": e.position, "start_year": e.start_year,
                "end_year": e.end_year, "is_current": e.is_current, "verified": by_id,
            },
        )
        if want_current is True:
            base = 1.0 if e.is_current else 0.35
        elif want_current is False:
            base = 1.0 if not e.is_current else 0.55
        else:
            base = 1.0
        strength = base * experience_weight(e, enabled=recency_on)
        if strength > best:
            best, ev = strength, [item]
    return best, ev


def _score_skill(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    target = norm(value)
    for s in facts.skills:
        if s.skill_name_norm == target or phrase_matches(s.skill_name_norm, target):
            if not s.is_inferred and s.source in (SkillSource.PROFILE, SkillSource.EXPERIENCE, SkillSource.EDUCATION):
                return 1.0, [EvidenceItem(type="skill", text=f"{s.skill_name} — listed on LinkedIn", detail={"source": s.source})]
    sem = facts.semantic
    for sk in sem.get("explicit_skills", []):
        if phrase_matches(sk, value):
            return 0.9, [EvidenceItem(type="skill", text=f"{sk} — stated on profile", detail={"source": "semantic_explicit"})]
    for isk in sem.get("inferred_skills", []):
        name = isk.get("skill") if isinstance(isk, dict) else None
        if name and phrase_matches(name, value):
            conf = float(isk.get("confidence", 0.7))
            return min(0.9, 0.5 + conf * 0.4), [
                EvidenceItem(type="semantic", text=f"{name} — inferred: {isk.get('evidence', '')[:160]}", detail={"confidence": conf, "inferred": True})
            ]
    for e in facts.experiences:
        exp_skill_text = " ".join(e.skills_json) if e.skills_json else ""
        if phrase_matches(e.description, value) or phrase_matches(exp_skill_text, value):
            w = experience_weight(e, enabled=settings.recency_weighting_enabled)
            return round(0.6 * w, 3), [EvidenceItem(type="experience", text=f"referenced in the {e.company_name or 'role'} description", detail={})]
    for dom in sem.get("technical_domains", []) + sem.get("domain_expertise", []):
        if phrase_matches(dom, value):
            return 0.6, [EvidenceItem(type="semantic", text=f"domain: {dom}", detail={"inferred": True})]
    hay = " ".join(filter(None, [facts.person.headline, facts.person.about]))
    if phrase_matches(hay, value):
        return 0.5, [EvidenceItem(type="headline", text="mentioned in headline / about", detail={})]
    return 0.0, []


def _score_domain(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    # legacy type — route to the concept evaluator so it uses the full semantic
    # profile instead of phrase_matches (spec §6)
    strength, ev, _status = _score_semantic_concept(facts, _synthetic_crit(value), ScoringContext())
    if strength > 0:
        return strength, ev
    return _score_skill(facts, value)


def _title_strength(title: str | None, value: str) -> float:
    if not title:
        return 0.0
    if phrase_matches(title, value):
        return 1.0
    return 0.6 if token_overlap(title, value) >= 0.34 else 0.0


def _score_title(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    p = facts.person
    cur = _title_strength(p.current_title, value)
    if cur > 0:
        return cur, [EvidenceItem(type="experience", text=f"current title: {p.current_title}", detail={"current": True})]
    best, ev = 0.0, []
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
    fields = [p.location_text, p.city, p.state, p.country]
    for candidate in expand_values([value]):
        if location_matches(fields, candidate):
            return 1.0, [EvidenceItem(type="location", text=f"located in {p.location_text or candidate}", detail={"matched": candidate})]
    return 0.0, []


def _score_seniority(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    v = norm(value)
    title = facts.person.current_title
    # "CXO"/executive — accept any real C-suite title, not a rank-gap game
    if "cxo" in v or v in {"executive", "c-suite", "c-level", "chief"}:
        if is_cxo_title(title) or seniority_rank(facts.semantic.get("seniority_level")) == 8:
            return 1.0, [EvidenceItem(type="experience", text=f"executive title: {title or facts.semantic.get('seniority_level')}", detail={})]
        rank = seniority_rank(facts.semantic.get("seniority_level")) or seniority_rank(title)
        return (0.5 if rank == 7 else 0.25 if rank == 6 else 0.0), (
            [EvidenceItem(type="semantic", text=f"seniority: {facts.semantic.get('seniority_level') or title}", detail={})] if rank and rank >= 6 else []
        )
    want = seniority_rank(value)
    have = seniority_rank(facts.semantic.get("seniority_level")) or seniority_rank(title)
    if want is None or have is None:
        return 0.0, []
    gap = abs(want - have)
    strength = 1.0 if gap == 0 else 0.6 if gap == 1 else 0.0
    if strength:
        return strength, [EvidenceItem(type="semantic", text=f"seniority: {facts.semantic.get('seniority_level') or title}", detail={})]
    return 0.0, []


def _score_certification(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    for c in facts.certifications:
        hay = " ".join(filter(None, [c.name, c.issuer]))
        if phrase_matches(hay, value) or phrase_matches(value, c.name or ""):
            return 1.0, [EvidenceItem(type="certification", text=f"{c.name}" + (f" — {c.issuer}" if c.issuer else "") + " (certification)", detail={"issuer": c.issuer, "issued_at": c.issued_at})]
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
            return 1.0, [EvidenceItem(type="language", text=f"speaks {lang.name}" + (f" ({lang.proficiency})" if lang.proficiency else ""), detail={"proficiency": lang.proficiency})]
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
        facts.person.headline, facts.person.about,
        " ".join(e.description or "" for e in facts.experiences),
        " ".join(s.skill_name for s in facts.skills),
        " ".join(facts.semantic.get("searchable_keywords", [])),
    ]
    if phrase_matches(" ".join(filter(None, parts)), value):
        return 0.8, [EvidenceItem(type="headline", text=f"profile mentions '{value}'", detail={})]
    return 0.0, []


# ─────────────────────── semantic-concept strategies ───────────────────────


def _synthetic_crit(concept: str) -> SearchCriterion:
    return SearchCriterion(id="_syn", type=CriterionType.SEMANTIC_CONCEPT, concept=concept, weight=1)


def _career_snippet(facts: ProfileFacts) -> str:
    bits = [facts.person.headline or ""]
    for e in facts.experiences[:6]:
        bits.append(f"{e.position or ''} at {e.company_name or ''}. {(e.description or '')[:200]}")
    sem = facts.semantic
    bits.append(" ".join(sem.get("industries", []) + sem.get("job_families", []) + sem.get("domain_expertise", [])))
    if sem.get("career_summary"):
        bits.append(sem["career_summary"])
    return "  ".join(b for b in bits if b.strip())[:1500]


_CATEGORY_FIELD = {
    "startup": "is_startup", "early stage": "is_startup", "early-stage": "is_startup",
    "big tech": "is_big_tech", "big technology": "is_big_tech", "faang": "is_big_tech",
    "major technology": "is_big_tech", "large tech": "is_big_tech",
    "tech": "is_technology_company", "technology": "is_technology_company",
    "software": "is_technology_company", "tech company": "is_technology_company",
}


def _category_field(concept: str) -> str | None:
    c = norm(concept)
    for key, fld in sorted(_CATEGORY_FIELD.items(), key=lambda kv: -len(kv[0])):
        if key in c:
            return fld
    return None


def _score_company_category(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str]:
    concept = (crit.concept or crit.value or "").strip()
    field_name = _category_field(concept)
    scoped = _experiences_in_scope(facts.experiences, crit.scope)

    saw_false = False
    for e in scoped:
        row = ctx.company_class.get(
            company_key(getattr(e, "company_id", None), getattr(e, "company_name", None))
        )
        if not row:
            continue
        val = row.get(field_name) if field_name else None
        if val is True:
            return 1.0, [
                EvidenceItem(
                    type="company_inference",
                    text=f"{e.company_name} classified as {concept}"
                    + (f" (“{row.get('reason')}”)" if row.get("reason") else ""),
                    detail={"confidence": row.get("confidence"), "provenance": row.get("provenance"),
                            "company": e.company_name, "role": e.position},
                )
            ], TriState.TRUE
        if val is False:
            saw_false = True
        # loose category / industry text fallback when the concept isn't one of
        # the mapped booleans (e.g. "consulting firm", "healthcare tech")
        if field_name is None:
            for x in (row.get("industries") or []) + (row.get("categories") or []):
                if concept_overlap(x, concept) >= 0.5:
                    return 0.85, [
                        EvidenceItem(type="company_inference", text=f"{e.company_name} is in {x}",
                                     detail={"confidence": row.get("confidence"), "provenance": row.get("provenance")})
                    ], TriState.TRUE

    if saw_false:
        return 0.0, [
            EvidenceItem(type="company_inference", text=f"employer(s) in scope are not classified as {concept}", detail={})
        ], TriState.FALSE
    return 0.0, [], TriState.UNKNOWN  # not classified yet / ambiguous — NOT a verified false


def _score_semantic_concept(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str]:
    concept = (crit.concept or crit.value or "").strip()
    if not concept:
        return 0.0, [], TriState.UNKNOWN
    sem = facts.semantic

    best_strength, best_ev, best_status = 0.0, [], TriState.UNKNOWN

    # 0. experience-level semantics (V4 §29 — most specific, scope-aware). For a
    #    ROLE_FUNCTION criterion match role_function/professional_domain/role_domains;
    #    for INDUSTRY_EXPERIENCE match employer_industries.
    want_industry = crit.type == CriterionType.INDUSTRY_EXPERIENCE
    exp_sem = _career.exp_semantics_by_id(sem)
    if exp_sem:
        for es in exp_sem.values():
            fields = (["employer_industries", "employer_categories"] if want_industry
                      else ["role_function", "professional_domain", "role_domains"])
            hit = 0.0
            for f in fields:
                v = es.get(f)
                for item in ([v] if isinstance(v, str) else (v or [])):
                    hit = max(hit, concept_overlap(str(item), concept))
            if hit >= 0.55:
                strength = min(0.9, 0.55 + hit * 0.35) * (0.6 + 0.4 * float(es.get("confidence", 0.6)))
                if strength > best_strength:
                    best_strength, best_status = strength, TriState.TRUE
                    best_ev = [EvidenceItem(type="semantic",
                                            text=f"{'industry' if want_industry else 'role'}: "
                                                 f"{es.get('role_function') or (es.get('employer_industries') or ['?'])[0]}",
                                            detail={"experience_id": es.get("experience_id"), "inferred": True})]

    # 1. a matching semantic assertion (strong — LLM-derived + evidence + source ids)
    for a in sem.get("semantic_assertions", []):
        if not isinstance(a, dict):
            continue
        ov = concept_overlap(a.get("concept", ""), concept)
        if ov >= 0.5:
            conf = float(a.get("confidence", 0.6))
            strength = min(0.95, 0.5 + conf * 0.45) * (0.6 + 0.4 * ov)
            ev = [EvidenceItem(
                type="semantic",
                text=f"{a.get('concept')} — {'; '.join(a.get('evidence', [])[:2])}",
                detail={"confidence": conf, "category": a.get("category"), "inferred": True},
            )]
            if strength > best_strength:
                best_strength, best_ev, best_status = strength, ev, TriState.TRUE

    # 2. semantic fields
    if best_strength < 0.8:
        for fld, label in (("industries", "industry"), ("job_families", "role"),
                           ("domain_expertise", "domain expertise"), ("leadership_experience", "leadership"),
                           ("role_keywords", "role")):
            for v in sem.get(fld, []):
                if concept_overlap(v, concept) >= 0.55 and 0.72 > best_strength:
                    best_strength, best_ev, best_status = 0.72, [
                        EvidenceItem(type="semantic", text=f"{label}: {v}", detail={"inferred": True})
                    ], TriState.TRUE

    # 3. concept-vs-career cross-encoder (criterion-level, not whole query, spec §26).
    #    Skipped for role_function / industry_experience when we HAVE experience-
    #    level semantics: structured role vs industry data is authoritative there,
    #    and a fuzzy similarity must not turn a clean "no" into a maybe (V4 §H.5).
    _structured_authoritative = bool(exp_sem) and crit.type in (
        CriterionType.ROLE_FUNCTION, CriterionType.INDUSTRY_EXPERIENCE)
    if best_strength < 0.6 and settings.reranker_enabled and not _structured_authoritative:
        try:
            from app.services.reranker import cross_encode

            snippet = _career_snippet(facts)
            ce = cross_encode(concept, [snippet])[0] if snippet else 0.0
        except Exception:  # noqa: BLE001
            ce = 0.0
        if ce >= 0.5:
            strength = 0.35 + ce * 0.3
            if strength > best_strength:
                best_strength, best_ev, best_status = strength, [
                    EvidenceItem(type="semantic_relevance",
                                 text="career profile is semantically consistent with this concept",
                                 detail={"cross_encoder": round(ce, 3)})
                ], TriState.UNKNOWN  # a similarity signal alone isn't a confident TRUE

    if best_strength < _REQUIRED_MIN:
        return 0.0, [], TriState.UNKNOWN
    return round(best_strength, 3), best_ev, best_status


def _score_semantic_multi(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str]:
    """Score a semantic criterion honouring ``values`` + ``operator`` (V4 §4).
    Single value -> plain concept scoring. ANY_OF -> best value. ALL_OF -> all
    values must hold (min). NOT -> invert best."""
    vals = [v for v in (crit.values or ([crit.value] if crit.value else [])) if v]
    # a rich free-text concept is scored whole; a short value list is scored per value
    if crit.concept and len(vals) <= 1:
        return _score_semantic_concept(facts, crit, ctx)
    if len(vals) <= 1:
        return _score_semantic_concept(facts, crit, ctx)

    results = [_score_semantic_concept(facts, _synthetic_crit(v), ctx) for v in vals]
    if crit.operator == Operator.NOT:
        best = max(results, key=lambda r: r[0])
        inv = 1.0 - best[0]
        status = TriState.FALSE if best[2] == TriState.TRUE else (
            TriState.TRUE if best[0] < _REQUIRED_MIN else TriState.UNKNOWN)
        return round(inv, 3), [], status
    if crit.operator == Operator.ALL_OF:
        strength = min(r[0] for r in results)
        if any(r[2] == TriState.FALSE for r in results):
            status = TriState.FALSE
        elif all(r[2] == TriState.TRUE for r in results):
            status = TriState.TRUE
        else:
            status = TriState.UNKNOWN
        ev = [e for r in results for e in r[1]][:4]
        return round(strength, 3), ev, status
    # ANY_OF
    best = max(results, key=lambda r: r[0])
    return best


def _score_chronology(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str]:
    """career_transition / years_experience — real implementation in PART D
    (career_chronology). Until then these stay UNKNOWN so a required transition
    criterion yields POSSIBLE_MATCH, never a false EXACT_MATCH or exclusion."""
    from app.services.career_chronology import score_transition, score_years_experience

    if crit.type == CriterionType.CAREER_TRANSITION:
        return score_transition(facts, crit)
    return score_years_experience(facts, crit)


# ─────────────────────── dispatch ───────────────────────

_STRATEGIES = {
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


def _score_one(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str | None]:
    """Returns ``(strength, evidence, status)``. ``status`` is a TriState for
    semantic criteria, else ``None``."""
    try:
        # judge verdict (upstream LLM pass) overrides for semantic criteria
        jr = ctx.judge_results.get(facts.person.id, {}).get(crit.id)
        if jr and crit.type in _SEMANTIC_TYPES:
            return (
                float(jr.get("match_strength", 0.0)),
                [EvidenceItem(type="semantic", text=jr.get("reason", "")[:220],
                              detail={"confidence": jr.get("confidence"), "judge": True,
                                      "evidence": jr.get("evidence", [])[:3]})]
                if jr.get("status") == TriState.TRUE else [],
                jr.get("status"),
            )

        if crit.type in (CriterionType.SEMANTIC_CONCEPT, CriterionType.PROFESSIONAL_CONCEPT,
                         CriterionType.INDUSTRY_EXPERIENCE, CriterionType.ROLE_FUNCTION):
            return _score_semantic_multi(facts, crit, ctx)
        if crit.type == CriterionType.COMPANY_CATEGORY:
            return _score_company_category(facts, crit, ctx)
        if crit.type in (CriterionType.CAREER_TRANSITION, CriterionType.YEARS_EXPERIENCE):
            return _score_chronology(facts, crit, ctx)

        if crit.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY):
            want = crit.type == CriterionType.CURRENT_COMPANY
            rids = ctx.company_ids_by_criterion.get(crit.id) if settings.company_id_matching else None
            s, e = _combine_over_values(
                crit, lambda v: _score_company(facts, v, want_current=want, resolved_ids=rids)
            )
            return s, e, None

        strat = _STRATEGIES.get(crit.type, _score_keyword)
        s, e = _combine_over_values(crit, lambda v: strat(facts, v))
        return s, e, None
    except Exception:  # noqa: BLE001 — a matcher bug must not kill a search
        return 0.0, [], None


def _combine_over_values(crit: SearchCriterion, scorer) -> tuple[float, list[EvidenceItem]]:
    """Apply ``scorer(value)`` across ``crit.values`` and combine per operator
    (spec §13). ANY_OF: best. ALL_OF: min (all must hold). NOT: 1 - best."""
    vals = crit.values or ([crit.value] if crit.value else [])
    if not vals:
        return 0.0, []
    results = [scorer(v) for v in vals]
    if crit.operator == Operator.ALL_OF:
        s = min(r[0] for r in results)
        ev = [item for r in results for item in r[1]]
        return s, ev
    if crit.operator == Operator.NOT:
        best = max(results, key=lambda r: r[0])
        # matching the value is BAD here — invert. Strong presence -> 0; absence -> 1.
        return (1.0 - best[0]), (
            [EvidenceItem(type="experience", text="does not match the excluded criterion", detail={})]
            if best[0] < _REQUIRED_MIN else best[1]
        )
    best = max(results, key=lambda r: r[0])  # ANY_OF
    return best


# ─────────────────────── relevance component ───────────────────────


def _cosine_norm(query_emb: bytes | None, profile_emb: bytes | None) -> float:
    if not query_emb or not profile_emb:
        return 0.0
    import numpy as np

    from app.services.embeddings import to_array

    q, v = to_array(query_emb), to_array(profile_emb)
    if q.shape != v.shape:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(q, v)) / 0.6))


def _relevance_component(facts: ProfileFacts, ctx: ScoringContext, weight: float) -> ScoreComponent:
    emb = _cosine_norm(ctx.query_embedding, facts.embedding)
    ce = ctx.reranker_scores.get(facts.person.id)
    if ce is not None:
        strength = settings.rerank_blend * ce + (1 - settings.rerank_blend) * emb
        note = "embedding + reranker"
    else:
        strength = emb
        note = "embedding similarity"
    strength = round(max(0.0, min(1.0, strength)), 3)
    return ScoreComponent(
        criterion="Overall relevance to your query", criterion_id="relevance", type="relevance",
        weight=round(weight, 2), match_strength=strength, score=round(weight * strength, 2), required=False,
        evidence=[EvidenceItem(type="semantic_relevance", text=f"Whole-profile semantic match to the query ({note})",
                               detail={"embedding": round(emb, 3), "reranker": round(ce, 3) if ce is not None else None})]
        if strength > 0 else [],
    )


_EXACT_TYPES = {
    CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY, CriterionType.EDUCATION,
    CriterionType.CERTIFICATION, CriterionType.LANGUAGE, CriterionType.LOCATION,
}


def _effective_relevance_weight(parsed: ParsedSearchQuery) -> float:
    base = max(0.0, min(60.0, settings.relevance_weight))
    if not parsed.criteria or base == 0.0:
        return 0.0
    strongest_exact = max((c.weight for c in parsed.criteria if c.type in _EXACT_TYPES), default=0.0)
    if strongest_exact >= 55:
        return base * 0.25
    if strongest_exact >= 35:
        return base * 0.55
    return base


def score_candidate(
    facts: ProfileFacts, parsed: ParsedSearchQuery, ctx: ScoringContext | None = None
) -> ScoredCandidate:
    ctx = ctx or ScoringContext()
    components: list[ScoreComponent] = []
    all_evidence: list[EvidenceItem] = []
    matched: list[str] = []
    total = 0.0

    rel_w = _effective_relevance_weight(parsed)
    scale = (100.0 - rel_w) / 100.0

    for crit in parsed.criteria:
        strength, ev, status = _score_one(facts, crit, ctx)
        eff_weight = crit.weight * scale
        score = round(eff_weight * strength, 2)
        components.append(ScoreComponent(
            criterion=_label(crit), criterion_id=crit.id, type=crit.type,
            weight=round(eff_weight, 2), match_strength=round(strength, 3), score=score,
            required=crit.required, evidence=ev,
        ))

        if crit.required:
            if crit.type in _SEMANTIC_TYPES:
                # spec §15/§28, V4 §24/§E.6 — a CONFIDENT FALSE always excludes
                # (negative evidence beats a weak similarity); UNKNOWN never does.
                if status == TriState.FALSE:
                    return _excluded(facts.person, components, crit)
            elif strength < _REQUIRED_MIN:
                return _excluded(facts.person, components, crit)

        total += score
        if strength >= _MATCHED_MIN:
            matched.append(_label(crit))
            all_evidence.extend(ev)

    if rel_w > 0:
        rc = _relevance_component(facts, ctx, rel_w)
        components.append(rc)
        total += rc.score
        if rc.match_strength >= 0.55:
            all_evidence.extend(rc.evidence)

    return ScoredCandidate(
        person=facts.person, match_score=round(min(100.0, total), 1), components=components,
        evidence=_dedupe_evidence(all_evidence), matched_criteria=matched,
    )


def _excluded(person: Person, components: list[ScoreComponent], crit: SearchCriterion) -> ScoredCandidate:
    return ScoredCandidate(
        person=person, match_score=0.0, components=components, evidence=[],
        excluded_reason=f"required criterion not met: {crit.concept or crit.value}",
    )


def _label(c: SearchCriterion) -> str:
    val = " or ".join(c.values) if len(c.values) > 1 else (c.value or c.concept or "")
    pretty = {
        CriterionType.CURRENT_COMPANY: f"Currently at {val}",
        CriterionType.PAST_COMPANY: f"Previously at {val}",
        CriterionType.SKILL: val,
        CriterionType.DOMAIN: val,
        CriterionType.TITLE: f"{val} role",
        CriterionType.EDUCATION: f"Studied / attended {val}",
        CriterionType.LOCATION: f"Located in {val}",
        CriterionType.SENIORITY: f"{val.title()} level",
        CriterionType.KEYWORD: val,
        CriterionType.CERTIFICATION: f"{val} certification",
        CriterionType.LANGUAGE: f"Speaks {val}",
        CriterionType.PUBLICATION: f"Published on {val}" if val else "Has publications",
        CriterionType.SEMANTIC_CONCEPT: c.concept or val,
        CriterionType.COMPANY_CATEGORY: f"Works at a {c.concept or val} company"
        + ({Scope.PAST_COMPANY: " (past)", Scope.CURRENT_COMPANY: " (current)"}.get(c.scope, "")),
    }
    if c.operator == Operator.NOT:
        return f"NOT {pretty.get(c.type, val)}"
    return pretty.get(c.type, val)


def _dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen, out = set(), []
    for it in items:
        key = (it.type, it.text)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out
