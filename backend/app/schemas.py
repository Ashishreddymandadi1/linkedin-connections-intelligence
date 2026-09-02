"""Pydantic models for API I/O and validated LLM output (spec §67).

Never trust free-form LLM output — every LLM call is parsed into one of the
``*Data`` / ``Parsed*`` models here and rejected on validation failure.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.constants import ALL_CRITERION_TYPES, ALL_OPERATORS, ALL_SCOPES, Operator

# ─────────────────────────── datasets ───────────────────────────


class DatasetSummary(BaseModel):
    dataset_id: str
    name: str
    connection_count: int
    status: str
    created_at: datetime
    updated_at: datetime


class UploadReport(BaseModel):
    dataset: DatasetSummary
    total_rows: int
    imported: int
    duplicates_removed: int
    skipped_no_url: int
    skipped: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []


class DatasetStatusReport(BaseModel):
    dataset_id: str
    name: str
    status: str
    connections: int
    counts: dict[str, int]
    ready: int
    partial: int
    failed: int
    pending: int
    waiting_for_llm: int
    progress_done: int
    progress_total: int
    progress_pct: int
    last_updated: datetime | None = None
    job: dict[str, Any] | None = None


# ─────────────────────────── people ───────────────────────────


class ExperienceOut(BaseModel):
    position: str | None = None
    company_name: str | None = None
    company_linkedin_url: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    is_current: bool = False
    duration_text: str | None = None
    description: str | None = None
    location: str | None = None


class EducationOut(BaseModel):
    school_name: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_year: int | None = None
    end_year: int | None = None


class SkillOut(BaseModel):
    skill_name: str
    source: str
    is_inferred: bool
    confidence: float
    evidence: str | None = None


class PersonOut(BaseModel):
    person_id: str
    is_connection: bool
    linkedin_url: str
    public_identifier: str | None = None
    full_name: str | None = None
    headline: str | None = None
    about: str | None = None
    location_text: str | None = None
    current_title: str | None = None
    current_company: str | None = None
    profile_picture_url: str | None = None
    connections_count: int | None = None
    followers_count: int | None = None
    profile_completeness: int = 0
    enrichment_state: str
    last_scraped_at: datetime | None = None
    experiences: list[ExperienceOut] = []
    education: list[EducationOut] = []
    skills: list[SkillOut] = []
    semantics: dict[str, Any] | None = None


class PersonListItem(BaseModel):
    person_id: str
    full_name: str | None = None
    headline: str | None = None
    current_title: str | None = None
    current_company: str | None = None
    linkedin_url: str
    profile_completeness: int = 0
    enrichment_state: str


# ─────────────────── validated LLM output: profile ───────────────────


class InferredSkill(BaseModel):
    skill: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str

    @field_validator("skill")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("skill must be non-empty")
        return v.strip()


def _coerce_str_list(v):
    """Tolerate the model returning [{'name': x}] / [{'skill': x}] / 'a, b'."""
    if v is None:
        return []
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    out = []
    for item in v if isinstance(v, list) else [v]:
        if isinstance(item, str):
            out.append(item.strip())
        elif isinstance(item, dict):
            val = item.get("name") or item.get("skill") or item.get("value") or item.get("keyword")
            if val:
                out.append(str(val).strip())
    return [x for x in out if x]


class SemanticAssertion(BaseModel):
    """One derived professional-concept assertion (spec §7, V4 §13). Provenance-
    tagged, never presented as a verified LinkedIn fact. Links back to the
    normalized rows it was derived from (source IDs beat matching evidence text)."""

    concept: str
    category: str = "industry_experience"
    scope: str = "career"
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)
    evidence: list[str] = []
    #: normalized row ids this assertion is grounded in (V4 §13) — validated in
    #: semantic_llm._ground(); invalid ids are dropped, never trusted.
    experience_ids: list[str] = []
    education_ids: list[str] = []
    certification_ids: list[str] = []

    @field_validator("concept")
    @classmethod
    def _nonempty_concept(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("concept must be non-empty")
        return v.strip()

    _norm_ids = field_validator(
        "evidence", "experience_ids", "education_ids", "certification_ids", mode="before",
    )(staticmethod(_coerce_str_list))


class ExperienceSemantic(BaseModel):
    """Meaning of ONE work experience (V4 §10/§11). Role function is tracked
    separately from employer industry so "Accountant at Google" and "Software
    Engineer at JPMorgan" behave differently per query."""

    experience_id: str
    role_function: str | None = None          # "software engineering", "accounting"
    professional_domain: str | None = None    # "software systems", "finance"
    role_domains: list[str] = []
    role_seniority: str | None = None
    employer_industries: list[str] = []       # "technology", "financial services"
    employer_categories: list[str] = []       # "big tech", "large bank" (advisory; company_intel is authoritative)
    leadership_signals: list[str] = []
    mentoring_signals: list[str] = []
    founder_signals: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)

    _norm_lists = field_validator(
        "role_domains", "employer_industries", "employer_categories",
        "leadership_signals", "mentoring_signals", "founder_signals", mode="before",
    )(staticmethod(_coerce_str_list))

    @field_validator("role_function", "professional_domain", "role_seniority", mode="before")
    @classmethod
    def _first_if_list(cls, v):
        # LLMs sometimes wrap a scalar in a list — don't let that sink the whole
        # profile's enrichment (review #7)
        if isinstance(v, (list, tuple)):
            return (str(v[0]).strip() if v else None) or None
        return v


class ProfileSemanticData(BaseModel):
    seniority_level: str | None = None
    job_families: list[str] = []
    technical_domains: list[str] = []
    industries: list[str] = []
    explicit_skills: list[str] = []
    inferred_skills: list[InferredSkill] = []
    leadership_experience: list[str] = []
    domain_expertise: list[str] = []
    career_summary: str | None = None
    years_of_experience: float | None = None
    current_role_summary: str | None = None
    past_company_names: list[str] = []
    current_company_names: list[str] = []
    education_keywords: list[str] = []
    role_keywords: list[str] = []
    searchable_keywords: list[str] = []
    #: derived professional-concept assertions (spec §7) — NOT verified LinkedIn
    #: facts. Each keeps its own evidence + confidence so provenance survives
    #: into search results.
    semantic_assertions: list["SemanticAssertion"] = []
    #: per-experience meaning (V4 §10) — role function vs employer industry, etc.
    experience_semantics: list["ExperienceSemantic"] = []

    _norm_lists = field_validator(
        "job_families",
        "technical_domains",
        "industries",
        "explicit_skills",
        "leadership_experience",
        "domain_expertise",
        "past_company_names",
        "current_company_names",
        "education_keywords",
        "role_keywords",
        "searchable_keywords",
        mode="before",
    )(staticmethod(_coerce_str_list))

    @field_validator("inferred_skills", mode="before")
    @classmethod
    def _coerce_inferred(cls, v):
        if not v:
            return []
        out = []
        for item in v if isinstance(v, list) else [v]:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str):
                out.append({"skill": item, "confidence": 0.5, "evidence": ""})
        return out

    @field_validator("semantic_assertions", mode="before")
    @classmethod
    def _coerce_assertions(cls, v):
        if not v:
            return []
        out = []
        for item in v if isinstance(v, list) else [v]:
            if isinstance(item, dict) and item.get("concept"):
                out.append(item)
            elif isinstance(item, str) and item.strip():
                out.append({"concept": item.strip(), "evidence": []})
        return out

    @field_validator("experience_semantics", mode="before")
    @classmethod
    def _coerce_exp_sem(cls, v):
        if not v:
            return []
        return [item for item in (v if isinstance(v, list) else [v])
                if isinstance(item, dict) and item.get("experience_id")]

    @field_validator("years_of_experience", mode="before")
    @classmethod
    def _sane_yoe(cls, v) -> float | None:
        if v is None or v == "":
            return None
        try:
            return max(0.0, min(60.0, float(v)))
        except (TypeError, ValueError):
            return None


# ─────────────────── validated LLM output: company classification ───────────────────


class CompanyClassificationItem(BaseModel):
    key: str  # echoes the request key so we can match batched responses back up
    industries: list[str] = []
    categories: list[str] = []
    is_technology_company: bool | None = None
    is_startup: bool | None = None
    is_big_tech: bool | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reason: str = ""

    _norm = field_validator("industries", "categories", mode="before")(staticmethod(_coerce_str_list))


class CompanyClassificationBatch(BaseModel):
    companies: list[CompanyClassificationItem] = []


# ─────────────────── validated LLM output: query ───────────────────


#: unknown types the LLM sometimes emits, mapped to a real first-class type (V4 §6)
_CRITERION_TYPE_ALIASES = {
    "industry": "industry_experience", "sector": "industry_experience",
    "industry_sector": "industry_experience", "employer_category": "company_category",
    "employer_industry": "industry_experience",
    "role": "role_function", "function": "role_function", "job_function": "role_function",
    "profession": "role_function", "domain": "role_function",
    "seniority_level": "seniority", "years_of_experience": "years_experience",
    "tenure": "years_experience", "experience_years": "years_experience",
    "transition": "career_transition", "career_change": "career_transition",
    "concept": "professional_concept", "leadership": "professional_concept",
    "mentorship": "professional_concept", "capability": "professional_concept",
}


class SearchCriterion(BaseModel):
    id: str
    type: str
    weight: float = Field(ge=0)
    required: bool = False

    # backward-compatible single value (every existing scorer reads .value)
    value: str = ""
    # v3 additions — optional, all existing code that only reads .value/.type
    # keeps working unchanged.
    values: list[str] = []
    operator: str = Operator.ANY_OF
    scope: str | None = None
    #: free-text semantic concept description — required for semantic_concept /
    #: company_category, ignored for exact-fact types.
    concept: str | None = None

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        if v in ALL_CRITERION_TYPES:
            return v
        if v in _CRITERION_TYPE_ALIASES:
            return _CRITERION_TYPE_ALIASES[v]
        # NEVER silently fall to `keyword` (V4 §6). An unrecognised semantic-ish
        # type is a professional_concept; only an explicit "keyword"/"text" type
        # produces a literal text search.
        if v in ("text", "phrase", "mention", "literal"):
            return "keyword"
        return "professional_concept"

    @field_validator("operator")
    @classmethod
    def _known_operator(cls, v: str) -> str:
        v = (v or Operator.ANY_OF).strip().upper()
        return v if v in ALL_OPERATORS else Operator.ANY_OF

    @field_validator("scope")
    @classmethod
    def _known_scope(cls, v: str | None) -> str | None:
        if not v:
            return None
        v = v.strip().lower()
        return v if v in ALL_SCOPES else None

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, v):
        return _coerce_str_list(v)

    @model_validator(mode="after")
    def _sync_value_and_values(self) -> "SearchCriterion":
        if not self.values and self.value:
            self.values = [self.value]
        if not self.value and self.values:
            self.value = self.values[0]
        if not self.value and not self.values and self.concept:
            self.value = self.concept
            self.values = [self.concept]
        return self


class ParsedSearchQuery(BaseModel):
    intent: str = "professional_recommendation"
    criteria: list[SearchCriterion]
    #: non-candidate framing (V4 §14) — e.g. {"purpose": "networking event"}. These
    #: words must NOT become criteria.
    context: dict[str, str] = {}
    #: one plain-English sentence describing how the query was read (V4 §18)
    interpretation_summary: str = ""
    #: 0..1 — lower for semantically ambiguous queries ("people who worked in tech")
    interpretation_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    #: hard upper bound set by the plan validator on an unresolved structural
    #: problem (V4 §9) — _finalize() must not raise confidence above this.
    interpretation_confidence_cap: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _normalize_weights(self) -> "ParsedSearchQuery":
        if not self.criteria:
            raise ValueError("query produced no criteria")
        total = sum(c.weight for c in self.criteria)
        if total <= 0:
            equal = 100.0 / len(self.criteria)
            for c in self.criteria:
                c.weight = equal
        elif abs(total - 100.0) > 0.5:
            for c in self.criteria:
                c.weight = round(c.weight / total * 100, 2)
        return self


# ─────────────────────── search results ───────────────────────


class EvidenceItem(BaseModel):
    type: str  # experience | education | skill | headline | about | semantic
    text: str
    detail: dict[str, Any] = {}


class ScoreComponent(BaseModel):
    criterion: str
    criterion_id: str
    type: str
    weight: float
    match_strength: float
    score: float
    required: bool = False
    evidence: list[EvidenceItem] = []


class SearchResultItem(BaseModel):
    rank: int
    person_id: str
    name: str | None = None
    linkedin_url: str
    profile_picture_url: str | None = None
    current_title: str | None = None
    current_company: str | None = None
    location: str | None = None
    is_connection: bool = True
    match_score: float
    data_confidence: int
    reason: str
    #: V4 §22-25 — exact_match / possible_match / not_match (near-matches only)
    qualification: str = "possible_match"
    #: required criteria still uncertain (why this is possible_match not exact)
    uncertain_criteria: list[str] = []
    #: required criteria this person FAILS (near-match rows only)
    unmet_criteria: list[str] = []
    matched_criteria: list[str] = []
    score_breakdown: list[ScoreComponent] = []
    evidence: list[EvidenceItem] = []
    relevant_experience: list[ExperienceOut] = []
    relevant_skills: list[SkillOut] = []
    relevant_education: list[EducationOut] = []


class ConnectionBucket(BaseModel):
    total_candidates: int
    returned: int
    results: list[SearchResultItem]
    #: V4 §22-26
    exact_match_count: int = 0
    possible_match_count: int = 0
    #: candidates that miss exactly one required criterion — clearly labelled,
    #: never mixed into `results` (V4 §26)
    near_matches: list[SearchResultItem] = []


class ExternalBucket(BaseModel):
    searched: bool = False
    total_candidates: int = 0
    returned: int = 0
    results: list[SearchResultItem] = []


class SearchRequest(BaseModel):
    dataset_id: str
    query: str = Field(min_length=2)


class SearchResponse(BaseModel):
    search_id: str
    query: str
    interpreted_query: dict[str, Any]
    connections: ConnectionBucket
    external: ExternalBucket
    llm_provider: str | None = None
    llm_model: str | None = None
