"""Pydantic models for API I/O and validated LLM output (spec §67).

Never trust free-form LLM output — every LLM call is parsed into one of the
``*Data`` / ``Parsed*`` models here and rejected on validation failure.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.constants import ALL_CRITERION_TYPES

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

    @field_validator("years_of_experience")
    @classmethod
    def _sane_yoe(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return max(0.0, min(60.0, float(v)))


# ─────────────────── validated LLM output: query ───────────────────


class SearchCriterion(BaseModel):
    id: str
    type: str
    value: str
    weight: float = Field(ge=0)
    required: bool = False

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ALL_CRITERION_TYPES:
            # tolerate unknown types by folding to keyword rather than erroring
            return "keyword"
        return v


class ParsedSearchQuery(BaseModel):
    intent: str = "professional_recommendation"
    criteria: list[SearchCriterion]

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
