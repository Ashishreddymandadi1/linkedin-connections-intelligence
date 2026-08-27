"""Shared string constants for state machines and enumerations.

Kept as plain strings (not DB enums) so SQLite stays flexible and Postgres can
adopt real enums later without a data migration.
"""
from __future__ import annotations


class EnrichmentState:
    """Per-person enrichment progress (spec §54)."""

    PENDING = "PENDING"
    APIFY_RUNNING = "APIFY_RUNNING"
    APIFY_COMPLETE = "APIFY_COMPLETE"
    NORMALIZED = "NORMALIZED"
    LLM_PENDING = "LLM_PENDING"
    LLM_COMPLETE = "LLM_COMPLETE"
    READY = "READY"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    WAITING_FOR_FREE_LLM = "WAITING_FOR_FREE_LLM"


#: States the worker treats as "done, do not reprocess this run".
TERMINAL_STATES = {
    EnrichmentState.READY,
    EnrichmentState.PARTIAL,
    EnrichmentState.FAILED,
}

#: States that still need the Apify scrape step.
NEEDS_APIFY = {EnrichmentState.PENDING, EnrichmentState.APIFY_RUNNING}


class JobStatus:
    """Enrichment job status (spec §8)."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class DatasetStatus:
    CREATED = "CREATED"
    READY_FOR_ENRICHMENT = "READY_FOR_ENRICHMENT"
    ENRICHING = "ENRICHING"
    READY = "READY"


class SkillSource:
    """Provenance of a skill row (spec §17)."""

    PROFILE = "linkedin_profile_skill"
    EXPERIENCE = "linkedin_experience_skill"
    EDUCATION = "linkedin_education_skill"
    DESCRIPTION = "linkedin_description"
    LLM = "llm_inference"


class RawSource:
    APIFY_HARVESTAPI = "apify_harvestapi"


class CriterionType:
    """Types the query interpreter may emit (spec §30)."""

    CURRENT_COMPANY = "current_company"
    PAST_COMPANY = "past_company"
    SKILL = "skill"
    DOMAIN = "domain"
    TITLE = "title"
    EDUCATION = "education"
    LOCATION = "location"
    SENIORITY = "seniority"
    KEYWORD = "keyword"
    CERTIFICATION = "certification"
    LANGUAGE = "language"
    PUBLICATION = "publication"


ALL_CRITERION_TYPES = {
    v for k, v in vars(CriterionType).items() if not k.startswith("_") and isinstance(v, str)
}


class LLMProviderName:
    GROQ_PRIMARY = "groq:primary"
    GROQ_FALLBACK = "groq:fallback"
    OPENROUTER = "openrouter:free"
    NONE = "none"
