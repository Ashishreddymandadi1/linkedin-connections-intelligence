"""Application configuration.

All values load from environment / ``backend/.env``. Secrets never have a
non-empty default and are never logged (see ``logging_config.redact``).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Apify ────────────────────────────────────────────────
    apify_api_token: str = ""
    apify_actor_id: str = "LpVuK3Zozwuipa5bp"
    apify_profile_scraper_mode: str = "Profile details no email ($4 per 1k)"

    # ── LLM providers ────────────────────────────────────────
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    groq_primary_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_max_retries: int = 2

    # Paid provider — only used when ENABLE_PAID_LLM=true AND a key is set. This is
    # an explicit opt-in; the app never reaches for a paid model on its own.
    enable_paid_llm: bool = False
    anthropic_api_key: str = ""
    anthropic_workspace_id: str = ""  # required for identity-linked keys
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_first: bool = True  # when enabled, try Anthropic before the free tier

    # ── Embeddings (local) ───────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embeddings_enabled: bool = True  # False -> deterministic hash vectors (tests / no-torch)

    # ── Database ─────────────────────────────────────────────
    database_url: str = "sqlite:///./data/app.db"

    # ── Enrichment / cost controls ───────────────────────────
    environment: str = "development"
    use_fixtures: bool = True
    development_batch_size: int = 5
    production_batch_size: int = 50
    apify_profile_batch_size: int = 50
    max_apify_retries: int = 2
    apify_max_charge_usd: float = 0.0  # 0 = no cap; e.g. 2.0 to hard-stop a run at $2
    profile_ttl_days: int = 30

    # ── Semantic enrichment ──────────────────────────────────
    semantic_enabled: bool = True  # False -> skip the LLM profile pass entirely
    semantic_profile_version: int = 1

    # ── Search ───────────────────────────────────────────────
    candidate_pool_size: int = 60
    top_connections: int = 20
    min_match_score: float = 12.0          # drop incidental / weak matches (spec §38)
    llm_query_interpretation: bool = True  # False -> deterministic query parser only
    llm_reason_generation: bool = True     # False -> deterministic reason templates
    llm_reason_top_n: int = 8              # LLM reasons for the top N; template for the rest

    # ── Search quality v2 ────────────────────────────────────
    relevance_weight: float = 20.0         # points reserved for whole-profile relevance
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_pool: int = 40                  # candidates re-scored by the cross-encoder
    rerank_blend: float = 0.6             # cross-encoder share within the relevance component
    llm_rerank_enabled: bool = False       # opt-in LLM reorder of the final top 20
    recency_weighting_enabled: bool = True
    company_id_matching: bool = True

    # ── Search quality v3 — semantic concepts ────────────────
    #: below this many connections, score EVERYONE — no SQL prefilter at all
    #: (spec §10/§31: correctness over shaving milliseconds on a ~1k network).
    full_scan_max_connections: int = 5000
    company_classification_enabled: bool = True
    #: candidates in this confidence band get a batched LLM semantic judge call
    #: for their semantic_concept/company_category criteria (spec §16-18).
    semantic_judge_enabled: bool = True
    semantic_judge_pool: int = 60           # max candidates considered for judging per search
    semantic_judge_batch_size: int = 10
    semantic_judge_low: float = 0.15        # below this concept strength: skip judging, already a clear miss
    semantic_judge_high: float = 0.75       # above this: skip judging, already a clear hit

    # ── App ──────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"

    @property
    def is_development(self) -> bool:
        return self.environment.lower().startswith("dev")

    @property
    def effective_batch_size(self) -> int:
        """Batch size used by the enrichment worker for Apify calls."""
        return self.development_batch_size if self.is_development else self.apify_profile_batch_size

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
