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

    # ── LLM providers (free only) ────────────────────────────
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    groq_primary_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    enable_paid_llm: bool = False
    llm_max_retries: int = 2

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
    profile_ttl_days: int = 30

    # ── Search ───────────────────────────────────────────────
    candidate_pool_size: int = 60
    top_connections: int = 20
    semantic_profile_version: int = 1

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
