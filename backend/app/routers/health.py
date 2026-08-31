from __future__ import annotations

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "environment": settings.environment,
        "use_fixtures": settings.use_fixtures,
        "apify_configured": bool(settings.apify_api_token),
        "groq_configured": bool(settings.groq_api_key),
        "openrouter_configured": bool(settings.openrouter_api_key),
        "paid_llm_enabled": settings.enable_paid_llm,
        "anthropic_active": settings.enable_paid_llm and bool(settings.anthropic_api_key),
        "llm_chain": [p.name for p in _chain_names()],
        "embedding_model": settings.embedding_model,
    }


def _chain_names():
    from app.services.llm.providers import default_chain

    return [p for p in default_chain() if p.available()]
