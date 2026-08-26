"""Free-LLM fallback chain (spec §24–§25).

Groq 120B → Groq 20B → OpenRouter free → give up (caller queues the unit).
Never calls a paid model. ``ENABLE_PAID_LLM`` exists only as a guard that this
module refuses to honor.
"""
from __future__ import annotations

import logging
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import settings
from app.services.llm.base import (
    LLMBadOutput,
    LLMProvider,
    LLMRateLimited,
    LLMUnavailable,
)
from app.services.llm.providers import default_chain

log = logging.getLogger("app.llm.router")

T = TypeVar("T", bound=BaseModel)

_MAX_BACKOFF = 5.0


def _sleep(seconds: float) -> None:
    time.sleep(min(seconds, _MAX_BACKOFF))


def generate_structured(
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    *,
    max_tokens: int = 1500,
    chain: list[LLMProvider] | None = None,
) -> tuple[T, str, str] | None:
    """Return ``(validated_model, provider_name, model_id)`` or ``None`` if every
    free provider is exhausted."""
    if settings.enable_paid_llm:
        log.warning("ENABLE_PAID_LLM is set but the router only ever uses free providers")

    providers = chain if chain is not None else default_chain()
    retries = max(0, settings.llm_max_retries)
    schema_hint = (
        "\n\nReturn ONLY a JSON object matching this schema (no prose, no markdown):\n"
        + _schema_skeleton(schema)
    )
    full_user = user_prompt + schema_hint

    for provider in providers:
        if not provider.available():
            continue
        for attempt in range(retries + 1):
            try:
                raw = provider.generate_json(system_prompt, full_user, max_tokens=max_tokens)
                model = schema.model_validate(raw)
                log.info("llm ok via %s (%s)", provider.name, provider.model)
                return model, provider.name, provider.model
            except LLMRateLimited as e:
                wait = e.retry_after if e.retry_after else 2 ** attempt
                log.warning("%s rate limited; wait %.1fs (attempt %d)", provider.name, wait, attempt + 1)
                if attempt < retries:
                    _sleep(wait)
                    continue
            except LLMUnavailable as e:
                log.warning("%s unavailable: %s (attempt %d)", provider.name, e, attempt + 1)
                if attempt < retries:
                    _sleep(2 ** attempt)
                    continue
            except (LLMBadOutput, ValidationError) as e:
                log.warning("%s bad/invalid output: %s (attempt %d)", provider.name, str(e)[:200], attempt + 1)
                if attempt < retries:
                    continue
            break  # this provider is done — move to the next

    log.error("all free LLM providers exhausted")
    return None


def _schema_skeleton(schema: type[BaseModel]) -> str:
    import json

    try:
        js = schema.model_json_schema()
        defs = js.get("$defs", {})
        props = js.get("properties", {})
        skeleton = {k: _example_for(v, defs) for k, v in props.items()}
        return json.dumps(skeleton, indent=2)
    except Exception:  # noqa: BLE001
        return "{}"


def _resolve(prop: dict, defs: dict) -> dict:
    ref = prop.get("$ref") or (prop.get("allOf", [{}])[0].get("$ref") if prop.get("allOf") else None)
    if ref:
        return defs.get(ref.split("/")[-1], {})
    return prop


def _example_for(prop: dict, defs: dict):
    prop = _resolve(prop, defs)
    t = prop.get("type")
    if "anyOf" in prop and not t:
        for opt in prop["anyOf"]:
            if opt.get("type") not in (None, "null"):
                return _example_for(opt, defs)
        return None
    if t == "array":
        items = _resolve(prop.get("items", {}), defs)
        if items.get("type") == "object" or items.get("properties"):
            return [{k: _example_for(v, defs) for k, v in items.get("properties", {}).items()}]
        return []
    if t in ("integer", "number"):
        return 0
    if t == "boolean":
        return False
    if t == "object" or prop.get("properties"):
        return {k: _example_for(v, defs) for k, v in prop.get("properties", {}).items()}
    return ""
