"""Shared OpenAI-compatible chat client (Groq and OpenRouter both speak it)."""
from __future__ import annotations

import json
import logging
import re

import httpx

from app.services.llm.base import LLMBadOutput, LLMRateLimited, LLMUnavailable

log = logging.getLogger("app.llm")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError as e:
                raise LLMBadOutput(f"model did not return valid JSON: {e}") from e
        raise LLMBadOutput("model response contained no JSON object")


def chat_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    extra_headers: dict | None = None,
    reasoning_effort: str | None = None,
    timeout: float = 60.0,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    try:
        resp = httpx.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=timeout)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise LLMUnavailable(f"transport error: {e}") from e

    if resp.status_code == 429:
        ra = resp.headers.get("retry-after")
        raise LLMRateLimited("429 from provider", retry_after=float(ra) if ra and ra.replace(".", "").isdigit() else None)
    if resp.status_code in (500, 502, 503, 504):
        raise LLMUnavailable(f"provider {resp.status_code}")
    if resp.status_code == 404:
        raise LLMUnavailable("model not found / unavailable")
    if resp.status_code == 400 and "json_validate_failed" in resp.text:
        # model produced malformed JSON (often truncated) — retryable as bad output
        raise LLMBadOutput("provider could not validate generated JSON (likely truncated)")
    if resp.status_code >= 400:
        # 401/403/other — config problem, not worth retrying this provider
        raise LLMUnavailable(f"provider error {resp.status_code}: {resp.text[:200]}")

    body = resp.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMBadOutput(f"unexpected response shape: {e}") from e
    return _extract_json(content)
