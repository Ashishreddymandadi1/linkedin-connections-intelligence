"""Generic adaptive batch-splitting for a batched structured-LLM call.

Shared by the semantic judge and the final auditor (hardening PART 3): if a
batch's response is truncated (the model hit ``max_tokens`` before producing
complete JSON), retrying the IDENTICAL request just truncates again — instead
the batch is split in half and each half retried independently. Recursion is
bounded, a single-packet leaf is never split further, every successful sibling
is kept, and a leaf that never resolves becomes UNKNOWN (not FALSE) — its
packets simply never appear in the caller's verdict/decision dict, which the
existing "fill missing -> UNKNOWN" step already covers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger("app.llm.adaptive_batch")

#: hard bound so a pathological input can never recurse forever — log2(any
#: realistic batch size) is well under this.
MAX_SPLIT_DEPTH = 6


@dataclass
class LeafResult:
    packets: list[dict]
    outcome: str                 # "ok" | "failed"
    payload: object = None       # the caller's parsed structured result (outcome == "ok" only)
    provider: str | None = None
    model: str | None = None


@dataclass
class BatchCallStats:
    batches_attempted: int = 0
    successful_batches: int = 0
    failed_batches: int = 0        # leaf batches that never produced a usable result
    truncations: int = 0           # individual attempts that came back truncated
    adaptive_splits: int = 0       # number of times a batch was cut in half after truncation
    providers: dict[str, int] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)


#: ``call_fn(packets) -> (outcome, payload, provider, model)``
#: outcome is one of "ok" / "truncated" / "failed". payload/provider/model are
#: only meaningful when outcome == "ok".
CallFn = Callable[[list[dict]], tuple[str, object, str | None, str | None]]


def run_adaptive(packets: list[dict], call_fn: CallFn, *, min_size: int = 1) -> tuple[list[LeafResult], BatchCallStats]:
    stats = BatchCallStats()
    leaves: list[LeafResult] = []
    if packets:
        _recurse(packets, call_fn, min_size, 0, leaves, stats)
    return leaves, stats


def _recurse(packets: list[dict], call_fn: CallFn, min_size: int, depth: int,
            leaves: list[LeafResult], stats: BatchCallStats) -> None:
    stats.batches_attempted += 1
    outcome, payload, provider, model = call_fn(packets)

    if outcome == "ok":
        stats.successful_batches += 1
        if provider:
            stats.providers[provider] = stats.providers.get(provider, 0) + 1
        if model and model not in stats.models:
            stats.models.append(model)
        leaves.append(LeafResult(packets=packets, outcome="ok", payload=payload,
                                 provider=provider, model=model))
        return

    if outcome == "truncated":
        stats.truncations += 1
        if len(packets) > min_size and depth < MAX_SPLIT_DEPTH:
            stats.adaptive_splits += 1
            mid = len(packets) // 2
            log.warning("adaptive_batch: truncated at size=%d depth=%d — splitting into %d/%d",
                        len(packets), depth, mid, len(packets) - mid)
            _recurse(packets[:mid], call_fn, min_size, depth + 1, leaves, stats)
            _recurse(packets[mid:], call_fn, min_size, depth + 1, leaves, stats)
            return
        log.error("adaptive_batch: truncation unresolved at size=%d depth=%d — leaving unjudged (UNKNOWN)",
                  len(packets), depth)

    # "failed", or truncation that could not be split further -> a genuine
    # failed leaf. Every person in it simply gets no verdict/decision, and the
    # caller's existing completeness step turns that into UNKNOWN, never FALSE.
    stats.failed_batches += 1
    leaves.append(LeafResult(packets=packets, outcome="failed"))
