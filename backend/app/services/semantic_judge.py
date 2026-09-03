"""Exhaustive batched LLM semantic judge (V4 PART 3 §8–§10, §19, §27–§31).

In ``all_viable`` mode EVERY candidate that passed the hard-fact gate is judged
— no ambiguity band, no 60-person cap. Candidates are sent in BATCHES
(``semantic_judge_batch_size``, split further if a batch is too large for the
provider) through the central LLM router. The model judges professional
MEANING against a compact, evidence-referenced packet; it may not invent
employers, roles, dates, skills, education or references.

Every verdict is validated downstream (``judge_validator``) before it is
allowed to change a score. A missing person / criterion becomes UNKNOWN, never
an assumed TRUE/FALSE. A partial batch failure keeps the verdicts already
obtained.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.constants import (
    CODE_AUTHORITATIVE_CRITERION_TYPES,
    JUDGEABLE_CRITERION_TYPES,
    JudgeMode,
    JudgeStatus,
    TriState,
)
from app.schemas import JudgeBatch, ParsedSearchQuery
from app.services.judge_packet import build_packets, plan_payload
from app.services.llm.router import generate_structured

log = logging.getLogger("app.judge")

_SYSTEM = (
    "You are a professional-network analyst. For each PERSON you decide whether they satisfy "
    "each listed CRITERION, using ONLY the evidence packet given for that person.\n\n"
    "FACTS ARE LOCKED. The backend already verified employers, job titles, dates, education "
    "rows, locations and company classifications. You judge MEANING and INTERPRETATION on top "
    "of those facts. You may NOT contradict a fact, and you may NOT invent an employer, role, "
    "date, skill, degree, publication or evidence reference that is not in the packet.\n\n"
    "MISSING EVIDENCE IS NOT FALSE. If the packet does not contain enough to decide, return "
    "\"unknown\". A weak or indirect implication is also \"unknown\", not \"true\". Use "
    "\"false\" only when the evidence CLEARLY contradicts the criterion.\n\n"
    "Ground every verdict: put the packet references that support it in "
    "supporting_evidence_refs (e.g. \"exp:<id>\", \"edu:<id>\", \"cert:<id>\", \"skill:<name>\", "
    "\"assertion:<n>\", \"company:<key>\"), anything that argues against it in "
    "contradicting_evidence_refs, and the relevant experience ids in experience_ids. A \"true\" "
    "verdict with no supporting reference will be rejected.\n\n"
    "SCOPE MATTERS. If a criterion is scoped 'current', it can only be true from a role marked "
    "is_current=true. 'past' needs a non-current role.\n\n"
    "COMMON FALSE POSITIVES — do NOT mark true for these:\n"
    "  - \"used AWS\" / \"built on AWS\"  != employed by Amazon\n"
    "  - \"sold to CXOs\" / \"advises executives\"  != is a CXO\n"
    "  - \"studied at MIT\" / a degree from a university  != a faculty / professor appointment\n"
    "  - \"participated in a mentorship program\" / \"was mentored\"  != is a mentor\n"
    "  - \"technology transformation\" / \"digital transformation\"  != employment in the tech industry\n"
    "  - employer is a healthcare company  != personal HIPAA / compliance expertise\n"
    "  - \"published a paper\" / a single publication  != a professor / career researcher\n"
    "  - a senior or staff title  != evidence of mentoring or people leadership\n"
    "  - a bootcamp / weekend event / hackathon at a startup  != working at a startup\n\n"
    "BOOLEAN STRUCTURE: when a criterion lists several values with operator ALL_OF, every value "
    "must hold; with ANY_OF, one is enough; with NOT, the value must be ABSENT.\n"
    "MODALITY: a criterion with modality 'possible' is a soft signal — 'true' if the evidence "
    "supports it, but its absence must never count against the person.\n\n"
    "Also give each person an overall_fit (strong / moderate / weak / not_fit) and a one-line "
    "overall_reason — informational only; it does not override individual criteria.\n"
    "Return JSON only."
)


@dataclass
class JudgeMetadata:
    mode: str
    network_size: int = 0
    candidate_pool_size: int = 0
    hard_fact_rejected_count: int = 0
    viable_candidate_count: int = 0
    judge_candidate_count: int = 0
    judge_batch_count: int = 0
    judge_successful_batches: int = 0
    judge_failed_batches: int = 0
    judge_status: str = JudgeStatus.NOT_USED
    capped: bool = False
    cap_limit: int = 0
    omitted_people: int = 0
    omitted_criteria: int = 0
    judgeable_criteria: list[str] = field(default_factory=list)
    providers: dict[str, int] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    prompt_chars: int = 0

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "status": self.judge_status,
            "network_size": self.network_size,
            "candidate_pool_size": self.candidate_pool_size,
            "hard_fact_rejected_count": self.hard_fact_rejected_count,
            "viable_candidate_count": self.viable_candidate_count,
            "judge_candidate_count": self.judge_candidate_count,
            "judge_batch_count": self.judge_batch_count,
            "judge_successful_batches": self.judge_successful_batches,
            "judge_failed_batches": self.judge_failed_batches,
            "capped": self.capped,
            "cap_limit": self.cap_limit,
            "omitted_people": self.omitted_people,
            "omitted_criteria": self.omitted_criteria,
            "judgeable_criteria": self.judgeable_criteria,
            "providers": self.providers,
            "models": self.models,
            "prompt_chars": self.prompt_chars,
        }


@dataclass
class JudgeRun:
    verdicts: dict[str, dict[str, dict]]          # person_id -> criterion_id -> raw verdict dict
    packets_by_id: dict[str, dict]
    metadata: JudgeMetadata


def judgeable_criteria(parsed: ParsedSearchQuery) -> list:
    return [
        c for c in parsed.criteria
        if c.type in JUDGEABLE_CRITERION_TYPES and c.type not in CODE_AUTHORITATIVE_CRITERION_TYPES
    ]


def run_judge(
    query: str,
    parsed: ParsedSearchQuery,
    bundle: list[tuple],
    ctx,
    *,
    network_size: int,
    pool_size: int,
    hard_rejected_count: int,
    local_scored: dict | None = None,
) -> JudgeRun:
    """``bundle``: ``[(person, ProfileFacts, {"volunteering":[...], "recommendations":[...]})]``
    — the candidates that passed the hard-fact gate."""
    mode = settings.semantic_judge_mode
    jcrits = judgeable_criteria(parsed)
    meta = JudgeMetadata(
        mode=mode, network_size=network_size, candidate_pool_size=pool_size,
        hard_fact_rejected_count=hard_rejected_count, viable_candidate_count=len(bundle),
        judgeable_criteria=[c.id for c in jcrits],
    )

    if not settings.semantic_judge_enabled or mode == JudgeMode.OFF or not jcrits or not bundle:
        meta.judge_status = JudgeStatus.NOT_USED
        return JudgeRun({}, {}, meta)

    if mode == JudgeMode.UNCERTAIN_ONLY:
        bundle = _uncertain_shortlist(bundle, jcrits, local_scored)
        meta.viable_candidate_count = len(bundle)
        if not bundle:
            meta.judge_status = JudgeStatus.NOT_USED
            return JudgeRun({}, {}, meta)

    cap = settings.semantic_judge_max_candidates
    if cap and cap > 0 and len(bundle) > cap:
        meta.capped = True
        meta.cap_limit = cap
        bundle = _cap_prioritise(bundle, local_scored)[:cap]

    packets = build_packets(bundle, parsed, ctx, query=query)
    packets_by_id = {pkt["person_id"]: pkt for pkt in packets}
    meta.judge_candidate_count = len(packets)

    payload = plan_payload(query, parsed, jcrits)
    batches = _make_batches(packets)
    meta.judge_batch_count = len(batches)

    verdicts: dict[str, dict[str, dict]] = {}
    for batch in batches:
        res = _judge_batch(payload, batch)
        if res is None:
            meta.judge_failed_batches += 1
            continue
        jb, provider, model = res
        meta.judge_successful_batches += 1
        meta.providers[provider] = meta.providers.get(provider, 0) + 1
        if model and model not in meta.models:
            meta.models.append(model)
        for pv in jb.people:
            verdicts[pv.person_id] = {
                cv.criterion_id: {**cv.model_dump(), "overall_fit": pv.overall_fit}
                for cv in pv.criteria
            }

    _fill_missing(verdicts, packets_by_id, jcrits, meta)

    if meta.judge_successful_batches == 0:
        meta.judge_status = JudgeStatus.UNAVAILABLE
    elif meta.judge_failed_batches or meta.omitted_people or meta.omitted_criteria:
        meta.judge_status = JudgeStatus.PARTIAL
    else:
        meta.judge_status = JudgeStatus.FULL

    log.info(
        "judge %s: %d viable, %d judged, %d/%d batches ok, status=%s",
        mode, meta.viable_candidate_count, meta.judge_candidate_count,
        meta.judge_successful_batches, meta.judge_batch_count, meta.judge_status,
    )
    return JudgeRun(verdicts, packets_by_id, meta)


# ─────────────────────── batching ───────────────────────


def _make_batches(packets: list[dict]) -> list[list[dict]]:
    size = max(1, settings.semantic_judge_batch_size)
    max_chars = max(2000, settings.semantic_judge_max_batch_chars)
    batches: list[list[dict]] = []
    cur: list[dict] = []
    cur_chars = 0
    for pkt in packets:
        pc = len(json.dumps(pkt, ensure_ascii=False, default=str))
        if cur and (len(cur) >= size or cur_chars + pc > max_chars):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(pkt)
        cur_chars += pc
    if cur:
        batches.append(cur)
    return batches


def _judge_batch(payload: dict, packets: list[dict]) -> tuple[JudgeBatch, str, str] | None:
    """One batched judge request through the router. Returns
    ``(JudgeBatch, provider, model)`` or ``None`` when every provider failed."""
    user = (
        "SEARCH PLAN (how to judge — never a phrase to match):\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
        + "\n\nPEOPLE (one evidence packet each):\n"
        + json.dumps(packets, ensure_ascii=False, default=str)
        + '\n\nReturn {"people":[{"person_id":"...","overall_fit":"strong|moderate|weak|not_fit",'
        '"overall_reason":"...","criteria":[{"criterion_id":"...","status":"true|false|unknown",'
        '"match_strength":0-1,"confidence":0-1,"reason":"...","supporting_evidence_refs":["exp:.."],'
        '"contradicting_evidence_refs":[],"experience_ids":[]}]}]} — one entry per person_id, '
        "one verdict per listed criterion.id."
    )
    result = generate_structured(
        _SYSTEM, user, JudgeBatch,
        max_tokens=min(4000, 400 + 320 * len(packets)),
        operation="semantic_judge",
    )
    if result is None:
        return None
    jb, provider, model = result
    return jb, provider, model


# ─────────────────────── completeness (§30) ───────────────────────


def _fill_missing(verdicts: dict, packets_by_id: dict, jcrits: list, meta: JudgeMetadata) -> None:
    judged = set(verdicts)
    for pid in packets_by_id:
        pcrit = verdicts.setdefault(pid, {})
        if pid not in judged:
            meta.omitted_people += 1
        for c in jcrits:
            if c.id not in pcrit:
                if pid in judged:
                    meta.omitted_criteria += 1
                pcrit[c.id] = {
                    "criterion_id": c.id, "status": TriState.UNKNOWN,
                    "match_strength": 0.0, "confidence": 0.0, "reason": "",
                    "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
                    "experience_ids": [], "judge_missing": True,
                }


# ─────────────────────── uncertain_only mode ───────────────────────


def _uncertain_shortlist(bundle: list[tuple], jcrits: list, local_scored: dict | None) -> list[tuple]:
    if not local_scored:
        return bundle[: settings.semantic_judge_pool]
    lo, hi = settings.semantic_judge_low, settings.semantic_judge_high
    jids = {c.id for c in jcrits}
    picked: list[tuple] = []
    for person, facts, extras in bundle:
        sc = local_scored.get(person.id)
        if sc is None:
            picked.append((person, facts, extras))
            continue
        if any(lo <= comp.match_strength <= hi for comp in sc.components if comp.criterion_id in jids):
            picked.append((person, facts, extras))
    return picked[: settings.semantic_judge_pool]


def _cap_prioritise(bundle: list[tuple], local_scored: dict | None) -> list[tuple]:
    if not local_scored:
        return bundle
    return sorted(
        bundle,
        key=lambda t: local_scored[t[0].id].match_score if t[0].id in local_scored else 0.0,
        reverse=True,
    )
