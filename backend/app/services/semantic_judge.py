"""Exhaustive batched LLM semantic judge (V4 PART 3 §8–§10, §19, §27–§31).

In ``all_viable`` mode EVERY candidate with genuine unresolved REQUIRED
semantic uncertainty is judged — no artificial cap, no reduction in recall.
"Genuine unresolved uncertainty" is a testable decision (``needs_semantic_
judge`` / ``candidate_needs_judge``, hardening PART 4/5): a candidate whose
required judgeable criteria are ALL already TRUE/FALSE from stored facts,
cached company classification, or stored ProfileSemantic v3 data never reaches
the judge at all — nor does one already sealed NOT_MATCH by a different
required criterion, since resolving this one couldn't change that outcome.
This is what keeps a ~1,000-person broad query from generating ~100 judge
requests: most candidates are already decided before any query-time LLM call
is even considered.

Candidates that DO need judging are sent in BATCHES (``semantic_judge_batch_
size``) through the central LLM router, with adaptive splitting on truncation
(``app.services.llm.adaptive_batch`` — shared with the final auditor): a batch
whose response hits ``max_tokens`` is cut in half and retried as two smaller
requests rather than being retried identically. The model judges professional
MEANING against a compact, evidence-referenced packet; it may not invent
employers, roles, dates, skills, education or references.

Every verdict is validated downstream (``judge_validator``) before it is
allowed to change a score. A missing person / criterion becomes UNKNOWN, never
an assumed TRUE/FALSE. A partial batch failure (or an unresolved truncation)
keeps the verdicts already obtained from every other batch.
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
    Qualification,
    TriState,
)
from app.schemas import JudgeBatch, ParsedSearchQuery
from app.services.judge_packet import build_packets, plan_payload
from app.services.llm.adaptive_batch import run_adaptive
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
    "supporting_evidence_refs — valid ref forms are \"exp:<id>\", \"edu:<id>\", \"cert:<id>\", "
    "\"skill:<name>\", \"assertion:<n>\", \"company:<key>\", \"pub:<id>\" (a publication), "
    "\"vol:<id>\" (a volunteering role), \"rec:<id>\" (a recommendation received). Put anything "
    "that argues AGAINST the criterion in contradicting_evidence_refs, and the relevant "
    "experience ids in experience_ids. A \"true\" verdict with no supporting reference is "
    "rejected. A \"false\" verdict needs a contradicting reference OR an explicit statement that "
    "the full history was reviewed and clearly lacks it — otherwise return \"unknown\".\n\n"
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
    #: packets too large for a single batch — left unjudged (subset of omitted_people)
    oversized_packets: int = 0
    judgeable_criteria: list[str] = field(default_factory=list)
    providers: dict[str, int] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    prompt_chars: int = 0
    #: hardening PART 3 — batches that came back truncated (max_tokens) and how
    #: many times a truncated batch was cut in half and retried as two.
    truncations: int = 0
    adaptive_splits: int = 0
    #: hardening PART 4 — candidates already fully decided from stored facts /
    #: semantics before any query-time LLM call was even considered.
    candidates_decided_locally: int = 0
    candidates_needing_llm: int = 0

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
            "oversized_packets": self.oversized_packets,
            "judgeable_criteria": self.judgeable_criteria,
            "providers": self.providers,
            "models": self.models,
            "prompt_chars": self.prompt_chars,
            "truncations": self.truncations,
            "adaptive_splits": self.adaptive_splits,
            "candidates_decided_locally": self.candidates_decided_locally,
            "candidates_needing_llm": self.candidates_needing_llm,
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


def needs_semantic_judge(prescored, crit) -> bool:
    """True when resolving THIS criterion for THIS candidate could actually
    change whether they end up Exact / Possible / Top-N (hardening PART 4/5 —
    the staged decision function the mission asked to make testable).

    Skip judging when:
      * the criterion isn't required — a semantic UNKNOWN on a non-required
        criterion never changes qualification, only a soft score contribution
      * the criterion isn't judgeable, or is code-authoritative (career
        transition / years experience are never judge-overridden)
      * the deterministic prescore for THIS criterion already resolved it to
        TRUE or FALSE — a validated judge verdict cannot do better than a
        grounded fact/company-classification/semantic-assertion already did
      * the candidate is ALREADY sealed NOT_MATCH by a DIFFERENT required
        criterion — this one's own outcome cannot change that result, so
        spending a call on it buys nothing (mission: "candidate could not
        affect the displayed result")
    """
    if not crit.required:
        return False
    if crit.type not in JUDGEABLE_CRITERION_TYPES or crit.type in CODE_AUTHORITATIVE_CRITERION_TYPES:
        return False
    if prescored is None:
        return True  # no prior signal — safest is to judge, never invented
    if prescored.status_by_criterion.get(crit.id) != TriState.UNKNOWN:
        return False
    return prescored.qualification != Qualification.NOT_MATCH


def candidate_needs_judge(prescored, jcrits: list) -> bool:
    return any(needs_semantic_judge(prescored, c) for c in jcrits)


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

    # ── STAGE E/F (hardening PART 4) — drop candidates already fully decided
    #    by stored facts / semantics. Applies in EVERY mode: "all_viable" means
    #    every candidate with genuine unresolved required uncertainty, never
    #    "everyone regardless of whether the judge could change anything." ──
    total_before_local = len(bundle)
    if local_scored is not None:
        bundle = [(p, f, x) for (p, f, x) in bundle
                 if candidate_needs_judge(local_scored.get(p.id), jcrits)]
    meta.candidates_decided_locally = total_before_local - len(bundle)
    meta.candidates_needing_llm = len(bundle)
    meta.viable_candidate_count = len(bundle)
    if not bundle:
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
    batches, oversized = _make_batches(
        packets,
        size=settings.semantic_judge_batch_size,
        max_chars=settings.semantic_judge_max_batch_chars,
    )
    meta.oversized_packets = len(oversized)
    if oversized:
        log.warning("judge: %d packet(s) too large for a batch — left unjudged: %s",
                    len(oversized), oversized[:10])

    verdicts: dict[str, dict[str, dict]] = {}
    for batch in batches:
        leaves, stats = run_adaptive(batch, lambda pkts: _call_judge(payload, pkts))
        meta.judge_batch_count += stats.batches_attempted
        meta.judge_successful_batches += stats.successful_batches
        meta.judge_failed_batches += stats.failed_batches
        meta.truncations += stats.truncations
        meta.adaptive_splits += stats.adaptive_splits
        for prov, n in stats.providers.items():
            meta.providers[prov] = meta.providers.get(prov, 0) + n
        for m in stats.models:
            if m not in meta.models:
                meta.models.append(m)
        for leaf in leaves:
            if leaf.outcome != "ok":
                continue
            jb = leaf.payload
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


def _pkt_chars(pkt: dict) -> int:
    return len(json.dumps(pkt, ensure_ascii=False, default=str))


def _make_batches(
    packets: list[dict], *, size: int, max_chars: int,
) -> tuple[list[list[dict]], list[str]]:
    """Pack packets into ``<= size`` / ``<= max_chars`` batches. A single packet
    that itself exceeds ``max_chars`` (or was flagged ``_packet_too_large``) is
    NEVER placed in a batch — it would create an oversized provider request. Its
    person_id is returned as ``oversized`` and the candidate is left
    unjudged / UNKNOWN, status PARTIAL (V4 PART 3.6 §8 / PART 5 §40). Shared by
    the exhaustive judge and the final auditor."""
    size = max(1, size)
    max_chars = max(2000, max_chars)
    batches: list[list[dict]] = []
    oversized: list[str] = []
    cur: list[dict] = []
    cur_chars = 0
    for pkt in packets:
        pc = _pkt_chars(pkt)
        if pkt.get("_packet_too_large") or pc > max_chars:
            oversized.append(pkt.get("person_id"))
            continue
        if cur and (len(cur) >= size or cur_chars + pc > max_chars):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(pkt)
        cur_chars += pc
    if cur:
        batches.append(cur)
    return batches, oversized


def _call_judge(payload: dict, packets: list[dict]) -> tuple[str, object, str | None, str | None]:
    """One batched judge request through the router — the ``CallFn`` the
    adaptive splitter drives. Returns ``("ok", JudgeBatch, provider, model)``,
    ``("truncated", None, None, None)`` (caller should split and retry the
    halves), or ``("failed", None, None, None)`` (every provider genuinely
    exhausted / non-truncation error)."""
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
        return_meta=True,
    )
    if result[0] is not None:
        model, provider, model_id, meta = result
        return "ok", model, provider, model_id
    _, meta = result
    if any(a.get("status") == "output_truncated" for a in meta.get("attempts", [])):
        return "truncated", None, None, None
    return "failed", None, None, None


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
    """The cheaper opt-in mode's pool cap. ``bundle`` has ALREADY been through
    the same ``candidate_needs_judge`` staged filter as ``all_viable`` (V4
    hardening PART 4/5) — this only trims that already-ambiguous set further
    when it's larger than ``semantic_judge_pool``, prioritising by prescore.
    (Previously used a raw match_strength ∈ [low,high] band, which silently
    missed a required criterion sitting at strength 0.0 — a real UNKNOWN is
    not always a "medium" number. The tri-state decision function is exact.)"""
    if not local_scored or len(bundle) <= settings.semantic_judge_pool:
        return bundle[: settings.semantic_judge_pool] if not local_scored else bundle
    return _cap_prioritise(bundle, local_scored)[: settings.semantic_judge_pool]


def _cap_prioritise(bundle: list[tuple], local_scored: dict | None) -> list[tuple]:
    if not local_scored:
        return bundle
    return sorted(
        bundle,
        key=lambda t: local_scored[t[0].id].match_score if t[0].id in local_scored else 0.0,
        reverse=True,
    )
