"""Offline retrieval metrics — computed ONLY when the query has human labels.

Every function returns ``None`` (rendered as "not available — ground truth not
labeled") when ``labels`` has nothing for the query. Nothing is fabricated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from eval.pilot.labels import QueryLabels

_UNAVAILABLE = "not available — ground truth not labeled"


@dataclass
class QueryMetrics:
    query_id: str
    labeled: bool
    precision_at: dict[int, float] | None = None
    recall_at_20: float | None = None
    mrr: float | None = None
    ndcg_at_20: float | None = None
    required_violation_rate: float | None = None
    exact_precision: float | None = None
    possible_precision: float | None = None
    audit_correction_rate: float | None = None
    audit_false_removal_rate: float | None = None
    note: str = ""

    def as_dict(self) -> dict:
        d = {"query_id": self.query_id, "labeled": self.labeled}
        if not self.labeled:
            d["note"] = _UNAVAILABLE
            return d
        d.update({
            "precision_at": self.precision_at,
            "recall_at_20": self.recall_at_20,
            "mrr": self.mrr,
            "ndcg_at_20": self.ndcg_at_20,
            "required_violation_rate": self.required_violation_rate,
            "exact_precision": self.exact_precision,
            "possible_precision": self.possible_precision,
            "audit_correction_rate": self.audit_correction_rate,
            "audit_false_removal_rate": self.audit_false_removal_rate,
        })
        if self.note:
            d["note"] = self.note
        return d


def _relevant(labels: QueryLabels) -> set[str]:
    return labels.must_match | labels.should_match


def _dcg(hits: list[int]) -> float:
    return sum(h / math.log2(i + 2) for i, h in enumerate(hits))


def compute(
    query_id: str,
    ranked_person_ids: list[str],
    *,
    labels: QueryLabels | None,
    qualifications: dict[str, str] | None = None,
    required_violations: dict[str, bool] | None = None,
    audit_changes: list[dict] | None = None,
) -> QueryMetrics:
    """``ranked_person_ids`` — main results in rank order (exact+possible).
    ``qualifications`` — person_id -> "exact_match" | "possible_match".
    ``required_violations`` — person_id -> True when a required criterion is unmet/uncertain.
    ``audit_changes`` — [{person_id, transition, correct?}] once labels exist.
    """
    if labels is None or not labels.any_labeled:
        return QueryMetrics(query_id=query_id, labeled=False)

    rel = _relevant(labels)
    bad = labels.must_not_match
    hits = [1 if pid in rel else 0 for pid in ranked_person_ids]

    def p_at(k: int) -> float:
        window = ranked_person_ids[:k]
        if not window:
            return 0.0
        return sum(1 for pid in window if pid in rel) / len(window)

    recall20 = (
        sum(hits[:20]) / len(labels.must_match) if labels.must_match else None
    )

    mrr = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            mrr = 1.0 / i
            break

    ideal = sorted(hits, reverse=True)[:20]
    idcg = _dcg(ideal)
    ndcg = (_dcg(hits[:20]) / idcg) if idcg else 0.0

    violation_rate = None
    if required_violations:
        vals = [required_violations.get(pid, False) for pid in ranked_person_ids]
        violation_rate = (sum(1 for v in vals if v) / len(vals)) if vals else 0.0

    exact_p = possible_p = None
    if qualifications:
        exact_ids = [pid for pid in ranked_person_ids if qualifications.get(pid) == "exact_match"]
        poss_ids = [pid for pid in ranked_person_ids if qualifications.get(pid) == "possible_match"]
        # a must_not_match landing in exact/possible is a precision failure
        if exact_ids:
            exact_p = sum(1 for pid in exact_ids if pid in rel and pid not in bad) / len(exact_ids)
        if poss_ids:
            possible_p = sum(1 for pid in poss_ids if pid in rel and pid not in bad) / len(poss_ids)

    corr_rate = false_removal_rate = None
    if audit_changes:
        graded = [c for c in audit_changes if "correct" in c]
        if graded:
            corr_rate = sum(1 for c in graded if c["correct"]) / len(graded)
        removals = [c for c in audit_changes if c.get("transition", "").endswith("removed") and "correct" in c]
        if removals:
            false_removal_rate = sum(1 for c in removals if not c["correct"]) / len(removals)

    return QueryMetrics(
        query_id=query_id,
        labeled=True,
        precision_at={5: p_at(5), 10: p_at(10), 20: p_at(20)},
        recall_at_20=recall20,
        mrr=mrr,
        ndcg_at_20=ndcg,
        required_violation_rate=violation_rate,
        exact_precision=exact_p,
        possible_precision=possible_p,
        audit_correction_rate=corr_rate,
        audit_false_removal_rate=false_removal_rate,
    )
