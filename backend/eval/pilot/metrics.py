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
    audit_grading: dict | None = None
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
            "audit_grading": self.audit_grading,
        })
        if self.note:
            d["note"] = self.note
        return d


# ─────────────────── audit-change grading (V4 PART 10.1 §2/§3) ───────────────────
#
# An audit change is graded ONLY when the affected person is labeled.
#   removed  (final == not_match): correct if person in must_not_match;
#                                  false removal if person in must_match;
#                                  should_match-only removals are recorded as
#                                  "questionable" and NOT counted as correct/incorrect.
#   exact->possible downgrade:     correct if person in must_not_match;
#                                  over-conservative (incorrect) if in must_match;
#                                  should_match-only downgrades are "questionable".
#   unlabeled person:              not graded.


def grade_audit_transitions(transitions: list[dict], labels: QueryLabels | None) -> dict:
    graded: list[dict] = []
    questionable: list[dict] = []
    ungraded = 0
    if labels is None or not labels.any_labeled:
        return {"policy": "no labels — audit changes not graded",
                "graded": [], "questionable": [], "ungraded": len(transitions)}

    for t in transitions:
        bucket = t.get("bucket", "")
        if bucket in ("exact_to_exact", "possible_to_possible"):
            continue  # not a change
        pid = t.get("person_id")
        removed = bucket.endswith("_to_removed")
        entry = {"person_id": pid, "bucket": bucket, "decision": t.get("audit_decision"),
                 "reason": t.get("reason")}
        if pid in labels.must_not_match:
            entry["correct"] = True
            graded.append(entry)
        elif pid in labels.must_match:
            entry["correct"] = False
            entry["kind"] = "false_removal" if removed else "over_conservative_downgrade"
            graded.append(entry)
        elif pid in labels.should_match:
            entry["policy"] = "should_match only — questionable, not graded correct/incorrect"
            questionable.append(entry)
        else:
            ungraded += 1

    removals = [g for g in graded if g["bucket"].endswith("_to_removed")]
    return {
        "policy": "graded via must_match / must_not_match; should_match-only changes are "
                  "'questionable' and not scored; unlabeled changes are not graded",
        "graded": graded,
        "questionable": questionable,
        "ungraded": ungraded,
        "correction_rate": (sum(1 for g in graded if g["correct"]) / len(graded)) if graded else None,
        "false_removal_rate": (sum(1 for g in removals if not g["correct"]) / len(removals)) if removals else None,
    }


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
    audit_transitions: list[dict] | None = None,
) -> QueryMetrics:
    """``ranked_person_ids`` — main results in rank order (exact+possible).
    ``qualifications`` — person_id -> "exact_match" | "possible_match".
    ``required_violations`` — person_id -> True when a required criterion is unmet/uncertain.
    ``audit_transitions`` — recorder's per-person first_pass->final transitions;
      graded against labels (never assumed correct just because the audit acted).
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

    audit_grading = grade_audit_transitions(audit_transitions or [], labels)

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
        audit_correction_rate=audit_grading.get("correction_rate"),
        audit_false_removal_rate=audit_grading.get("false_removal_rate"),
        audit_grading=audit_grading,
    )
