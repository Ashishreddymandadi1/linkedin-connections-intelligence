"""Local ground-truth label file — filled in BY A HUMAN after inspecting results.

Format (one entry per evaluation query)::

    {
      "query_id": "q01_big_tech_bay_area",
      "must_match":     ["person_..."],   # a correct result set MUST contain these
      "should_match":   ["person_..."],   # good to surface; not penalised if deep
      "must_not_match": ["person_..."]    # must NOT appear as exact/possible
    }

Labels are NEVER auto-generated as if they were truth. ``write_template`` emits
empty arrays plus a ``_person_directory`` so the reviewer can map ids to people.
The whole ``labels/`` directory is gitignored (contains real names).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS = PILOT_DIR / "labels" / "labels.json"


@dataclass
class QueryLabels:
    query_id: str
    must_match: set[str]
    should_match: set[str]
    must_not_match: set[str]

    @property
    def any_labeled(self) -> bool:
        return bool(self.must_match or self.should_match or self.must_not_match)


def write_template(queries: list[dict], person_directory: list[dict], path: Path | str = DEFAULT_LABELS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "_README": "Fill must_match / should_match / must_not_match with person_ids "
                   "after inspecting the pilot profiles and results. Leave empty to "
                   "skip metrics for that query.",
        "_person_directory": person_directory,
        "labels": [
            {"query_id": q["id"], "query": q["query"],
             "must_match": [], "should_match": [], "must_not_match": []}
            for q in queries
        ],
    }
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_labels(path: Path | str = DEFAULT_LABELS) -> dict[str, QueryLabels]:
    path = Path(path)
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, QueryLabels] = {}
    for row in doc.get("labels", []):
        out[row["query_id"]] = QueryLabels(
            query_id=row["query_id"],
            must_match=set(row.get("must_match") or []),
            should_match=set(row.get("should_match") or []),
            must_not_match=set(row.get("must_not_match") or []),
        )
    return out
