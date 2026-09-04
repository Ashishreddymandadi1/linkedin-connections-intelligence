"""Deterministic, stratified selection of the pilot sample from EXISTING data.

Not "first N", not "random N". Each person is tagged with professional facets
derived only from their already-stored normalized rows, then the sample is built
by:

  1. reserving a few sparse and medium-completeness profiles (hash-ordered), and
  2. round-robining across facets (hash-ordered within each) for the remainder,

so the sample spans role diversity AND data-quality tiers. The ordering key is
``md5(seed + person_id)`` — fully reproducible, never name-based.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import repositories as repo
from app.models import Person
from app.services.company_intel import company_key

DEFAULT_SEED = "v4-part10-pilot"
DEFAULT_TARGET = 40
RESERVED_SPARSE = 3        # profile_completeness < 50
RESERVED_MEDIUM = 6        # 50 <= profile_completeness < 85

# facet -> lowercase keyword fragments matched against role text / company / field / skills
FACET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "software_engineering": ("software engineer", "swe", "sde", "developer", "backend", "frontend",
                             "full stack", "full-stack", "platform engineer", "programmer"),
    "engineering_management": ("engineering manager", "eng manager", "head of engineering",
                               "director of engineering", "vp engineering", "vp of engineering",
                               "engineering lead", "tech lead manager"),
    "product": ("product manager", "product management", "head of product", "product owner",
                "group product manager", "director of product"),
    "sales": ("sales", "account executive", "business development", "revenue", "quota",
              "account manager", "sdr", "bdr"),
    "finance": ("finance", "financial analyst", "investment", "banking", "accountant",
                "accounting", "controller", "cfo", "private equity", "venture capital", "fp&a"),
    "consulting": ("consultant", "consulting", "advisory", "mckinsey", "bcg", "bain",
                   "deloitte", "accenture", "pwc", "ey ", "kpmg"),
    "healthcare": ("health", "healthcare", "clinical", "hospital", "patient", "medical",
                   "pharma", "biotech", "life sciences", "nurse", "physician", "hipaa"),
    "cybersecurity": ("security", "cybersecurity", "infosec", "appsec", "soc analyst",
                      "penetration", "threat", "iam", "ciso", "vulnerability"),
    "ai_ml": ("machine learning", "ml engineer", "deep learning", "data scientist", "ai ",
              "artificial intelligence", "nlp", "computer vision", "mlops", "llm"),
    "academia_research": ("professor", " phd", "ph.d", "postdoc", "research scientist",
                          "research fellow", "research associate", "research assistant",
                          "lecturer", "faculty", "dissertation", "thesis advisor"),
    "founder_startup": ("founder", "co-founder", "cofounder", "founding engineer", "entrepreneur",
                        "startup", "self-employed"),
    "big_tech": ("google", "amazon", "microsoft", "meta", "apple", "netflix", "nvidia",
                 "salesforce", "oracle", "adobe"),
    "nonprofit_volunteering": ("nonprofit", "non-profit", "ngo", "volunteer", "foundation",
                               "charity", "501(c)"),
    "senior_exec": ("chief", "cxo", "ceo", "cto", "coo", "cfo", "ciso", "president", "vp ",
                    "vice president", "head of", "partner", "managing director"),
}

FACET_ORDER = list(FACET_KEYWORDS.keys())


@dataclass
class PilotPerson:
    person_id: str
    public_identifier: str | None
    full_name: str | None
    completeness: int
    completeness_tier: str
    enrichment_state: str
    semantic_version: int | None
    has_semantic: bool
    has_embedding: bool
    facets: list[str] = field(default_factory=list)
    selected_via: str = ""


@dataclass
class PilotSample:
    dataset_id: str
    seed: str
    target: int
    people: list[PilotPerson]

    @property
    def person_ids(self) -> list[str]:
        return [p.person_id for p in self.people]


def _order_key(seed: str, pid: str) -> str:
    return hashlib.md5(f"{seed}:{pid}".encode()).hexdigest()


def _tier(completeness: int) -> str:
    if completeness >= 85:
        return "strong"
    if completeness >= 50:
        return "medium"
    return "sparse"


def _facets_for(
    person: Person,
    exps: list,
    edus: list,
    skills: list,
    vols: list,
    company_class: dict,
) -> list[str]:
    hay_parts: list[str] = [
        (person.current_title or ""), (person.headline or ""), (person.current_company or ""),
    ]
    for e in exps:
        hay_parts.append(f"{e.position or ''} {e.company_name or ''} {e.description or ''}")
    for e in edus:
        hay_parts.append(f"{e.degree or ''} {e.field_of_study or ''} {e.school_name or ''}")
    for s in skills:
        hay_parts.append(s.skill_name or "")
    hay = " ".join(hay_parts).lower()

    found: list[str] = []
    for facet, kws in FACET_KEYWORDS.items():
        if any(kw in hay for kw in kws):
            found.append(facet)

    if vols and "nonprofit_volunteering" not in found:
        found.append("nonprofit_volunteering")

    # company intelligence (already cached) sharpens big_tech / founder_startup
    for e in exps:
        if not e.company_name:
            continue
        cc = company_class.get(company_key(e.company_id, e.company_name))
        if not cc:
            continue
        if cc.get("is_big_tech") and "big_tech" not in found:
            found.append("big_tech")
        if cc.get("is_startup") and "founder_startup" not in found:
            found.append("founder_startup")
    return found


def build_people(db: Session, dataset_id: str) -> list[PilotPerson]:
    """Tag every READY/PARTIAL person in the dataset with facets + data-quality flags."""
    people = [
        p for p in repo.list_people(db, dataset_id, is_connection=True)
        if p.enrichment_state in ("READY", "PARTIAL")
    ]
    pids = [p.id for p in people]
    exp_by = repo.bulk_experiences(db, pids)
    edu_by = repo.bulk_education(db, pids)
    skill_by = repo.bulk_skills(db, pids)
    vol_by = repo.bulk_volunteering(db, pids)
    sem_by = repo.bulk_semantics(db, pids)
    emb_by = repo.bulk_embeddings_by_person(db, pids)

    # cache-only company classification for the whole pool (no LLM)
    seen: dict = {}
    for exps in exp_by.values():
        for e in exps:
            if e.company_name:
                seen.setdefault((e.company_id, e.company_name), True)
    keys = [company_key(cid, nm) for (cid, nm) in seen]
    from app.services.company_intel import to_dict
    rows = repo.get_company_semantics(db, keys)
    company_class = {k: to_dict(r) for k, r in rows.items()}

    out: list[PilotPerson] = []
    for p in people:
        facets = _facets_for(
            p, exp_by.get(p.id, []), edu_by.get(p.id, []), skill_by.get(p.id, []),
            vol_by.get(p.id, []), company_class,
        )
        out.append(PilotPerson(
            person_id=p.id,
            public_identifier=p.public_identifier,
            full_name=p.full_name,
            completeness=p.profile_completeness or 0,
            completeness_tier=_tier(p.profile_completeness or 0),
            enrichment_state=p.enrichment_state,
            semantic_version=p.semantic_version,
            has_semantic=p.id in sem_by,
            has_embedding=p.id in emb_by,
            facets=facets,
        ))
    return out


def select_pilot(
    db: Session,
    dataset_id: str,
    *,
    target: int = DEFAULT_TARGET,
    seed: str = DEFAULT_SEED,
) -> PilotSample:
    tagged = build_people(db, dataset_id)
    by_id = {p.person_id: p for p in tagged}
    ordered_ids = sorted(by_id, key=lambda pid: _order_key(seed, pid))

    selected: list[str] = []
    seen: set[str] = set()

    def take(pid: str, via: str) -> None:
        if pid in seen:
            return
        seen.add(pid)
        by_id[pid].selected_via = via
        selected.append(pid)

    # 1. reserved data-quality slots (hash-ordered within tier)
    sparse = [pid for pid in ordered_ids if by_id[pid].completeness_tier == "sparse"]
    medium = [pid for pid in ordered_ids if by_id[pid].completeness_tier == "medium"]
    for pid in sparse[:RESERVED_SPARSE]:
        take(pid, "reserved:sparse")
    for pid in medium[:RESERVED_MEDIUM]:
        take(pid, "reserved:medium")

    # 2. facet round-robin for the remainder
    facet_pool: dict[str, list[str]] = {
        f: [pid for pid in ordered_ids if f in by_id[pid].facets] for f in FACET_ORDER
    }
    facet_cursor = {f: 0 for f in FACET_ORDER}
    stalled = 0
    while len(selected) < target and stalled < len(FACET_ORDER):
        progressed = False
        for f in FACET_ORDER:
            if len(selected) >= target:
                break
            pool = facet_pool[f]
            while facet_cursor[f] < len(pool) and pool[facet_cursor[f]] in seen:
                facet_cursor[f] += 1
            if facet_cursor[f] < len(pool):
                take(pool[facet_cursor[f]], f"facet:{f}")
                facet_cursor[f] += 1
                progressed = True
        stalled = 0 if progressed else stalled + 1

    # 3. last resort — fill from the global hash order (rarely needed)
    for pid in ordered_ids:
        if len(selected) >= target:
            break
        take(pid, "fill:hash-order")

    return PilotSample(
        dataset_id=dataset_id, seed=seed, target=target,
        people=[by_id[pid] for pid in selected],
    )
