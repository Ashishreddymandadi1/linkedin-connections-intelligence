"""Deterministic query-fact layer (V4 §14–§18).

Runs on EVERY query, regardless of whether an LLM answered:

  strip_context()      pull out non-candidate framing ("... networking event")
  extract_facts()      high-precision structured facts: locations (incl. OR),
                       named companies with current/past/NOT scope, boolean
                       operators, "N+ years"
  merge_into_plan()    fold those hard facts into the LLM's SearchPlan — the LLM
                       may enrich meaning, it may not delete an explicit
                       location / company / NOT (V4 §15/§17)
  validate_and_repair()catch a plan that dropped or inverted an explicit
                       constraint and fix it deterministically (V4 §17)
  build_summary()      one-sentence interpretation + a confidence score (V4 §18)

MEANING over words still holds: this layer only asserts the parts of a query
that are literally checkable (a city, a company name, an operator). Industry /
role / seniority *concepts* are left to the LLM + semantic scorer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.constants import CriterionType, Operator, Scope
from app.schemas import ParsedSearchQuery, SearchCriterion

# ─────────────────────────── context (V4 §14) ───────────────────────────

_EVENT_NOUNS = (
    "networking event", "event", "conference", "meetup", "meet-up", "panel",
    "webinar", "summit", "mixer", "dinner", "luncheon", "gathering", "gala",
    "reception", "roundtable", "fireside chat", "workshop", "session", "party",
    "happy hour",
)
_EVENT_RE = re.compile(
    r"\b(?:a|an|the|our|my|this|for)\s+([a-z][\w&/-]*(?:\s+[a-z][\w&/-]*){0,3}?)\s+"
    r"(" + "|".join(re.escape(n) for n in _EVENT_NOUNS) + r")\b",
    re.I,
)
_PURPOSE_VERB_RE = re.compile(r"\b(invite|speak|present|recommend|nominate|refer)\b", re.I)


#: modifier words that are pure event framing (drop with the noun); anything
#: else in front of the event noun (e.g. "CXO") stays a candidate requirement.
_FRAMING_WORDS = {"networking", "social", "professional", "industry", "alumni", "community"}


def strip_context(query: str) -> tuple[str, dict[str, str]]:
    """Return ``(query_without_context_phrase, context)``. ``context['purpose']``
    holds the event description; only the event noun (and any pure-framing word
    right before it) is removed — real requirements like 'CXO' are left in."""
    context: dict[str, str] = {}
    m = _EVENT_RE.search(query)
    if m:
        modifier, noun = m.group(1).strip(), m.group(2).strip()
        context["purpose"] = f"{modifier} {noun}".strip()
        cut_start = m.start(2)
        # also swallow an immediately-preceding pure-framing word
        pre = query[m.start(1):m.start(2)].strip().split()
        if pre and pre[-1].lower() in _FRAMING_WORDS:
            cut_start = m.start(1) + query[m.start(1):m.start(2)].rfind(pre[-1])
        query = (query[:cut_start] + " " + query[m.end(2):]).strip()
    elif _PURPOSE_VERB_RE.search(query):
        context["purpose"] = "professional recommendation"
    return re.sub(r"\s{2,}", " ", query).strip(" ,."), context


def context_terms(context: dict[str, str]) -> set[str]:
    """Lower-cased words that came from context and must never be a criterion."""
    out: set[str] = set()
    for v in context.values():
        out.update(re.findall(r"[a-z][a-z+.#-]{2,}", v.lower()))
    return out


# ─────────────────────────── fact extraction (V4 §15/§16) ───────────────────────────

_REGION_WORDS = (
    "bay area", "silicon valley", "sf bay area", "san francisco bay area",
    "greater new york", "new york city", "nyc", "greater boston", "greater seattle",
    "greater los angeles", "greater chicago", "dfw", "research triangle",
    "pacific northwest", "tri-state area", "socal", "norcal",
)
_LOC_OR_RE = re.compile(
    r"\b(?:in|from|near|around|based in|located in)\s+"
    r"((?:[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*)"
    r"(?:\s*,\s*(?:or\s+)?(?:[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*)"
    r"|\s+or\s+(?:[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*))+)",
)
#: single "in Nashville" / "in Austin, Texas" — capitalised, case-SENSITIVE so
#: "in big tech" is not mistaken for a place. Regions handled separately.
_LOC_ONE_RE = re.compile(
    r"\b(?:in|from|near|around|based in|located in)\s+"
    r"(?:the\s+)?([A-Z][\w.\-]+(?:\s+[A-Z][\w.\-]+){0,3})"
)
_REGION_IN_RE = re.compile(
    r"\b(?:in|from|near|around|based in|located in)\s+(?:the\s+)?("
    + "|".join(re.escape(r) for r in _REGION_WORDS) + r")\b",
    re.I,
)

_FORMER_RE = re.compile(
    r"(?i:\b(?:former|formerly|previously|ex[- ]|used to work (?:at|for))\s*"
    r"(?:employees?\s+of\s+|people\s+(?:at|from)\s+|worked\s+(?:at|for)\s+)?)"
    r"([A-Z][\w&.\-]*(?:\s+(?:[A-Z][\w&.\-]*|or|and|&|of))*)"
)
_CURRENT_RE = re.compile(
    r"(?i:\b(?:currently|now|presently)\s+(?:at|with|working at|employed at)\s+|"
    r"\bworks?\s+at\s+)"
    r"([A-Z][\w&.\-]*(?:\s+(?:[A-Z][\w&.\-]*|or|and|&))*)"
)
_NOT_AT_RE = re.compile(
    r"(?i:\bnot\s+(?:currently\s+)?(?:at|working at|employed at|with)\s+)"
    r"([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*)*)"
)
_YEARS_RE = re.compile(r"\b(\d{1,2})\s*\+?\s*years?\b", re.I)
_LAST_N_RE = re.compile(r"\b(?:last|past|recent)\s+(\d{1,2})\s+years?\b", re.I)

_NOT_A_COMPANY = {
    "tech", "big tech", "faang", "startup", "startups", "consulting", "fintech",
    "healthcare", "industry", "product", "engineering",
}
_STOP_TAIL_RE = re.compile(r"\s+(?:who|that|which|and now|,|\.).*$", re.I)


def _clean_company(s: str) -> str:
    s = _STOP_TAIL_RE.sub("", s).strip(" .,&")
    s = re.sub(r"\s+(?:of|and|&|or)$", "", s).strip()
    return s


def _split_or(blob: str) -> list[str]:
    return [p.strip(" .,") for p in re.split(r"\s*,\s*(?:or\s+)?|\s+or\s+|\s+and\s+", blob) if p.strip(" .,")]


@dataclass
class FactSet:
    criteria: list[SearchCriterion] = field(default_factory=list)
    consumed: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def has_type(self, *types: str) -> bool:
        return any(c.type in types for c in self.criteria)


def extract_facts(query: str, *, context: dict[str, str] | None = None) -> FactSet:
    fs = FactSet()
    ctx_words = context_terms(context or {})
    q = query

    loc_m = _LOC_OR_RE.search(q)
    places: list[str] = []
    if loc_m:
        places = _split_or(loc_m.group(1))
    else:
        region_m = _REGION_IN_RE.search(q)
        if region_m:
            places = [region_m.group(1).strip()]
        else:
            for one in _LOC_ONE_RE.finditer(q):
                cand = one.group(1).strip()
                if _looks_like_place(cand):
                    places = [cand]
                    break
    if places:
        fs.criteria.append(SearchCriterion(
            id="loc", type=CriterionType.LOCATION, values=places,
            operator=Operator.ANY_OF, weight=30, required=True,
        ))
        fs.consumed.update(p.lower() for p in places)

    for m in _NOT_AT_RE.finditer(q):
        name = _clean_company(m.group(1))
        if name and name.lower() not in _NOT_A_COMPANY:
            fs.criteria.append(SearchCriterion(
                id=f"not_{_slug(name)}", type=CriterionType.CURRENT_COMPANY, value=name,
                operator=Operator.NOT, scope=Scope.CURRENT_COMPANY, weight=25, required=True,
            ))
            fs.consumed.add(name.lower())

    for m in _FORMER_RE.finditer(q):
        blob = _clean_company(m.group(1))
        if not blob:
            continue
        names = [n for n in _split_or(blob) if n.lower() not in _NOT_A_COMPANY]
        if not names or any(n.lower() in fs.consumed for n in names):
            continue
        fs.criteria.append(SearchCriterion(
            id=f"past_{_slug(names[0])}", type=CriterionType.PAST_COMPANY,
            values=names, operator=Operator.ANY_OF, scope=Scope.PAST_COMPANY,
            weight=35, required=True,
        ))
        fs.consumed.update(n.lower() for n in names)

    for m in _CURRENT_RE.finditer(q):
        blob = _clean_company(m.group(1))
        names = [n for n in _split_or(blob) if n.lower() not in _NOT_A_COMPANY]
        if not names or any(n.lower() in fs.consumed for n in names):
            continue
        fs.criteria.append(SearchCriterion(
            id=f"cur_{_slug(names[0])}", type=CriterionType.CURRENT_COMPANY,
            values=names, operator=Operator.ANY_OF, scope=Scope.CURRENT_COMPANY,
            weight=35, required=True,
        ))
        fs.consumed.update(n.lower() for n in names)

    last_n = _LAST_N_RE.search(q)
    if last_n:
        fs.notes.append(f"recency window: last {last_n.group(1)} years")
    else:
        ym = _YEARS_RE.search(q)
        if ym and "year" in q.lower():
            n = int(ym.group(1))
            fs.criteria.append(SearchCriterion(
                id="years_exp", type=CriterionType.SEMANTIC_CONCEPT,
                concept=f"at least {n} years of professional experience",
                scope=Scope.CAREER, weight=15, required=False,
            ))

    fs.criteria = [c for c in fs.criteria if (c.concept or c.value or "").lower() not in ctx_words]
    return fs


def _looks_like_place(s: str) -> bool:
    if s.lower() in _REGION_WORDS:
        return True
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:,\s*[A-Z][a-z]+| [A-Z]{2})?$", s):
        return s.lower() not in _NOT_A_COMPANY
    return False


# ─────────────────────────── merge + repair (V4 §15/§17) ───────────────────────────

_FACT_TYPES = {
    CriterionType.LOCATION, CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY,
}


def _values_of(c: SearchCriterion) -> set[str]:
    vs = set(c.values or [])
    if c.value:
        vs.add(c.value)
    return {v.strip().lower() for v in vs if v.strip()}


def _same_fact(a: SearchCriterion, b: SearchCriterion) -> bool:
    if a.type != b.type and {a.type, b.type} != {CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY}:
        return False
    return bool(_values_of(a) & _values_of(b))


def merge_into_plan(plan: ParsedSearchQuery, facts: FactSet, *, context: dict[str, str]) -> tuple[ParsedSearchQuery, list[str]]:
    """Fold deterministic hard facts into an LLM plan. The LLM keeps its semantic
    criteria; explicit facts it dropped are re-added; explicit facts it mangled
    (wrong scope / OR->AND / lost NOT) are corrected toward the deterministic
    reading."""
    issues: list[str] = []
    crits = list(plan.criteria)
    ctx_words = context_terms(context)

    fact_values = {v for f in facts.criteria for v in _values_of(f)} | facts.consumed
    kept: list[SearchCriterion] = []
    for c in crits:
        cv = (c.value or c.concept or "").lower()
        if c.type in (CriterionType.SKILL, CriterionType.KEYWORD) and cv in ctx_words:
            issues.append(f"dropped context word '{c.value}' parsed as a {c.type}")
            continue
        # a weak one-word concept/keyword that just restates a hard fact is noise
        if c.type in (CriterionType.SKILL, CriterionType.KEYWORD, CriterionType.SEMANTIC_CONCEPT) \
                and cv in fact_values and len(cv.split()) <= 2 and not c.required:
            issues.append(f"dropped redundant {c.type} '{cv}' (already a hard fact)")
            continue
        kept.append(c)
    crits = kept

    for f in facts.criteria:
        match = next((c for c in crits if _same_fact(c, f)), None)
        if match is None:
            issues.append(f"re-added explicit {f.type} {sorted(_values_of(f))} the plan omitted")
            crits.append(f)
            continue
        # union any explicit values the plan missed ("Google or Meta" -> both)
        missing = _values_of(f) - _values_of(match)
        if missing:
            have = match.values or ([match.value] if match.value else [])
            add = [v for v in (f.values or [f.value]) if v.lower() in missing]
            match.values = have + add
            match.value = ""
            issues.append(f"added missing values {sorted(missing)} to {match.type}")
        if f.scope and match.scope != f.scope and f.type in (CriterionType.PAST_COMPANY, CriterionType.CURRENT_COMPANY):
            issues.append(f"fixed scope of {sorted(_values_of(match))}: {match.scope} -> {f.scope}")
            match.scope = f.scope
            match.type = f.type
        if f.operator == Operator.ANY_OF and match.operator == Operator.ALL_OF and len(_values_of(f)) > 1:
            issues.append(f"restored ANY_OF for {sorted(_values_of(match))} (plan had ALL_OF)")
            match.operator = Operator.ANY_OF
        if f.operator == Operator.NOT and match.operator != Operator.NOT:
            issues.append(f"restored NOT for {sorted(_values_of(match))}")
            match.operator = Operator.NOT
        if f.required and not match.required:
            issues.append(f"made {sorted(_values_of(match))} required (explicit in the query)")
            match.required = True

    plan.criteria = crits
    plan.context = {**plan.context, **context}
    return _renorm(plan), issues


def validate_and_repair(plan: ParsedSearchQuery, query: str, facts: FactSet, *, context: dict[str, str]) -> tuple[ParsedSearchQuery, list[str]]:
    """Final safety net (V4 §17)."""
    plan, issues = merge_into_plan(plan, facts, context=context)

    ql = query.lower()
    if re.search(r"\b\w+\s+or\s+\w+", ql) and not any(
        c.operator == Operator.ANY_OF and len(_values_of(c)) > 1 for c in plan.criteria
    ):
        issues.append("query contains 'or' but no OR criterion was produced")
    if re.search(r"\bnot\b|\bnever\b|\bexcept\b|\bexcluding\b", ql) and not any(
        c.operator == Operator.NOT for c in plan.criteria
    ):
        issues.append("query contains a negation but no NOT criterion was produced")
    return plan, issues


# ─────────────────────────── summary + confidence (V4 §18) ───────────────────────────

_AMBIGUOUS_RE = re.compile(
    r"\b(worked in tech|in tech|technical people|technology executives?|good people|"
    r"startup experience|strong profile|the best people)\b", re.I,
)


def build_summary(plan: ParsedSearchQuery, query: str) -> tuple[str, float]:
    parts: list[str] = []
    for c in plan.criteria:
        vals = sorted(_values_of(c)) or ([c.concept] if c.concept else [])
        joiner = " or " if c.operator == Operator.ANY_OF else " and "
        label = joiner.join(v for v in vals if v) or c.type
        prefix = "not " if c.operator == Operator.NOT else ""
        scope = f" ({c.scope})" if c.scope and c.scope != Scope.CAREER else ""
        tag = "required" if c.required else "preferred"
        parts.append(f"{prefix}{c.type.replace('_', ' ')}: {label}{scope} [{tag}]")

    summary = "Interpreted as — " + "; ".join(parts) if parts else "No structured criteria found."
    if plan.context.get("purpose"):
        summary += f". Context (not a filter): {plan.context['purpose']}."

    confidence = 0.85
    if _AMBIGUOUS_RE.search(query):
        confidence -= 0.25
    required_facts = sum(1 for c in plan.criteria if c.required and c.type in _FACT_TYPES)
    if required_facts == 0 and any(
        c.type in (CriterionType.SEMANTIC_CONCEPT, CriterionType.COMPANY_CATEGORY) for c in plan.criteria
    ):
        confidence -= 0.1
    return summary, round(max(0.2, min(1.0, confidence)), 2)


# ─────────────────────────── helpers ───────────────────────────

def _renorm(plan: ParsedSearchQuery) -> ParsedSearchQuery:
    total = sum(c.weight for c in plan.criteria) or 1.0
    if abs(total - 100.0) > 0.5:
        for c in plan.criteria:
            c.weight = round(c.weight / total * 100, 2)
    return plan


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:24] or "x"
