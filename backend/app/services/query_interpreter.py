"""Natural-language query → weighted structured criteria (spec §30–§32).

Primary path: free-LLM chain producing a ``ParsedSearchQuery`` (weights forced to
sum to 100 by the schema validator). Fallback: a deterministic regex/keyword
parser that always runs and is good enough on the common query shapes.
"""
from __future__ import annotations

import logging
import re

from app.config import settings
from app.constants import CriterionType
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.llm.router import generate_structured
from app.services.matching import seniority_rank

log = logging.getLogger("app.query")

_KNOWN_SKILLS = {
    "aws", "gcp", "azure", "java", "python", "golang", "go", "rust", "c++", "typescript",
    "javascript", "react", "node", "node.js", "kubernetes", "docker", "terraform", "kafka",
    "spark", "sql", "postgresql", "mysql", "redis", "mongodb", "graphql", "distributed systems",
    "machine learning", "ml", "deep learning", "nlp", "llm", "llms", "pytorch", "tensorflow",
    "data engineering", "mlops", "devops", "security", "cryptography", "blockchain",
    "microservices", "system design", "cloud", "cloud infrastructure", "networking",
    "product management", "design", "ui", "ux", "fintech", "payments", "fraud",
}

_SYSTEM = (
    "You convert a recruiter/networking search into weighted structured criteria. "
    "Output JSON only. Each criterion has: id (short slug), type (one of "
    "current_company, past_company, skill, domain, title, education, location, seniority, "
    "certification, language, publication, keyword), "
    "value (the concrete thing to match), weight (number), required (boolean). "
    "Use `certification` for 'has an AWS cert' (value 'AWS'), `language` for 'speaks "
    "Mandarin' (value 'Mandarin'), `publication` for 'has published' (value '' or the topic). "
    "The query's own emphasis decides the weights — 'people who went to Stanford' is mostly an "
    "education criterion; 'backend engineers currently at Microsoft' weights current_company and "
    "title highly. A criterion is required only when the query clearly demands it "
    "('currently at Google who know Java' → Google required, Java preferred). "
    "Weights MUST sum to 100."
)


_MODAL_RE = re.compile(r"\b(must|only|required|has to|need to have|exclusively)\b", re.I)
_SOFT_TYPES = {CriterionType.SKILL, CriterionType.DOMAIN, CriterionType.KEYWORD, CriterionType.SENIORITY, CriterionType.TITLE}


def _soften_requirements(parsed: ParsedSearchQuery, query: str) -> ParsedSearchQuery:
    """A skill/domain/title is a hard requirement only when the query says so
    ('must know X'). Company/education/location keep the model's flag. Also cap
    total required criteria so an over-eager model can't exclude everyone."""
    hard_modal = bool(_MODAL_RE.search(query))
    for c in parsed.criteria:
        if c.type in _SOFT_TYPES and not hard_modal:
            c.required = False
    required = [c for c in parsed.criteria if c.required]
    if len(required) > 2:
        for c in sorted(required, key=lambda x: x.weight)[:-2]:
            c.required = False
    return parsed


def interpret_query(query: str) -> tuple[ParsedSearchQuery, str, str | None]:
    """Return ``(parsed, provider_name, model)``. provider = 'deterministic' on fallback."""
    if settings.llm_query_interpretation:
        result = generate_structured(
            _SYSTEM,
            f"Search query: {query!r}\nProduce the criteria JSON.",
            ParsedSearchQuery,
            max_tokens=900,
        )
        if result is not None:
            parsed, provider, model = result
            if parsed.criteria:
                return _soften_requirements(parsed, query), provider, model
        log.info("query interpreter: falling back to deterministic parser")

    return _soften_requirements(_deterministic_parse(query), query), "deterministic", None


# ─────────────────────── deterministic parser ───────────────────────

_PAST_RE = re.compile(r"(?:previously|formerly|ex[- ]|used to (?:work|be)|before)\s+(?:worked\s+(?:at|for)\s+)?", re.I)
_CURRENT_RE = re.compile(r"currently\s+(?:works?\s+)?(?:at|for|with)\s+", re.I)
_AT_RE = re.compile(r"\b(?:works?\s+at|worked\s+at|at)\s+([A-Z][\w&.\- ]+?)(?:\s+(?:who|that|and|,|\.|$))", re.I)
_SCHOOL_RE = re.compile(r"(?:studied at|went to|graduated from|degree from|alum(?:ni|nus)? of|attended)\s+([A-Z][\w&.\- ]+)", re.I)
_STUDY_RE = re.compile(r"stud(?:ied|ying)\s+([a-z][\w ]+?)(?:\s+(?:at|and|,|\.|$))", re.I)


def _clean_name(s: str) -> str:
    return re.sub(r"\s+(who|that|and|with)\b.*$", "", s.strip(" .,")).strip()


def _deterministic_parse(query: str) -> ParsedSearchQuery:
    q = query.strip()
    ql = q.lower()
    crits: list[SearchCriterion] = []
    used_spans: list[str] = []

    for m in re.finditer(r"(previously|formerly|ex[- ])\s*(?:worked\s+(?:at|for)\s+)?([A-Z][\w&.\- ]+)", q, re.I):
        name = _clean_name(m.group(2))
        if name:
            crits.append(SearchCriterion(id=f"past_{_slug(name)}", type=CriterionType.PAST_COMPANY, value=name, weight=30, required=False))
            used_spans.append(name.lower())

    for m in _CURRENT_RE.finditer(q):
        tail = q[m.end():]
        name = _clean_name(re.split(r"\b(who|that|and)\b", tail)[0])
        if name:
            crits.append(SearchCriterion(id=f"cur_{_slug(name)}", type=CriterionType.CURRENT_COMPANY, value=name, weight=35, required="currently" in ql and " who " in ql))
            used_spans.append(name.lower())

    for m in _SCHOOL_RE.finditer(q):
        name = _clean_name(m.group(1))
        if name and name.lower() not in used_spans:
            crits.append(SearchCriterion(id=f"edu_{_slug(name)}", type=CriterionType.EDUCATION, value=name, weight=40, required=False))
            used_spans.append(name.lower())

    for m in _STUDY_RE.finditer(q):
        field = m.group(1).strip()
        if field and len(field) > 2:
            crits.append(SearchCriterion(id=f"field_{_slug(field)}", type=CriterionType.EDUCATION, value=field, weight=25, required=False))

    for m in re.finditer(r"([A-Za-z][\w+.\- ]{1,30}?)\s+(?:certified|certification|cert)\b", q, re.I):
        val = _clean_name(m.group(1))
        if val and val.lower() not in used_spans:
            crits.append(SearchCriterion(id=f"cert_{_slug(val)}", type=CriterionType.CERTIFICATION, value=val, weight=35, required=False))
            used_spans.append(val.lower())
    for m in re.finditer(r"\bspeaks?\s+([A-Za-z]+(?:\s+(?:or|and)\s+[A-Za-z]+)*)", q, re.I):
        val = m.group(1).strip()
        crits.append(SearchCriterion(id=f"lang_{_slug(val)}", type=CriterionType.LANGUAGE, value=val, weight=40, required=False))
    if re.search(r"\b(publish(?:ed|es)?|publication|research paper|wrote a paper)\b", ql):
        topic_m = re.search(r"publish(?:ed)?(?:\s+(?:research|papers?|on))?\s+(?:on\s+|about\s+)?([a-z][\w ]{2,40})", ql)
        topic = topic_m.group(1).strip() if topic_m else ""
        crits.append(SearchCriterion(id="pub", type=CriterionType.PUBLICATION, value=topic, weight=30, required=False))

    if not any(c.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY) for c in crits):
        for m in _AT_RE.finditer(q + " "):
            name = _clean_name(m.group(1))
            if name and name.lower() not in used_spans and name.lower() not in _KNOWN_SKILLS:
                crits.append(SearchCriterion(id=f"co_{_slug(name)}", type=CriterionType.CURRENT_COMPANY, value=name, weight=30, required=False))
                used_spans.append(name.lower())

    for skill in sorted(_KNOWN_SKILLS, key=lambda s: -len(s)):
        if re.search(rf"\b{re.escape(skill)}\b", ql):
            if any(skill in u for u in used_spans):
                continue
            crits.append(SearchCriterion(id=f"skill_{_slug(skill)}", type=CriterionType.SKILL, value=skill, weight=20, required=False))
            used_spans.append(skill)

    rank = seniority_rank(ql)
    if rank is not None and rank >= 3:
        label = next((w for w in ["principal", "staff", "senior", "director", "vp", "lead", "founder", "head"] if w in ql), "senior")
        crits.append(SearchCriterion(id="seniority", type=CriterionType.SENIORITY, value=label, weight=15, required=False))

    for m in re.finditer(r"\b([a-z]+)\s+(engineer|engineers|developer|developers|manager|managers|scientist|scientists|designer|designers|architect|architects)\b", ql):
        title = f"{m.group(1)} {m.group(2)}".rstrip("s")
        crits.append(SearchCriterion(id=f"title_{_slug(title)}", type=CriterionType.TITLE, value=title, weight=20, required=False))

    if not crits:
        for kw in _keywords(q):
            crits.append(SearchCriterion(id=f"kw_{_slug(kw)}", type=CriterionType.KEYWORD, value=kw, weight=10, required=False))
    if not crits:
        crits.append(SearchCriterion(id="kw_all", type=CriterionType.KEYWORD, value=q[:60], weight=100, required=False))

    return ParsedSearchQuery(intent="professional_recommendation", criteria=crits)


_STOP = {
    "who", "should", "i", "reach", "out", "to", "for", "about", "people", "person", "in", "my",
    "network", "and", "the", "a", "an", "with", "know", "knows", "at", "of", "that", "currently",
    "previously", "worked", "works", "both", "someone", "find", "list", "give", "me",
}


def _keywords(q: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z+.#-]{2,}", q.lower())
    return list(dict.fromkeys(w for w in words if w not in _STOP))[:6]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:24] or "x"
