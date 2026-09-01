"""Geographic interpretation — a small normalization/ontology layer (spec §12).

Not a paid API, not a giant per-query dictionary. The query planner is asked to
expand a metro/region into its cities in the criterion's ``values``; this module
provides (a) a compact static fallback for the most common US metros so the
deterministic path and an under-expanding LLM still work, and (b) a
token-subset location matcher so "San Jose" matches
"San Jose, California, United States" (not the strict 75%-phrase rule that
broke "Bay Area" vs "Palo Alto").
"""
from __future__ import annotations

from app.services.matching import norm

#: region alias -> canonical member cities/areas (lower-cased). Kept deliberately
#: small — the LLM planner does the general case; this covers the metros that
#: show up constantly and must never silently fail.
_REGIONS: dict[str, list[str]] = {
    "bay area": [
        "san francisco", "san jose", "oakland", "palo alto", "mountain view", "sunnyvale",
        "santa clara", "menlo park", "redwood city", "cupertino", "fremont", "berkeley",
        "san mateo", "foster city", "south san francisco", "emeryville", "burlingame",
        "silicon valley",
    ],
    "silicon valley": [
        "san jose", "palo alto", "mountain view", "sunnyvale", "santa clara", "menlo park",
        "cupertino", "redwood city", "san francisco",
    ],
    "sf bay area": ["san francisco", "san jose", "oakland", "palo alto", "mountain view", "sunnyvale"],
    "greater new york": ["new york", "brooklyn", "jersey city", "newark", "manhattan", "queens"],
    "nyc": ["new york", "brooklyn", "manhattan", "jersey city"],
    "new york city": ["new york", "brooklyn", "manhattan"],
    "greater seattle": ["seattle", "bellevue", "redmond", "kirkland", "tacoma"],
    "greater boston": ["boston", "cambridge", "somerville", "waltham"],
    "greater los angeles": ["los angeles", "santa monica", "pasadena", "burbank", "el segundo"],
    "greater chicago": ["chicago", "evanston", "naperville"],
    "dmv": ["washington", "arlington", "alexandria", "bethesda", "reston", "mclean"],
    "washington dc area": ["washington", "arlington", "alexandria", "bethesda"],
    "research triangle": ["raleigh", "durham", "chapel hill", "cary"],
    "dfw": ["dallas", "fort worth", "plano", "irving"],
}


def expand_region(value: str) -> list[str]:
    """Return member cities for a region name, or ``[value]`` if it isn't a
    known region. Always includes the original value too."""
    v = norm(value)
    members = _REGIONS.get(v)
    if members:
        return [value, *members]
    return [value]


def expand_values(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values or []:
        for x in expand_region(v):
            if x not in out:
                out.append(x)
    return out


def location_matches(location_fields: list[str | None], value: str) -> bool:
    """True if ``value``'s significant tokens are all present in any of the
    person's location strings (token-subset, order-free). 'San Jose' matches
    'San Jose, California, United States'; 'Bay Area' matches 'San Francisco
    Bay Area'."""
    val_tokens = {t for t in norm(value).split() if len(t) > 1}
    if not val_tokens:
        return False
    for field in location_fields:
        hay = norm(field)
        if hay and val_tokens <= set(hay.split()):
            return True
    return False
