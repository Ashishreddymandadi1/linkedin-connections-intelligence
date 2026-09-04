"""Heuristic flags on the SearchPlan itself (V4 PART 10 §7).

These are ADVISORY signals for the human reviewer — not assertions and not
auto-fixes. No per-query logic: every check reads only the query text and the
produced plan.
"""
from __future__ import annotations

import re

_EVENT_WORDS = ("event", "networking", "conference", "meetup", "party", "dinner",
                "gala", "summit", "reception", "mixer", "invite to")
_PAST_WORDS = ("former", "ex-", "previously", "used to", "past ", "onetime", "one-time")
_TRANSITION_WORDS = ("moved from", "move from", "transition", "switched from", "pivot",
                     "left academia", "into industry", "into management", "trying to move")
_POSSIBLE_WORDS = ("might", "maybe", "possibly", "could have", "may have", "perhaps")
_AND_DOMAIN = re.compile(r"\b(and|both|plus|as well as|combined with)\b", re.I)


def flag_plan(query: str, interp: dict) -> list[dict]:
    q = query.lower()
    crits = interp.get("criteria", [])
    flags: list[dict] = []

    def add(code: str, detail: str) -> None:
        flags.append({"code": code, "detail": detail})

    ctypes = [c.get("type") for c in crits]
    values_text = " ".join(
        f"{c.get('value') or ''} {c.get('concept') or ''} {' '.join(c.get('values') or [])}"
        for c in crits
    ).lower()

    # 1. event / framing context leaking into candidate criteria
    if any(w in q for w in _EVENT_WORDS):
        leaked = [w for w in _EVENT_WORDS if w in values_text]
        if leaked:
            add("context_leak", f"event framing may be in criteria values: {leaked}")
        if not interp.get("context"):
            add("context_missing", "query has event framing but plan.context is empty")

    # 2. past-tense company/role with no past scope anywhere
    if any(w in q for w in _PAST_WORDS):
        if not any("past" in (c.get("scope") or "") for c in crits):
            add("scope_past_missing", "query says former/previously but no criterion has a past scope")

    # 3. career transition language with no career_transition criterion
    if any(w in q for w in _TRANSITION_WORDS):
        if "career_transition" not in ctypes:
            add("transition_missing", "query describes a move/transition but no career_transition criterion")

    # 4. hedged phrasing with no 'possible' modality
    if any(w in q for w in _POSSIBLE_WORDS):
        if not any(c.get("modality") == "possible" for c in crits):
            add("modality_missing", "query hedges (might/maybe) but no criterion modality=possible")

    # 5. cross-domain AND: two content words joined by 'and' but no ALL_OF / <2 required
    if _AND_DOMAIN.search(q):
        required_content = [c for c in crits if c.get("required") and c.get("type") not in ("location",)]
        all_of = any(c.get("operator") == "ALL_OF" for c in crits)
        if len(required_content) < 2 and not all_of:
            add("weak_and", "query joins concepts with AND but plan has <2 required content criteria and no ALL_OF")

    # 6. nothing required at all
    if crits and not any(c.get("required") for c in crits):
        add("no_required", "plan has zero required criteria")

    # 7. unresolved context surfaced
    if interp.get("unresolved"):
        add("unresolved", f"interpreter could not resolve: {interp['unresolved']}")

    # 8. relational query with empty target_person_context
    if ("mentor" in q or "advice" in q or "my field" in q) and not interp.get("target_person_context"):
        add("target_context_empty", "relational/advice query but target_person_context is empty")

    return flags
