"""Hardening PART 13 — ONE authoritative current-employer view.

A real live result showed the final audit naming one current employer while
the display "reason" text named a different one (``Person.current_company``
had gone stale relative to the normalized, backend-maintained experiences
table). Every consumer of "current employer" — scoring, the judge/audit
packet, the API response, and reason generation — must resolve to the SAME
value: the normalized ``is_current`` experience row wins over the
denormalized scrape field.
"""
from __future__ import annotations

from app.models import Dataset, Experience, Person
from app.schemas import EvidenceItem, ParsedSearchQuery, SearchCriterion
from app.services.profile_authority import current_employer, current_employer_from
from app.services.reason_generator import generate_reason
from app.services.scoring import ProfileFacts, ScoredCandidate
from app.services.search_service import _to_result_item


class _Facts:
    def __init__(self, person, experiences):
        self.person = person
        self.experiences = experiences


def _person(**kw) -> Person:
    return Person(
        id=kw.pop("id", "p1"), dataset_id="ds1", is_connection=True,
        linkedin_url=kw.pop("linkedin_url", "https://www.linkedin.com/in/p1"),
        full_name=kw.pop("full_name", "Test Person"), **kw,
    )


def _exp(**kw) -> Experience:
    return Experience(id=kw.pop("id", "exp1"), person_id="p1", **kw)


def test_current_employer_prefers_normalized_current_experience_over_stale_person_field():
    p = _person(current_company="Cox Automotive")
    cur = _exp(company_name="Clever Vixen Media", position="Manager", is_current=True)
    facts = _Facts(p, [cur])
    assert current_employer_from(p, [cur]) == "Clever Vixen Media"
    assert current_employer(facts) == "Clever Vixen Media"


def test_current_employer_falls_back_to_person_field_when_no_current_experience_row():
    p = _person(current_company="Cox Automotive")
    past = _exp(company_name="Old Co", position="Analyst", is_current=False)
    assert current_employer_from(p, [past]) == "Cox Automotive"
    assert current_employer_from(p, []) == "Cox Automotive"
    assert current_employer_from(p, None) == "Cox Automotive"


def test_current_employer_ignores_past_experiences_even_when_listed_first():
    p = _person(current_company=None)
    past = _exp(id="e1", company_name="Old Co", is_current=False)
    cur = _exp(id="e2", company_name="New Co", is_current=True)
    assert current_employer_from(p, [past, cur]) == "New Co"


def test_result_item_current_company_matches_the_current_experience_not_the_stale_person_field(db):
    """End-to-end through the same function search_service uses to build a
    result: the displayed current_company must be the normalized experience,
    never the stale Person.current_company (the exact live bug)."""
    db.add(Dataset(id="ds1"))
    db.flush()
    p = Person(
        id="p1", dataset_id="ds1", is_connection=True,
        linkedin_url="https://www.linkedin.com/in/p1", full_name="Nina Jennings",
        current_title="Marketing Manager", current_company="Cox Automotive",
    )
    db.add(p)
    db.flush()
    cur = Experience(
        id="exp1", person_id="p1", position="Marketing Manager",
        company_name="Clever Vixen Media", is_current=True, start_year=2022,
    )
    db.add(cur)
    db.commit()

    facts = ProfileFacts(person=p, experiences=[cur], education=[], skills=[], semantic={}, embedding=None)
    cand = ScoredCandidate(
        person=p, match_score=90.0, components=[],
        evidence=[EvidenceItem(type="experience", text="Marketing Manager at Clever Vixen Media", detail={})],
        matched_criteria=["Marketing"], qualification="exact_match",
    )
    parsed = ParsedSearchQuery(
        criteria=[SearchCriterion(id="c", type="title", value="marketing manager", weight=100)]
    )
    item = _to_result_item(db, 1, cand, parsed, "marketing manager",
                           reason="matches on marketing experience", facts=facts)
    assert item.current_company == "Clever Vixen Media"
    assert item.current_company != p.current_company


def test_reason_generator_payload_uses_the_authoritative_current_employer(monkeypatch):
    """The LLM reason path must be given the SAME resolved current employer —
    it must never be left to guess between two names."""
    p = Person(
        id="p1", dataset_id="ds1", is_connection=True,
        linkedin_url="https://www.linkedin.com/in/p1", full_name="Nina Jennings",
        current_company="Cox Automotive",
    )
    cur = Experience(id="exp1", person_id="p1", company_name="Clever Vixen Media", is_current=True)
    facts = ProfileFacts(person=p, experiences=[cur], education=[], skills=[], semantic={}, embedding=None)
    cand = ScoredCandidate(
        person=p, match_score=90.0, components=[],
        evidence=[EvidenceItem(type="experience", text="Marketing Manager at Clever Vixen Media", detail={})],
    )

    captured = {}

    def _fake_generate_structured(system, user, schema, **kw):
        captured["user"] = user
        return None  # force the deterministic template so no network path is used

    monkeypatch.setattr("app.services.reason_generator.generate_structured", _fake_generate_structured)
    monkeypatch.setattr("app.services.reason_generator.settings.llm_reason_generation", True)
    generate_reason(cand, "marketing manager", facts=facts)
    assert "current_employer: Clever Vixen Media" in captured["user"]
    assert "Cox Automotive" not in captured["user"]
