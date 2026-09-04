"""V4 PART 2 — universal query representation.

Deterministic path only (conftest sets LLM_QUERY_INTERPRETATION=false): these
prove that arbitrary professional-network questions get a real intent + real
candidate criteria + relational context WITH NO LLM — not that parsing merely
"returns something".

Every test checks at least one of: intent, requiredness, AND semantics, context
separation, scope, modality, no accidental keyword fallback, "my field"
unresolved behaviour, academia-vs-education distinction, mentor relationality.
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.constants import CriterionType, Modality, Operator, QueryIntent
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.query_interpreter import _dedupe_semantic_duplicates, interpret_query


def _plan(q):
    parsed, _prov, _ = interpret_query(q)
    return parsed


def _types(plan):
    return {c.type for c in plan.criteria}


def _by_type(plan, t):
    return [c for c in plan.criteria if c.type == t]


def _vals(c):
    return {v.lower() for v in (c.values or ([c.value] if c.value else []))}


def _concepts(plan):
    return " || ".join((c.concept or c.value or "").lower() for c in plan.criteria)


def _has_keyword_fallback(plan):
    return any(c.type == CriterionType.KEYWORD for c in plan.criteria) or any(
        c.type == CriterionType.SEMANTIC_CONCEPT and len((c.concept or c.value or "").split()) <= 1
        for c in plan.criteria
    )


# ─────────────────────────── §1 reusable intent ───────────────────────────


@pytest.mark.parametrize(
    "query,intent",
    [
        ("Who are the data scientists in Seattle", QueryIntent.FIND_PEOPLE),
        ("Big tech in Bay Area", QueryIntent.PROFESSIONAL_RECOMMENDATION),
        ("career mentors who can give advice", QueryIntent.MENTOR_RECOMMENDATION),
        ("Who could mentor a backend engineer moving into management?", QueryIntent.MENTOR_RECOMMENDATION),
        ("anyone in my field who can give advice", QueryIntent.MENTOR_RECOMMENDATION),
        ("HIPAA compliance experts", QueryIntent.SUBJECT_MATTER_EXPERTISE),
        ("professors in AI who are world-class in reinforcement learning", QueryIntent.SUBJECT_MATTER_EXPERTISE),
        ("academia to industry transitions", QueryIntent.CAREER_TRANSITION),
        ("academia → industry transitions", QueryIntent.CAREER_TRANSITION),
        ("Who should I invite to a CXO networking event in Nashville?", QueryIntent.NETWORKING_INVITATION),
    ],
)
def test_intent_is_classified_reusably(query, intent):
    assert _plan(query).intent == intent


def test_intent_is_a_known_value_for_unseen_phrasings():
    for q in ["former Amazon people now at startups", "nonprofit experience in Chicago",
              "cybersecurity AND healthcare backgrounds", "FAANG"]:
        assert _plan(q).intent in {
            QueryIntent.FIND_PEOPLE, QueryIntent.PROFESSIONAL_RECOMMENDATION,
            QueryIntent.MENTOR_RECOMMENDATION, QueryIntent.SUBJECT_MATTER_EXPERTISE,
            QueryIntent.CAREER_TRANSITION, QueryIntent.NETWORKING_INVITATION,
        }


# ─────────────────────── §2/§3 candidate criteria vs relational context ───────────────────────


def test_mentor_query_is_relational_not_a_search_phrase():
    plan = _plan("Who could mentor a backend engineer trying to move into management?")
    assert not any(
        "backend engineer trying to move" in (c.concept or c.value or "").lower()
        for c in plan.criteria
    )
    tpc = plan.target_person_context
    assert tpc.get("current_role", "").startswith("backend engineer")
    assert "engineering" in tpc.get("field", "")
    assert "management" in tpc.get("goal", "")
    assert not any("my field" in (c.value or c.concept or "").lower() for c in plan.criteria)


def test_mentor_query_candidate_criteria_are_evidence_based():
    plan = _plan("Who could mentor a backend engineer moving into management?")
    concepts = _concepts(plan)
    assert any(
        c.required and ("management" in (c.concept or "").lower() or "leadership" in (c.concept or "").lower())
        for c in plan.criteria
    )
    assert any(c.required and "mentor" in (c.concept or "").lower() and c.type in
               {CriterionType.PROFESSIONAL_CONCEPT, CriterionType.SEMANTIC_CONCEPT}
               for c in plan.criteria)
    assert not any(c.type in {CriterionType.KEYWORD, CriterionType.SKILL, CriterionType.TITLE}
                   and "mentor" in (c.value or c.concept or "").lower() for c in plan.criteria)
    assert "backend engineering" in concepts


def test_seniority_alone_does_not_imply_mentor():
    plan = _plan("Who could mentor a backend engineer moving into management?")
    assert not any(c.type == CriterionType.SENIORITY and c.required for c in plan.criteria)


def test_networking_event_words_are_context_not_criteria():
    plan = _plan("Who should I invite to a CXO networking event in Memphis or Nashville?")
    assert plan.intent == QueryIntent.NETWORKING_INVITATION
    assert "networking" in plan.context.get("purpose", "").lower()
    assert not any((c.value or c.concept or "").lower() == "networking" for c in plan.criteria)
    assert _by_type(plan, CriterionType.SENIORITY) and _by_type(plan, CriterionType.SENIORITY)[0].required
    loc = _by_type(plan, CriterionType.LOCATION)[0]
    assert loc.required and _vals(loc) == {"memphis", "nashville"} and loc.operator == Operator.ANY_OF


# ─────────────────────────── §3 "my field" ───────────────────────────


def test_my_field_is_unresolved_without_a_configured_profile():
    plan = _plan("anyone in my field who can give advice")
    assert "field" in plan.unresolved
    assert not plan.target_person_context.get("field")
    assert not any("backend" in (c.concept or c.value or "").lower() for c in plan.criteria)
    assert not _has_keyword_fallback(plan)
    assert plan.interpretation_confidence <= 0.6


def test_my_field_resolves_from_configured_profile(monkeypatch):
    monkeypatch.setattr(settings, "user_field", "clinical genomics")
    monkeypatch.setattr(settings, "user_current_role", "bioinformatics scientist")
    plan = _plan("anyone in my field who can give advice")
    assert plan.target_person_context.get("field") == "clinical genomics"
    assert plan.target_person_context.get("current_role") == "bioinformatics scientist"
    assert "field" not in plan.unresolved


# ─────────────────────────── §4 cross-domain AND ───────────────────────────


@pytest.mark.parametrize(
    "query,a,b",
    [
        ("cybersecurity AND healthcare backgrounds", "cybersecurity", "healthcare"),
        ("AI and healthcare", "artificial intelligence", "healthcare"),
        ("AI AND healthcare", "artificial intelligence", "healthcare"),
        ("research PLUS industry experience", "research", "industry"),
    ],
)
def test_cross_domain_becomes_two_required_dimensions(query, a, b):
    plan = _plan(query)
    sem = [c for c in plan.criteria if c.type in
           {CriterionType.PROFESSIONAL_CONCEPT, CriterionType.INDUSTRY_EXPERIENCE,
            CriterionType.SEMANTIC_CONCEPT}]
    concepts = _concepts(plan)
    assert a in concepts and b in concepts
    required_dims = [c for c in sem if c.required]
    assert len(required_dims) >= 2, f"{query}: expected 2 required dims, got {[c.concept for c in sem]}"
    assert not any(c.operator == Operator.ANY_OF and len(c.values) >= 2 for c in required_dims)
    assert not _has_keyword_fallback(plan)


def test_llm_shaped_semantic_duplicates_collapse_to_two_required_dims():
    """hardening PART 8 — the exact live failure: the LLM path (not the
    deterministic parser) returned 4 criteria for "research plus industry
    experience" because "research" was asked about under two types and
    "industry" under two types. This directly exercises the post-processing
    dedup against that hand-built shape, independent of how either parser
    produces criteria — it must collapse to exactly the 2 real dimensions
    without inventing/dropping the query's intent, and never touch a
    genuinely different cross-domain AND (cybersecurity/healthcare)."""
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="research_pc", type=CriterionType.PROFESSIONAL_CONCEPT,
                        concept="research experience", scope="career", weight=25, required=True),
        SearchCriterion(id="research_rf", type=CriterionType.ROLE_FUNCTION,
                        concept="research", scope="career", weight=25, required=True),
        SearchCriterion(id="industry_ie", type=CriterionType.INDUSTRY_EXPERIENCE,
                        concept="industry experience", scope="career", weight=25, required=True),
        SearchCriterion(id="industry_sc", type=CriterionType.SEMANTIC_CONCEPT,
                        concept="industry", scope="career", weight=25, required=False),
    ])
    _dedupe_semantic_duplicates(plan)
    assert len(plan.criteria) == 2
    concepts = {(c.concept or "").lower() for c in plan.criteria}
    assert any("research" in c for c in concepts)
    assert any("industry" in c for c in concepts)
    # the generic semantic_concept duplicate's required=False must not weaken
    # the surviving industry_experience criterion — OR semantics, not AND
    assert all(c.required for c in plan.criteria)
    assert sum(c.weight for c in plan.criteria) == 100

    # a real cross-domain AND is untouched — near-zero concept overlap
    cross = ParsedSearchQuery(criteria=[
        SearchCriterion(id="cyber", type=CriterionType.PROFESSIONAL_CONCEPT,
                        concept="cybersecurity experience", scope="career", weight=50, required=True),
        SearchCriterion(id="health", type=CriterionType.INDUSTRY_EXPERIENCE,
                        concept="healthcare industry experience", scope="career", weight=50, required=True),
    ])
    _dedupe_semantic_duplicates(cross)
    assert len(cross.criteria) == 2


def test_role_noun_form_stays_one_all_of_criterion():
    plan = _plan("AI and security leaders")
    sem = [c for c in plan.criteria if c.type in
           {CriterionType.PROFESSIONAL_CONCEPT, CriterionType.ROLE_FUNCTION}]
    assert len(sem) == 1 and sem[0].operator == Operator.ALL_OF
    assert {"ai", "security"} <= _vals(sem[0])


# ─────────────────────────── §5 modality ───────────────────────────


def test_possible_experience_keeps_weaker_modality():
    maybe = _plan("possible HIPAA compliance experience")
    certain = _plan("HIPAA compliance experts")

    m = [c for c in maybe.criteria if "hipaa" in (c.concept or c.value or "").lower()]
    assert m and m[0].modality == Modality.POSSIBLE and not m[0].required

    c = [x for x in certain.criteria if "hipaa" in (x.concept or x.value or "").lower()]
    assert c and c[0].modality == Modality.CERTAIN and c[0].required

    assert (m[0].modality, m[0].required) != (c[0].modality, c[0].required)
    assert not _has_keyword_fallback(maybe)


def test_might_have_and_possible_phrasings_agree():
    a = _plan("might have experience with HIPAA compliance")
    b = _plan("possible HIPAA compliance experience")
    for plan in (a, b):
        hits = [c for c in plan.criteria if "hipaa" in (c.concept or c.value or "").lower()]
        assert hits and hits[0].modality == Modality.POSSIBLE and not hits[0].required


# ─────────────────────────── §6 academia semantics ───────────────────────────


def test_professors_in_ai_is_faculty_employment_not_a_degree():
    plan = _plan("professors in AI")
    assert CriterionType.EDUCATION not in _types(plan)
    concepts = _concepts(plan)
    assert "faculty" in concepts or "professor" in concepts
    assert "ai" in concepts
    assert any(c.required and ("faculty" in (c.concept or "").lower() or "professor" in (c.concept or "").lower())
               for c in plan.criteria)


def test_studying_at_a_university_is_education_not_academic_employment():
    plan = _plan("people who studied computer science at Stanford")
    assert CriterionType.EDUCATION in _types(plan)
    assert not any("faculty appointment" in (c.concept or "").lower()
                   or "employment in academia" in (c.concept or "").lower()
                   for c in plan.criteria)


def test_publications_do_not_imply_professor():
    plan = _plan("connections who have published peer-reviewed papers")
    assert CriterionType.PUBLICATION in _types(plan)
    assert not any("faculty" in (c.concept or "").lower() or "professor" in (c.concept or "").lower()
                   for c in plan.criteria)


def test_academia_to_industry_is_an_ordered_transition():
    for q in ("academia to industry transitions", "academia → industry transitions"):
        plan = _plan(q)
        tr = _by_type(plan, CriterionType.CAREER_TRANSITION)
        assert tr and tr[0].required
        assert "academia" in tr[0].concept.lower() and "industry" in tr[0].concept.lower()
        assert tr[0].concept.lower().index("academia") < tr[0].concept.lower().index("industry")


# ─────────────────────────── §7 mentor / advice semantics ───────────────────────────


def test_mentor_query_never_requires_the_literal_word_mentor():
    plan = _plan("career mentors who can give advice")
    assert plan.intent == QueryIntent.MENTOR_RECOMMENDATION
    assert not any(c.type in {CriterionType.KEYWORD, CriterionType.SKILL}
                   and (c.value or "").lower() in {"mentor", "mentors", "advice"}
                   for c in plan.criteria)
    assert any(c.required and c.type in {CriterionType.PROFESSIONAL_CONCEPT, CriterionType.SEMANTIC_CONCEPT}
               and any(w in (c.concept or "").lower() for w in ("mentor", "coach", "advis", "leadership", "manage"))
               for c in plan.criteria)
    assert not _has_keyword_fallback(plan)


def test_mentor_evidence_covers_multiple_signal_types():
    plan = _plan("Who could mentor a backend engineer moving into management?")
    evidence = next(c for c in plan.criteria if "mentor" in (c.concept or "").lower() and c.required)
    low = evidence.concept.lower()
    assert sum(w in low for w in ("mentor", "coach", "advis", "management", "leadership")) >= 3


# ─────────────────────────── §8 interpretation output ───────────────────────────


def test_plan_exposes_the_full_representation():
    plan = _plan("Who could mentor a backend engineer moving into management?")
    assert plan.intent == QueryIntent.MENTOR_RECOMMENDATION
    assert plan.criteria
    assert plan.target_person_context
    assert plan.interpretation_summary.startswith("Interpreted as")
    assert "intent" in plan.interpretation_summary
    assert 0.0 <= plan.interpretation_confidence <= 1.0
    dumped = plan.model_dump()
    for key in ("intent", "criteria", "context", "target_person_context",
                "unresolved", "interpretation_summary", "interpretation_confidence"):
        assert key in dumped


def test_summary_mentions_relational_context_and_unresolved():
    plan = _plan("anyone in my field who can give advice")
    assert "unresolved" in plan.interpretation_summary.lower()
    assert "field" in plan.interpretation_summary.lower()


# ─────────────────────────── §9 deterministic fallback safety ───────────────────────────


def test_unexplained_concepts_never_become_keyword_matching():
    for q in ["career mentors who can give advice",
              "anyone in my field who can give advice",
              "possible HIPAA compliance experience",
              "cybersecurity AND healthcare backgrounds",
              "professors in AI"]:
        plan = _plan(q)
        assert not _has_keyword_fallback(plan), f"{q!r} produced a keyword/1-word fallback"


def test_explicit_facts_and_booleans_survive_part2():
    plan = _plan("former Google or Meta engineers not currently at Amazon")
    past = _by_type(plan, CriterionType.PAST_COMPANY)[0]
    assert _vals(past) == {"google", "meta"} and past.operator == Operator.ANY_OF and past.required
    assert any(c.operator == Operator.NOT and "amazon" in _vals(c) for c in plan.criteria)


def test_relational_query_lowers_confidence_when_unresolved():
    resolved = _plan("former Google or Meta engineers")
    unresolved = _plan("anyone in my field who can give advice")
    assert unresolved.interpretation_confidence < resolved.interpretation_confidence


# ─────────────────────── unseen examples (not in the spec list) ───────────────────────


def test_unseen_cross_domain_climate_and_finance():
    plan = _plan("climate and finance experience")
    concepts = _concepts(plan)
    assert "climate" in concepts and "finance" in concepts
    req = [c for c in plan.criteria if c.required and c.type in
           {CriterionType.PROFESSIONAL_CONCEPT, CriterionType.INDUSTRY_EXPERIENCE}]
    assert len(req) >= 2


def test_unseen_mentor_for_a_designer_going_into_leadership():
    plan = _plan("mentor for a product designer moving into design leadership")
    assert plan.intent == QueryIntent.MENTOR_RECOMMENDATION
    tpc = plan.target_person_context
    assert "designer" in tpc.get("current_role", "")
    assert "leadership" in tpc.get("goal", "")
    assert any(c.required and "leadership" in (c.concept or "").lower() for c in plan.criteria)


def test_unseen_subject_matter_expertise_is_required_and_certain():
    plan = _plan("experts in Kubernetes networking")
    assert plan.intent == QueryIntent.SUBJECT_MATTER_EXPERTISE
    sme = [c for c in plan.criteria if c.required and c.modality == Modality.CERTAIN
           and c.type in {CriterionType.PROFESSIONAL_CONCEPT, CriterionType.SEMANTIC_CONCEPT, CriterionType.SKILL}]
    assert sme and "kubernetes" in _concepts(plan)


def test_unseen_faculty_query_keeps_education_out():
    plan = _plan("tenure-track faculty in computational biology")
    assert CriterionType.EDUCATION not in _types(plan)
    assert any("faculty" in (c.concept or "").lower() and c.required for c in plan.criteria)


# ─────────────────── hardening PART 17 — bounded interpretation timeout ───────────────────


def test_llm_interpretation_uses_the_configured_timeout(monkeypatch):
    """Query interpretation is foundational (never metered by SEARCH_LLM_MAX_CALLS)
    so it must not be able to hold a search hostage on a slow provider — it must
    pass a bounded timeout into the router, distinct from every other LLM call
    (judge/audit/reason), which leave it at the provider default."""
    monkeypatch.setattr(settings, "llm_query_interpretation", True)
    monkeypatch.setattr(settings, "query_interpretation_timeout_seconds", 12.5)
    captured = {}

    def _fake_generate_structured(system, user, schema, **kw):
        captured.update(kw)
        return None  # falls through to the deterministic parser

    monkeypatch.setattr("app.services.query_interpreter.generate_structured", _fake_generate_structured)
    parsed, provider, _model = interpret_query("cybersecurity AND healthcare backgrounds")
    assert captured.get("timeout") == 12.5
    assert provider == "deterministic"  # fallback still produced a real plan
    assert parsed.criteria
