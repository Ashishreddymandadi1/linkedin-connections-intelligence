"""The PART 10 real-data evaluation query suite.

These are TEST CASES — plain natural-language queries a user might type. There is
NO query-specific code anywhere; the search pipeline sees only ``query``.

``id``      stable identifier used in result files and the label file
``query``   the exact user text
``group``   ``core`` = the 14 spec queries, ``new`` = the 6+ unseen queries
``note``    what the query is probing (for the human reviewer only)
"""
from __future__ import annotations

CORE_QUERIES: list[dict] = [
    {"id": "q01_big_tech_bay_area", "group": "core",
     "query": "Big tech in Bay Area",
     "note": "company_category + location scope; terse phrasing"},
    {"id": "q02_faang", "group": "core",
     "query": "FAANG",
     "note": "single acronym -> company_category expansion, no location"},
    {"id": "q03_cxo_event_memphis_nashville", "group": "core",
     "query": "Who should I invite to a CXO networking event in Memphis or Nashville?",
     "note": "event context must NOT leak into criteria; seniority + OR location"},
    {"id": "q04_career_mentors", "group": "core",
     "query": "Any career mentors who can give me good advice?",
     "note": "relational/mentor query; mentoring evidence as a concept"},
    {"id": "q05_my_field_advice", "group": "core",
     "query": "Anyone in my field who can give me good advice?",
     "note": "'my field' unresolved unless a current-user profile is configured"},
    {"id": "q06_nonprofit_chicago", "group": "core",
     "query": "People with nonprofit experience in Chicago",
     "note": "volunteering/nonprofit concept + location"},
    {"id": "q07_professors_ai", "group": "core",
     "query": "Professors in AI",
     "note": "academia role_function + AI domain"},
    {"id": "q08_hipaa_compliance", "group": "core",
     "query": "People who might have experience with HIPAA compliance",
     "note": "modality 'might' -> possible, soft signal; healthcare/compliance concept"},
    {"id": "q09_ex_amazon_now_startup", "group": "core",
     "query": "Former Amazon people now at startups",
     "note": "past_company (Amazon) AND current company_category (startup)"},
    {"id": "q10_cyber_and_healthcare", "group": "core",
     "query": "People with cybersecurity and healthcare backgrounds",
     "note": "cross-domain AND: security + healthcare"},
    {"id": "q11_academia_to_industry", "group": "core",
     "query": "Who has moved from academia into industry?",
     "note": "career_transition academia -> industry"},
    {"id": "q12_backend_to_management_mentor", "group": "core",
     "query": "Who could mentor a backend engineer trying to move into management?",
     "note": "target_person_context (backend eng -> management) shapes criteria; mentoring"},
    {"id": "q13_ai_and_healthcare", "group": "core",
     "query": "People who understand AI and healthcare",
     "note": "cross-domain AND: AI + healthcare (understand, not necessarily employed)"},
    {"id": "q14_research_plus_industry", "group": "core",
     "query": "People with research plus industry experience",
     "note": "two experience types both required"},
]

NEW_QUERIES: list[dict] = [
    {"id": "n01_cloud_and_finance", "group": "new",
     "query": "People with cloud infrastructure experience in the finance industry",
     "note": "skill/domain (cloud) + industry_experience (finance)"},
    {"id": "n02_sales_leadership_healthcare", "group": "new",
     "query": "Sales leaders in healthcare",
     "note": "role_function sales + seniority (leader) + industry healthcare"},
    {"id": "n03_founders_engineering_background", "group": "new",
     "query": "Founders who started out as engineers",
     "note": "founder_signals + career origin in engineering"},
    {"id": "n04_technical_leaders_consulting", "group": "new",
     "query": "Technical leaders who also have consulting experience",
     "note": "engineering leadership AND past consulting role_function"},
    {"id": "n05_researchers_now_at_startups", "group": "new",
     "query": "Researchers who are now at startups",
     "note": "research background + current company_category startup (transition)"},
    {"id": "n06_product_and_engineering", "group": "new",
     "query": "People who have done both product management and engineering",
     "note": "two role_functions across a career"},
    {"id": "n07_ml_engineers_open_source", "group": "new",
     "query": "Machine learning engineers who contribute to open source",
     "note": "role_function ML + open-source contribution concept"},
]

ALL_QUERIES: list[dict] = CORE_QUERIES + NEW_QUERIES


def by_id(qid: str) -> dict:
    for q in ALL_QUERIES:
        if q["id"] == qid:
            return q
    raise KeyError(qid)
