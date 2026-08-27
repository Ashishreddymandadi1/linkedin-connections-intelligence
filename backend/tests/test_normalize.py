from __future__ import annotations

from app.services.completeness import compute_completeness
from app.services.normalize import normalize_profile, normalize_skill_name

JANE = {
    "id": "abc",
    "publicIdentifier": "jane-smith",
    "firstName": "Jane",
    "lastName": "Smith",
    "headline": "SWE",
    "about": "Built things.",
    "topSkills": "AWS, Java , Distributed Systems",
    "location": {"linkedinText": "Seattle, WA", "parsed": {"city": "Seattle", "state": "Washington", "country": "United States", "countryCode": "US"}},
    "currentPosition": [{"companyName": "Google"}],
    "experience": [
        {
            "position": "Senior SWE",
            "companyName": "Google",
            "startDate": {"month": "Jan", "year": 2024},
            "endDate": {"text": "Present"},
            "skills": ["Go"],
        },
        {
            "position": "SDE II",
            "companyName": "Amazon",
            "startDate": {"month": 7, "year": 2021},
            "endDate": {"month": "Dec", "year": 2023},
            "description": "AWS ECS + DynamoDB.",
        },
    ],
    "education": [{"schoolName": "Georgia Tech", "degree": "MS", "fieldOfStudy": "CS", "period": "2019 - 2021"}],
    "skills": [{"name": "Amazon Web Services (AWS)"}, {"name": "Java"}],
}


def test_identity_and_location():
    n = normalize_profile(JANE)
    p = n["person"]
    assert p["full_name"] == "Jane Smith"
    assert p["city"] == "Seattle"
    assert p["country_code"] == "US"


def test_current_experience_detection():
    n = normalize_profile(JANE)
    exps = n["experiences"]
    assert exps[0]["company_name"] == "Google" and exps[0]["is_current"] is True
    assert exps[1]["company_name"] == "Amazon" and exps[1]["is_current"] is False
    assert n["person"]["current_company"] == "Google"
    assert n["person"]["current_title"] == "Senior SWE"


def test_date_parsing_month_name_and_number():
    exps = normalize_profile(JANE)["experiences"]
    assert exps[0]["start_month"] == 1 and exps[0]["start_year"] == 2024
    assert exps[1]["start_month"] == 7
    assert exps[1]["end_month"] == 12 and exps[1]["end_year"] == 2023


def test_education_period_split():
    edu = normalize_profile(JANE)["education"][0]
    assert edu["start_year"] == 2019 and edu["end_year"] == 2021
    assert edu["school_name"] == "Georgia Tech"


def test_topskills_string_and_experience_skills_merge():
    skills = {s["skill_name_norm"] for s in normalize_profile(JANE)["skills"]}
    assert "aws" in skills  # from topSkills + skills[], deduped
    assert "java" in skills
    assert "distributed systems" in skills
    assert "go" in skills  # from experience skills


def test_missing_and_null_fields_do_not_crash():
    n = normalize_profile({"publicIdentifier": "x", "experience": None, "education": [], "skills": None, "topSkills": None})
    assert n["person"]["full_name"] is None
    assert n["experiences"] == [] and n["skills"] == []


def test_empty_profile_lowers_completeness():
    empty = normalize_profile({"publicIdentifier": "x"})
    score, detail = compute_completeness(empty, {})
    assert score < 20
    assert detail["experience"] is False

    full = normalize_profile(JANE)
    fscore, _ = compute_completeness(full, JANE)
    assert fscore > score


def test_skill_name_normalization():
    assert normalize_skill_name("Amazon Web Services (AWS)") == "aws"
    assert normalize_skill_name("Go (Programming Language)") == "go"


def test_extra_sections_extracted():
    raw = {
        "publicIdentifier": "x",
        "certifications": [
            {"title": "AWS Certified Solutions Architect", "issuedBy": "Amazon Web Services", "issuedAt": "2022"},
            {"name": "no title key", "authority": "Somewhere"},
        ],
        "languages": [{"name": "English", "proficiency": "Native"}, "Spanish", {"proficiency": "only prof"}],
        "publications": [
            {"title": "A Paper on X", "publishedAt": "Journal of Y · Jan 2024", "link": "http://x"},
        ],
        "receivedRecommendations": [
            {"givenBy": "Jane Doe", "givenByHeadline": "CTO", "text": "Great engineer.", "givenAt": "2021"},
        ],
        "patents": None,
    }
    n = normalize_profile(raw)
    assert len(n["certifications"]) == 2
    assert n["certifications"][0]["issuer"] == "Amazon Web Services"
    langs = {row["name"] for row in n["languages"]}
    assert langs == {"English", "Spanish"}
    assert n["publications"][0]["publisher"] == "Journal of Y"
    assert n["recommendations"][0]["recommender_name"] == "Jane Doe"
    assert n["patents"] == []
