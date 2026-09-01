# Search architecture fix — final report (spec §44)

Branch `main`, commits `161074e … 92f439e` (9 commits). 88 backend tests + 2
frontend tests + `tsc --noEmit` all green.

---

## A. Architecture — before

```
query ──► query_interpreter (LLM)                 keyword-biased plan
             │  types: keyword / title / company / school / skill /
             │         domain / seniority / location
             ▼
       candidate_pool.get_candidates              SQL ILIKE + embedding NN,
             │                                    HARD-CAPPED to ~60 rows
             ▼
       score_candidate  (per candidate)
             │  every criterion → string/substring test against profile text
             │  "domain"/"seniority" → weak keyword overlap
             │  embeddings only used as a candidate filter, never scored well
             ▼
       top 20  ──► reason (LLM one-liner)
```

Consequences the user hit:

| Symptom | Mechanism |
|---|---|
| "CXO event in Memphis or Nashville" → 0 people | LLM emitted two separate required `location` criteria (`Memphis` AND `Nashville`); nobody is in both. Location matched against headline/about text, not the person's location fields. |
| "Former Amazon people now at startups" → people whose *bio contains the word* "startup" | `startup` became a `keyword`; scorer just checked whether "startup" appeared anywhere in the profile blob. The actual current employer was never classified. |
| "big tech in Bay Area" → 1 match out of ~990 | `big tech` → `company` criterion literally matching a company *named* "big tech"; "Bay Area" → `location` substring, missing San Jose / Palo Alto / Mountain View / etc. |
| "people who worked in tech" → keyword noise | `tech` → `keyword`; matched "fintech", "edtech", "technician", "biotechnology"… |
| Excel export → `IllegalCharacterError` | openpyxl rejects control chars (`\x00`–`\x1f`) that HarvestAPI leaves in bio text; no sanitisation. |
| Only ~60 of 990 people ever considered | `candidate_pool` prefilter cap. |

## B. Architecture — after

```
query ──► query_interpreter                        SEARCH PLAN
             │   LLM plan  (free Groq)  ── OR ──  deterministic parser (always
             │                                     runs when every LLM is down)
             │   criterion types now include:
             │     exact facts:  current_company / past_company / school /
             │                   title / location / skill / certification …
             │     concepts:     semantic_concept  (meaning, scope-aware)
             │                   company_category  (startup / big tech / …)
             │   each criterion carries: values[] + operator (ANY_OF/ALL_OF/NOT)
             │                           + scope (current / past / any_experience / career)
             ▼
   candidate_pool.get_candidates
             │   ≤ full_scan_max_connections (5000) ⇒ EVERYONE, no prefilter
             │   else UNION( SQL facts ∪ semantic-JSON scan ∪ geo-expansion
             │               ∪ embedding-NN ∪ top-up )
             ▼
   company_intel.get_or_classify          one classification per employer,
             │   (LinkedIn company_id → else normalised name)   cached forever
             │   returns TRUE / FALSE / UNKNOWN  in `company_semantics` table
             ▼
   score_candidate(facts, plan, ctx)       bulk-loaded facts (no N+1)
             │   exact fact  → deterministic match (company_id preferred)
             │   semantic_concept → profile's semantic_assertions, then semantic
             │        fields, then concept-vs-career cross-encoder   (TRI-STATE)
             │   company_category → the ACTUAL employer's classification (TRI-STATE)
             │   location → geo layer expands regions, matches Person.location_*
             │   operator/scope aware; required-gate is tri-state aware
             │        (UNKNOWN never excludes; only a confident FALSE does)
             ▼
   _maybe_judge   shortlist whose semantic component is uncertain
             │   (0.15–0.75) → ONE batched LLM call / 10 people with a compact
             │   evidence packet → merge verdicts → re-score those people
             ▼
   two-pass relevance (embedding + local cross-encoder) ─► top 20 ─► reason
```

## C. Root causes (the user's list of 10 + what was found)

Confirmed all 10 suspected causes. Additional causes found:

1. `candidate_pool` hard cap (60) — a semantic bullseye phrased differently never entered scoring.
2. `location` scored against headline/about, not `Person.location_text/city/state/country`.
3. LLM prompt actively *encouraged* keyword criteria ("tech", "startup", "FAANG").
4. No company-intelligence layer at all — "startup"/"big tech" had nowhere to resolve.
5. Multi-value criteria impossible — the schema had a single `value: str`, so "Memphis or Nashville" could only be two criteria.
6. No operator concept (ANY_OF / ALL_OF / NOT).
7. No scope concept — "former Amazon" and "currently at Amazon" scored identically.
8. `domain`/`seniority` were keyword-overlap, not semantic.
9. Semantic enrichment produced only coarse `industries`/`job_families`; no provenance-preserving assertions, so "led a technology transformation" (a consultant) looked identical to "Software Engineer at Google".
10. N+1: `score_candidate` lazily loaded experiences/skills/education per candidate — fine at 60, quadratic at 990.
11. `llm_available` was derived from `provider is not None`; `groq:fallback` / `openrouter` were wrongly treated as "LLM down", disabling the judge & reason.
12. Excel: no control-char sanitisation.

## D. Exact files modified

New services: `company_intel.py`, `geo.py`, `semantic_judge.py`.
Rewritten: `scoring.py` (+532/−… ), `query_interpreter.py`, `candidate_pool.py`,
`compact_profile.py`, `search_text.py`, `semantic_llm.py`.
Touched: `constants.py`, `schemas.py`, `models.py`, `repositories.py`,
`config.py`, `matching.py`, `search_service.py`, `export_service.py`,
`routers/enrich.py`, `services/enrichment_runner.py`.
New scripts: `scripts/backfill_v3.py`, `scripts/verify_v3.py`.
New tests: `tests/test_company_intel.py`, `tests/test_search_semantic.py`;
extended `test_export.py`, `test_search.py`.
Frontend: `api/types.ts`, `components/ResultCard.tsx`, `pages/ResultsPage.tsx`,
`pages/DashboardPage.tsx`.

## E. DB / schema changes

- **New table `company_semantics`** — `company_key` (unique, `id:<company_id>`
  or `name:<norm>`), `industries` / `categories` (JSON), `is_technology_company`
  / `is_startup` / `is_big_tech` (nullable Boolean = tri-state), `confidence`,
  `reason`, `provenance`, `llm_provider`, timestamps. Created by
  `Base.metadata.create_all` on startup (no migration framework in this repo).
- **`schemas.ProfileSemanticData`** gains `semantic_assertions: [{concept,
  category, scope, confidence, evidence[]}]` — stored inside the existing
  `profile_semantics.data` JSON, no column change.
- **`config.semantic_profile_version` 1 → 2** — bump forces re-enrichment via
  `people_missing_semantics(current_version=2)`. Existing v1 rows stay
  searchable (search does not gate on version), they just lack assertions until
  re-run.
- No destructive changes; no existing column altered.

## F. How each query type works now

| Query | Plan (LLM path, verified) | Evaluation |
|---|---|---|
| **people who worked in tech** | `semantic_concept "…technology industry…"` scope=any_experience | assertion `technology industry experience` → semantic `industries` → concept-vs-career cross-encoder. Tri-state. |
| **Former Amazon people now at startups** | `past_company=[Amazon]` **required** ∧ `company_category=startup` scope=current_company **required** | past_company against experience rows in *past* scope; startup judged from the **current employer's** `company_semantics` row. A profile that merely says "startup" scores 0 on that criterion. |
| **CXO event in Memphis or Nashville** | `location=[Memphis,Nashville]` ANY_OF **required** + `seniority=cxo` + `skill=networking` | one location criterion, best-of over values, matched against `Person.location_text/city/state`. `cxo` → `is_cxo_title()` (CEO/CTO/…/"chief") or semantic seniority rank. |
| **big tech in Bay Area** | `company_category="big tech"` scope=current_company **required** + `location=["Bay Area"]` **required** | employer classification `is_big_tech`; `geo.expand_region("bay area")` → {SF, San Jose, Palo Alto, Mountain View, Sunnyvale, Oakland, Berkeley, …}. |
| **senior engineering mentors in tech** | `seniority=senior` + `semantic_concept engineering` + `semantic_concept mentoring` + `semantic_concept technology` | mentoring needs an actual mentoring/coaching assertion or leadership signal; "mentor program attendee" alone won't clear the bar. |

## G. Operators & scope

`SearchCriterion.values: list[str]` + `operator ∈ {ANY_OF, ALL_OF, NOT}` +
`scope ∈ {current, past, any_experience, career, current_company,
past_company}`. `_combine_over_values`: ANY_OF = best, ALL_OF = min, NOT =
1−best. `_experiences_in_scope` filters experience rows before a company/role
criterion is evaluated. Backward compatible: old single-`value` plans still
parse (`_sync_value_and_values`).

## H. Company intelligence (§4/§5)

`company_intel.get_or_classify(db, companies)` — bulk cache lookup by
`company_key`, LLM-classifies only the misses in batches of 20, writes
`company_semantics`, returns an UNKNOWN stub (never omitted) for anything it
can't classify. Prompt: *"use general knowledge of well-known companies … use
true/false only when you clearly recognise the company … do NOT invent funding
stage, employee count, revenue, or founding year."* No hardcoded
`STARTUP_WORDS`. Classifications observed on the real data:

```
Google              → is_big_tech=true   "Google is a well-known large technology company"
Amazon Web Services → is_big_tech=true   "…core big-tech cloud platform"
NVIDIA              → is_big_tech=true   "major technology hardware and AI company"
Meta                → is_big_tech=true   "major technology company (Facebook, Instagram)"
Sortment            → is_startup=true    "AI-driven retail analytics, a startup"
Stealth Startup     → is_startup=true    "generic early-stage stealth startup; no public info to confirm sector"
```

## I. Semantic assertions (§7) — real output

From a re-enriched (v2) profile in `suraj_1000_connections`:

```json
[{"concept":"technology industry experience","category":"industry_experience",
  "scope":"career","confidence":0.95,
  "evidence":["Director of Technology at Aziro","Test Architect at MSys Technologies","QA Lead at Calsoft"]},
 {"concept":"engineering leadership","category":"leadership","scope":"career","confidence":0.9,
  "evidence":["Managing multiple projects","managing Test team of 24 peoples"]},
 {"concept":"storage systems expertise","category":"domain_expertise","scope":"career","confidence":0.95,
  "evidence":["Specialties: SAN, Virtualization & Cloud Domain","Worked on vvol and VASA Certification"]}]
```

Assertions with no evidence text are dropped at ground time. The false-positive
guard is in the prompt and unit-tested: a consultant who "led a technology
transformation" gets `industries=[consulting, retail]` and **no**
technology-industry assertion.

## J. Tri-state (§15/§28)

`TriState.{TRUE, FALSE, UNKNOWN}` flows from `company_semantics` and from the
judge through `_score_*` into `score_candidate`. Required-gate:

```
semantic / company_category criterion, required:
    exclude  ⇔  status == FALSE            (a confident negative)
    keep     ⇔  status in {TRUE, UNKNOWN}  (UNKNOWN gets partial/zero credit,
                                            never a hard exclude)
exact fact criterion, required:
    exclude  ⇔  strength < _REQUIRED_MIN
```

## K. Batched semantic judge (§16–18)

`semantic_judge.judge()` — only candidates whose semantic component is
*uncertain* (`0.15 ≤ s ≤ 0.75`), pooled to `semantic_judge_pool` (60), sent in
batches of 10 with a compact evidence packet (current/past roles, company
classifications, industries, job families, leadership, career summary,
assertions, skills, location). One structured call per batch. Prompt: *"you may
interpret the meaning of roles and text; you may NOT invent employers, roles,
dates, skills, education, or company facts."* Verdicts merged into
`ctx.judge_results`, affected candidates re-scored. Off ⇒ pure deterministic +
embedding path still works.

## L. Geographic layer (§12)

`geo._REGIONS`: bay area / silicon valley / sf bay area / greater new york /
nyc / greater seattle / greater boston / dfw / socal / … → member cities.
`expand_values` used in both `candidate_pool` (retrieval) and `scoring`
(match). `location_matches` = token-subset test against
`location_text | city | state | country`. No `if query == "Bay Area"`.

## M. Retrieval / FULL_SCAN (§10/§31)

`candidate_pool.get_candidates`: `total ≤ full_scan_max_connections (5000)` ⇒
**return every person** (the 990-network is a full scan — verified
`candidates=987` on "worked in tech"). Above the cap ⇒ UNION of SQL-fact
matches, semantic-JSON scan, geo matches, embedding NN, and a relevance top-up.
`scoring` consumes `repo.bulk_experiences/education/skills/certifications/
languages/publications/semantics/embeddings` — dict keyed by person_id, one
query each, N+1 gone.

## N. `llm_available` (§29)

`search_service`: `llm_available = provider != "deterministic"` — any real
provider (`groq:primary`, `groq:fallback`, `openrouter`, `anthropic:paid`)
counts as available and enables the judge + LLM reason.

## O. Excel fix (§32)

`export_service._ILLEGAL_XLSX_RE = [\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`,
`sanitize_excel_value()` applied to **every** cell via an `_append()` wrapper
on all 7 sheets. Unicode, bullets, en/em dashes, and newlines are preserved.
Regression tests feed `\x00 \x01 \x0b \x0c \x1f` through a profile and assert
the workbook builds and the visible text survives.

## P. Tests

**Backend 88 pass** (was 76). New `tests/test_search_semantic.py` — 12
deterministic regression tests, no live LLM:

- concept satisfied without the literal word (Google SWE via assertion)
- "technology transformation" consultant scores **below** the real engineer (§34 trap)
- ex-Amazon + classified-startup matches; ex-Amazon + classified-non-startup **excluded** even with "I advise startups" in the bio
- unclassified current company → **not** excluded (UNKNOWN ≠ FALSE)
- location OR: Nashville & Memphis match, Atlanta excluded when required
- CXO matches a real CEO/CTO title; "I sell to CXO customers" does **not**
- past_company ANY_OF: ex-Google matches, ex-Google∧ex-Meta not required, ex-Oracle excluded
- "Amazon Selling Partner API" in a description ≠ employment at Amazon
- "Attended Startup Weekend" at Deloitte ≠ working at a startup
- deterministic fallback still yields location-OR and company_category (not a company named "startup")

`tests/test_company_intel.py` — id-over-name keying, UNKNOWN≠FALSE, cache hit,
disabled flag skips the LLM.

**Frontend** — 2 pass, `tsc --noEmit` clean.

Commands run:
```
cd backend && ./.venv/Scripts/python.exe -m pytest -q          # 88 passed
cd frontend && npx tsc --noEmit                                # exit 0
cd frontend && npx vitest run                                  # 2 passed
```

## Q. Real verification & remaining limitations

**Ran** — all 5 mandatory queries in-process against `suraj_1000_connections`
(991 people). Deterministic-parser path (LLM quota exhausted mid-run — see
below), company classifications served from the `company_semantics` cache:

- **worked in tech** — 987 candidates scanned, top 5 = Director of Technology,
  COO, Software Engineer (GenAI), Founder/CEO of a tech company, Senior BA at a
  tech-services firm. Evidence = `technology industry experience — <grounded
  roles>`. No literal-word matches.
- **Former Amazon → startups** — 69 candidates (people at classified
  startups); top results currently at "Sortment" (AI retail analytics),
  "Engage People Inc.", "Stealth Startup" — each with `[company_inference]
  classified as startup (<reason>)`.
- **CXO in Memphis or Nashville** — exactly 2 people in the whole network are
  in Memphis/Nashville; both returned, both actually in Nashville. **The 0-result
  bug is fixed.** (Neither is a CXO — the network simply has none there.)
- **big tech in Bay Area** — 36 candidates; top 5 = Google, AWS, AWS, NVIDIA,
  Meta, all "San Francisco Bay Area", each `[company_inference] classified as
  big tech (<reason>)`. **Was 1 result, now 20.**
- **senior engineering mentors in tech** — top 5 are SVP/Senior Director of
  Engineering / Architect at tech-services firms with `seniority: senior` +
  `role: … engineering` semantic evidence.

**LLM query-interpreter path** (verified separately before quota ran out):

```
"people working in big tech in Bay Area"
   → company_category "big tech" (current, required) + location ["Bay Area"] (required)
"Former Amazon people now at startups"
   → past_company [Amazon] ALL_OF (past, required) + company_category startup (current, required)
```

Both plans are correct — the Bay-Area geo criterion and the Amazon∧startup
double-required are what the LLM path adds over the deterministic fallback.

### Limitations

1. **Full semantic backfill of the 991-network is incomplete.** Groq's
   free tier rate-limited then exhausted after ~4 profiles and ~200 company
   classifications this session. `python -m scripts.backfill_v3
   dataset_0ba27cae09d4` is resumable and idempotent — run it over time (or
   raise the Groq quota) to give every profile v2 assertions. Until then,
   profiles enriched under v1 are still searchable via their `industries` /
   `job_families` / `leadership` fields; they just lack the fine-grained
   assertion evidence. New datasets enriched from scratch get v2 directly.
2. **Deterministic parser doesn't extract a bare "in \<region\>"** (no "or") as
   a geo criterion — the LLM path does. When both LLMs are down, "big tech in
   Bay Area" still returns Bay-Area people (geo-aware retrieval + embeddings)
   but without an explicit required location gate.
3. **Deterministic parser makes `past_company` soft** for "Former X people
   now …" (weight ~46, not required). The LLM path makes it required. So with
   LLMs down, "Former Amazon people now at startups" ranks current-startup
   people first and ex-Amazon is a tie-breaker rather than a filter.
4. **`_maybe_llm_rerank`** still receives a thin one-line packet (the fuller
   §17 packet went to the judge, not the reranker). It's off by default
   (`llm_rerank_enabled=false`), so low priority.
5. **Company classification quality** depends on the free model's world
   knowledge — a genuinely obscure employer resolves to UNKNOWN (correctly not
   excluded, but also not credited). LinkedIn `company_id` keying means a
   re-run after enrichment picks up the canonical entity.

### Cost controls preserved

No paid APIs enabled (`ENABLE_PAID_LLM=false`, `paid_llm_enabled:false` in
`/health`). No Apify calls in the search path — `backfill_v3` reads stored
`raw_profiles.raw_json` + normalised tables only. Company classification and
the judge are free-Groq, cached, and batched.
