# V4 — B.5 + PART C + PART D + PART E — final report

Branch `main`, commits `a3ef980 … 849d634`. Stopped after PART E as instructed
(no vector redesign, no UI redesign, no new paid services).

---

**A. Starting commit** — `378eebe` (parent `a3ef980`). Verified: on `main`, clean
tree, PART A (Anthropic-first routing + circuit breaker) and PART B (context vs
candidate, deterministic fact merge, OR/NOT protection, interpretation summary +
confidence) present in source.

**B. Baseline** — `111 passed` before any change.

**C. B.5 query problems confirmed**
1. `parse` used one splitter that discarded the AND/OR connective — "Amazon and
   Microsoft" and "Google or Meta" produced the same operator.
2. Boolean structure worked only for company/location, not semantic concepts.
3. `validate_and_repair` only *warned* about a dropped OR/NOT; it never repaired.
4. `SearchCriterion._known_type` silently folded an unknown type to `keyword`
   (-> "worked in tech" could become a literal text search).
5. `TITLE` was auto-softened unless the query said "must" — so "software
   engineers at fintech companies" made the role optional.

**D. AND / OR / NOT fixes**
- `query_facts.parse_value_group(blob) -> (values, operator)` — reports the
  connective: `and`/`&`/`+`/`both` (and no `or`) -> `ALL_OF`; otherwise `ANY_OF`.
  Comma defaults to `ANY_OF`.
- Applied to former/current company extraction, a new `"X and Y experience"`
  pattern, and `_extract_semantic_boolean` ("security or cloud experts" ->
  `role_function`/`professional_concept` `values=[security, cloud]` `ANY_OF`;
  "AI and security leaders" -> `ALL_OF`).
- `_NOT_AT_RE` -> `operator=NOT`, `scope=current_company`, `required`.
- `merge_into_plan` now unions missing OR values ("Google or Meta" when the LLM
  kept only Google) and sets the operator to the deterministic reading when the
  query was explicit.
- `_score_semantic_multi` in scoring evaluates `values` per-operator (ANY_OF =
  best, ALL_OF = min + tri-state combine, NOT = invert).

**E. SearchPlan validator changes** — `validate_and_repair` now:
- reconstructs a semantic `ANY_OF` from the raw "X or Y <noun>" shape when the
  plan produced no OR criterion;
- if it still can't repair an OR/NOT mismatch, it *lowers*
  `interpretation_confidence` to <= 0.5 and keeps a safe broad criterion (no
  bounded LLM repair call was added — the deterministic reconstruction covers
  the tested cases; a one-shot LLM repair is a documented follow-up).

**F. Unsupported-criterion-type behaviour** — `_known_type`: exact match ->
kept; known LLM alias (`industry`->`industry_experience`, `role`/`job_function`->
`role_function`, `leadership`/`mentorship`->`professional_concept`, `tenure`->
`years_experience`, `career_change`->`career_transition`, ...) -> mapped;
explicit `text`/`phrase`/`mention`/`literal` -> `keyword`; **anything else ->
`professional_concept`, never `keyword`**.

**G. Files modified for PART C** — `app/schemas.py` (ExperienceSemantic,
assertion source-ids, `experience_semantics[]`), `app/services/compact_profile.py`
(experience_id / education_id / certification_id), `app/services/semantic_llm.py`
(prompt + `_ground`), `app/services/search_text.py` (role/industry in the
embedding doc), `app/services/career_chronology.py` (`exp_semantics_by_id`),
`app/services/scoring.py` (experience-semantics step), `app/config.py` (version
3). No DB migration.

**H. ProfileSemanticData changes** — all existing v2/v3 fields preserved
(`seniority_level`, `job_families`, `technical_domains`, `industries`,
`explicit_skills`, `inferred_skills`, `leadership_experience`,
`domain_expertise`, `career_summary`, `current_role_summary`,
`semantic_assertions`, ...). Added:
- `experience_semantics: list[ExperienceSemantic]`
- `SemanticAssertion.experience_ids / education_ids / certification_ids`

**I. semantic_profile_version** — `2 -> 3`. v2 rows stay searchable; only a
backfill (or a fresh enrichment) fills in the v3 fields.

**J. Experience-ID changes** — `compact_profile` sends the real normalized row
id on every experience (`experience_id`), education (`education_id`) and
certification (`{certification_id, name}`). The LLM keys `experience_semantics`
by `experience_id` and cites `experience_ids` on assertions.

**K. Grounding behaviour** — `semantic_llm._ground`: collects the valid id sets
from the compact payload; drops every `experience_id` / `education_id` /
`certification_id` not in those sets; drops an `experience_semantics` row whose
`experience_id` is invalid; drops an assertion left with *neither* an evidence
phrase *nor* a valid source id. (Tested: `fake-123` removed, `exp-real` kept.)

**L. role_function design** — first-class `CriterionType.ROLE_FUNCTION`, a
member concept ("software engineering", "product management"). Scored in
`_score_semantic_concept` step 0 against `experience_semantics[].role_function`
/ `professional_domain` / `role_domains`. The fuzzy cross-encoder fallback is
**skipped** for `role_function` when structured experience semantics exist —
similarity must not turn a clean "no" into a "maybe" (V4 §H.5).

**M. employer_industry design** — `CriterionType.INDUSTRY_EXPERIENCE`, scored
against `experience_semantics[].employer_industries` / `employer_categories`,
kept strictly separate from `role_function`. Company *category* (startup / big
tech) still comes from the `company_semantics` cache, never the profile LLM.

**N. Example semantics**

| Profile | role_function | employer_industries | "worked at tech companies" | "technical engineering background" |
|---|---|---|---|---|
| Accountant @ Google | accounting | technology | strong (industry) | weak — NOT a technical role |
| Software Engineer @ JPMorgan | software engineering | financial services | weak — not a tech employer | strong (role) |

Verified by `tests/test_experience_semantics.py`: the accountant scores
`> 0.5` for `industry_experience:technology` and `< 0.4` for
`role_function:software engineering`; the JPMorgan engineer is the mirror, and
also scores `> 0.5` for `industry_experience:financial services`.

**O. career_transition schema** — `CriterionType.CAREER_TRANSITION`, structure
in `concept = "from <A> to <B>"`, `required`. Emitted by
`query_facts._extract_transition` from "moved/left/switched/pivoted from X to
Y", "left X for Y", "former X now in Y".

**P. Chronology algorithm** — `career_chronology.ordered_experiences()` sorts by
`(start_year, start_month)` oldest-first, falling back to reversed `order_index`
when dates are missing. `score_transition()`: find the earliest experience
matching the *from* concept and the earliest matching the *to* concept; TRUE
(0.9) only if a `from` experience **ends no later than** a later `to`
experience starts; if both concepts are present but never in that order ->
**FALSE** (V4 §20 — "tech then a 2022 consulting side-role" is not the same
transition). Uses `experience_semantics` when present, else role/company/
description text.

**Q. years-experience algorithm** — `total_years_matching()` collects the
date-intervals of experiences matching the domain predicate, **merges
overlapping intervals**, sums -> years (concurrent roles counted once, V4 §21).
`>= minimum` -> TRUE; some-but-short -> FALSE; no relevant role -> UNKNOWN (not
a false FALSE). Conservative when months are missing (assume Jan start / Dec
end).

**R. EXACT_MATCH** — every required criterion satisfied: a required *semantic*
criterion is `TriState.TRUE`; a required *non-semantic* criterion has
`match_strength >= _EXACT_MIN (0.55)`.

**S. POSSIBLE_MATCH** — no required criterion FALSE / unmet, but >= 1 required
semantic criterion is `UNKNOWN` (recorded in `uncertain_required`).

**T. NOT_MATCH** — a required semantic criterion is a confident `FALSE`, OR a
required non-semantic criterion is below `_EXACT_MIN` (a "Director" when the
query asked for "CXO" is a NOT_MATCH / near-match, not an exact match).

**U. Ranking-tier behaviour** — `search_service._tier_key = (rank[qualification],
-match_score)`. EXACT before POSSIBLE; NOT_MATCH excluded from `results`. Every
sort (pass-1, post-judge, post-rerank) uses it, and the cross-encoder re-sort is
followed by a tier re-sort so relevance **cannot** reorder across tiers (V4
§25). Verified: a verified EXACT_MATCH with score 40 ranks above a
POSSIBLE_MATCH with score 99.

**V. Near-match behaviour** — a NOT_MATCH candidate that misses **exactly one**
required criterion goes to `ConnectionBucket.near_matches` (capped at 5),
carrying `unmet_criteria` (e.g. *"Cxo level"*, *"Located in Bay Area"*). Normal
`results` never contain a NOT_MATCH. `exact_match_count` / `possible_match_count`
exposed on the bucket.

**W. Company-intelligence changes** — none in this phase (PART G is post-STOP).
The existing `company_semantics` cache remains authoritative for company
category; the profile LLM is now explicitly told **not** to classify companies.

**X. Embedding / vector changes** — none structural (PART H is post-STOP). One
behavioural change: the criterion-level cross-encoder fallback is suppressed for
`role_function` / `industry_experience` when experience semantics exist, so a
fuzzy similarity can't override structured role-vs-industry data.

**Y. Semantic judge / reranker changes** — none in this phase (PART I is
post-STOP). `llm_rerank` stays default-off. The judge still routes through the
PART A chain (Anthropic -> Groq -> OpenRouter).

**Z. Live API calls made** — ONE small structured provider-verification call
(`{"ok": true}` schema). Result:
```
anthropic:paid  -> configuration_error (no ANTHROPIC_WORKSPACE_ID)  [no retry, circuit opened 900s]
groq:primary    -> bad_output x3
groq:fallback   -> SUCCESS   (openai/gpt-oss-20b)
```
This proves the PART A contract end to end: Anthropic-first, auth/config = no
retry + immediate fallthrough, circuit breaker trips, Groq p->f fallthrough,
search continues.

**AA. Estimated paid operations during implementation** — **zero**. The
Anthropic key is identity-linked and rejects every request without a workspace
id, so the one verification call cost nothing on Anthropic; Groq is free-tier.
No dollar figure because no paid call succeeded.

**AB. ZERO Apify calls during this task** — confirmed. `tests/test_search.py`
and the search path make no Apify calls; `run_connection_search` and
`verify_v4.py` touch only stored DB data. No enrichment / re-scrape was run.

**AC. No Bright Data / Zyte / Firecrawl / paid scraping plugin used** —
confirmed. None invoked.

**AD. Commands for the optional semantic v3 backfill** (NOT run — needs your
approval per §44):
```bash
cd backend && python -m scripts.backfill_v3 dataset_0ba27cae09d4
# resumable, idempotent, version-aware, no Apify: classifies distinct employers
# (cache-first) then re-runs the semantic pass for every profile below
# semantic_profile_version=3 and re-embeds. Groq free-tier rate limits make a
# full 987-profile run take hours; add ANTHROPIC_WORKSPACE_ID to .env to make it
# fast (and paid).
```

**AE. Real searches** — `python -m scripts.verify_v4 dataset_0ba27cae09d4`
(deterministic, `LLM_QUERY_INTERPRETATION=false`, company classification
cache-only). Highlights:

| Query | Plan | EXACT / POSSIBLE | Verdict |
|---|---|---|---|
| CXO event in Memphis or Nashville | `location[Memphis,Nashville] ANY_OF req` + `seniority:cxo req`; "networking event" -> context | **0 / 0** | OK §37 — nobody is a CXO in those cities. 3 near-matches: 2 Nashville non-CXOs ("FAILS: Cxo level"), 1 COO ("FAILS: location"). |
| big tech in Bay Area | `company_category:big tech (current) req` + `location[Bay Area] req` | 13 / 103 | OK — EXACT = Google/AWS/NVIDIA/Meta in SF Bay Area with classification reasons. Near-matches = Amazon staff in other cities. |
| Former Amazon -> startups | `past_company[Amazon] req` + `company_category:startup (current) req` + `career_transition req` | 0 / 10 | PARTIAL — POSSIBLE not EXACT because company classification was cache-only this run (current employers UNKNOWN). Tri-state is correct; needs the v3 backfill to promote to EXACT. Near-matches "FAILS: Works at a startup" / "FAILS: Previously at Amazon". |
| moved from consulting to tech | `career_transition:"from consulting to technology" req` + soft `tech` | 48 / 909 | OK — EXACT rows show real ordered evidence ("Consultant at X -> Director at Y"). |
| software engineers in financial services | `role_function:"software engineering" req` | 43 / 944 | GAP — "in financial services" was **dropped** by the deterministic parser (no `industry_experience` criterion built for that phrase). The LLM path would keep it. |
| worked in tech / worked at tech companies / technical backgrounds | 3 near-identical `semantic_concept` plans | 987 / 0 each | GAP — not differentiated (§31). Interpretation confidence *is* lowered (0.5 for "worked in tech"). Real §30 two-dimension ranking needs the v3 `experience_semantics` + the LLM. |

**AF. Tests added**
- `tests/test_criterion_types.py` (6) — unknown type -> `professional_concept`, never `keyword`.
- `tests/test_query_facts.py` (+14) — AND vs OR, semantic booleans, NOT, role
  requiredness, "at fintech" required, transition + years emission.
- `tests/test_experience_semantics.py` (6) — role_function vs employer_industry
  (Accountant@Google / SWE@JPMorgan), id grounding.
- `tests/test_career_chronology.py` (8) — ordering, overlap merge, transition
  order, years-relevance.
- `tests/test_match_tiers.py` (5) — §36 A/B/C tiers, §25 tier-beats-score,
  §37 CXO-event-no-CXO.
- `scripts/verify_v4.py` — the real-data harness.

**AG. Backend test count** — `144 passed` (was 111). 0 failures.

**AH. Frontend tests** — `2 passed` (unchanged — no UI redesign).

**AI. TypeScript** — `tsc --noEmit` exit 0. The new response fields
(`qualification`, `near_matches`, `interpretation_summary`, ...) are read
structurally; no frontend change required by §46.

**AJ. Remaining limitations — what still needs real-dataset evaluation**

1. **The v3 semantic backfill has NOT been run** (0.4 % coverage — 4 / 991).
   Everything experience-level (`role_function` vs `employer_industry`
   separation, id-grounded assertions, `experience_semantics` in scoring and in
   `career_transition`) is **unit-verified with fixtures but not exercised on
   the real network**. Company classification is also mostly un-backfilled.
   Until `scripts.backfill_v3` runs, "worked in tech" ranking and
   Amazon->startup EXACT counts on real data will not reflect the design.
2. **Deterministic parser gaps** (LLM path handles these; deterministic does
   not): "software engineers **in financial services**" drops the industry;
   "worked in tech" / "worked at tech companies" / "technical backgrounds"
   produce near-identical plans (§31 wants them distinct); the bounded one-shot
   LLM repair call (§5 option 2) was not implemented — only deterministic
   reconstruction + confidence-lowering.
3. **No LLM-path real run** — the Anthropic key needs a workspace id and Groq is
   currently rate-limited, so the *LLM* query-planner was not exercised on the 8
   real queries. The plans above are the deterministic fallback (the worst case).
4. **Match-tier thresholds are heuristic** — `_EXACT_MIN = 0.55` for
   non-semantic criteria was chosen, not calibrated against labelled data (that
   is PART M, post-STOP).
5. **`career_transition` can be slightly redundant** — "Former Amazon people
   now at startups" produces `past_company` + `company_category` + a
   `career_transition:"from Amazon to startups"`; the transition adds a mild
   extra constraint rather than replacing the other two.

**Not claimed:** that search quality is "fixed". What is verified: the query
planner preserves AND/OR/NOT and never emits `keyword` for a concept; role
function is scored separately from employer industry (fixtures); transitions use
chronology and reject wrong-order careers; EXACT/POSSIBLE/NOT tiers rank before
score and a verified TRUE can't be out-ranked by embedding similarity; the CXO
query yields zero exact matches when no CXO exists in the requested cities. The
end-to-end quality on the real 991-person network still depends on running the
v3 backfill and the LLM planner.
