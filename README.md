# LinkedIn Connections Intelligence

Upload your LinkedIn Connections CSV → enrich every connection with Apify HarvestAPI
profile data → search your network in plain English and get the **Top 20 matching
people you already know**, each with an evidence-backed 0–100 match score, a grounded
reason, exact supporting evidence, and a **separate** data-confidence score.

Built to minimise cost: the `$4 / 1,000` HarvestAPI *profile details – no email* tier,
free LLMs only (Groq → Groq → OpenRouter), and **local** `sentence-transformers`
embeddings. `ENABLE_PAID_LLM=false` is enforced — no paid model is ever called.

## Stack

| | |
|---|---|
| Backend | Python 3.11 · FastAPI · Pydantic v2 · SQLAlchemy 2 · SQLite (Postgres-ready via `DATABASE_URL`) |
| Frontend | Vite · React · TypeScript · Tailwind · TanStack Query |
| Profile data | Apify actor `harvestapi/linkedin-profile-scraper` (`LpVuK3Zozwuipa5bp`), *Profile details no email* |
| LLM | Groq `openai/gpt-oss-120b` → `openai/gpt-oss-20b` → OpenRouter `:free` → queue (`WAITING_FOR_FREE_LLM`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, local, numpy brute-force cosine |

## Run it

```bash
# backend  →  http://localhost:8010   (docs at /docs)
cd backend
py -3.11 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp ../.env.example .env          # then fill APIFY_API_TOKEN + GROQ_API_KEY
.venv/Scripts/python -m uvicorn app.main:app --port 8010
```

```bash
# frontend  →  http://localhost:5182   (proxies /api/* → :8010)
cd frontend
npm install
npm run dev
```

Or use `run.ps1` from the repo root to start both.

## How it works

```
CSV → parse (skip export preamble) → canonicalize URLs → dedupe by public id
    → dataset + people (is_connection=true)
    → enrichment worker (resumable, batched): Apify → raw_profiles (verbatim)
      → deterministic normalize → experiences/education/skills + completeness
      → free-LLM semantic pass (seniority, domains, inferred skills w/ evidence)
      → local embedding
    → NL query → LLM (or deterministic) → weighted criteria (Σ = 100)
      → candidate pool (SQL + embeddings, no Apify) → deterministic scoring
      → Top 20 connections, each with score breakdown + evidence + data confidence
```

Scoring is done **in code** — the LLM only turns the deterministic evidence into a
sentence and may not add claims. Fact vs. AI-inferred is shown distinctly in the UI.

## Cost

~300 connections ≈ **one-time $1.20** on Apify (fixture mode during development is $0).
The external "Expand beyond my connections" search is **not** in this build (inert
hooks are in place for it).

## Tests

```bash
cd backend && .venv/Scripts/python -m pytest      # 59 tests
cd frontend && npm test                            # 2 tests
```

## Doing the real run (your ~240 connections)

**Apify credit first.** The scraper is pay-per-event ($0.004/profile). Check
`https://console.apify.com/billing` — the Free plan gives **$5/month** of usage that
resets on your billing-cycle date. ~240 profiles ≈ **$0.96**, so the free credit
covers a full run *if* it hasn't already been spent this cycle. If usage is near $5,
either wait for the reset or add a pay-as-you-go card.

1. In `backend/.env`: `USE_FIXTURES=false`, keep `DEVELOPMENT_BATCH_SIZE=5` /
   `ENVIRONMENT=development` for the first pass. Optionally set
   `APIFY_MAX_CHARGE_USD=2` as a hard stop.
2. Start backend + frontend, open http://localhost:5182, upload your `Connections.csv`.
3. Enrichment runs in resumable batches. If the free **Groq** quota runs out mid-run,
   the semantic step is skipped for the rest of the run (profiles still get scraped,
   normalized, embedded and marked READY). Click **Resume** later and it runs a
   `backfill_semantics` pass — nothing is re-scraped.
4. Bump `ENVIRONMENT=production` (batch size 50) once the first batches look good.
5. Refresh a stale profile from its page; `PROFILE_TTL_DAYS=30` guards re-scrapes.

`scripts/reset_dataset.py <dataset_id>` wipes a dataset's derived data back to PENDING
(keeps the CSV rows) if a run went wrong.

> Note on latency: when the shared Groq free tier is rate-limited, query
> interpretation and reason generation fall back to `gpt-oss-20b` and then to
> deterministic templates — correct, just slower. Add `OPENROUTER_API_KEY` to widen
> free capacity.

## Config knobs (`.env`)

`USE_FIXTURES` · `ENVIRONMENT` · `DEVELOPMENT_BATCH_SIZE` / `APIFY_PROFILE_BATCH_SIZE`
· `MAX_APIFY_RETRIES` · `LLM_MAX_RETRIES` · `PROFILE_TTL_DAYS` · `CANDIDATE_POOL_SIZE`
· `TOP_CONNECTIONS` · `MIN_MATCH_SCORE` · `SEMANTIC_ENABLED` · `LLM_QUERY_INTERPRETATION`
· `LLM_REASON_GENERATION` · `EMBEDDINGS_ENABLED` · `ENABLE_PAID_LLM` (must stay `false`).
