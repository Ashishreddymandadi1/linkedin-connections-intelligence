# LinkedIn Connections Intelligence

Upload your LinkedIn Connections CSV → enrich every connection with Apify HarvestAPI
profile data → search your network in plain English and get the **Top 20 matching
people you already know**, each with an evidence-backed 0–100 match score, a grounded
reason, and a separate data-confidence score.

Built to minimise cost: the `$4 / 1,000` HarvestAPI *profile details – no email* tier,
free LLMs only (Groq → Groq → OpenRouter), and **local** `sentence-transformers`
embeddings.

## Stack

| | |
|---|---|
| Backend | Python 3.11 · FastAPI · Pydantic v2 · SQLAlchemy 2 · SQLite (Postgres-ready) |
| Frontend | Vite · React · TypeScript · Tailwind |
| Profile data | Apify actor `harvestapi/linkedin-profile-scraper` (`LpVuK3Zozwuipa5bp`) |
| LLM | Groq `gpt-oss-120b` → `gpt-oss-20b` → OpenRouter free (never a paid model) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, local, brute-force cosine |

## Setup

```bash
# backend
cd backend
py -3.11 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp ../.env.example .env      # then fill APIFY_API_TOKEN + GROQ_API_KEY
.venv/Scripts/python -m uvicorn app.main:app --port 8010

# frontend  (new terminal)
cd frontend
npm install
npm run dev                  # http://localhost:5182
```

The frontend proxies `/api/*` → `http://localhost:8010`.

## Cost

~300 connections ≈ **one-time $1.20** on Apify. Everything else is free.
`ENABLE_PAID_LLM=false` is enforced — no paid model is ever called automatically.

## Tests

```bash
cd backend && .venv/Scripts/python -m pytest
cd frontend && npm test
```

## Status

Vertical slice in progress — see `../` build plan. External "Expand beyond my
connections" search is deliberately out of this build (hooks left in place).
