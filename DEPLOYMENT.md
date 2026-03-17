# Deployment Guide

## Prerequisites
- GitHub account
- Render.com account (free)
- Vercel account (free)
- Gemini API key (free at https://ai.google.dev)

---

## Step 1 — Push to GitHub

```bash
cd shl-recommender
git init
git add .
git commit -m "Initial commit: SHL Assessment Recommender"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/shl-recommender.git
git push -u origin main
```

---

## Step 2 — Run the data pipeline locally first

You MUST run Steps 1–2 locally to generate the data files before deploying.
The scraped data and ChromaDB index need to be committed to your repo.

```bash
# Install deps
pip install -r requirements.txt
playwright install chromium

# Set up .env
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_key_here

# Run full pipeline (scrape → embed → evaluate → predict)
python run.py

# Commit the generated data
git add data/assessments.json data/chroma_db/
git commit -m "Add scraped assessments and vector index"
git push
```

> Note: chroma_db/ can be large (~50MB). Add to .gitignore if you prefer
> to rebuild on the server. Set `REBUILD_INDEX=true` env var on Render
> and add startup logic to call build_index.py before uvicorn starts.

---

## Step 3 — Deploy API to Render.com

1. Go to https://render.com → **New** → **Web Service**
2. Connect your GitHub repo
3. Configure:
   - **Name:** `shl-recommender-api`
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable:
   - Key: `GEMINI_API_KEY`
   - Value: your Gemini API key
5. Click **Create Web Service**

Your API will be live at: `https://shl-recommender-api.onrender.com`

Test it:
```bash
# Health check
curl https://shl-recommender-api.onrender.com/health

# Recommendation
curl -X POST https://shl-recommender-api.onrender.com/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "I need a Java developer who can collaborate with teams"}'
```

---

## Step 4 — Deploy Frontend to Vercel

1. Update the `API_BASE` in `frontend/index.html`:
   ```js
   const API_BASE = "https://shl-recommender-api.onrender.com";
   ```

2. Go to https://vercel.com → **New Project** → Import your GitHub repo
3. Configure:
   - **Framework Preset:** Other
   - **Root Directory:** `frontend`
   - **Output Directory:** `.` (current)
4. Click **Deploy**

Your frontend will be live at: `https://shl-recommender.vercel.app`

---

## Step 5 — Verify Submission Requirements

Run this checklist before submitting:

```bash
# 1. API Health Check
curl https://YOUR-API-URL/health
# Expected: {"status": "healthy"}

# 2. Recommendation Endpoint
curl -X POST https://YOUR-API-URL/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Java developer with collaboration skills"}'
# Expected: {"recommended_assessments": [...]} with 5-10 items

# 3. Check CSV format
head -5 data/test_predictions.csv
# Expected:
# Query,Assessment_url
# Looking to hire mid-level...,https://www.shl.com/...
# Looking to hire mid-level...,https://www.shl.com/...
```

---

## Submission Checklist

- [ ] **API URL** — `https://YOUR-API-URL` (Render)
  - [ ] GET `/health` returns `{"status": "healthy"}`
  - [ ] POST `/recommend` returns correct JSON format
- [ ] **Frontend URL** — `https://YOUR-FRONTEND-URL` (Vercel)
- [ ] **GitHub URL** — `https://github.com/YOUR_USERNAME/shl-recommender`
  - [ ] All code including experiments visible
  - [ ] README with setup instructions
- [ ] **2-page document** — `APPROACH.md` (convert to PDF for submission)
- [ ] **CSV file** — `data/test_predictions.csv`
  - [ ] Columns: `Query`, `Assessment_url`
  - [ ] 9 test queries × up to 10 predictions each

---

## Troubleshooting

**Render cold starts:** Free tier spins down after 15 min inactivity. First request takes ~30s. Use UptimeRobot (free) to ping `/health` every 14 minutes to keep it warm.

**ChromaDB on Render:** If you get file permission errors, set the CHROMA_DB_PATH to `/tmp/chroma_db` and rebuild on startup.

**Gemini rate limits:** Free tier allows 15 requests/minute. The pipeline uses 2 LLM calls per query, so you can handle ~7 queries/minute. Add a simple in-memory cache for repeated queries if needed.
