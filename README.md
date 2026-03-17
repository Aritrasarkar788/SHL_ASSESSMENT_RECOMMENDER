# SHL Assessment Recommendation System

> AI-powered tool that recommends relevant SHL assessments from a natural language query or job description.
> Built with Gemini LLM + sentence-transformers + ChromaDB + FastAPI.

## Architecture

```
User Query / JD / URL
      ↓
[Gemini: Query Expansion]  → extracts skills, test types, duration
      ↓
[Sentence-Transformer Embed] → query vector
      ↓
[ChromaDB Vector Search]    → top 20-30 candidate assessments
      ↓
[Duration Filter]           → respects time constraints
      ↓
[Gemini: Re-rank + Balance] → picks best 10, balances K-type vs P-type
      ↓
FastAPI /recommend → JSON response
```

## Project Structure
```
shl-recommender/
├── scraper/
│   └── scrape_catalog.py       # Step 1: Scrape SHL catalog (377+ assessments)
├── embeddings/
│   └── build_index.py          # Step 2: Build vector embeddings + ChromaDB index
├── api/
│   ├── main.py                 # Step 3: FastAPI server (/health + /recommend)
│   └── recommender.py          # Core recommendation logic
├── frontend/
│   └── index.html              # Step 4: Simple web UI
├── evaluation/
│   ├── evaluate.py             # Step 5: Compute Mean Recall@10 on train set
│   └── generate_predictions.py # Step 6: Generate CSV for test set submission
├── data/
│   ├── assessments.json        # Output of scraper (generated)
│   ├── train_data.json         # Parsed from Excel
│   └── test_queries.json       # Parsed from Excel
├── requirements.txt
└── .env.example
```

## Setup & Run (in order)

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Set your Gemini API key
cp .env.example .env
# Edit .env and add GEMINI_API_KEY=your_key

# 3. Scrape SHL catalog
python scraper/scrape_catalog.py

# 4. Build embeddings index
python embeddings/build_index.py

# 5. Start API
uvicorn api.main:app --reload --port 8000

# 6. Evaluate on train set
python evaluation/evaluate.py

# 7. Generate test set predictions CSV
python evaluation/generate_predictions.py
```
