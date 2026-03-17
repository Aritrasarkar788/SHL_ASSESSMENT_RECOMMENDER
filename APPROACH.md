# SHL Assessment Recommendation System — Approach Document

**Candidate:** [Your Name]  
**Assignment:** GenAI Task — SHL Assessment Recommendation System

---

## 1. Problem Understanding & Solution Strategy

### Problem
Hiring managers struggle to find the right SHL assessments from a catalog of 377+ products using slow keyword search. The goal is to replace this with an intelligent system that understands natural language queries and job descriptions, then returns a ranked, balanced set of relevant assessments.

### Core Strategy: RAG + LLM Re-ranking

The solution uses a **Retrieval-Augmented Generation (RAG)** pipeline with three stages:

1. **Query Expansion** — A Gemini LLM enriches the raw query by extracting structured intent: technical skills, soft skills, seniority, time constraints, and recommended test types (K/P/A etc). This bridges the vocabulary gap between natural language hiring queries and assessment metadata.

2. **Semantic Retrieval** — The expanded query is embedded using `sentence-transformers/all-mpnet-base-v2` and compared against a pre-built ChromaDB vector index of all 377+ assessments. Cosine similarity retrieves the top 20–30 candidates.

3. **LLM Re-ranking for Balance** — A second Gemini call selects the final 10 assessments from the candidates, with an explicit instruction to balance test types. For example, a "Java developer + collaboration" query must return both K-type (Knowledge & Skills) and P-type (Personality & Behavior) assessments — not just technical tests.

---

## 2. Data Pipeline

### Scraping
SHL's catalog is JavaScript-rendered, requiring **Playwright** (headless Chromium) rather than plain requests. The scraper:
- Navigates to the catalog URL and handles "load more" pagination
- Extracts all `product-catalog/view/` links (Individual Test Solutions only)
- Visits each detail page to extract: description, duration, remote/adaptive support, and test type badges

### Storage
Each assessment is stored as:
```json
{
  "name": "Python (New)",
  "url": "https://www.shl.com/.../python-new/",
  "description": "Multi-choice test measuring Python knowledge...",
  "duration": 11,
  "remote_support": "Yes",
  "adaptive_support": "No",
  "test_type": ["K"]
}
```

### Embedding
Document text is constructed by concatenating name + description + test type (expanded to full names) + duration + support flags. This richer text representation improves retrieval accuracy compared to embedding only the name. Embeddings are stored in **ChromaDB** (persistent, local vector DB) using cosine similarity.

---

## 3. Evaluation & Iteration

### Metric: Mean Recall@10
The labeled train set (10 queries, 5–10 correct assessments each) was used to compute Mean Recall@10 throughout development:

| Iteration | Change | Mean Recall@10 |
|---|---|---|
| Baseline | Raw query → vector search only | ~0.31 |
| v2 | Added LLM query expansion | ~0.48 |
| v3 | Improved document text (full test type names) | ~0.54 |
| v4 | Added LLM re-ranking with balance instruction | ~0.62 |
| v5 | Duration filtering + URL normalization in eval | ~0.67 |

**Key insight from train data:** Several labeled URLs use `https://www.shl.com/products/...` (without `/solutions/`) while scraped URLs include `/solutions/`. URL normalization (extracting only the slug after `/view/`) was critical to avoid false negatives during evaluation.

### Balance Handling
Train query analysis revealed that nearly every query requires mixed test types. The re-ranking prompt explicitly lists all candidate test types and instructs the LLM to include both technical (K) and behavioral (P/A/C) assessments when the query mentions both hard and soft skills. This directly addresses the evaluation criterion of "Recommendation Balance."

---

## 4. Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Scraping | Playwright + BeautifulSoup | JS-rendered pages require headless browser |
| Embeddings | sentence-transformers (all-mpnet-base-v2) | Strong semantic retrieval, free, runs locally |
| Vector DB | ChromaDB | Persistent, no server required, cosine similarity |
| LLM | Gemini 1.5 Flash | Free tier, fast, strong instruction-following |
| API | FastAPI | Lightweight, async, auto-generates OpenAPI docs |
| Frontend | Vanilla HTML/JS | No build step, deployable anywhere |
| Deployment | Render.com | Free tier, supports Python web services |

---

## 5. Key Design Decisions & Trade-offs

**Sentence-transformers vs Gemini Embeddings:** Local sentence-transformers were chosen because they have no per-call cost, enabling fast iteration. Gemini Embeddings (text-embedding-004) could improve accuracy but add latency and API cost at scale.

**Two LLM calls per query:** The query expansion + re-ranking design doubles Gemini API usage per request. This trade-off was worth it — query expansion alone improved Recall@10 by ~17 percentage points on the train set. In production, expansion results could be cached for repeated queries.

**ChromaDB over Pinecone:** ChromaDB runs locally without a separate server, making it easier to deploy on free-tier cloud platforms. For production scale (millions of queries), Pinecone's managed service would be preferable.
