import json
import os
import re
import requests
from typing import Optional
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

# ── Paths ──
CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chroma_db")
)
ASSESSMENTS_PATH = os.path.join(os.path.dirname(__file__), "../data/assessments.json")

# ── Constants ──
TEST_TYPE_NAMES = {
    "A": "Ability and Aptitude",
    "B": "Biodata and Situational Judgement",
    "C": "Competencies",
    "D": "Development and 360",
    "E": "Assessment Exercises",
    "K": "Knowledge and Skills",
    "P": "Personality and Behavior",
    "S": "Simulations",
}


class SHLRecommender:
    def __init__(self):
        print("Initializing SHL Recommender...")

        # Set up Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env file")
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel("gemini-2.5-flash")

        # Load embedding model
        print("  Loading embedding model...")
        self.embed_model = SentenceTransformer("all-mpnet-base-v2")

        # Load ChromaDB
        print("  Connecting to ChromaDB...")
        if not os.path.exists(CHROMA_DB_PATH):
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DB_PATH}. "
                "Run: python embeddings/build_index.py"
            )
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = client.get_collection("shl_assessments")
        print(f"  Loaded collection with {self.collection.count()} assessments ✅")

        # Load raw assessments for fallback lookups
        with open(ASSESSMENTS_PATH) as f:
            self.assessments = json.load(f)
        self.url_to_assessment = {a["url"].rstrip("/") + "/": a for a in self.assessments}

  
    # Step 1: If input is a URL, fetch the JD text
  
    def resolve_input(self, query: str) -> str:
        """If query looks like a URL, fetch its text content."""
        url_pattern = re.compile(r"^https?://", re.I)
        if url_pattern.match(query.strip()):
            try:
                resp = requests.get(query.strip(), timeout=10)
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "lxml")
                # Remove scripts/styles
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                return text[:3000]  # Limit to 3000 chars
            except Exception as e:
                print(f"  [WARN] Could not fetch URL: {e}")
                return query
        return query


    # Step 2: LLM Query Expansion
  
    def expand_query(self, query: str) -> dict:
        """
        Use Gemini to extract structured intent from the query.
        Returns: skills, soft_skills, test_types, max_duration, seniority
        """
        prompt = f"""
You are an SHL assessment expert. Analyze this hiring query carefully.

Query: "{query[:1500]}"

Return ONLY valid JSON — no markdown, no explanation:
{{
  "technical_skills": ["exact skill names like Python, SQL, JavaScript, Java"],
  "soft_skills": ["collaboration", "communication"],
  "job_role": "role in 2-3 words",
  "seniority": "entry or mid or senior",
  "max_duration_minutes": null,
  "min_duration_minutes": null,
  "test_types_needed": ["K", "P"],
  "expanded_query": "Write a detailed search query. Include EXACT skill names. Example: Python programming knowledge test SQL database JavaScript web development mid-level professional assessment Knowledge Skills test"
}}

Examples of test_types_needed:
- Java developer + collaboration → ["K", "P"]
- Sales role → ["P", "B", "K"]  
- COO / leadership → ["P", "C", "A"]
- Data analyst → ["K", "A"]
- Admin role → ["K", "A", "B"]
"""
        try:
            response = self.llm.generate_content(prompt)
            text = response.text.strip()
            # Clean up any accidental markdown
            text = re.sub(r"```json|```", "", text).strip()
            parsed = json.loads(text)
            return parsed
        except Exception as e:
            print(f"  [WARN] LLM expansion failed: {e}. Using raw query.")
            return {
                "technical_skills": [],
                "soft_skills": [],
                "job_role": "",
                "seniority": "mid",
                "max_duration_minutes": None,
                "min_duration_minutes": None,
                "test_types_needed": [],
                "expanded_query": query,
            }

    # ────────────────────────────────────────────────
    # Step 3: Vector Search
    # ────────────────────────────────────────────────
    def vector_search(self, expanded_query: str, n_results: int = 20) -> list:
        """Search ChromaDB using the expanded query embedding."""
        query_embedding = self.embed_model.encode(expanded_query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["metadatas", "distances", "documents"],
        )
        candidates = []
        for i, meta in enumerate(results["metadatas"][0]):
            candidates.append({
                "name": meta.get("name", ""),
                "url": meta.get("url", ""),
                "description": meta.get("description", ""),
                "duration": meta.get("duration", 0),
                "remote_support": meta.get("remote_support", "No"),
                "adaptive_support": meta.get("adaptive_support", "No"),
                "test_type": meta.get("test_type", "").split(",") if meta.get("test_type") else [],
                "similarity_score": 1 - results["distances"][0][i],  # cosine → similarity
            })
        return candidates

    # ────────────────────────────────────────────────
    # Step 4: Filter by Duration
    # ────────────────────────────────────────────────
    def filter_by_duration(
        self,
        candidates: list,
        max_duration: Optional[int],
        min_duration: Optional[int],
    ) -> list:
        """Filter candidates by duration constraints."""
        if not max_duration and not min_duration:
            return candidates

        filtered = []
        for c in candidates:
            dur = c.get("duration", 0)
            if max_duration and dur > 0 and dur > max_duration:
                continue
            if min_duration and dur > 0 and dur < min_duration:
                continue
            filtered.append(c)

        # If filtering removed too many, relax and return top candidates
        if len(filtered) < 5:
            return candidates
        return filtered

    # ────────────────────────────────────────────────
    # Step 5: LLM Re-ranking for Balance
    # ────────────────────────────────────────────────
    def rerank_for_balance(
        self,
        candidates: list,
        original_query: str,
        expansion: dict,
        top_k: int = 10,
    ) -> list:
        """
        Ask Gemini to pick the best balanced set of assessments.
        Ensures mix of technical (K) and behavioral (P) types when relevant.
        """
        # Prepare a compact summary of candidates for the LLM
        candidate_summaries = []
        for i, c in enumerate(candidates[:25]):  # Give LLM top 25 to choose from
            types = ", ".join(c["test_type"]) if c["test_type"] else "Unknown"
            dur = f"{c['duration']} min" if c["duration"] else "unknown duration"
            candidate_summaries.append(
                f"{i+1}. [{types}] {c['name']} ({dur}) - {c['description'][:120]}"
            )

        candidates_text = "\n".join(candidate_summaries)
        needed_types = ", ".join(expansion.get("test_types_needed", [])) or "mixed"
        
        prompt = f"""
You are an expert HR assessment advisor. Select the BEST {top_k} assessments for this hiring need.

ORIGINAL QUERY: "{original_query}"

IDENTIFIED NEEDS:
- Role: {expansion.get('job_role', 'unspecified')}
- Technical skills: {', '.join(expansion.get('technical_skills', [])) or 'none'}
- Soft skills: {', '.join(expansion.get('soft_skills', [])) or 'none'}
- Seniority: {expansion.get('seniority', 'mid')}
- Test types needed: {needed_types}

CANDIDATE ASSESSMENTS (numbered):
{candidates_text}

RULES:
1. Select exactly {top_k} assessments (or fewer if not enough candidates)
2. BALANCE is critical: if both technical and behavioral skills are needed, include BOTH K-type and P-type assessments
3. Avoid selecting duplicate or nearly identical assessments
4. Prioritize relevance to the specific role and skills

Return ONLY a JSON array of the selected assessment numbers, e.g.: [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
No explanation, no markdown, just the JSON array.
"""
        try:
            response = self.llm.generate_content(prompt)
            text = response.text.strip()
            text = re.sub(r"```json|```", "", text).strip()
            selected_indices = json.loads(text)
            
            # Convert 1-based indices to 0-based and select candidates
            result = []
            for idx in selected_indices:
                if 1 <= idx <= len(candidates[:25]):
                    result.append(candidates[idx - 1])
            return result[:top_k]
        except Exception as e:
            print(f"  [WARN] LLM re-ranking failed: {e}. Using similarity scores.")
            return candidates[:top_k]

    # ────────────────────────────────────────────────
    # Main Recommend Function
    # ────────────────────────────────────────────────
    def recommend(self, query: str, top_k: int = 10) -> list:
        """
        Full pipeline:
        URL resolve → LLM expand → vector search → duration filter → LLM rerank
        """
        print(f"\n[Recommender] Query: {query[:80]}...")

        # Step 1: Resolve URL if needed
        resolved_query = self.resolve_input(query)

        # Step 2: LLM expansion
        print("  [1/4] Expanding query with LLM...")
        expansion = self.expand_query(resolved_query[:2000])
        print(f"        Role: {expansion.get('job_role')}, "
              f"Types: {expansion.get('test_types_needed')}, "
              f"Max duration: {expansion.get('max_duration_minutes')} min")

        # Step 3: Vector search using expanded query
        print("  [2/4] Searching vector index...")
        search_query = expansion.get("expanded_query", resolved_query)
        candidates = self.vector_search(search_query, n_results=30)
        print(f"        Found {len(candidates)} candidates")

        # Step 4: Duration filter
        print("  [3/4] Applying duration filter...")
        candidates = self.filter_by_duration(
            candidates,
            max_duration=expansion.get("max_duration_minutes"),
            min_duration=expansion.get("min_duration_minutes"),
        )
        print(f"        {len(candidates)} candidates after filter")

        # Step 5: LLM re-ranking
        print("  [4/4] Re-ranking for relevance and balance...")
        final = self.rerank_for_balance(candidates, query, expansion, top_k=top_k)
        print(f"        Selected {len(final)} final assessments")

        return final

    def format_response(self, assessments: list) -> dict:

        TYPE_NAMES = {
            "A": "Ability & Aptitude",
            "B": "Biodata & Situational Judgement",
            "C": "Competencies",
            "D": "Development & 360",
            "E": "Assessment Exercises",
            "K": "Knowledge & Skills",
            "P": "Personality & Behaviour",
            "S": "Simulations",
        }

        return {
            "recommended_assessments": [
                {
                    "url"             : a["url"],
                    "name"            : a["name"],
                    "adaptive_support": a.get("adaptive_support", "No"),
                    "description"     : a.get("description", ""),
                    "duration"        : int(a.get("duration", 0)),
                    "remote_support"  : a.get("remote_support", "No"),
                    "test_type"       : [
                        TYPE_NAMES.get(t, t)
                        for t in a.get("test_type", [])
                    ],
                }
                for a in assessments
            ]
        }


# Singleton instance
_recommender_instance = None


def get_recommender() -> SHLRecommender:
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = SHLRecommender()
    return _recommender_instance
