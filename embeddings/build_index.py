"""
Step 2: Build Vector Embeddings Index
---------------------------------------
Reads data/assessments.json, creates sentence embeddings,
and stores them in a ChromaDB vector database for fast retrieval.

Run: python embeddings/build_index.py
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from sentence_transformers import SentenceTransformer

ASSESSMENTS_PATH = os.path.join(os.path.dirname(__file__), "../data/assessments.json")
CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chroma_db")
)

# Test type full names for richer embeddings
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


def build_document_text(assessment: dict) -> str:
    """
    Create a rich text representation of each assessment for embedding.
    The more descriptive this text, the better the retrieval quality.
    """
    name = assessment.get("name", "")
    description = assessment.get("description", "")
    duration = assessment.get("duration", 0)
    remote = assessment.get("remote_support", "No")
    adaptive = assessment.get("adaptive_support", "No")
    test_types = assessment.get("test_type", [])

    # Expand test type codes to full names
    type_descriptions = []
    for t in test_types:
        full_name = TEST_TYPE_NAMES.get(t, t)
        type_descriptions.append(f"{t} ({full_name})")

    parts = [
        f"Assessment Name: {name}",
        f"Description: {description}" if description else "",
        f"Test Types: {', '.join(type_descriptions)}" if type_descriptions else "",
        f"Duration: {duration} minutes" if duration else "",
        f"Remote Testing: {remote}",
        f"Adaptive Testing: {adaptive}",
    ]

    return " | ".join([p for p in parts if p])


def build_index():
    # ── Load assessments ──
    print(f"Loading assessments from {ASSESSMENTS_PATH}...")
    if not os.path.exists(ASSESSMENTS_PATH):
        print("ERROR: assessments.json not found. Run scraper first:")
        print("  python scraper/scrape_catalog.py")
        sys.exit(1)

    with open(ASSESSMENTS_PATH, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    print(f"  Loaded {len(assessments)} assessments")

    # ── Load embedding model ──
    print("Loading sentence transformer model...")
    # all-mpnet-base-v2 is slower but more accurate than MiniLM
    # Use 'all-MiniLM-L6-v2' if you want faster build time
    model = SentenceTransformer("all-mpnet-base-v2")
    print("  Model loaded ✅")

    # ── Build ChromaDB ──
    print(f"Setting up ChromaDB at {CHROMA_DB_PATH}...")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("shl_assessments")
        print("  Deleted existing collection (rebuilding fresh)")
    except:
        pass

    collection = client.create_collection(
        name="shl_assessments",
        metadata={"hnsw:space": "cosine"},  # Cosine similarity
    )

    # ── Embed and store ──
    print("Building embeddings and storing in ChromaDB...")
    documents = []
    embeddings_list = []
    metadatas = []
    ids = []

    for i, assessment in enumerate(assessments):
        doc_text = build_document_text(assessment)
        documents.append(doc_text)
        metadatas.append({
            "name": assessment.get("name", ""),
            "url": assessment.get("url", ""),
            "duration": assessment.get("duration", 0),
            "remote_support": assessment.get("remote_support", "No"),
            "adaptive_support": assessment.get("adaptive_support", "No"),
            "test_type": ",".join(assessment.get("test_type", [])),
            "description": assessment.get("description", "")[:400],
        })
        ids.append(f"assessment_{i}")

    # Batch encode for efficiency
    print("  Encoding documents (this may take a minute)...")
    embeddings_list = model.encode(documents, batch_size=32, show_progress_bar=True).tolist()

    # Store in ChromaDB in batches of 100
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            embeddings=embeddings_list[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  Stored batch {start}–{end}")

    print(f"\n✅ Index built with {len(documents)} assessments")
    print(f"   ChromaDB saved at: {CHROMA_DB_PATH}")
    return collection


if __name__ == "__main__":
    print("=" * 60)
    print("Building SHL Assessment Embeddings Index")
    print("=" * 60)
    build_index()
    print("\nDone! You can now run the API:")
    print("  uvicorn api.main:app --reload --port 8000")
