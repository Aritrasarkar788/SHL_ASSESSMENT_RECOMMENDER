"""
Step 5: Evaluate on Train Set
-------------------------------
Computes Mean Recall@10 against the labeled train set from Gen_AI_Dataset.xlsx.

Uses YOUR dataset directly — no manual input needed.

Run: python evaluation/evaluate.py
"""

import json
import os
import sys
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import openpyxl
from api.recommender import get_recommender

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../data/Gen_AI_Dataset.xlsx")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../data/train_evaluation_results.json")


def normalize_url(url: str) -> str:
    """
    Normalize SHL URLs for comparison.
    Both https://www.shl.com/products/... and https://www.shl.com/solutions/products/...
    should match. We extract just the slug.
    """
    if not url:
        return ""
    # Extract the slug after /product-catalog/view/
    match = re.search(r"/product-catalog/view/([^/]+)/?", url)
    if match:
        return match.group(1).rstrip("/").lower()
    return url.rstrip("/").lower()


def load_train_data() -> dict:
    """Load train set from Excel. Returns {query: [list of correct URLs]}"""
    wb = openpyxl.load_workbook(DATASET_PATH)
    ws = wb["Train-Set"]
    
    queries = {}
    for row in ws.iter_rows(values_only=True):
        if row[0] == "Query":
            continue
        q, url = row
        if q not in queries:
            queries[q] = []
        queries[q].append(url)
    
    return queries


def recall_at_k(predicted_urls: list, true_urls: list, k: int = 10) -> float:
    """Compute Recall@K for a single query."""
    if not true_urls:
        return 0.0
    
    # Normalize all URLs to slugs for fair comparison
    pred_slugs = set(normalize_url(u) for u in predicted_urls[:k])
    true_slugs = set(normalize_url(u) for u in true_urls)
    
    hits = len(pred_slugs & true_slugs)
    return hits / len(true_slugs)


def evaluate():
    print("=" * 60)
    print("SHL Recommender — Train Set Evaluation")
    print("=" * 60)

    # Load data
    print("\nLoading train data from dataset...")
    train_data = load_train_data()
    print(f"  {len(train_data)} labeled queries loaded")

    # Load recommender
    print("\nLoading recommender...")
    recommender = get_recommender()

    # Run evaluation
    results = []
    recall_scores = []

    for i, (query, true_urls) in enumerate(train_data.items()):
        print(f"\n[{i+1}/{len(train_data)}] Query: {query[:70]}...")
        print(f"  Ground truth: {len(true_urls)} assessments")

        # Get predictions
        try:
            predictions = recommender.recommend(query, top_k=10)
            predicted_urls = [p["url"] for p in predictions]
        except Exception as e:
            print(f"  ERROR: {e}")
            predicted_urls = []

        # Compute recall
        score = recall_at_k(predicted_urls, true_urls, k=10)
        recall_scores.append(score)

        print(f"  Predicted: {len(predicted_urls)} assessments")
        print(f"  Recall@10: {score:.3f}")

        # Show hits and misses
        pred_slugs = set(normalize_url(u) for u in predicted_urls)
        true_slugs = {normalize_url(u): u for u in true_urls}
        
        hits = [u for slug, u in true_slugs.items() if slug in pred_slugs]
        misses = [u for slug, u in true_slugs.items() if slug not in pred_slugs]
        
        if hits:
            print(f"  ✅ HITS: {[h.split('/view/')[-1].rstrip('/') for h in hits]}")
        if misses:
            print(f"  ❌ MISSED: {[m.split('/view/')[-1].rstrip('/') for m in misses]}")

        results.append({
            "query": query[:100],
            "true_count": len(true_urls),
            "predicted_count": len(predicted_urls),
            "recall_at_10": score,
            "predicted_urls": predicted_urls,
            "true_urls": true_urls,
        })

    # Final score
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    print("\n" + "=" * 60)
    print(f"MEAN RECALL@10: {mean_recall:.4f}  ({mean_recall*100:.1f}%)")
    print("=" * 60)
    print("\nPer-query breakdown:")
    for i, (q, s) in enumerate(zip(train_data.keys(), recall_scores)):
        bar = "█" * int(s * 20)
        print(f"  Q{i+1}: {s:.3f} {bar} | {q[:50]}...")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "mean_recall_at_10": mean_recall,
            "queries": results
        }, f, indent=2)
    print(f"\n✅ Detailed results saved to: {RESULTS_PATH}")

    return mean_recall


if __name__ == "__main__":
    evaluate()
