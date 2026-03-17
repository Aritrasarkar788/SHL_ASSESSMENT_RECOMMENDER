"""
Step 6: Generate Test Set Predictions CSV
------------------------------------------
Runs your recommender on all 9 test queries from Gen_AI_Dataset.xlsx
and saves predictions in the required submission format:

  Query,Assessment_url
  Query 1,https://www.shl.com/...
  Query 1,https://www.shl.com/...
  Query 2,https://www.shl.com/...
  ...

Run: python evaluation/generate_predictions.py
"""

import os
import sys
import csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import openpyxl
from api.recommender import get_recommender

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../data/Gen_AI_Dataset.xlsx")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "../data/test_predictions.csv")


def load_test_queries() -> list:
    """Load test queries from the Excel dataset."""
    wb = openpyxl.load_workbook(DATASET_PATH)
    ws = wb["Test-Set"]
    queries = []
    for row in ws.iter_rows(values_only=True):
        if row[0] == "Query":
            continue
        queries.append(row[0])
    return queries


def generate_predictions():
    print("=" * 60)
    print("Generating Test Set Predictions")
    print("=" * 60)

    # Load test queries
    test_queries = load_test_queries()
    print(f"\nLoaded {len(test_queries)} test queries")

    # Load recommender
    print("Loading recommender...")
    recommender = get_recommender()

    # Generate predictions
    all_rows = []

    for i, query in enumerate(test_queries):
        print(f"\n[{i+1}/{len(test_queries)}] {query[:80]}...")
        
        try:
            predictions = recommender.recommend(query, top_k=10)
            urls = [p["url"] for p in predictions]
        except Exception as e:
            print(f"  ERROR: {e}")
            urls = []

        print(f"  → {len(urls)} assessments recommended")
        for url in urls:
            print(f"     {url}")
            all_rows.append({"Query": query, "Assessment_url": url})

    # Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Query", "Assessment_url"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✅ Predictions saved to: {OUTPUT_CSV}")
    print(f"   Total rows: {len(all_rows)} (across {len(test_queries)} queries)")
    print("\nReady to submit!")


if __name__ == "__main__":
    generate_predictions()
