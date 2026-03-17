"""
run.py — Full Pipeline Orchestrator
-------------------------------------
Runs all steps in order with clear progress output.
Run this after setting GEMINI_API_KEY in your .env file.

Usage:
  python run.py                 # Run all steps
  python run.py --skip-scrape   # Skip scraping (if assessments.json already exists)
  python run.py --eval-only     # Only run evaluation on train set
  python run.py --predict-only  # Only generate test predictions CSV
"""

import argparse
import os
import sys
import subprocess
import json

ROOT = os.path.dirname(__file__)


def header(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def check_env():
    """Verify .env and GEMINI_API_KEY are set."""
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("❌  GEMINI_API_KEY not found.")
        print("    1. Copy .env.example to .env")
        print("    2. Add your Gemini API key (free at https://ai.google.dev)")
        sys.exit(1)
    print(f"✅  GEMINI_API_KEY found ({key[:8]}...)")


def check_assessments():
    """Check if assessments.json exists and has enough entries."""
    path = os.path.join(ROOT, "data", "assessments.json")
    if not os.path.exists(path):
        return False, 0
    with open(path) as f:
        data = json.load(f)
    return True, len(data)


def check_chromadb():
    """Check if ChromaDB index exists."""
    path = os.path.join(ROOT, "data", "chroma_db")
    return os.path.exists(path)


def run_step(title, script_path, *args):
    """Run a Python script as a subprocess with live output."""
    header(title)
    cmd = [sys.executable, script_path] + list(args)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n❌  Step failed: {title}")
        print(f"    Script: {script_path}")
        sys.exit(1)
    print(f"\n✅  {title} completed")


def main():
    parser = argparse.ArgumentParser(description="SHL Recommender Pipeline")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip scraping if assessments.json already exists")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on train set")
    parser.add_argument("--predict-only", action="store_true",
                        help="Only generate test set predictions CSV")
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("  SHL Assessment Recommendation System — Pipeline Runner")
    print("█" * 60)

    # ── Pre-flight checks ──
    header("Pre-flight Checks")
    check_env()

    exists, count = check_assessments()
    if exists:
        print(f"✅  assessments.json found ({count} assessments)")
    else:
        print("ℹ️   assessments.json not found — will scrape")

    if check_chromadb():
        print("✅  ChromaDB index found")
    else:
        print("ℹ️   ChromaDB not found — will build")

    dataset_path = os.path.join(ROOT, "data", "Gen_AI_Dataset.xlsx")
    if os.path.exists(dataset_path):
        print("✅  Gen_AI_Dataset.xlsx found")
    else:
        print("❌  Gen_AI_Dataset.xlsx not found in data/")
        print("    Place your dataset at: data/Gen_AI_Dataset.xlsx")
        sys.exit(1)

    # ── Eval/predict only shortcuts ──
    if args.eval_only:
        run_step(
            "Step 5 · Evaluate on Train Set (Recall@10)",
            os.path.join(ROOT, "evaluation", "evaluate.py"),
        )
        return

    if args.predict_only:
        run_step(
            "Step 6 · Generate Test Set Predictions",
            os.path.join(ROOT, "evaluation", "generate_predictions.py"),
        )
        print(f"\n📄  Predictions saved to: data/test_predictions.csv")
        return

    # ── Full pipeline ──

    # Step 1: Scrape
    if args.skip_scrape and exists and count >= 377:
        header("Step 1 · Scraper")
        print(f"⏭️   Skipping — {count} assessments already in assessments.json")
    else:
        if exists and count < 377:
            print(f"⚠️   Only {count} assessments found (need 377+). Re-scraping...")
        run_step(
            "Step 1 · Scrape SHL Catalog",
            os.path.join(ROOT, "scraper", "scrape_catalog.py"),
        )

    # Step 2: Parse dataset
    run_step(
        "Step 1b · Parse Dataset (Excel → JSON)",
        os.path.join(ROOT, "data", "parse_dataset.py"),
    )

    # Step 3: Build embeddings
    if args.skip_scrape and check_chromadb():
        header("Step 2 · Build Embeddings")
        print("⏭️   Skipping — ChromaDB index already exists")
    else:
        run_step(
            "Step 2 · Build Vector Embeddings Index",
            os.path.join(ROOT, "embeddings", "build_index.py"),
        )

    # Step 4: Evaluate on train set
    run_step(
        "Step 5 · Evaluate on Train Set (Recall@10)",
        os.path.join(ROOT, "evaluation", "evaluate.py"),
    )

    # Step 5: Generate test predictions
    run_step(
        "Step 6 · Generate Test Set Predictions CSV",
        os.path.join(ROOT, "evaluation", "generate_predictions.py"),
    )

    # ── Final summary ──
    header("Pipeline Complete!")
    print("📊  Results:")
    results_path = os.path.join(ROOT, "data", "train_evaluation_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        score = results.get("mean_recall_at_10", 0)
        print(f"    Mean Recall@10 on train set: {score:.4f} ({score*100:.1f}%)")

    pred_path = os.path.join(ROOT, "data", "test_predictions.csv")
    if os.path.exists(pred_path):
        import csv
        with open(pred_path) as f:
            rows = list(csv.reader(f))
        print(f"    Test predictions CSV: {len(rows)-1} rows")
        print(f"    Path: {pred_path}")

    print("\n📦  Submission checklist:")
    print("    ☐  Deploy API → get /health + /recommend URL")
    print("    ☐  Deploy frontend → get web app URL")
    print("    ☐  Push code to GitHub → get repo URL")
    print("    ☐  Submit: 3 URLs + approach doc + test_predictions.csv")
    print("\n🚀  To start the API server:")
    print("    uvicorn api.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
