"""
Unit Tests
-----------
Tests for API endpoints and core logic.

Run: pytest tests/ -v
"""

import pytest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Test evaluation metrics ──

def test_recall_at_k_perfect():
    """Perfect recall: all ground truth items are predicted."""
    from evaluation.evaluate import recall_at_k
    predicted = [
        "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/sql-server-new/",
    ]
    ground_truth = [
        "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/sql-server-new/",
    ]
    score = recall_at_k(predicted, ground_truth, k=10)
    assert score == 1.0


def test_recall_at_k_zero():
    """Zero recall: no predicted items are in ground truth."""
    from evaluation.evaluate import recall_at_k
    predicted = ["https://www.shl.com/solutions/products/product-catalog/view/java-8-new/"]
    ground_truth = ["https://www.shl.com/solutions/products/product-catalog/view/python-new/"]
    score = recall_at_k(predicted, ground_truth, k=10)
    assert score == 0.0


def test_recall_at_k_partial():
    """Partial recall: 1 of 2 ground truth items predicted."""
    from evaluation.evaluate import recall_at_k
    predicted = [
        "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/tableau-new/",
    ]
    ground_truth = [
        "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/sql-server-new/",
    ]
    score = recall_at_k(predicted, ground_truth, k=10)
    assert score == 0.5


def test_url_normalization():
    """URLs with and without /solutions/ should match."""
    from evaluation.evaluate import normalize_url
    url1 = "https://www.shl.com/solutions/products/product-catalog/view/python-new/"
    url2 = "https://www.shl.com/products/product-catalog/view/python-new/"
    assert normalize_url(url1) == normalize_url(url2)


def test_url_normalization_trailing_slash():
    """URLs with and without trailing slashes should match."""
    from evaluation.evaluate import normalize_url
    url1 = "https://www.shl.com/solutions/products/product-catalog/view/python-new/"
    url2 = "https://www.shl.com/solutions/products/product-catalog/view/python-new"
    assert normalize_url(url1) == normalize_url(url2)


# ── Test API endpoints ──

@pytest.fixture
def client():
    """Create test client without loading the full recommender."""
    from fastapi.testclient import TestClient
    # Patch the recommender so tests don't need Gemini/ChromaDB
    import api.main as main_module

    class MockRecommender:
        def recommend(self, query, top_k=10):
            return [
                {
                    "name": "Python (New)",
                    "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
                    "description": "Multi-choice test for Python programming.",
                    "duration": 11,
                    "remote_support": "Yes",
                    "adaptive_support": "No",
                    "test_type": ["K"],
                }
            ]

        def format_response(self, assessments):
            return {"recommended_assessments": assessments}

    main_module.recommender = MockRecommender()
    return TestClient(main_module.app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_recommend_endpoint_basic(client):
    response = client.post(
        "/recommend",
        json={"query": "I need a Python developer"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "recommended_assessments" in data
    assert len(data["recommended_assessments"]) >= 1


def test_recommend_endpoint_empty_query(client):
    response = client.post("/recommend", json={"query": ""})
    assert response.status_code == 400


def test_recommend_response_fields(client):
    """Each assessment must have all required fields."""
    response = client.post(
        "/recommend",
        json={"query": "hiring a data analyst"},
    )
    data = response.json()
    assessment = data["recommended_assessments"][0]

    required_fields = ["url", "name", "adaptive_support", "description",
                       "duration", "remote_support", "test_type"]
    for field in required_fields:
        assert field in assessment, f"Missing field: {field}"


def test_recommend_max_10(client):
    """API should return at most 10 assessments."""
    response = client.post(
        "/recommend",
        json={"query": "hiring for any role"},
    )
    data = response.json()
    assert len(data["recommended_assessments"]) <= 10


# ── Test document text builder ──

def test_build_document_text():
    """Document text should include all key fields for embedding quality."""
    from embeddings.build_index import build_document_text
    assessment = {
        "name": "Python (New)",
        "description": "Tests Python programming knowledge.",
        "duration": 11,
        "remote_support": "Yes",
        "adaptive_support": "No",
        "test_type": ["K"],
    }
    text = build_document_text(assessment)
    assert "Python" in text
    assert "Knowledge" in text   # K expanded to "Knowledge and Skills"
    assert "11" in text
    assert "Remote" in text
