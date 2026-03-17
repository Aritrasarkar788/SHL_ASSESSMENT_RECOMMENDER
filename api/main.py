from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os

TEST_TYPE_FULL_NAMES = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.recommender import get_recommender

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends relevant SHL assessments based on job descriptions or natural language queries.",
    version="1.0.0",
)

# Allow all origins for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I am hiring for Java developers who can also collaborate effectively with my business teams."
            }
        }


# ── Load recommender on startup ──
recommender = None

@app.on_event("startup")
async def startup_event():
    global recommender
    recommender = get_recommender()



@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}



@app.post("/recommend")
def recommend(request: QueryRequest):
    """
    Accepts a job description or natural language query.
    Returns 5–10 most relevant SHL Individual Test Solutions.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = recommender.recommend(request.query.strip(), top_k=10)

        # Ensure minimum 1, maximum 10
        results = results[:10]
        if not results:
            raise HTTPException(
                status_code=404, detail="No assessments found for this query"
            )

        return recommender.format_response(results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
