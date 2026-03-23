FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set your Gemini API key directly here
ENV GEMINI_API_KEY=REMOVED_API_KEY
ENV CHROMA_DB_PATH=/app/data/chroma_db
ENV ASSESSMENTS_PATH=/app/data/assessments.json
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
