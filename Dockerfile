FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CHROMA_DB_PATH=/app/data/chroma_db
ENV ASSESSMENTS_PATH=/app/data/assessments.json
ENV PYTHONPATH=/app

EXPOSE 7860

# Build index first then start server
CMD ["sh", "-c", "python embeddings/build_index.py && uvicorn api.main:app --host 0.0.0.0 --port 7860"]
