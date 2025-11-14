# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Add essentials + libgomp (needed by torch CPU wheel); then clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# UPGRADE pip/setuptools/wheel so prebuilt wheels are used (no source builds)
RUN python -m pip install --upgrade pip setuptools wheel

# Install deps
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Cache model weights at build time so first request isn't slow
ENV HF_HOME=/var/tmp/hf-cache
ENV SENTENCE_TRANSFORMERS_HOME=/var/tmp/hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('Models cached')"


# Copy source
COPY app ./app

ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
