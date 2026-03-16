# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs (certs for TLS, zbar optional if you use QR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install opencv-python-headless
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Optional: run as non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV PROMPT_PATH="data/prompt.txt" \
    QUICK_PROMPT_PATH="data/quick_prompt.txt" \
    VENUE_PROFILES_PATH="data/venue_profiles.json"

CMD ["sh", "-c", "uvicorn app.main_gemini:app --host 0.0.0.0 --port ${PORT:-8000}"]
