# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs (zbar for QR, build tools for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# If you have a requirements.txt keep this step early for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Optional: run as non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Default envs (can be overridden by docker-compose or env)
# PROMPT files live under data/ by default per your code
ENV PROMPT_PATH="data/prompt.txt" \
    QUICK_PROMPT_PATH="data/quick_prompt.txt" \
    VENUE_PROFILES_PATH="data/venue_profiles.json"

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
