FROM python:3.11-slim

# System deps only (no build tools needed — use binary wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer caching — only reinstalls 
# when requirements.txt changes)
COPY requirements.txt .

# Install CPU-only torch first (200MB vs 2GB GPU version)
# then remaining deps — all binary wheels, no compilation
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    --only-binary :all: \
    -r requirements.txt

# Copy source (this layer changes often — keep it last)
COPY src/ ./src/
COPY main.py .
COPY .env.example .env

# Data directory (real data mounted as volume at runtime)
RUN mkdir -p data

EXPOSE 8000

# --workers 1 is REQUIRED: SemanticCache is an in-memory 
# data structure. Multiple workers = multiple independent 
# caches that don't share state. Single worker ensures 
# cache coherence.
CMD ["uvicorn", "src.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD wget -q -O- http://localhost:8000/health | grep -q ok \
    || exit 1
