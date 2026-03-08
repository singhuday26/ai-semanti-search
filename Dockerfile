FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY main.py .

EXPOSE 8000

# wget is available in python:3.11-slim; curl is not.
HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget -q -O- http://localhost:8000/health | grep ok || exit 1

# Justification for CMD:
# - uvicorn is run directly because gunicorn is not needed for a lightweight dedicated container if managed externally
# - workers=1 is used because the service relies on a stateful semantic cache and sharing state across workers without Redis is complex
CMD ["uvicorn","src.api:app","--host","0.0.0.0","--port","8000","--workers","1"]
