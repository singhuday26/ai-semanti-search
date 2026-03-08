"""
API integration tests for src/api.py using FastAPI TestClient.

No real index is built — heavy ML dependencies are stubbed in conftest.py and
state is overridden in fixtures AFTER the lifespan has run so the override is
not clobbered by startup logic.

Fixture pattern:
  Both fixtures use `with TestClient(app) as client:` so the ASGI lifespan
  runs on __enter__, then state is overridden, then the test executes, then
  the lifespan teardown runs on __exit__.  This is the only reliably ordered
  way to set post-startup state under Starlette's lifespan contract.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app, state
from src.cache import SemanticCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client_degraded(monkeypatch):
    """
    TestClient in degraded mode (index not built).

    get_model() is replaced with a no-op so the lifespan does not attempt to
    load sentence-transformers.  embeddings_exist() is forced False so the
    lifespan never tries to pickle.load() cluster models through the mocked
    sklearn — making the startup path deterministic regardless of what data
    files exist on disk.  All overrides use monkeypatch so teardown reverts
    state cleanly for the next test.
    """
    monkeypatch.setattr("src.api.get_model", lambda: None)
    monkeypatch.setattr("src.api.embeddings_exist", lambda: False)
    with TestClient(app) as client:
        monkeypatch.setattr(state, "ready", False)
        yield client


@pytest.fixture
def client_ready(monkeypatch):
    """
    TestClient with a fully initialised (but empty) SemanticCache.

    get_model() is silenced; embeddings_exist() is forced False to prevent
    pickle-loading real model files through mocked sklearn.  collection_size()
    is stubbed to return 0 so the /health endpoint (which calls it when
    ready=True) can JSON-serialise the response instead of hitting the
    MagicMock chromadb stub.  state.ready and state.cache are overridden
    after the lifespan completes so request handlers see a ready service.
    """
    monkeypatch.setattr("src.api.get_model", lambda: None)
    monkeypatch.setattr("src.api.embeddings_exist", lambda: False)
    monkeypatch.setattr("src.api.collection_size", lambda: 0)
    with TestClient(app) as client:
        monkeypatch.setattr(state, "ready", True)
        monkeypatch.setattr(state, "cache", SemanticCache(threshold=0.85))
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_degraded(client_degraded):
    """
    Proves GET /health always returns HTTP 200 even when the index has not
    been built.  The 'status' field must be 'degraded' so callers can
    distinguish an uninitialised service from a healthy one without relying
    on the HTTP status code (required by the Docker HEALTHCHECK contract).
    """
    resp = client_degraded.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["index_built"] is False


def test_query_503_when_not_ready(client_degraded):
    """
    Proves POST /query returns 503 Service Unavailable when the index has not
    been built.  The require_ready() guard must fire before any embedding or
    cluster work is attempted, preventing a silent failure deep in the pipeline.
    """
    resp = client_degraded.post("/query", json={"query": "What is spaceflight?"})
    assert resp.status_code == 503


def test_cache_stats_empty(client_ready):
    """
    Proves GET /cache/stats returns 200 with a zeroed-out stats snapshot
    immediately after initialisation, before any queries have been processed.
    Confirms the endpoint correctly exposes the empty-cache baseline state.
    """
    resp = client_ready.get("/cache/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_entries"] == 0
    assert body["hit_rate"] == 0.0


def test_flush_empty_cache(client_ready):
    """
    Proves DELETE /cache succeeds (200, status='ok') even when the cache is
    already empty.  Flushing an empty cache must be idempotent — no error,
    no state corruption.
    """
    resp = client_ready.delete("/cache")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_query_schema_validation(client_ready):
    """
    Proves the QueryRequest schema rejects an empty string.  min_length=3 on
    the 'query' field must cause FastAPI/Pydantic to return 422 Unprocessable
    Entity before the handler is invoked, enforcing the input contract at the
    boundary.
    """
    resp = client_ready.post("/query", json={"query": ""})
    assert resp.status_code == 422


def test_n_results_range(client_ready):
    """
    Proves n_results is validated to the range [1, 10].  Both boundary
    violations (n_results=0 and n_results=11) must return 422, confirming
    the ge=1 / le=10 constraints are enforced before handler execution.
    """
    resp_low = client_ready.post("/query", json={"query": "space exploration", "n_results": 0})
    assert resp_low.status_code == 422

    resp_high = client_ready.post("/query", json={"query": "space exploration", "n_results": 11})
    assert resp_high.status_code == 422
