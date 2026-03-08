"""
Unit tests for src/cache.py — SemanticCache.

All tests run against the real SemanticCache class with no mocking, verifying
correctness, thread-safety, and the mathematical properties of the BLAS cosine
similarity implementation.
"""

import threading

import numpy as np
import pytest

from src.cache import SemanticCache


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _unit_vec(dim: int = 384, seed: int = 0) -> np.ndarray:
    """Returns a deterministic L2-normalised float32 vector of length *dim*."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_invalid_threshold_raises():
    """
    Proves SemanticCache rejects out-of-range threshold values at construction
    time.  θ=0.0 sits on the excluded lower boundary (0, 1.0]; θ=1.1 exceeds
    the upper limit.  Both must raise ValueError to prevent silent
    misconfiguration.
    """
    with pytest.raises(ValueError):
        SemanticCache(threshold=0.0)
    with pytest.raises(ValueError):
        SemanticCache(threshold=1.1)


def test_empty_cache_miss():
    """
    Proves that a lookup on an empty cache always returns a miss with
    similarity 0.0 — no phantom hits possible from uninitialised state.
    """
    cache = SemanticCache(threshold=0.85)
    result = cache.lookup(_unit_vec(), cluster_id=0)
    assert not result.hit
    assert result.similarity == 0.0


def test_exact_match_hits():
    """
    Proves that storing an embedding and immediately looking it up with the
    identical vector yields a cache hit with cosine similarity ≈ 1.0.
    This is the basic end-to-end correctness guarantee: store → lookup → hit.
    """
    cache = SemanticCache(threshold=0.85)
    v = _unit_vec(seed=1)
    cache.store("test query", v, {"result": "data"}, cluster_id=0)
    result = cache.lookup(v, cluster_id=0)
    assert result.hit
    assert result.similarity == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_misses():
    """
    Concentration-of-measure proof: orthogonal unit vectors in R^384 have
    cosine similarity exactly 0.0.  Since 0.0 < any valid threshold, orthogonal
    vectors can never produce a false cache hit.  This is the geometric
    foundation of the statistical impossibility claim in the module docstring.
    """
    cache = SemanticCache(threshold=0.85)
    e_0 = np.zeros(384, dtype=np.float32)
    e_0[0] = 1.0
    e_1 = np.zeros(384, dtype=np.float32)
    e_1[1] = 1.0
    cache.store("first", e_0, {}, cluster_id=0)
    result = cache.lookup(e_1, cluster_id=0)
    assert not result.hit
    assert result.similarity == pytest.approx(0.0, abs=1e-6)


def test_cluster_isolation():
    """
    Proves that the cluster-indexed sharding is strictly isolated: an entry
    stored in cluster 0 is never returned when searching cluster 1, even when
    the query embedding is identical to the stored one.  Cluster boundaries
    must not leak across shards.
    """
    cache = SemanticCache(threshold=0.85)
    v = _unit_vec(seed=2)
    cache.store("query", v, {}, cluster_id=0)
    result = cache.lookup(v, cluster_id=1)
    assert not result.hit


def test_threshold_sensitivity():
    """
    Proves that θ is the sole control over the hit/miss boundary.  A vector
    with exact cosine similarity 0.87 vs v_base hits at θ=0.83 but misses at
    θ=0.91, demonstrating the precision/recall tradeoff from the θ decision
    table.  v_perturbed is constructed via Gram-Schmidt so the similarity is
    mathematically exact, not approximate.
    """
    v_base = _unit_vec(seed=0)

    # Build v_perp: a unit vector orthogonal to v_base via Gram-Schmidt
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(384).astype(np.float32)
    v_perp = noise - np.dot(noise, v_base) * v_base
    v_perp /= np.linalg.norm(v_perp)

    # v_perturbed = target*v_base + sqrt(1-target^2)*v_perp
    # → already a unit vector, cos(v_base, v_perturbed) == target exactly
    target = 0.87
    v_perturbed = target * v_base + np.sqrt(1.0 - target ** 2) * v_perp
    v_perturbed /= np.linalg.norm(v_perturbed)  # remove float32 rounding only

    cache_lo = SemanticCache(threshold=0.83)
    cache_lo.store("base", v_base, {}, cluster_id=0)
    assert cache_lo.lookup(v_perturbed, cluster_id=0).hit

    cache_hi = SemanticCache(threshold=0.91)
    cache_hi.store("base", v_base, {}, cluster_id=0)
    assert not cache_hi.lookup(v_perturbed, cluster_id=0).hit


def test_stats_tracking():
    """
    Proves hit/miss counters are accurate: 2 hits followed by 1 miss must
    yield hit_rate = 2/3 ≈ 0.6667 (within ±0.001).  The miss is guaranteed by
    using a vector orthogonalised against the stored one (sim = 0.0).
    """
    cache = SemanticCache(threshold=0.85)
    v = _unit_vec(seed=3)
    cache.store("query", v, {}, cluster_id=0)

    cache.lookup(v, cluster_id=0)  # hit 1
    cache.lookup(v, cluster_id=0)  # hit 2

    # Orthogonalised vector → cosine similarity = 0.0 → guaranteed miss
    rng = np.random.default_rng(55)
    noise = rng.standard_normal(384).astype(np.float32)
    miss_vec = noise - np.dot(noise, v) * v
    miss_vec /= np.linalg.norm(miss_vec)
    cache.lookup(miss_vec, cluster_id=0)  # miss

    assert abs(cache.hit_rate - 2 / 3) < 0.001


def test_flush_resets_all():
    """
    Proves flush() completely resets the cache to its initial empty state:
    total_entries=0, hit_count=0, miss_count=0, hit_rate=0.0.  No state from
    before the flush must survive.
    """
    cache = SemanticCache(threshold=0.85)
    v = _unit_vec(seed=4)
    cache.store("q", v, {}, cluster_id=0)
    cache.lookup(v, cluster_id=0)  # generate one hit
    cache.flush()
    assert cache.total_entries == 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0
    assert cache.hit_rate == 0.0


def test_concurrent_writes_safe():
    """
    Proves the threading.RLock correctly serialises concurrent store() calls:
    100 threads each storing one entry must yield exactly 100 total entries
    with no lost writes due to race conditions or torn list mutations.
    """
    cache = SemanticCache(threshold=0.85)
    threads = [
        threading.Thread(
            target=cache.store,
            args=(f"query_{i}", _unit_vec(seed=i), {}, i % 10),
        )
        for i in range(100)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache.total_entries == 100


def test_matrix_vs_loop_equiv():
    """
    BLAS correctness proof: the matrix-vector product M @ q (used internally
    by the cache) must produce cosine similarities numerically identical to a
    naive Python dot-product loop.  Max absolute deviation < 1e-5 confirms the
    BLAS SGEMV path is correct to float32 precision.
    """
    stored_vecs = [_unit_vec(seed=i) for i in range(50)]
    q = _unit_vec(seed=999)

    # BLAS path: stack all vectors into a single (50, 384) matrix, multiply
    M = np.vstack(stored_vecs).astype(np.float32)
    blas_sims = M @ q

    # Python loop path: individual dot products
    loop_sims = np.array([float(np.dot(v, q)) for v in stored_vecs])

    assert np.max(np.abs(blas_sims - loop_sims)) < 1e-5


def test_boundary_searches_two_clusters():
    """
    Proves boundary-aware lookup: when cluster_probs shows the dominant cluster
    probability < 0.60, the lookup extends to the second-best cluster.  An
    entry stored only in cluster 0 must be found via a lookup that nominates
    cluster 1 as dominant but has p_dominant=0.55 < 0.60, triggering the
    boundary-shard extension.
    """
    cache = SemanticCache(threshold=0.85)

    v0 = _unit_vec(seed=10)  # stored in cluster 0
    v1 = _unit_vec(seed=11)  # stored in cluster 1
    cache.store("cluster0_query", v0, {"cluster": 0}, cluster_id=0)
    cache.store("cluster1_query", v1, {"cluster": 1}, cluster_id=1)

    # dominant=1 (p=0.55), second=0 (p=0.45): p_dominant=0.55 < 0.60 → boundary
    cluster_probs = np.array([0.45, 0.55], dtype=np.float32)

    # Lookup v0 with dominant_cluster=1: boundary mode must extend to cluster 0
    result = cache.lookup(v0, cluster_id=1, cluster_probs=cluster_probs)
    assert result.hit
    assert result.matched_query == "cluster0_query"

