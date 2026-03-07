"""
Clustering Pipeline - Semantic Search Pipeline

Why UMAP before GMM: 
In 384D, all pairwise distances concentrate around sqrt(2d/3), making 
GMM covariance matrices near-singular. UMAP reduces dimensionality to 50D 
while preserving local manifold structure via a fuzzy topological graph.

Why NOT PCA: 
PCA is purely linear and distorts non-linear cluster topologies present 
in complex semantic embeddings.

Why NOT 2D: 
2D UMAP aggressively collapses global structure (useful for visualization, 
but destroys inter-cluster relationships needed for density estimation).

UMAP HYPERPARAMETERS:
- n_components=50 : Optimal for clustering; preserves structure while reducing GMM noise and curse of dimensionality.
- n_neighbors=15  : Balances local/global structure; 5 is too local (fragments clusters), 50 is too global (blurs boundaries).
- min_dist=0.1    : Encourages tight packing within clusters; 0.5+ reduces separation between distinct topics.
- metric='cosine' : Consistent with our L2-normalized embedding computation from MiniLM.
- random_state=42 : Reproducibility CRITICAL — ensures cluster IDs match saved ChromaDB metadata exactly across API restarts.
EXPECTED RESULT FOR 20 NEWSGROUPS:
K = 12-15. The 20 gold labels over-specify semantic structure. 
(e.g. talk.politics.guns + talk.politics.misc + talk.politics.mideast merge into 1-2 clusters due to overlapping vocabulary).

BIC FORMULA & K-SELECTION:
BIC = -2 * log_likelihood + k * log(N)
For GMM with DIAGONAL covariance, D dims, K components:
  k = K(2D + 1) - 1
At D=50, K=12: k = 12(2*50 + 1) - 1 = 1211 free parameters.
Why diagonal: full covariance at D=50 -> k = 15911 params (13x more), which causes overfitting and near-singular matrices.
BIC is consistent — at N=18000, evaluates to true model if in candidates.
"""

import os
import time
import pickle
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from typing import Dict, Tuple


def fit_umap(embeddings: np.ndarray, n_components: int = 50) -> tuple:
    """
    Fits a UMAP reducer on the provided embeddings.
    
    1. Applies StandardScaler to homogeneise the gradient landscape and speed convergence.
    2. Fits UMAP on the scaled embeddings.
    3. Saves the scaler for inference time to robustly transform queries.
    
    Returns:
        (umap.UMAP, np.ndarray): The fitted UMAP model and the reduced embeddings matrix.
    """
    print(f"Fitting UMAP (Input shape: {embeddings.shape})...")
    t0 = time.time()
    
    # 1. Apply StandardScaler before UMAP
    # Speeds convergence even for L2-normalized embeddings by homogenizing the gradient landscape
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    # 2. Fit UMAP on scaled embeddings
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    
    reduced_embeddings = reducer.fit_transform(scaled_embeddings)
    
    # 3. Save scaler and reducer to disk for inference transform
    os.makedirs("data", exist_ok=True)
    with open("data/umap_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("data/umap_model.pkl", "wb") as f:
        pickle.dump(reducer, f)
        
    print(f"UMAP reduction complete. Output shape: {reduced_embeddings.shape} (Took {time.time() - t0:.2f}s)")
    
    return reducer, reduced_embeddings


def transform_umap(reducer: umap.UMAP, embeddings: np.ndarray) -> np.ndarray:
    """
    Transforms new single query embeddings (or batches) during inference time.
    IMPORTANT: The scaler is loaded from disk to ensure the exact same transform
    is applied before UMAP reduction.
    """
    scaler_path = "data/umap_scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing UMAP scaler at {scaler_path}. Cannot transform safely without it.")
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    # Scale exactly using the statistics from the fit corpus
    scaled_embeddings = scaler.transform(embeddings)
    
    # Reduce
    reduced_embeddings = reducer.transform(scaled_embeddings)
    
    return reduced_embeddings


def select_k_with_bic(
    reduced: np.ndarray,
    k_candidates: list[int] = [8, 10, 12, 15, 18, 20, 25],
    random_state: int = 42
) -> Tuple[int, Dict[int, float]]:
    """
    Fits a GaussianMixture for each K in k_candidates and selects the optimal K
    using the Bayesian Information Criterion (BIC) elbow method.

    BIC FORMULA PROOF:
      BIC = -2 * log_likelihood + k * log(N)
      For GMM with DIAGONAL covariance, D dims, K components:
        k = K(2D + 1) - 1
        At D=50, K=12: k = 12(2*50 + 1) - 1 = 1211 free parameters.
      Why diagonal covariance:
        Full covariance at D=50 requires k = K*(D*(D+1)/2 + D) - 1
        At K=12, D=50: k = 12*(1275 + 50) - 1 = 15911 params — 13x more.
        This causes overfitting and near-singular covariance matrices.
      BIC is 'consistent' — at N=18000, BIC converges to the true model
      if it exists among the candidates.

    EXPECTED RESULT FOR 20 NEWSGROUPS:
      K = 12-15. The 20 gold labels over-specify semantic structure.
      e.g. talk.politics.guns + talk.politics.misc + talk.politics.mideast
      merge into 1-2 clusters due to heavily overlapping vocabulary.

    Elbow detection: all candidates are evaluated; best_k is selected
    post-loop as the last K where relative improvement was still >= 2%.
    Evaluating all candidates avoids premature termination when BIC
    improvement is temporarily noisy across adjacent K values.

    Returns:
        (best_k, {K: bic_score}) — best_k is the last K where relative
        improvement was still >= 2%.
    """
    print(f"\nEvaluating GMM cluster counts (K) via BIC...")
    print(f"{'K':<4} | {'BIC Score':<14} | Rel. Impr %")
    print("-" * 38)

    bic_scores = {}

    for i, K in enumerate(k_candidates):
        gmm = GaussianMixture(
            n_components=K,
            covariance_type='diag',
            n_init=5,
            max_iter=200,
            random_state=random_state + i
        )
        gmm.fit(reduced)
        score = gmm.bic(reduced)
        bic_scores[K] = score

        if i > 0:
            prev_score = bic_scores[k_candidates[i - 1]]
            # Lower BIC is better; compute relative improvement
            improvement = ((prev_score - score) / abs(prev_score)) * 100
            print(f"{K:<4} | {score:<14.1f} | {improvement:.2f}%")
        else:
            print(f"{K:<4} | {score:<14.1f} | -")

    print("-" * 38)

    # Post-loop elbow detection: find the last K where improvement >= 2%
    best_k = k_candidates[0]
    for i in range(1, len(k_candidates)):
        prev_score = bic_scores[k_candidates[i - 1]]
        score = bic_scores[k_candidates[i]]
        improvement = ((prev_score - score) / abs(prev_score)) * 100
        if improvement < 2.0:
            break
        best_k = k_candidates[i]

    # Print the final formatted summary table
    print_bic_table(bic_scores)

    return best_k, bic_scores


def print_bic_table(bic_scores: Dict[int, float]) -> None:
    """
    Prints the final formatted BIC cluster selection table to stdout.
    Reconstructs best_k using the same elbow logic as select_k_with_bic
    and marks the selected K with an arrow.
    """
    k_list = sorted(bic_scores.keys())

    # Reconstruct best_k using identical elbow logic
    best_k = k_list[0]
    for i in range(1, len(k_list)):
        prev_score = bic_scores[k_list[i - 1]]
        score = bic_scores[k_list[i]]
        improvement = ((prev_score - score) / abs(prev_score)) * 100
        if improvement < 2.0:
            break
        best_k = k_list[i]

    print("\n--- BIC Cluster Selection Summary ---")
    print(f"{'K':<4} | {'BIC Score':<14} | Rel. Impr %")
    print("-" * 38)

    for i, K in enumerate(k_list):
        score = bic_scores[K]
        if i > 0:
            prev_score = bic_scores[k_list[i - 1]]
            imp = ((prev_score - score) / abs(prev_score)) * 100
            rel_imp = f"{imp:.2f}%"
        else:
            rel_imp = "-"
        print(f"{K:<4} | {score:<14.1f} | {rel_imp}")

    print("-" * 38)
    print(f"  --> K={best_k}  BIC={bic_scores[best_k]:.0f}  (selected)\n")


# ---------------------------------------------------------------------------
# Persistence paths — overridable via environment variables
# ---------------------------------------------------------------------------
CLUSTER_MODEL_PATH = os.getenv("CLUSTER_MODEL_PATH", "data/gmm_model.pkl")
UMAP_MODEL_PATH    = os.getenv("UMAP_MODEL_PATH",    "data/umap_model.pkl")
CLUSTER_LABELS_PATH = os.getenv("CLUSTER_LABELS_PATH", "data/cluster_labels.npy")
CLUSTER_PROBS_PATH  = os.getenv("CLUSTER_PROBS_PATH",  "data/cluster_probs.npy")


def fit_gmm(reduced: np.ndarray, n_components: int) -> GaussianMixture:
    """
    Fits a GaussianMixture model on the UMAP-reduced embeddings.

    GMM CONFIGURATION:
      covariance_type='diag'  — axis-aligned ellipsoids; 13x fewer free
                                parameters than 'full' at D=50 (1211 vs
                                15911), preventing overfitting and near-
                                singular covariance matrices.
      max_iter=300            — more EM iterations than the BIC scan
                                (200) to ensure the final model converges
                                fully on the chosen K.
      n_init=5                — 5 independent random initialisations; EM
                                keeps the run with the best log-likelihood,
                                reducing the risk of poor local optima.
      random_state=42         — reproducibility is critical: cluster IDs
                                must be identical across API restarts so
                                that ChromaDB metadata stays consistent.
    """
    print(f"Fitting GMM (K={n_components})...")
    t0 = time.time()

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='diag',  # axis-aligned ellipsoids; 13x fewer params than full
        max_iter=300,            # more iterations for final fit vs BIC scan
        n_init=5,                # 5 random inits, keep best log-likelihood; prevents bad local optima in EM
        random_state=42          # reproducibility — cluster IDs must be stable
    )
    gmm.fit(reduced)

    print(f"GMM fitted. Log-likelihood: {gmm.lower_bound_:.4f} (Took {time.time() - t0:.2f}s)")
    return gmm


def get_soft_assignments(
    gmm: GaussianMixture,
    reduced: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns fuzzy cluster memberships for every document.

    Returns:
        dominant_labels: shape (N,) int — argmax of posterior per document.
        probs:           shape (N, K) float32 — full GMM posterior matrix.
                         Each row is a probability distribution over K clusters.
    """
    # Full posterior P(cluster_k | doc_i) from the E-step
    probs = gmm.predict_proba(reduced).astype(np.float32)

    # Hard assignment = highest-probability cluster
    dominant_labels = np.argmax(probs, axis=1)

    return dominant_labels, probs


def membership_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Computes normalised Shannon entropy for each document's membership vector.

    H_i = -sum_k( p_ik * log(p_ik) ) / log(K)

    Normalised to [0, 1] by dividing by log(K) so the scale is independent
    of the number of clusters:
      H_i -> 0 : document belongs firmly to one cluster (low uncertainty).
      H_i -> 1 : probability mass spread uniformly across all clusters.

    Interpretation thresholds:
      H_i > 0.7 — boundary / multi-topic document; spans cluster edges.
      H_i < 0.3 — clearly within a single dominant topic cluster.
    """
    K = probs.shape[1]

    # Clip to avoid log(0); values are already probabilities so clip is safe
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))

    raw_entropy = -np.sum(probs * log_probs, axis=1)

    # Normalise by log(K) to get [0, 1] range
    normalised = raw_entropy / np.log(K)

    return normalised.astype(np.float32)


def assign_cluster(
    query_embedding: np.ndarray,
    gmm: GaussianMixture,
    umap_reducer
) -> Tuple[int, np.ndarray]:
    """
    Live inference path — called on every POST /query request.

    Transforms a single query embedding through the fitted UMAP reducer
    and returns its dominant cluster ID and full posterior vector.

    Args:
        query_embedding: shape (384,) L2-normalised float32 from embed_query().
        gmm:             The fitted GaussianMixture model.
        umap_reducer:    The fitted UMAP reducer (loaded from UMAP_MODEL_PATH).

    Returns:
        (dominant_k, probs) where dominant_k is int and probs is shape (K,).
    """
    # UMAP expects 2D input — reshape single vector to (1, D)
    embedding_2d = query_embedding.reshape(1, -1)

    # transform_umap loads the scaler from disk and applies it before reducing
    reduced = transform_umap(umap_reducer, embedding_2d)

    # Get full posterior for the single query point
    probs = gmm.predict_proba(reduced)[0].astype(np.float32)
    dominant_k = int(np.argmax(probs))

    return dominant_k, probs


def run_clustering_pipeline(
    embeddings: np.ndarray,
    force_refit: bool = False
) -> Tuple[np.ndarray, np.ndarray, GaussianMixture, object]:
    """
    Full clustering orchestrator.

    Steps:
      1. Check for cached models on disk; load and return early if valid
         and force_refit=False.
      2. fit_umap(embeddings, n_components=50) → (umap_reducer, reduced)
      3. select_k_with_bic(reduced) → best_k
      4. fit_gmm(reduced, best_k) → gmm
      5. get_soft_assignments(gmm, reduced) → (dominant_labels, probs)
      6. Persist: gmm → CLUSTER_MODEL_PATH, umap_reducer → UMAP_MODEL_PATH,
         dominant_labels → CLUSTER_LABELS_PATH, probs → CLUSTER_PROBS_PATH.
      7. Print summary: K, mean entropy, high-uncertainty document count.

    Returns:
        (dominant_labels, probs, gmm, umap_reducer)
    """
    os.makedirs("data", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Cache check — skip refit when all artefacts are present
    # ------------------------------------------------------------------
    all_cached = all(os.path.exists(p) for p in [
        CLUSTER_MODEL_PATH, UMAP_MODEL_PATH,
        CLUSTER_LABELS_PATH, CLUSTER_PROBS_PATH
    ])

    if all_cached and not force_refit:
        print("Cached clustering models found. Loading from disk...")
        with open(CLUSTER_MODEL_PATH, "rb") as f:
            gmm = pickle.load(f)
        with open(UMAP_MODEL_PATH, "rb") as f:
            umap_reducer = pickle.load(f)
        dominant_labels = np.load(CLUSTER_LABELS_PATH)
        probs = np.load(CLUSTER_PROBS_PATH)
        print(f"Loaded K={gmm.n_components} cluster model from cache.")
        return dominant_labels, probs, gmm, umap_reducer

    # ------------------------------------------------------------------
    # 2. UMAP dimensionality reduction
    # ------------------------------------------------------------------
    umap_reducer, reduced = fit_umap(embeddings, n_components=50)

    # ------------------------------------------------------------------
    # 3. BIC-based K selection
    # ------------------------------------------------------------------
    best_k, bic_scores = select_k_with_bic(reduced)

    # ------------------------------------------------------------------
    # 4. Final GMM fit on selected K
    # ------------------------------------------------------------------
    gmm = fit_gmm(reduced, best_k)

    # ------------------------------------------------------------------
    # 5. Soft assignments and entropy
    # ------------------------------------------------------------------
    dominant_labels, probs = get_soft_assignments(gmm, reduced)
    entropies = membership_entropy(probs)

    # ------------------------------------------------------------------
    # 6. Persist all artefacts
    # ------------------------------------------------------------------
    with open(CLUSTER_MODEL_PATH, "wb") as f:
        pickle.dump(gmm, f)
    with open(UMAP_MODEL_PATH, "wb") as f:
        pickle.dump(umap_reducer, f)
    np.save(CLUSTER_LABELS_PATH, dominant_labels)
    np.save(CLUSTER_PROBS_PATH, probs)

    print(f"Clustering artefacts saved to data/")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    high_uncertainty_count = int(np.sum(entropies > 0.7))
    print(f"\n--- Clustering Summary ---")
    print(f"Optimal K:              {best_k}")
    print(f"Mean membership entropy: {np.mean(entropies):.4f}")
    print(f"High-uncertainty docs (entropy > 0.7): "
          f"{high_uncertainty_count} ({high_uncertainty_count / len(embeddings) * 100:.1f}%)")

    return dominant_labels, probs, gmm, umap_reducer
