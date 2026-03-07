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
    
    # 3. Save scaler to disk for inference transform
    os.makedirs("data", exist_ok=True)
    with open("data/umap_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
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
    k_candidates: list = [8, 10, 12, 15, 18, 20, 25],
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

    Elbow detection: iteration halts when relative BIC improvement drops
    below 2.0%, since gains beyond that threshold are not statistically
    meaningful relative to the penalty for adding more components.

    Returns:
        (best_k, {K: bic_score}) — best_k is the last K where relative
        improvement was still >= 2%.
    """
    print(f"\nEvaluating GMM cluster counts (K) via BIC...")
    print(f"{'K':<4} | {'BIC Score':<14} | Rel. Impr %")
    print("-" * 38)

    bic_scores = {}
    best_k = k_candidates[0]

    for i, K in enumerate(k_candidates):
        gmm = GaussianMixture(
            n_components=K,
            covariance_type='diag',
            n_init=3,
            max_iter=200,
            random_state=random_state
        )
        gmm.fit(reduced)
        score = gmm.bic(reduced)
        bic_scores[K] = score

        if i > 0:
            prev_K = k_candidates[i - 1]
            prev_score = bic_scores[prev_K]
            # Lower BIC is better; compute relative improvement
            improvement = ((prev_score - score) / abs(prev_score)) * 100
            rel_imp_str = f"{improvement:.2f}%"
            print(f"{K:<4} | {score:<14.1f} | {rel_imp_str}")

            if improvement < 2.0:
                # Elbow reached — improvement is no longer meaningful
                break
        else:
            print(f"{K:<4} | {score:<14.1f} | -")

        best_k = K

    print("-" * 38)

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


def run_clustering_pipeline(embeddings: np.ndarray):
    """
    Mock implementation for build_index script testing.
    This will be fully implemented in Prompt 3.2.
    """
    n_docs = len(embeddings)
    k = 5
    
    # Mock labels and distributions
    labels = np.random.randint(0, k, size=n_docs)
    
    # Generate mock probabilities
    probs = np.random.rand(n_docs, k)
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    # Generate mock entropies
    entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1) / np.log(k)
    
    # Enforce some high-uncertainty
    for i in range(10):
        if i < n_docs:
            entropies[i] = 0.8  # forced high entropy
            probs[i] = [0.4, 0.4, 0.1, 0.05, 0.05]
            
    return {"n_components": k}, {"labels": labels, "probs": probs, "entropies": entropies}
