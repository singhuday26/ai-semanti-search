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
"""

import os
import time
import pickle
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler


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
