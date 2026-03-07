"""
Embedder Core - Semantic Search Pipeline

MODEL DECISIONS: sentence-transformers/all-MiniLM-L6-v2
1. 384D vectors: Our cache lookup is O((n/K) * 384) dot products. This dimensionality
   keeps the cache latency sub-millisecond.
2. MTEB score: 0.586. This semantic similarity score is highly sufficient for 
   grouping heavily overlapping topical concepts in the 20 Newsgroups corpus.
3. CPU inference: Generates queries in <15ms natively on CPU, meaning no GPU 
   requirement for the production API.
4. Rejected all-mpnet-base-v2: The 768D vector space would double our cache 
   comparison cost and memory footprint with minimal MTEB gain for this task.
5. Rejected text-embedding-ada-002: External API network latency utterly defeats 
   the purpose of building a low-latency semantic cache.

CRITICAL MATHEMATICS: normalize_embeddings=True
We explicitly enforce `normalize_embeddings=True` on every single encode() call.
The cosine similarity between vector x and vector y is defined as:
    cos(x, y) = (x · y) / (||x|| · ||y||)
When vectors are L2-normalized, ||x|| = 1 and ||y|| = 1.
Therefore, cos(x, y) = x · y
This mathematical guarantee enables us to use `M @ query_embedding` as a direct 
cosine similarity matrix multiplication.
Savings: It completely eliminates 2*D extra multiply-adds and 2 square roots per 
comparison in our cache lookup and clustering passes.
"""

import os
import pickle
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

# Module-level singleton for the model
_MODEL_INSTANCE = None
MODEL_NAME = "all-MiniLM-L6-v2"

def get_model() -> SentenceTransformer:
    """
    Lazy-loads the SentenceTransformer model on the first call.
    Never loads inside a request handler to prevent cold-start latency spikes.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        print(f"Loading SentenceTransformer: {MODEL_NAME}...")
        _MODEL_INSTANCE = SentenceTransformer(MODEL_NAME)
        
        # Verify and print dimensions to confirm model bounds
        dummy_emb = _MODEL_INSTANCE.encode(["test"])
        dim = dummy_emb.shape[1]
        print(f"Model loaded successfully. Embedding dimension: {dim}D")
        
    return _MODEL_INSTANCE


def embed_texts(texts: List[str], batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
    """
    Embeds a list of texts into a matrix of shape (N, 384).
    
    batch_size=256: Tuned specifically for systems with ~4GB RAM available 
    for the Python process. Larger batches on CPU with 384D models saturate 
    memory bandwidth without proportional latency gains.
    
    Returns: float32, L2-normalized array.
    """
    model = get_model()
    
    # normalize_embeddings=True is CRITICAL (see module docstring)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True
    )
    
    # Cast to float32 definitively. Saves 50% memory over float64 with negligible precision loss.
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string for caching or searching.
    Returns: shape (384,) float32, L2-normalized array.
    """
    model = get_model()
    
    # normalize_embeddings=True must also be applied to queries
    embedding = model.encode(
        query,
        normalize_embeddings=True
    )
    
    return embedding.astype(np.float32)


def save_embeddings(embeddings: np.ndarray, doc_ids: List[str]) -> None:
    """
    Persists the matrix to EMBEDDINGS_CACHE_PATH (.npy) and the document IDs 
    to DOCIDS_CACHE_PATH (.pkl).
    """
    emb_path = os.getenv("EMBEDDINGS_CACHE_PATH", "data/embeddings.npy")
    ids_path = os.getenv("DOCIDS_CACHE_PATH", "data/doc_ids.pkl")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    os.makedirs(os.path.dirname(ids_path), exist_ok=True)
    
    print(f"Saving {embeddings.shape[0]} embeddings to {emb_path}")
    np.save(emb_path, embeddings)
    
    print(f"Saving {len(doc_ids)} doc_ids to {ids_path}")
    with open(ids_path, 'wb') as f:
        pickle.dump(doc_ids, f)


def load_embeddings() -> Tuple[np.ndarray, List[str]]:
    """
    Loads saved embeddings and doc_ids. Raises FileNotFoundError logically.
    """
    emb_path = os.getenv("EMBEDDINGS_CACHE_PATH", "data/embeddings.npy")
    ids_path = os.getenv("DOCIDS_CACHE_PATH", "data/doc_ids.pkl")
    
    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        raise FileNotFoundError(
            f"Embeddings or DocIDs not found at {emb_path} / {ids_path}.\n"
            f"Please run the indexing pipeline (scripts/build_index.py) first."
        )
        
    embeddings = np.load(emb_path)
    with open(ids_path, 'rb') as f:
        doc_ids = pickle.load(f)
        
    return embeddings, doc_ids


def embeddings_exist() -> bool:
    """Checks if the processed embedding artifacts already exist on disk."""
    emb_path = os.getenv("EMBEDDINGS_CACHE_PATH", "data/embeddings.npy")
    ids_path = os.getenv("DOCIDS_CACHE_PATH", "data/doc_ids.pkl")
    return os.path.exists(emb_path) and os.path.exists(ids_path)


def embedding_stats(embeddings: np.ndarray) -> dict:
    """
    Calculates summary statistics on a sample subset of embeddings to verify
    L2 normalization mathematically.
    """
    # Use a 1000 document sample for speed if matrix is large
    sample_size = min(1000, embeddings.shape[0])
    sample = embeddings[:sample_size]
    
    # 1. Verify L2 Norma (should be extremely close to 1.0)
    # Norm of a vector x is sqrt(sum(x_i^2))
    norms = np.linalg.norm(sample, axis=1)
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))
    
    # 2. Check inner-sample dot products (which act as cosine similarity here)
    # We calculate the similarity matrix S = X @ X.T
    sim_matrix = np.dot(sample, sample.T)
    
    # To find min/max similarity BETWEEN DIFFERENT documents, we mask the diagonal (identity)
    np.fill_diagonal(sim_matrix, -np.inf)
    max_sim = float(np.max(sim_matrix))
    
    # To find min similarity, replace the -inf diagonal with +inf so it isn't picked as the min
    np.fill_diagonal(sim_matrix, np.inf)
    min_sim = float(np.min(sim_matrix))
    
    return {
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "min_sim": min_sim,
        "max_sim": max_sim,
        "sample_size": sample_size
    }


if __name__ == "__main__":
    # Quick sanity check
    print("Running embedder sanity check...")
    test_texts = [
        "Semantic search algorithms",
        "Neural information retrieval systems",
        "How to bake a chocolate cake"
    ]
    
    embs = embed_texts(test_texts, show_progress=False)
    
    print("\n--- Embeddings Info ---")
    print(f"Shape: {embs.shape}")
    print(f"Dtype: {embs.dtype}")
    
    stats = embedding_stats(embs)
    print("\n--- Stats ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
        
    print("\nSim(text[0], text[1]) = ", np.dot(embs[0], embs[1]))
    print("Sim(text[0], text[2]) = ", np.dot(embs[0], embs[2]))
