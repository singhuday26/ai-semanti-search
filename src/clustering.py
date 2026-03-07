import numpy as np

def run_clustering_pipeline(embeddings: np.ndarray):
    """
    Mock implementation for build_index script testing.
    This will be fully implemented in Prompt 2.4.
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
