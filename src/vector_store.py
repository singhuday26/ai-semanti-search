"""
Vector Store Integration - Semantic Search Pipeline

STORE CHOICE JUSTIFICATION:
1. ChromaDB: Selected because it provides an embedded, persistent vector database 
   with metadata where-filters built-in. Critically, it runs entirely in-process 
   with zero external server requirements, which satisfies our constraint for a 
   single-command `uvicorn` startup.
2. Rejected FAISS: While fast for pure vector search, FAISS lacks built-in metadata 
   storage or pre-filtering capabilities necessary for our cluster-based searches.
3. Rejected Qdrant/Pinecone: These require either a separate local server process 
   or network calls to a SaaS API, breaking the zero-configuration and local-first 
   requirements.

COLLECTION CONFIGURATION:
- Name: 'newsgroups'
- Distance Metric: 'cosine' (essential as our embeddings are L2-normalized)
- Storage: Persistent local directory from CHROMA_PERSIST_PATH environment variable.

METADATA SCHEMA:
Each document must include the following metadata fields:
- original_label (int): 0-19 gold label — for confusion heatmap analysis
- label_name (str): e.g. 'sci.space' — human-readable category
- dominant_cluster_id (int): argmax of GMM posterior — used as an O(1) cache shard key
- cluster_entropy (float): normalized Shannon entropy [0,1] — uncertainty measure
- is_high_uncertainty (bool): entropy > 0.7 — enables fast boundary search queries
- second_cluster_id (int): 2nd highest GMM component
- second_cluster_prob (float): P(cluster_2|doc) — quantifies ambiguity between topics
"""

import os
import chromadb
from chromadb.api.models.Collection import Collection
import numpy as np

_CHROMA_CLIENT = None
_COLLECTION = None

def get_collection() -> Collection:
    """
    Module-level singleton for the ChromaDB client and collection.
    Uses the get_or_create pattern to ensure persistence and configuration.
    """
    global _CHROMA_CLIENT, _COLLECTION
    
    if _CHROMA_CLIENT is None:
        persist_path = os.getenv("CHROMA_PERSIST_PATH", "data/chroma_db")
        os.makedirs(persist_path, exist_ok=True)
        
        # Initialize the persistent client
        _CHROMA_CLIENT = chromadb.PersistentClient(path=persist_path)
        
    if _COLLECTION is None:
        # Get or create the newsgroups collection using cosine distance
        _COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
            name="newsgroups",
            metadata={"hnsw:space": "cosine"}
        )
        
    return _COLLECTION


def index_documents(doc_ids: list, texts: list, embeddings: list, metadatas: list, batch_size: int = 512):
    """
    Inserts documents into ChromaDB in batches.
    batch_size=512: Tuned because single-document inserts are ~100x slower due to overhead.
    
    Skips insertion if the collection already contains all documents.
    """
    collection = get_collection()
    
    current_count = collection.count()
    if current_count == len(doc_ids):
        print("Collection already fully indexed. Skipping.")
        return

    total_docs = len(doc_ids)
    
    # We batch insertions as single-doc inserts into Chroma are very inefficient
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        
        batch_ids = doc_ids[i:end_idx]
        batch_texts = texts[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]
        
        # Validate embedding dimensions before insert
        if np.shape(batch_embeddings)[-1] != 384:
            raise ValueError("Embedding dimension mismatch. Expected 384.")

        # Convert embeddings directly to list (avoids Python iteration overhead)
        batch_embeddings_lists = batch_embeddings.tolist() if hasattr(batch_embeddings, 'tolist') else batch_embeddings
        
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings_lists,
            documents=batch_texts,
            metadatas=batch_metadatas
        )
        
        if (i + batch_size) % 2048 == 0 or end_idx == total_docs:
            print(f"Indexed {end_idx}/{total_docs} documents.")


def query_similar(query_embedding, n_results: int = 5, cluster_filter: int = None) -> dict:
    """
    Queries ChromaDB for the most similar documents to the provided embedding.
    Optionally applies a pre-filter by dominant_cluster_id.
    """
    collection = get_collection()
    
    query_emb_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
    
    where_clause = None
    if cluster_filter is not None:
        where_clause = {'dominant_cluster_id': {'$eq': int(cluster_filter)}}
        
    results = collection.query(
        query_embeddings=[query_emb_list],
        n_results=n_results,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results.get("ids", []) or not results["ids"][0]:
        return {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "similarities": []
        }
    
    # Calculate cosine similarity from cosine distance
    distances = results["distances"][0]
    similarities = [max(0.0, min(1.0, 1 - d)) for d in distances]
    
    # Flatten structure
    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": distances,
        "similarities": similarities
    }


def collection_size() -> int:
    """Returns the total number of documents currently in the collection."""
    return get_collection().count()


def query_boundary_docs(min_entropy: float = 0.7, n: int = 20) -> list:
    """
    Queries ChromaDB for documents situated near decision boundaries.
    Uses the exact match where={'is_high_uncertainty': True} filter.
    
    Returns the top-n boundary documents, sorted descending by cluster_entropy.
    Used for the /clusters/boundary API endpoint and notebook analysis.
    """
    collection = get_collection()
    
    # We use get() instead of query() since we are filtering purely by metadata, 
    # not searching by vector distance.
    results = collection.get(
        where={'cluster_entropy': {'$gte': min_entropy}},
        limit=200,
        include=["documents", "metadatas"]
    )
    
    if not results or not results['ids']:
        return []
        
    docs = []
    for i in range(len(results['ids'])):
        doc_info = {
            'id': results['ids'][i],
            'text': results['documents'][i],
            'metadata': results['metadatas'][i]
        }
        docs.append(doc_info)
        
    # Sort descending by cluster_entropy
    docs.sort(key=lambda x: x['metadata'].get('cluster_entropy', 0.0), reverse=True)
    
    return docs[:n]


def reset_collection() -> None:
    """
    Deletes the current collection entirely.
    Helpful during development for index rebuilds.
    """
    global _CHROMA_CLIENT, _COLLECTION
    if _CHROMA_CLIENT:
        try:
            _CHROMA_CLIENT.delete_collection("newsgroups")
            _COLLECTION = None
            print("Collection 'newsgroups' reset successfully.")
        except ValueError:
            pass # Collection does not exist
