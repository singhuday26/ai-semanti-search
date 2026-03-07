import os
import sys
import time
import numpy as np
from collections import Counter
from typing import List

# Ensure we can import from src when running from the scripts/ folder natively
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all processing pipeline components
from src.preprocessing import load_and_clean
from src.embedder import embed_texts, save_embeddings, embeddings_exist, load_embeddings
from src.clustering import run_clustering_pipeline, membership_entropy
from src.vector_store import index_documents, collection_size

def print_stage(title: str):
    print(f"\n{'='*50}\n[{time.strftime('%H:%M:%S')}] {title}\n{'='*50}")

def main():
    print_stage("PIPELINE START: 20 Newsgroups Semantic Indexing")
    start_time = time.time()
    
    # Ensure reproducible clustering computations
    np.random.seed(42)
    
    # ---------------------------------------------------------
    # STAGE 1: Load and Clean Corpus
    # ---------------------------------------------------------
    print_stage("STAGE 1: Loading & Cleaning")
    t0 = time.time()
    docs, texts = load_and_clean(subset='all')
    print(f"Loaded {len(docs)} clean documents in {time.time()-t0:.2f}s.")
    
    if len(docs) == 0:
         print("No documents returned. Exiting.")
         return
         
    # ---------------------------------------------------------
    # STAGE 2: Generate or Load Embeddings
    # ---------------------------------------------------------
    print_stage("STAGE 2: Embeddings Generation")
    t0 = time.time()
    
    if embeddings_exist():
        print("Embeddings already exist on disk. Loading cached arrays...")
        embeddings, cached_ids = load_embeddings()
        
        # Verify cache integrity
        expected_ids = [doc.doc_id for doc in docs]
        
        if len(embeddings) != len(docs) or cached_ids != expected_ids:
             print("Embedding cache mismatch detected. Rebuilding embeddings...")
             embeddings = embed_texts(texts, batch_size=256, show_progress=True)
             save_embeddings(embeddings, [doc.doc_id for doc in docs])
        else:
             print(f"Successfully loaded {len(embeddings)} embeddings from cache.")
    else:
        print("No embeddings found. Generating embeddings natively on CPU...")
        embeddings = embed_texts(texts, batch_size=256, show_progress=True)
        save_embeddings(embeddings, [doc.doc_id for doc in docs])
        
    stage_time = time.time() - t0
    print(f"Stage completed in {stage_time:.2f}s")
    
    # ---------------------------------------------------------
    # STAGE 3: Clustering Pipeline (UMAP + GMM)
    # ---------------------------------------------------------
    print_stage("STAGE 3: Clustering Pipeline")
    t0 = time.time()
    dominant_labels, probs, gmm, umap_reducer = run_clustering_pipeline(embeddings)

    entropies = membership_entropy(probs)
    k = gmm.n_components
    print(f"Identified K={k} optimal clusters via BIC.")
    stage_time = time.time() - t0
    print(f"Stage completed in {stage_time:.2f}s")
    
    # ---------------------------------------------------------
    # STAGE 4: Build Metadata and Batch
    # ---------------------------------------------------------
    print_stage("STAGE 4: Building Metadata Schema")
    t0 = time.time()
    
    doc_ids = []
    metadatas = []
    high_uncertainty_docs = []
    
    for i, doc in enumerate(docs):
        # Calculate second cluster fallback efficiently
        top2 = np.argsort(probs[i])[-2:]
        second_cluster_id = int(top2[0])
        second_cluster_prob = float(probs[i][second_cluster_id])
        
        entropy_val = float(entropies[i])
        is_high_uncertainty = bool(entropy_val > 0.7)
        
        metadata = {
            "original_label": int(doc.original_label),
            "label_name": str(doc.label_name),
            "dominant_cluster_id": int(dominant_labels[i]),
            "cluster_entropy": entropy_val,
            "is_high_uncertainty": is_high_uncertainty,
            "second_cluster_id": second_cluster_id,
            "second_cluster_prob": second_cluster_prob
        }
        
        doc_ids.append(str(doc.doc_id))
        metadatas.append(metadata)
        
        if is_high_uncertainty:
             # Stash for the top-5 boundary doc summary later
             high_uncertainty_docs.append({
                  "text": doc.text[:150].replace('\n', ' ') + "...",
                  "entropy": entropy_val,
                  "top1_cluster": int(dominant_labels[i]),
                  "top2_cluster": second_cluster_id
             })
             
             # Limit memory overhead to strictly the top 5 highest-entropy docs
             high_uncertainty_docs.sort(key=lambda x: x['entropy'], reverse=True)
             if len(high_uncertainty_docs) > 5:
                  high_uncertainty_docs.pop()
             
    stage_time = time.time() - t0
    print(f"Stage completed in {stage_time:.2f}s")
    
    # ---------------------------------------------------------
    # STAGE 5: Vector Store Insertion
    # ---------------------------------------------------------
    print_stage("STAGE 5: ChromaDB Insertion")
    t0 = time.time()
    
    # Batch size is configured to 512 natively in index_documents()
    index_documents(doc_ids, texts, embeddings, metadatas)
    
    stage_time = time.time() - t0
    print(f"Stage completed in {stage_time:.2f}s")
    
    # ---------------------------------------------------------
    # STAGE 6: Reporting and Pipeline Metrics
    # ---------------------------------------------------------
    print_stage("PIPELINE COMPLETE: Final Metrics Report")
    
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print(f"\n--- Corpus ---")
    print(f"Total documents indexed: {len(doc_ids)}")
    print(f"Discard rate: Printed in STAGE 1 downstream telemetry.") 
    
    print(f"\n--- Clustering ---")
    print(f"Optimal Clusters (K): {k} (BIC-selected)")
    print(f"Mean membership entropy: {np.mean(entropies):.4f}")
    print(f"High-uncertainty docs (>0.7): {len(high_uncertainty_docs)} ({(len(high_uncertainty_docs)/len(docs))*100:.1f}%)")
    
    print(f"\n--- Cluster Size Distribution ---")
    size_dist = Counter(dominant_labels)
    for cluster_id in range(k): # Enforce 0 to K order 
         count = size_dist.get(cluster_id, 0)
         print(f"Cluster {cluster_id:02d}: {count:5d} docs | {count/len(docs)*100:5.1f}%")
         
    print(f"\n--- Top 5 Boundary Documents (Highest Entropy) ---")
    for i, b_doc in enumerate(high_uncertainty_docs):
         print(f"{i+1}. [Entropy: {b_doc['entropy']:.3f} | C1: {b_doc['top1_cluster']}, C2: {b_doc['top2_cluster']}]")
         print(f"   {b_doc['text']}")
         print("-" * 50)
    
    print(f"\nTotal Pipeline Runtime: {minutes}m {seconds}s")
    print("\n" + "="*50)
    print("Build complete. Run: uvicorn src.api:app --port 8000")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
