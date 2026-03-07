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
from src.clustering import run_clustering_pipeline
from src.vector_store import index_documents, collection_size

def print_stage(title: str):
    print(f"\n{'='*50}\n[{time.strftime('%H:%M:%S')}] {title}\n{'='*50}")

def main():
    print_stage("PIPELINE START: 20 Newsgroups Semantic Indexing")
    start_time = time.time()
    
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
        if len(embeddings) != len(docs):
             print(f"WARNING: Cache size ({len(embeddings)}) mismatched with corpus size ({len(docs)}). Re-embedding...")
             embeddings = embed_texts(texts, batch_size=256, show_progress=True)
             save_embeddings(embeddings, [doc.doc_id for doc in docs])
        else:
             print(f"Successfully loaded {len(embeddings)} embeddings from cache.")
    else:
        print("No embeddings found. Generating embeddings natively on CPU...")
        embeddings = embed_texts(texts, batch_size=256, show_progress=True)
        save_embeddings(embeddings, [doc.doc_id for doc in docs])
        
    print(f"Stage 2 completed in {time.time()-t0:.2f}s.")
    
    # ---------------------------------------------------------
    # STAGE 3: Clustering Pipeline (UMAP + GMM)
    # ---------------------------------------------------------
    print_stage("STAGE 3: Clustering Pipeline")
    t0 = time.time()
    params, results = run_clustering_pipeline(embeddings)
    
    dominant_labels = results['labels']
    probs = results['probs']
    entropies = results['entropies']
    
    k = params['n_components']
    print(f"Identified K={k} optimal clusters via BIC.")
    print(f"Stage 3 completed in {time.time()-t0:.2f}s.")
    
    # ---------------------------------------------------------
    # STAGE 4: Build Metadata and Batch
    # ---------------------------------------------------------
    print_stage("STAGE 4: Building Metadata Schema")
    t0 = time.time()
    
    doc_ids = []
    metadatas = []
    high_uncertainty_docs = []
    
    for i, doc in enumerate(docs):
        # Calculate second cluster fallback
        sorted_indices = np.argsort(probs[i])
        second_cluster_id = int(sorted_indices[-2])
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
             
    print(f"Constructed metadata schemas for {len(metadatas)} documents in {time.time() - t0:.2f}s.")
    
    # ---------------------------------------------------------
    # STAGE 5: Vector Store Insertion
    # ---------------------------------------------------------
    print_stage("STAGE 5: ChromaDB Insertion")
    t0 = time.time()
    
    # Deduplicate before insertion to avert ChromaDB duplicate key errors
    unique_ids = set()
    dedup_ids, dedup_texts, dedup_embs, dedup_metas = [], [], [], []
    
    for _id, _text, _emb, _meta in zip(doc_ids, texts, list(embeddings), metadatas):
         if _id not in unique_ids:
              unique_ids.add(_id)
              dedup_ids.append(_id)
              dedup_texts.append(_text)
              dedup_embs.append(_emb)
              dedup_metas.append(_meta)
              
    duplicates_removed = len(doc_ids) - len(dedup_ids)
    if duplicates_removed > 0:
         print(f"Removed {duplicates_removed} hash duplicates prior to insertion.")
    
    # Batch size is configured to 512 natively in index_documents()
    index_documents(dedup_ids, dedup_texts, dedup_embs, dedup_metas)
    
    print(f"Vector Store insertion synced in {time.time() - t0:.2f}s.")
    
    # ---------------------------------------------------------
    # STAGE 6: Reporting and Pipeline Metrics
    # ---------------------------------------------------------
    print_stage("PIPELINE COMPLETE: Final Metrics Report")
    
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print(f"\n--- Corpus ---")
    print(f"Total documents indexed: {len(docs)}")
    if hasattr(docs[0], 'is_discarded'): # Mock check for discard variables
         print(f"Discard rate tracking: Computed downstream") 
    
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
    high_uncertainty_docs.sort(key=lambda x: x['entropy'], reverse=True)
    for i, b_doc in enumerate(high_uncertainty_docs[:5]):
         print(f"{i+1}. [Entropy: {b_doc['entropy']:.3f} | C1: {b_doc['top1_cluster']}, C2: {b_doc['top2_cluster']}]")
         print(f"   {b_doc['text']}")
         print("-" * 50)
    
    print(f"\nTotal Pipeline Runtime: {minutes}m {seconds}s")
    print("\n" + "="*50)
    print("Build complete. Run: uvicorn src.api:app --port 8000")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
