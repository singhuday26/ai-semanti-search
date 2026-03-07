"""
Script to analyze the Twenty Newsgroups dataset structure.
"""
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from collections import Counter

def main():
    print("Loading the 20 Newsgroups dataset (all subsets, without stripping headers)...")
    # Using remove=() to keep raw text, allowing manual stripping downstream
    dataset = fetch_20newsgroups(subset='all', remove=())
    
    docs = dataset.data
    labels = dataset.target
    target_names = dataset.target_names
    
    total_docs = len(docs)
    
    # Compute category counts
    counts = Counter(labels)
    
    # Compute lengths
    lengths = [len(doc) for doc in docs]
    avg_length = np.mean(lengths)
    
    # Estimate short documents (< 50 chars)
    short_docs = sum(1 for l in lengths if l < 50)
    
    print("\n--- Dataset Analysis ---")
    print(f"Total documents: {total_docs}")
    print(f"Average document length (characters): {avg_length:.2f}")
    print(f"Documents shorter than 50 characters: {short_docs}")
    
    print("\n--- Category Breakdown ---")
    for category_idx, count in counts.most_common():
        print(f"{target_names[category_idx]:<30} {count} docs")

if __name__ == "__main__":
    main()
