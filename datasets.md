# Twenty Newsgroups Dataset

## Overview
This repository uses the **Twenty Newsgroups** dataset, an extensively used dataset for experiments in text applications of machine learning techniques, such as text classification and text clustering. 

- **Instances**: ~20,000 messages
- **Subject Area**: Computer, Science, Recreation, Society, Talk, Misc
- **Missing Values**: No
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Local Directory Structure
The datasets are managed locally within the `data/` directory. 

```text
data/
├── raw/                    # Immutable raw dataset downloads
│   ├── 20_newsgroups/      # Full dataset (19,997 documents)
│   ├── mini_newsgroups/    # Subset of the dataset (2,000 documents)
```

*Note: The `data/raw/` directory exclusively contains immutable datasets. Any processed artifacts like chunked texts, vector embeddings, cluster metadata, or cache objects will live separately in the respective `data/` root or subdirectories to prevent mutating the original source.*

### The 20 Categories (Classes)
The dataset is partitioned evenly across 20 different newsgroups, representing 6 broad themes:

1. **comp.** (Computers): `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`
2. **rec.** (Recreation): `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey`
3. **sci.** (Science): `sci.crypt`, `sci.electronics`, `sci.med`, `sci.space`
4. **soc.** (Society): `soc.religion.christian`
5. **talk.** (Talk/Politics/Religion): `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`, `talk.religion.misc`, `alt.atheism`
6. **misc.** (Miscellaneous): `misc.forsale`

---

## Semantic Overlap & Suitability for Semantic Search 
This dataset is highly suitable for building and testing our semantic search and fuzzy clustering system due to the intentional **semantic overlap** between certain categories. 

- **High Overlap (Fuzzy Boundaries):** For instance, `comp.sys.ibm.pc.hardware` and `comp.sys.mac.hardware` share significant vocabulary (e.g., "motherboard", "RAM", "CPU"). Similarly, `talk.religion.misc`, `soc.religion.christian`, and `alt.atheism` structurally discuss the same foundational concepts from different vantage points.
- **Low Overlap (Distinct Boundaries):** Conversely, `rec.sport.baseball` and `sci.crypt` rarely share underlying contextual embeddings. 

Because keyword-based systems perform poorly at differentiating nuanced overlap (e.g. knowing when "apple" refers to a computer vs a fruit/recipe based on surrounding vectors), this exact property allows us to test the efficacy of our **sentence-transformer embeddings** and **Gaussian Mixture Model (GMM) fuzzy clustering**. GMM probabilistically assigns documents to multiple overlapping clusters rather than rigidly throwing them in just one.

### Expected Statistics
- **Total Documents:** ~18,846 (after removing empty lines/headers)
- **Average Length:** Varies heavily, typically ~200-300 words.
- **Short Documents:** There are numerous documents under 50 characters (often just quotes or signatures) which require careful preprocessing to avoid low-quality embeddings.

---

## Dataset Loading Strategy
Our pipeline loads data using the standard `scikit-learn` dataset fetcher:

```python
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='all', remove=())
```

**Why `remove=()`?**
Even though the `fetch_20newsgroups` utility has built-in arguments `remove=('headers', 'footers', 'quotes')` to strip metadata automatically, our architecture requires us to use `remove=()` (keeping the raw text intact). We fetch the raw text first because our custom preprocessing pipeline strips these manual artifacts itself before passing them into the embedder. This ensures full lineage and visibility into text mutation logic inside `src/preprocessing.py`, allowing us to optimize for semantic density directly.
