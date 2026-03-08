"""
Shared pytest configuration.

Heavy ML dependencies (sentence_transformers, chromadb, umap, scikit-learn)
are stubbed out via sys.modules at collection time — BEFORE any src.* module
is imported — so that unit and integration tests run without the full ML stack
installed.
"""

import sys
from unittest.mock import MagicMock

_HEAVY_MODULES = [
    "sentence_transformers",
    "chromadb",
    "chromadb.api",
    "chromadb.api.models",
    "chromadb.api.models.Collection",
    "umap",
    "sklearn",
    "sklearn.preprocessing",
    "sklearn.mixture",
]

for _m in _HEAVY_MODULES:
    sys.modules.setdefault(_m, MagicMock())
