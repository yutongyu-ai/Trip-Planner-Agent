"""Sentence-embedding helper shared by the online reranker and the offline
evaluation scripts, so both use the exact same embedding model/logic.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

# Multilingual (50+ languages, incl. Chinese) so a zh preference string and
# zh place descriptions embed into the same space. Still 384-dim, so nothing
# downstream (rank.py, the eval scripts) changes.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache
def _get_model() -> SentenceTransformer:
    # CSF3 GPUs are only available to scheduled jobs; a login-node process
    # that lets sentence-transformers auto-detect CUDA hits
    # cudaErrorDevicesUnavailable. This model is small and fast enough on
    # CPU, so pin it explicitly instead of auto-detecting.
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return an (n, d) L2-normalized embedding matrix for `texts`."""
    if not texts:
        return np.empty((0, 0))
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
