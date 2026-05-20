"""Shared utilities for eisenstein-embed."""

import re
import math
import hashlib
import unicodedata
from typing import List

import numpy as np


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, and normalize whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization after normalization."""
    return normalize_text(text).split()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    norm_a[norm_a == 0] = 1.0
    norm_b[norm_b == 0] = 1.0
    dots = np.sum((a / norm_a) * (b / norm_b), axis=1)
    return float(np.clip(dots, -1.0, 1.0)[0])


def smooth_inverse_frequency(prob: float, a: float = 1e-3) -> float:
    """Compute SIF weight for a word probability."""
    return a / (a + prob)


def hash_embedding(text: str, dim: int = 256, seed: int = 42) -> np.ndarray:
    """Create a deterministic pseudo-embedding from text using hashing.

    This provides a zero-dependency `encode()` fallback when no
    Model2Vec model is loaded. Vectors are L2-normalized.
    """
    words = tokenize(text)
    if not words:
        return np.zeros(dim, dtype=np.float32)

    vectors = []
    for word in words:
        # Deterministic hash-based vector
        h = hashlib.blake2b(key=seed.to_bytes(4, "little"), digest_size=32)
        h.update(word.encode("utf-8"))
        digest = h.digest()
        # Expand digest to dim using repeated hashing with different seeds
        vec = np.zeros(dim, dtype=np.float32)
        for i in range(0, dim, 32):
            chunk_size = min(32, dim - i)
            h2 = hashlib.blake2b(key=(seed + i).to_bytes(4, "little"), digest_size=32)
            h2.update(digest)
            chunk = np.frombuffer(h2.digest()[:chunk_size], dtype=np.uint8).astype(np.float32)
            vec[i : i + chunk_size] = chunk / 255.0 * 2.0 - 1.0
        vectors.append(vec)

    emb = np.mean(vectors, axis=0)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)
