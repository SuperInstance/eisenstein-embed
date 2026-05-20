"""Cosine similarity cache with configurable threshold."""

from typing import List, Optional, Tuple

import numpy as np

from eisenstein_embed.utils import normalize_text
from eisenstein_embed.bitvector import text_fingerprint, bitvector_similarity


class DeadbandCache:
    """Cache that skips re-encoding when a query is similar to a cached one.

    In conversational settings, users often rephrase or repeat queries.
    The deadband cache stores (normalized_text, vector) pairs and returns
    a cached vector when textual similarity (using bitvector proxy) exceeds
    the threshold. This approximates a cosine-similarity deadband without
    requiring an expensive embedding step.
    """

    def __init__(self, threshold: float = 0.90, max_size: int = 1000):
        self.threshold = threshold
        self.max_size = max_size
        self._entries: List[Tuple[str, int, np.ndarray]] = []

    def get(self, text: str) -> Optional[np.ndarray]:
        """Return a cached vector if a sufficiently similar text exists.

        First checks exact text match, then falls back to bitvector similarity
        as a fast proxy for semantic cosine similarity.
        """
        text = normalize_text(text)
        query_fp = text_fingerprint(text)

        best_match: Optional[np.ndarray] = None
        best_sim = -1.0

        for cached_text, cached_fp, cached_vec in self._entries:
            if text == cached_text:
                return cached_vec
            sim = bitvector_similarity(query_fp, cached_fp)
            if sim > best_sim:
                best_sim = sim
                best_match = cached_vec

        if best_sim >= self.threshold:
            return best_match
        return None

    def put(self, text: str, vector: np.ndarray) -> None:
        """Store a text-vector pair in the cache."""
        text = normalize_text(text)
        fp = text_fingerprint(text)
        # Avoid duplicates
        for i, (cached_text, _, _) in enumerate(self._entries):
            if cached_text == text:
                self._entries[i] = (text, fp, np.asarray(vector))
                return
        self._entries.append((text, fp, np.asarray(vector)))
        if len(self._entries) > self.max_size:
            self._entries.pop(0)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
