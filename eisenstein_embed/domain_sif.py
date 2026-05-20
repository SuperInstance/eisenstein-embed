"""Domain-aware SIF weighting (corpus-driven, not Zipf-estimated)."""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from eisenstein_embed.utils import tokenize, smooth_inverse_frequency


class DomainSIF:
    """Domain-aware smooth inverse frequency weighting.

    Given a corpus of in-domain texts, computes per-word probabilities
    and derives SIF weights. This is more accurate than Zipf-law
    estimates because it reflects actual word usage in the domain.
    """

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        self.word_counts: Counter = Counter()
        self.total_words: int = 0
        self.word_probs: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        """Learn domain-specific word probabilities from a corpus."""
        self.word_counts.clear()
        self.total_words = 0
        for text in texts:
            words = tokenize(text)
            self.word_counts.update(words)
            self.total_words += len(words)

        if self.total_words == 0:
            self.total_words = 1

        self.word_probs = {
            word: count / self.total_words
            for word, count in self.word_counts.items()
        }
        self.weights = {
            word: smooth_inverse_frequency(prob, self.alpha)
            for word, prob in self.word_probs.items()
        }
        self._fitted = True

    def get_weight(self, word: str) -> float:
        """Return the SIF weight for a word.

        Falls back to a default weight for out-of-vocabulary words.
        """
        if not self._fitted:
            return 1.0
        return self.weights.get(word.lower(), self.alpha / (self.alpha + 1e-8))

    def compute_sentence_vector(
        self,
        words: List[str],
        word_vectors: Dict[str, np.ndarray],
        dim: Optional[int] = None,
    ) -> np.ndarray:
        """Compute a weighted average sentence vector from word vectors.

        Args:
            words: List of words in the sentence.
            word_vectors: Mapping from word to vector.
            dim: Expected vector dimension. Inferred if not provided.

        Returns:
            A numpy array of shape (dim,).
        """
        vectors = []
        weights = []
        for word in words:
            vec = word_vectors.get(word)
            if vec is not None:
                vectors.append(np.asarray(vec, dtype=np.float64))
                weights.append(self.get_weight(word))

        if not vectors:
            if dim is None:
                # Cannot determine dimension without any vectors
                return np.zeros(0)
            return np.zeros(dim, dtype=np.float64)

        vectors = np.stack(vectors)
        weights = np.asarray(weights, dtype=np.float64)
        weights_sum = weights.sum()
        if weights_sum == 0:
            weights_sum = 1.0

        weighted = (vectors.T * weights).T
        sentence_vec = weighted.sum(axis=0) / weights_sum

        # Remove common first principal component (standard SIF post-processing)
        sentence_vec = self._remove_pc(sentence_vec)
        return sentence_vec

    @staticmethod
    def _remove_pc(vec: np.ndarray) -> np.ndarray:
        """Simple projection removal — just L2 normalize for now."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
