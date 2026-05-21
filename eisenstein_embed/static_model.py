"""Drop-in StaticModel replacement (extends Model2Vec)."""

import threading
from typing import List, Optional, Union

import numpy as np

from eisenstein_embed.utils import normalize_text, cosine_similarity, hash_embedding
from eisenstein_embed.bitvector import find_best_bitvector_match
from eisenstein_embed.deadband_cache import DeadbandCache
from eisenstein_embed.domain_sif import DomainSIF
from eisenstein_embed.bma_monitor import BMAMonitor
from eisenstein_embed.cascade import CascadeMatcher


class MatchResult:
    """Result from EisensteinModel.match()."""
    def __init__(
        self,
        best_match: Optional[str] = None,
        score: float = 0.0,
        method: str = "none",
    ) -> None:
        self.best_match: Optional[str] = best_match
        self.score: float = score
        self.method: str = method
    
    def __repr__(self):
        return f"MatchResult(best_match={self.best_match!r}, score={self.score:.3f}, method={self.method!r})"

    def __eq__(self, other):
        if not isinstance(other, MatchResult):
            return NotImplemented
        return (self.best_match == other.best_match
                and abs(self.score - other.score) < 1e-6
                and self.method == other.method)

    def __hash__(self):
        return hash((self.best_match, round(self.score, 6), self.method))


class EisensteinModel:
    """Drop-in replacement for Model2Vec's StaticModel.

    Supports zero-config bitvector-only matching or full semantic
    encoding when a Model2Vec model is loaded.
    """

    def __init__(
        self,
        semantic_model=None,
        bitvector_threshold: float = 0.85,
        semantic_threshold: float = 0.3,
        deadband_threshold: float = 0.90,
        deadband_max_size: int = 1000,
        use_stemming: bool = False,
    ):
        self._lock = threading.Lock()
        self.semantic_model = semantic_model
        self.domain_sifs: dict = {}
        self.active_domain: Optional[str] = None
        self.deadband_cache = DeadbandCache(
            threshold=deadband_threshold, max_size=deadband_max_size
        )
        self.bma_monitor: Optional[BMAMonitor] = None
        self.cascade = CascadeMatcher(
            semantic_encoder=self._encode_fn if semantic_model is not None else None,
            domain_sif=None,
            deadband_cache=self.deadband_cache,
            bma_monitor=None,
            bitvector_threshold=bitvector_threshold,
            semantic_threshold=semantic_threshold,
        )
        self._self_tuning = False
        self.use_stemming = use_stemming

    @classmethod
    def from_model2vec(cls, model_name: str, **kwargs):
        """Load an EisensteinModel backed by a Model2Vec StaticModel."""
        try:
            from model2vec import StaticModel
        except ImportError as exc:
            raise ImportError(
                "model2vec is required for from_model2vec. "
                "Install it with: pip install eisenstein-embed[model2vec]"
            ) from exc

        semantic_model = StaticModel.from_pretrained(model_name)
        return cls(semantic_model=semantic_model, **kwargs)

    def _encode_fn(self, texts: List[str]) -> np.ndarray:
        """Internal encoder callable used by the cascade."""
        return self.encode(texts)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts into dense vectors.

        Args:
            texts: A single string or list of strings to encode.

        Returns:
            numpy array of shape (n_texts, dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.semantic_model is not None:
            vectors = self.semantic_model.encode(texts)
            # Apply domain SIF if active
            if self.active_domain is not None:
                dsif = self.domain_sifs.get(self.active_domain)
                if dsif is not None and dsif._fitted:
                    vectors = np.stack([
                        self.cascade._apply_domain_to_text(t, v)
                        for t, v in zip(texts, vectors)
                    ])
            return vectors

        # Fallback: hash-based deterministic embeddings
        dim = 64
        return np.stack([hash_embedding(t, dim=dim) for t in texts])

    def match(
        self,
        query: str,
        candidates: List[str],
        threshold: Optional[float] = None,
        use_stemming: Optional[bool] = None,
    ) -> MatchResult:
        """Find the best match for a query among candidates.

        Args:
            query: The search string.
            candidates: List of candidate strings to match against.
            threshold: Minimum similarity score to return a match.
            use_stemming: Whether to use morphological stemming.
                Defaults to model setting.

        Returns:
            MatchResult with best_match, score, and method used.

        Raises:
            TypeError: If query or candidates are not strings/lists.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if not isinstance(candidates, list):
            raise TypeError(
                f"candidates must be list, got {type(candidates).__name__}"
            )
        for i, c in enumerate(candidates):
            if not isinstance(c, str):
                raise TypeError(
                    f"candidates[{i}] must be str, got {type(c).__name__}"
                )
        if not candidates:
            return MatchResult(None, 0.0, "none")

        stemming = use_stemming if use_stemming is not None else self.use_stemming
        if threshold is not None:
            old_thresh = self.cascade.bitvector_threshold
            self.cascade.bitvector_threshold = threshold
            best, score, layer = self.cascade.match(query, candidates, use_stemming=stemming)
            self.cascade.bitvector_threshold = old_thresh
            return MatchResult(best_match=best, score=score, method=layer)
        best, score, layer = self.cascade.match(query, candidates, use_stemming=stemming)
        return MatchResult(best_match=best, score=score, method=layer)

    def add_knowledge(self, key: str, value: str) -> None:
        """Store a key-value knowledge pair in the model.

        Args:
            key: The knowledge key string.
            value: The knowledge value string.

        Raises:
            TypeError: If key or value are not strings.
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}")
        if not isinstance(value, str):
            raise TypeError(f"value must be str, got {type(value).__name__}")
        with self._lock:
            if not hasattr(self, '_knowledge'):
                self._knowledge = {}
            self._knowledge[key] = value

    def remove_knowledge(self, key: str) -> bool:
        """Remove a knowledge entry by key.

        Args:
            key: The key to remove.

        Returns:
            True if the key was found and removed, False otherwise.
        """
        with self._lock:
            if hasattr(self, '_knowledge') and key in self._knowledge:
                del self._knowledge[key]
                return True
            return False

    def clear_knowledge(self) -> None:
        """Remove all stored knowledge entries."""
        with self._lock:
            self._knowledge = {}

    def __len__(self) -> int:
        """Return the number of stored knowledge entries."""
        return len(getattr(self, '_knowledge', {}))

    def match_all(
        self,
        query: str,
        candidates: List[str],
        use_stemming: Optional[bool] = None,
    ) -> List[MatchResult]:
        """Score all candidates and return results sorted by score (desc).

        Args:
            query: The search string.
            candidates: List of candidate strings to score.
            use_stemming: Whether to use morphological stemming.

        Returns:
            List of MatchResult for all candidates, best first.

        Raises:
            TypeError: If query or candidates are invalid types.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if not isinstance(candidates, list):
            raise TypeError(f"candidates must be list, got {type(candidates).__name__}")
        for i, c in enumerate(candidates):
            if not isinstance(c, str):
                raise TypeError(f"candidates[{i}] must be str, got {type(c).__name__}")

        stemming = use_stemming if use_stemming is not None else self.use_stemming
        results = []
        for candidate in candidates:
            best, score, layer = self.cascade.match(
                query, [candidate], use_stemming=stemming
            )
            results.append(MatchResult(
                best_match=candidate, score=score, method=layer
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def __getstate__(self):
        """Return serializable state for pickling."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Restore state from pickled data."""
        self.__dict__.update(state)

    def save(self, path: str) -> None:
        """Save model to file.

        Args:
            path: File path to save the serialized model.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "EisensteinModel":
        """Load model from file.

        Args:
            path: File path to load the model from.

        Returns:
            Deserialized EisensteinModel instance.
        """
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    def add_domain(self, name: str, texts: List[str]) -> None:
        """Learn a domain-specific SIF profile from example texts."""
        dsif = DomainSIF()
        dsif.fit(texts)
        with self._lock:
            self.domain_sifs[name] = dsif
            self.active_domain = name
            self.cascade.domain_sif = dsif

    def set_domain(self, name: Optional[str]) -> None:
        """Activate a previously learned domain profile.

        Args:
            name: Domain name to activate, or None to deactivate.
        """
        with self._lock:
            self.active_domain = name
            self.cascade.domain_sif = self.domain_sifs.get(name)

    def enable_self_tuning(self) -> None:
        """Enable BMA drift detection and adaptive thresholds.

        Activates the Bayesian Moving Average monitor that tracks
        score distributions and adjusts thresholds adaptively.
        """
        self._self_tuning = True
        if self.bma_monitor is None:
            self.bma_monitor = BMAMonitor()
        self.cascade.bma_monitor = self.bma_monitor

    def disable_self_tuning(self) -> None:
        """Disable BMA drift detection.

        Deactivates the BMA monitor and resets adaptive thresholds.
        """
        self._self_tuning = False
        self.bma_monitor = None
        self.cascade.bma_monitor = None

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        if self.semantic_model is not None:
            return self.semantic_model.dim
        return 64

    def similarity(self, texts1: Union[str, List[str]], texts2: Union[str, List[str]]) -> np.ndarray:
        """Compute pairwise cosine similarities between two lists of texts.

        Args:
            texts1: First set of texts (string or list of strings).
            texts2: Second set of texts (string or list of strings).

        Returns:
            numpy array of shape (len(texts1), len(texts2)) with cosine similarities.
        """
        if isinstance(texts1, str):
            texts1 = [texts1]
        if isinstance(texts2, str):
            texts2 = [texts2]
        enc1 = self.encode(texts1)
        enc2 = self.encode(texts2)
        # Normalized dot product
        enc1 = enc1 / (np.linalg.norm(enc1, axis=1, keepdims=True) + 1e-12)
        enc2 = enc2 / (np.linalg.norm(enc2, axis=1, keepdims=True) + 1e-12)
        return np.dot(enc1, enc2.T)
