"""Drop-in StaticModel replacement (extends Model2Vec)."""

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
    def __init__(self, best_match=None, score=0.0, method="none"):
        self.best_match = best_match
        self.score = score
        self.method = method


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
    ):
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

        Returns a numpy array of shape (n_texts, dim).
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
    ) -> MatchResult:
        """Find the best candidate for *query*.

        Returns:
            MatchResult(best_match, score, method)
        """
        if threshold is not None:
            old_thresh = self.cascade.bitvector_threshold
            self.cascade.bitvector_threshold = threshold
            best, score, layer = self.cascade.match(query, candidates)
            self.cascade.bitvector_threshold = old_thresh
            return MatchResult(best_match=best, score=score, method=layer)
        best, score, layer = self.cascade.match(query, candidates)
        return MatchResult(best_match=best, score=score, method=layer)

    def add_domain(self, name: str, texts: List[str]) -> None:
        """Learn a domain-specific SIF profile from example texts."""
        dsif = DomainSIF()
        dsif.fit(texts)
        self.domain_sifs[name] = dsif
        self.active_domain = name
        self.cascade.domain_sif = dsif

    def set_domain(self, name: Optional[str]) -> None:
        """Activate a previously learned domain profile."""
        self.active_domain = name
        self.cascade.domain_sif = self.domain_sifs.get(name)

    def enable_self_tuning(self) -> None:
        """Enable BMA drift detection and adaptive thresholds."""
        self._self_tuning = True
        if self.bma_monitor is None:
            self.bma_monitor = BMAMonitor()
        self.cascade.bma_monitor = self.bma_monitor

    def disable_self_tuning(self) -> None:
        """Disable BMA drift detection."""
        self._self_tuning = False
        self.cascade.bma_monitor = None

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        if self.semantic_model is not None:
            return self.semantic_model.dim
        return 64

    def similarity(self, texts1: Union[str, List[str]], texts2: Union[str, List[str]]) -> np.ndarray:
        """Compute pairwise cosine similarities between two lists of texts."""
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
