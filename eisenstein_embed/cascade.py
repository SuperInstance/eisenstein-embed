"""CascadeMatcher — the 5-layer cascade."""

from typing import List, Optional, Tuple

import numpy as np

from eisenstein_embed.utils import normalize_text, cosine_similarity
from eisenstein_embed.bitvector import find_best_bitvector_match
from eisenstein_embed.deadband_cache import DeadbandCache
from eisenstein_embed.domain_sif import DomainSIF
from eisenstein_embed.bma_monitor import BMAMonitor


class CascadeMatcher:
    """5-layer matching cascade for finding the best candidate.

    Layers (in order of application):
        1. EXACT    — case-insensitive string match
        2. BITVECTOR — 64-bit Hamming distance on text fingerprints
        3. DEADBAND — cosine-similarity cache (skips redundant encoding)
        4. SEMANTIC — dense vector cosine similarity (Model2Vec)
        5. DOMAIN   — domain-aware SIF re-weighting of semantic vectors
    """

    def __init__(
        self,
        semantic_encoder=None,
        domain_sif: Optional[DomainSIF] = None,
        deadband_cache: Optional[DeadbandCache] = None,
        bma_monitor: Optional[BMAMonitor] = None,
        bitvector_threshold: float = 0.85,
        semantic_threshold: float = 0.3,
    ):
        self.semantic_encoder = semantic_encoder
        self.domain_sif = domain_sif
        self.deadband_cache = deadband_cache
        self.bma_monitor = bma_monitor
        self.bitvector_threshold = bitvector_threshold
        self.semantic_threshold = semantic_threshold

    def match(
        self,
        query: str,
        candidates: List[str],
        use_stemming: bool = False,
    ) -> Tuple[Optional[str], float, str]:
        """Find the best candidate for *query* through the cascade.

        Returns:
            (best_candidate, score, layer_name)
        """
        if not candidates:
            return None, 0.0, "none"

        norm_query = normalize_text(query)
        if not norm_query:
            return None, 0.0, "none"

        norm_candidates = [normalize_text(c) for c in candidates]

        # 1. EXACT
        for i, nc in enumerate(norm_candidates):
            if norm_query == nc:
                score = 1.0
                if self.bma_monitor is not None:
                    self.bma_monitor.record(query, score, "exact")
                return candidates[i], score, "exact"

        # 2. BITVECTOR
        bv_cand, bv_score = find_best_bitvector_match(query, candidates, use_stemming=use_stemming)
        if bv_score >= self.bitvector_threshold:
            if self.bma_monitor is not None:
                self.bma_monitor.record(query, bv_score, "bitvector")
            return bv_cand, bv_score, "bitvector"

        # 3–5. SEMANTIC (+ DEADBAND + DOMAIN)
        if self.semantic_encoder is not None:
            # 3. DEADBAND — check cache before encoding
            query_vec = None
            if self.deadband_cache is not None:
                query_vec = self.deadband_cache.get(query)

            if query_vec is None:
                query_vec = self._encode(query)
                if self.deadband_cache is not None:
                    self.deadband_cache.put(query, query_vec)

            cand_vecs = self._encode_candidates(candidates)

            best_idx = 0
            best_sim = -1.0
            for i, cand_vec in enumerate(cand_vecs):
                sim = cosine_similarity(query_vec, cand_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            layer = "semantic"
            if self.domain_sif is not None and self.domain_sif._fitted:
                layer = "domain"

            if best_sim >= self.semantic_threshold:
                if self.bma_monitor is not None:
                    self.bma_monitor.record(query, best_sim, layer)
                return candidates[best_idx], best_sim, layer

            # Semantic found something but below threshold — still record
            if self.bma_monitor is not None:
                self.bma_monitor.record(query, best_sim, layer)

        # Fallback to bitvector if nothing else beat the threshold
        if bv_cand is not None:
            return bv_cand, bv_score, "bitvector"

        # Absolute fallback — best semantic match even if below threshold
        if self.semantic_encoder is not None:
            query_vec = self._encode(query)
            cand_vecs = self._encode_candidates(candidates)
            best_idx = int(np.argmax([cosine_similarity(query_vec, cv) for cv in cand_vecs]))
            best_sim = cosine_similarity(query_vec, cand_vecs[best_idx])
            layer = "semantic"
            if self.domain_sif is not None and self.domain_sif._fitted:
                layer = "domain"
            return candidates[best_idx], best_sim, layer

        return None, 0.0, "none"

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single text, applying domain SIF if available."""
        vec = self.semantic_encoder([text])[0]
        if self.domain_sif is not None and self.domain_sif._fitted:
            # If the encoder exposes token-level embeddings, re-weight them.
            # Otherwise, this is a no-op fallback.
            vec = self._apply_domain_to_text(text, vec)
        return vec

    def _encode_candidates(self, candidates: List[str]) -> List[np.ndarray]:
        """Encode a list of candidate texts."""
        # Check deadband cache for each candidate
        vecs = []
        to_encode = []
        to_encode_idx = []
        for i, c in enumerate(candidates):
            cached = None
            if self.deadband_cache is not None:
                cached = self.deadband_cache.get(c)
            if cached is not None:
                vecs.append((i, cached))
            else:
                vecs.append((i, None))
                to_encode.append(c)
                to_encode_idx.append(i)

        if to_encode:
            encoded = self.semantic_encoder(to_encode)
            if self.deadband_cache is not None:
                for c, v in zip(to_encode, encoded):
                    self.deadband_cache.put(c, v)
            for idx, v in zip(to_encode_idx, encoded):
                vecs[idx] = (idx, v)

        # Apply domain SIF
        result = []
        for i, c in enumerate(candidates):
            vec = vecs[i][1]
            if self.domain_sif is not None and self.domain_sif._fitted:
                vec = self._apply_domain_to_text(c, vec)
            result.append(vec)
        return result

    def _apply_domain_to_text(self, text: str, sentence_vec: np.ndarray) -> np.ndarray:
        """Attempt to apply domain SIF weights to a sentence vector.

        If the semantic encoder is a Model2Vec StaticModel and exposes
        token-level embeddings, we recompute the weighted average using
        domain SIF weights. Otherwise we return the vector unchanged.
        """
        encoder = self.semantic_encoder
        if encoder is None:
            return sentence_vec

        # Check if this looks like a Model2Vec StaticModel
        if hasattr(encoder, "tokenize") and hasattr(encoder, "embedding"):
            try:
                tokens = encoder.tokenize(text)
                token_ids = [encoder.token_mapping.get(t, encoder.unk_token_id) for t in tokens]
                # Filter unknown tokens
                valid = [(tid, t) for tid, t in zip(token_ids, tokens) if tid != encoder.unk_token_id]
                if not valid:
                    return sentence_vec
                tids, token_strs = zip(*valid)
                embs = encoder.embedding[np.array(tids)]
                weights = [self.domain_sif.get_weight(t) for t in token_strs]
                weights_arr = np.array(weights, dtype=np.float64)
                weights_sum = weights_arr.sum()
                if weights_sum == 0:
                    weights_sum = 1.0
                weighted = (embs.T * weights_arr).T
                new_vec = weighted.sum(axis=0) / weights_sum
                norm = np.linalg.norm(new_vec)
                if norm > 0:
                    new_vec = new_vec / norm
                return new_vec.astype(np.float32)
            except Exception:
                pass
        return sentence_vec
