"""Tests for the 5-layer cascade matcher."""

import numpy as np
import pytest

from eisenstein_embed.cascade import CascadeMatcher
from eisenstein_embed.domain_sif import DomainSIF
from eisenstein_embed.deadband_cache import DeadbandCache
from eisenstein_embed.bma_monitor import BMAMonitor


def dummy_encoder(texts):
    """A dummy semantic encoder that returns deterministic vectors."""
    dim = 4
    results = []
    for t in texts:
        # Simple hash-based deterministic vector
        h = hash(t) % 10000
        np.random.seed(h)
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        results.append(vec)
    return np.stack(results)


class TestCascadeMatcherExact:
    def test_exact_match(self):
        cm = CascadeMatcher()
        cand, score, layer = cm.match("hello", ["hello", "world"])
        assert cand == "hello"
        assert score == 1.0
        assert layer == "exact"

    def test_exact_case_insensitive(self):
        cm = CascadeMatcher()
        cand, score, layer = cm.match("Hello", ["hello", "world"])
        assert cand == "hello"
        assert score == 1.0
        assert layer == "exact"


class TestCascadeMatcherBitvector:
    def test_typo_match(self):
        cm = CascadeMatcher(bitvector_threshold=0.80)
        cand, score, layer = cm.match("triangel", ["triangle", "square", "circle"])
        assert cand == "triangle"
        assert layer == "bitvector"
        assert score > 0.80

    def test_no_match_empty_candidates(self):
        cm = CascadeMatcher()
        cand, score, layer = cm.match("hello", [])
        assert cand is None
        assert score == 0.0
        assert layer == "none"


class TestCascadeMatcherSemantic:
    def test_semantic_match(self):
        cm = CascadeMatcher(semantic_encoder=dummy_encoder, semantic_threshold=0.0)
        # Use distinct words to avoid exact or bitvector match
        cand, score, layer = cm.match("foo", ["bar", "baz", "qux"])
        assert cand is not None
        assert layer in ("semantic", "domain", "bitvector")  # bitvector may match first for short words
        assert score > -1.0

    def test_semantic_above_threshold(self):
        cm = CascadeMatcher(semantic_encoder=dummy_encoder, semantic_threshold=0.99)
        # With threshold very high, bitvector should win or semantic should still return something
        cand, score, layer = cm.match("triangel", ["triangle", "square"])
        # "triangel" vs "triangle" should be close bitvector-wise
        assert cand is not None


class TestCascadeMatcherDeadband:
    def test_deadband_skips_encoding(self):
        cache = DeadbandCache()
        vec = dummy_encoder(["hello"])[0]
        cache.put("hello", vec)

        call_count = [0]

        def counting_encoder(texts):
            call_count[0] += 1
            return dummy_encoder(texts)

        cm = CascadeMatcher(semantic_encoder=counting_encoder, deadband_cache=cache)
        cand, score, layer = cm.match("hello", ["world", "foo"])
        # Since "hello" is in cache, encoding should be skipped for query
        assert call_count[0] <= 1  # candidates still need encoding


class TestCascadeMatcherDomain:
    def test_domain_layer_active(self):
        dsif = DomainSIF()
        dsif.fit(["deploy micro model", "plato room tile", "npu acceleration"])
        cm = CascadeMatcher(
            semantic_encoder=dummy_encoder,
            domain_sif=dsif,
            semantic_threshold=0.0,
            bitvector_threshold=1.0,  # disable bitvector layer
        )
        cand, score, layer = cm.match("foo", ["bar", "baz"])
        assert layer == "domain"


class TestCascadeMatcherBMA:
    def test_bma_records(self):
        bma = BMAMonitor()
        cm = CascadeMatcher(
            semantic_encoder=dummy_encoder,
            bma_monitor=bma,
            semantic_threshold=0.0,
        )
        cm.match("foo", ["bar", "baz"])
        assert len(bma._scores) >= 1
