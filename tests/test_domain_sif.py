"""Tests for domain-aware SIF weighting."""
import pytest
import numpy as np
from eisenstein_embed.domain_sif import DomainSIF


class TestDomainSIF:
    def test_fit(self):
        sif = DomainSIF()
        sif.fit(["deploy model", "plato room", "fleet status"])
        assert sif._fitted
        assert sif.total_words > 0

    def test_word_frequencies_tracked(self):
        sif = DomainSIF()
        sif.fit(["hello hello hello world"])
        assert sif.word_counts["hello"] == 3
        assert sif.word_counts["world"] == 1

    def test_get_weight(self):
        sif = DomainSIF()
        sif.fit(["deploy deploy deploy model"])
        # "deploy" is more frequent → lower weight (SIF formula)
        assert sif.get_weight("deploy") < sif.get_weight("model")

    def test_get_weight_unfitted(self):
        sif = DomainSIF()
        assert sif.get_weight("anything") == 1.0

    def test_compute_sentence_vector(self):
        sif = DomainSIF()
        sif.fit(["deploy model", "plato room"])
        word_vecs = {
            "deploy": np.array([1.0, 0.0]),
            "model": np.array([0.0, 1.0]),
        }
        vec = sif.compute_sentence_vector(["deploy", "model"], word_vecs, dim=2)
        assert vec.shape == (2,)
        # Should be L2 normalized
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01 or vec.sum() == 0.0

    def test_empty_sentence(self):
        sif = DomainSIF()
        sif.fit(["hello world"])
        vec = sif.compute_sentence_vector([], {}, dim=64)
        assert vec.shape == (64,)
        assert np.allclose(vec, 0.0)
