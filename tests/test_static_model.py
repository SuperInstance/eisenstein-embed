"""Tests for EisensteinModel (drop-in StaticModel replacement)."""
import pytest
import numpy as np
from eisenstein_embed.static_model import EisensteinModel, MatchResult


class TestEisensteinModelNoDeps:
    """Test without Model2Vec installed."""

    def test_create_without_model2vec(self):
        model = EisensteinModel()
        assert model is not None

    def test_encode_bitvector_fallback(self):
        model = EisensteinModel()
        vecs = model.encode(["hello world", "test"])
        assert vecs.shape == (2, 64)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_match_exact(self):
        model = EisensteinModel()
        result = model.match("triangle", ["square", "triangle"])
        assert isinstance(result, MatchResult)
        assert result.best_match == "triangle"
        assert result.method == "exact"

    def test_match_typo(self):
        model = EisensteinModel()
        result = model.match("triangel", ["square", "triangle"])
        assert result.best_match == "triangle"

    def test_dim_bitvector(self):
        model = EisensteinModel()
        assert model.dim == 64


class TestEisensteinQuantize:
    def test_import(self):
        from eisenstein_embed.eisenstein_quantize import SplineLinearQuantizer
        assert callable(SplineLinearQuantizer)

    def test_compress_decompress(self):
        from eisenstein_embed.eisenstein_quantize import SplineLinearQuantizer
        table = np.random.randn(100, 32).astype(np.float32)
        q = SplineLinearQuantizer(n_segments=8)
        q.fit(table)
        compressed = q.compress(table)
        result = q.decompress(compressed)
        assert isinstance(result, np.ndarray)
        assert result.shape == table.shape


class TestEisensteinModelWithModel2Vec:
    """Tests that require model2vec — skipped if not installed."""

    @pytest.fixture
    def model(self):
        try:
            return EisensteinModel.from_model2vec("minishlab/potion-base-8M")
        except ImportError:
            pytest.skip("model2vec not installed")

    def test_encode_with_model2vec(self, model):
        vecs = model.encode(["hello world", "greetings earth"])
        assert vecs.shape[0] == 2
        assert vecs.shape[1] == model.dim

    def test_match_with_model2vec(self, model):
        result = model.match("how many tests", ["how many tests", "different"])
        assert isinstance(result, MatchResult)
        assert result.best_match is not None
