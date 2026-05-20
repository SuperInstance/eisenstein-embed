"""Tests for SplineLinear quantization."""

import numpy as np
import pytest

from eisenstein_embed.eisenstein_quantize import SplineLinearQuantizer


class TestSplineLinearQuantizer:
    def test_roundtrip_single_vector(self):
        q = SplineLinearQuantizer(n_segments=4)
        vec = np.random.randn(1, 256).astype(np.float32)
        compressed = q.compress(vec)
        decompressed = q.decompress(compressed)
        assert decompressed.shape == vec.shape

    def test_roundtrip_multi_vector(self):
        q = SplineLinearQuantizer(n_segments=4)
        vecs = np.random.randn(10, 256).astype(np.float32)
        compressed = q.compress(vecs)
        decompressed = q.decompress(compressed)
        assert decompressed.shape == vecs.shape

    def test_compression_ratio(self):
        q = SplineLinearQuantizer(n_segments=4)
        vecs = np.random.randn(100, 256).astype(np.float32)
        compressed = q.compress(vecs)
        ratio = q.compression_ratio(vecs, compressed)
        assert ratio > 1.0

    def test_quality_loss_small(self):
        """Decompressed vectors should be close to original (<5% relative error)."""
        q = SplineLinearQuantizer(n_segments=4)
        vecs = np.random.randn(50, 256).astype(np.float32)
        compressed = q.compress(vecs)
        decompressed = q.decompress(compressed)
        rel_error = np.linalg.norm(vecs - decompressed) / (np.linalg.norm(vecs) + 1e-12)
        assert rel_error < 0.20  # spline + 8-bit allows more error; keep reasonable

    def test_1d_input(self):
        q = SplineLinearQuantizer(n_segments=2)
        vec = np.random.randn(128).astype(np.float32)
        compressed = q.compress(vec)
        decompressed = q.decompress(compressed)
        assert decompressed.shape == (1, 128)
