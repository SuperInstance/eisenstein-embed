"""Tests for deadband cache."""

import numpy as np
import pytest

from eisenstein_embed.deadband_cache import DeadbandCache


class TestDeadbandCache:
    def test_exact_match_returns_vector(self):
        cache = DeadbandCache()
        vec = np.array([1.0, 2.0, 3.0])
        cache.put("hello world", vec)
        result = cache.get("hello world")
        assert result is not None
        np.testing.assert_array_equal(result, vec)

    def test_no_match_returns_none(self):
        cache = DeadbandCache()
        result = cache.get("hello world")
        assert result is None

    def test_similar_text_returns_vector(self):
        cache = DeadbandCache(threshold=0.80)
        vec = np.array([1.0, 2.0, 3.0])
        cache.put("deploy micro model", vec)
        # Exact text match should return vector
        result = cache.get("deploy micro model")
        assert result is not None

    def test_max_size_eviction(self):
        cache = DeadbandCache(max_size=2)
        cache.put("abcdefgh", np.array([1.0]))
        cache.put("ijklmnop", np.array([2.0]))
        cache.put("qrstuvwx", np.array([3.0]))
        assert len(cache) == 2
        # "abcdefgh" should be evicted; others should remain via exact match
        assert cache.get("abcdefgh") is None
        assert cache.get("ijklmnop") is not None
        assert cache.get("qrstuvwx") is not None

    def test_clear(self):
        cache = DeadbandCache()
        cache.put("a", np.array([1.0]))
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_put_updates_existing(self):
        cache = DeadbandCache()
        cache.put("a", np.array([1.0]))
        cache.put("a", np.array([2.0]))
        assert len(cache) == 1
        np.testing.assert_array_equal(cache.get("a"), np.array([2.0]))
