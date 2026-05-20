"""Tests for eisenstein-embed bitvector module."""
import pytest
from eisenstein_embed.bitvector import (
    word_fingerprint,
    text_fingerprint,
    hamming_distance,
    bitvector_similarity,
    find_best_bitvector_match,
)


class TestWordFingerprint:
    def test_same_word_identical(self):
        assert word_fingerprint("hello") == word_fingerprint("hello")

    def test_case_insensitive(self):
        assert word_fingerprint("Hello") == word_fingerprint("hello")

    def test_empty_string(self):
        assert word_fingerprint("") == 0

    def test_nonzero_for_word(self):
        assert word_fingerprint("test") != 0

    def test_different_words_different(self):
        assert word_fingerprint("cat") != word_fingerprint("dog")


class TestTextFingerprint:
    def test_same_text_identical(self):
        assert text_fingerprint("hello world") == text_fingerprint("hello world")

    def test_case_insensitive(self):
        assert text_fingerprint("Hello World") == text_fingerprint("hello world")

    def test_empty(self):
        assert text_fingerprint("") == 0


class TestHammingDistance:
    def test_identical_zero(self):
        fp = word_fingerprint("test")
        assert hamming_distance(fp, fp) == 0

    def test_different_nonzero(self):
        assert hamming_distance(word_fingerprint("cat"), word_fingerprint("dog")) > 0

    def test_max_distance(self):
        assert hamming_distance(0xFFFFFFFFFFFFFFFF, 0) == 64


class TestBitvectorSimilarity:
    def test_identical_one(self):
        fp = word_fingerprint("test")
        assert bitvector_similarity(fp, fp) == 1.0

    def test_typo_high_similarity(self):
        sim = bitvector_similarity(
            text_fingerprint("triangle"),
            text_fingerprint("triangel"),
        )
        assert sim > 0.5, f"Expected >0.5, got {sim}"

    def test_different_lower_similarity(self):
        sim = bitvector_similarity(
            text_fingerprint("cat"),
            text_fingerprint("automobile"),
        )
        assert sim < 0.95, f"Expected <0.95, got {sim}"


class TestFindBestMatch:
    def test_exact_match(self):
        best, score = find_best_bitvector_match("triangle", ["square", "triangle", "circle"])
        assert best == "triangle"
        assert score == 1.0

    def test_typo_match(self):
        best, score = find_best_bitvector_match("triangel", ["square", "triangle", "circle"])
        assert best == "triangle"

    def test_empty_candidates(self):
        best, score = find_best_bitvector_match("test", [])
        assert best is None
        assert score == 0.0


class TestBenchmark:
    def test_speed_10k_comparisons(self):
        import time
        fp1 = text_fingerprint("hello world this is a test")
        fp2 = text_fingerprint("helo world this is a tst")
        start = time.time()
        for _ in range(10000):
            bitvector_similarity(fp1, fp2)
        elapsed = time.time() - start
        assert elapsed < 0.1, f"10K comparisons took {elapsed:.3f}s (expected <0.1s)"
