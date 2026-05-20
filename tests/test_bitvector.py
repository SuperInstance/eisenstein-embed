"""Tests for eisenstein-embed bitvector module."""
import pytest
from eisenstein_embed.bitvector import (
    word_fingerprint,
    text_fingerprint,
    stem_word,
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


class TestStemWord:
    """Tests for morphological stem hashing."""

    def test_deploy_variants_share_stem(self):
        """deploy, deployment, deploying should all stem to 'deploy'."""
        assert stem_word("deploy") == stem_word("deployment")
        assert stem_word("deploy") == stem_word("deploying")
        assert stem_word("deploy") == "deploy"

    def test_strips_ing(self):
        assert stem_word("running") == "runn"
        assert stem_word("walking") == "walk"

    def test_strips_tion(self):
        assert stem_word("detection") == "detec"
        assert stem_word("collection") == "collec"

    def test_strips_ment(self):
        assert stem_word("deployment") == "deploy"
        assert stem_word("statement") == "state"

    def test_strips_ness(self):
        assert stem_word("happiness") == "happi"  # drops 'ness', leaves 'happi'

    def test_strips_ed(self):
        assert stem_word("walked") == "walk"

    def test_strips_er(self):
        assert stem_word("runner") == "runn"

    def test_strips_ly(self):
        assert stem_word("quickly") == "quick"

    def test_strips_able(self):
        assert stem_word("readable") == "read"

    def test_strips_ous(self):
        assert stem_word("dangerous") == "danger"

    def test_strips_ive(self):
        assert stem_word("active") == "act"

    def test_strips_ful(self):
        assert stem_word("useful") == "use"

    def test_strips_less(self):
        assert stem_word("useless") == "use"

    def test_strips_ity(self):
        assert stem_word("ability") == "abil"

    def test_strips_ize(self):
        assert stem_word("modernize") == "modern"

    def test_strips_al(self):
        assert stem_word("formal") == "form"

    def test_no_strip_if_too_short(self):
        """Words that would be shorter than 3 chars after stripping should be left alone."""
        assert stem_word("red") == "red"  # stripping 'ed' leaves 'r' (too short)
        assert stem_word("able") == "able"  # stripping 'able' leaves '' (too short)

    def test_case_insensitive(self):
        assert stem_word("Deploying") == stem_word("deploying")

    def test_unstemmed_word_unchanged(self):
        assert stem_word("cat") == "cat"
        assert stem_word("house") == "house"


class TestStemHashingFingerprint:
    """Tests for stem hashing integrated into fingerprints."""

    def test_stemmed_variants_same_fingerprint(self):
        """deploy/deployment/deploying should produce the same fingerprint with stemming on."""
        fp1 = text_fingerprint("deploy the model", use_stemming=True)
        fp2 = text_fingerprint("deployment of model", use_stemming=True)
        fp3 = text_fingerprint("deploying the model", use_stemming=True)
        assert fp1 == fp2 == fp3

    def test_backward_compatible_no_stemming(self):
        """Without stemming, variants should produce different fingerprints."""
        fp1 = text_fingerprint("deploy the model")
        fp2 = text_fingerprint("deployment of model")
        # They're different words, so fingerprints should differ
        assert fp1 != fp2

    def test_default_no_stemming(self):
        """Default behavior should be no stemming (backward compatible)."""
        fp_default = text_fingerprint("deploying")
        fp_explicit_off = text_fingerprint("deploying", use_stemming=False)
        assert fp_default == fp_explicit_off

    def test_stemming_improves_detection_match(self):
        """Stem hashing should improve score for morphological variants."""
        candidates = ["deploy fleet nodes", "check weather", "bake bread"]
        # With stemming, score for the correct candidate should be higher
        from eisenstein_embed.bitvector import text_fingerprint as tfp
        query_stemmed = tfp("deploying to fleet", use_stemming=True)
        cand_stemmed = tfp("deploy fleet nodes", use_stemming=True)
        query_plain = tfp("deploying to fleet", use_stemming=False)
        cand_plain = tfp("deploy fleet nodes", use_stemming=False)
        sim_stemmed = bitvector_similarity(query_stemmed, cand_stemmed)
        sim_plain = bitvector_similarity(query_plain, cand_plain)
        assert sim_stemmed >= sim_plain


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
