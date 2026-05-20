"""Test Unicode support across all matching layers."""
import pytest
from eisenstein_embed import EisensteinModel, MatchResult
from eisenstein_embed.bitvector import word_fingerprint, text_fingerprint, bitvector_similarity
from eisenstein_embed.utils import normalize_text, tokenize


class TestUnicodeNormalize:
    def test_cjk_preserved(self):
        result = normalize_text("日本語テスト")
        assert result == "日本語テスト"

    def test_cyrillic_preserved(self):
        result = normalize_text("Привет мир")
        assert "Привет" in result
        assert "мир" in result

    def test_korean_preserved(self):
        result = normalize_text("한국어 테스트")
        assert "한국어" in result

    def test_accents_stripped(self):
        """Combining marks removed but base chars kept."""
        result = normalize_text("café résumé")
        assert result == "cafe resume"

    def test_mixed_unicode_ascii(self):
        result = normalize_text("Python编程语言")
        assert "python" in result
        assert "编程语言" in result

    def test_tokenize_cjk(self):
        tokens = tokenize("日本語 テスト")
        assert len(tokens) >= 1

    def test_empty_after_strip(self):
        """Pure combining marks should produce empty."""
        result = normalize_text("\u0301\u0302")  # combining acute + circumflex
        assert result == ""


class TestUnicodeBitvector:
    def test_cjk_fingerprint_nonzero(self):
        fp = text_fingerprint("日本語")
        assert fp != 0

    def test_cjk_same_word_same_fingerprint(self):
        assert word_fingerprint("中文") == word_fingerprint("中文")

    def test_cjk_similarity_exact(self):
        fp = text_fingerprint("日本語")
        assert bitvector_similarity(fp, fp) == 1.0


class TestUnicodeMatching:
    def test_exact_cjk_match(self):
        model = EisensteinModel()
        result = model.match("日本語", ["日本語"])
        assert result.best_match == "日本語"
        assert result.score == 1.0

    def test_cjk_distinguish(self):
        model = EisensteinModel()
        result = model.match("日本語", ["日本語", "中文", "한국어"])
        assert result.best_match == "日本語"

    def test_mixed_unicode_ascii(self):
        model = EisensteinModel()
        result = model.match("Python编程", ["Python编程", "Python programming"])
        assert result.best_match == "Python编程"

    def test_match_all_unicode(self):
        model = EisensteinModel()
        results = model.match_all("日本語", ["日本語", "中文", "hello"])
        assert len(results) == 3
        assert results[0].best_match == "日本語"
        assert results[0].score >= results[1].score
