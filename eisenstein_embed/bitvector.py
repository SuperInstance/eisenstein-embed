"""TUTOR-style 64-bit word fingerprints + Hamming distance.

Delegates to plato-training's tutor_judge when available for the
canonical TUTOR bitvector implementation. Falls back to a local
implementation using rolling hash over character bigrams.

Supports morphological stem hashing (V3 breakthrough) — stripping
common suffixes so related forms (deploy/deployment/deploying) share
the same fingerprint.
"""

import re
from typing import List, FrozenSet

# Common English stopwords that inflate bitvector similarity
_STOPWORDS: FrozenSet[str] = frozenset({
    "what", "is", "the", "a", "an", "how", "does", "do", "tell", "me",
    "about", "it", "that", "this", "of", "for", "in", "on", "to", "and", "or",
})

# Morphological suffixes ordered longest-first so greedy stripping works
_STEM_SUFFIXES: List[str] = [
    "ization", "isation",  # -ize → -ization
    "ation", "ition",      # -ate → -ation
    "ment",                # state → statement
    "ness",                # happy → happiness
    "able", "ible",        # read → readable
    "ful",                 # use → useful
    "less",                # use → useless
    "ous",                 # danger → dangerous
    "ive",                 # act → active
    "ing",                 # run → running
    "tion", "sion",        # act → action
    "ity",                 # able → ability
    "ize", "ise",          # modern → modernize
    "est",                 # big → biggest
    "ed",                  # walk → walked
    "er",                  # run → runner
    "ly",                  # quick → quickly
    "al",                  # form → formal
]

# Minimum stem length to avoid stripping too aggressively
_MIN_STEM_LEN = 3


def stem_word(word: str) -> str:
    """Strip common English suffixes to get a morphological stem.

    This is intentionally simple — not a full Porter stemmer — just enough
    to collapse common morphological variants (deploy/deployment/deploying)
    into the same stem so they produce identical fingerprints.

    Args:
        word: Lowercase word to stem.

    Returns:
        The stemmed form (at least _MIN_STEM_LEN chars).
    """
    word = word.lower()
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM_LEN:
            return word[: -len(suffix)]
    return word

# Try importing plato-training's tutor_judge for canonical bitvector logic
try:
    from plato_training.tutor_judge import (
        word_to_bitvector as _plato_word_to_bitvector,
        hamming_distance as _plato_hamming_distance,
        word_similarity as _plato_word_similarity,
    )
    HAS_PLATO_TUTOR = True
except ImportError:
    HAS_PLATO_TUTOR = False


def word_fingerprint(word: str) -> int:
    """Compute a 64-bit fingerprint for a single word.

    Uses a simple rolling hash over character bigrams to set bits
    in a 64-bit integer. This is fast and captures local character
    structure, making it robust to small typos.
    """
    fp = 0
    word = word.lower()
    if not word:
        return fp

    # Set bits based on character n-grams (unigrams and bigrams)
    for i, ch in enumerate(word):
        # Unigram hash
        h = hash(ch) & 0xFFFFFFFFFFFFFFFF
        bit = h % 64
        fp |= 1 << bit

        # Bigram hash
        if i + 1 < len(word):
            bigram = ch + word[i + 1]
            h = hash(bigram) & 0xFFFFFFFFFFFFFFFF
            bit = h % 64
            fp |= 1 << bit

    # Add length-based bit for extra discrimination
    length_bit = (len(word) * 7) % 64
    fp |= 1 << length_bit

    return fp


def text_fingerprint(text: str, use_stemming: bool = False) -> int:
    """Compute a 64-bit fingerprint for a full text.

    Aggregates word fingerprints with a simple XOR/mix.
    Stopwords are filtered so common function words don't inflate
    similarity scores.

    Args:
        text: Input text.
        use_stemming: If True, apply morphological stem hashing so that
            related word forms (deploy/deployment/deploying) produce the
            same fingerprint. Default False for backward compatibility.
    """
    from eisenstein_embed.utils import tokenize

    words = tokenize(text)
    if not words:
        return 0

    fp = 0
    for word in words:
        word_lower = word.lower()
        if word_lower in _STOPWORDS:
            continue
        lookup = stem_word(word_lower) if use_stemming else word_lower
        wfp = word_fingerprint(lookup)
        # Mix using XOR and rotation to avoid cancellation
        fp ^= wfp
        fp = ((fp << 1) | (fp >> 63)) & 0xFFFFFFFFFFFFFFFF

    return fp


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 64-bit fingerprints."""
    if HAS_PLATO_TUTOR:
        return _plato_hamming_distance(a, b)
    x = a ^ b
    # Brian Kernighan's algorithm for popcount
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


def bitvector_similarity(a: int, b: int) -> float:
    """Normalized similarity between two 64-bit fingerprints.

    Returns a value in [0, 1] where 1 means identical.
    """
    dist = hamming_distance(a, b)
    # Using a sigmoid-like decay for softer matching
    return 1.0 - (dist / 64.0)


def find_best_bitvector_match(query: str, candidates: List[str], use_stemming: bool = False) -> tuple:
    """Find the best candidate match using bitvector fingerprints.

    Args:
        query: Query text.
        candidates: List of candidate texts.
        use_stemming: If True, use stem hashing for matching.

    Returns (best_candidate, similarity_score) or (None, 0.0) if no candidates.
    """
    if not candidates:
        return None, 0.0

    query_fp = text_fingerprint(query, use_stemming=use_stemming)
    best_candidate = None
    best_score = -1.0

    for cand in candidates:
        cand_fp = text_fingerprint(cand, use_stemming=use_stemming)
        score = bitvector_similarity(query_fp, cand_fp)
        if score > best_score:
            best_score = score
            best_candidate = cand

    return best_candidate, best_score
