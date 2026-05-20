"""TUTOR-style 64-bit word fingerprints + Hamming distance."""

from typing import List


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


def text_fingerprint(text: str) -> int:
    """Compute a 64-bit fingerprint for a full text.

    Aggregates word fingerprints with a simple XOR/mix.
    """
    from eisenstein_embed.utils import tokenize

    words = tokenize(text)
    if not words:
        return 0

    fp = 0
    for word in words:
        wfp = word_fingerprint(word)
        # Mix using XOR and rotation to avoid cancellation
        fp ^= wfp
        fp = ((fp << 1) | (fp >> 63)) & 0xFFFFFFFFFFFFFFFF

    return fp


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 64-bit fingerprints."""
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


def find_best_bitvector_match(query: str, candidates: List[str]) -> tuple:
    """Find the best candidate match using bitvector fingerprints.

    Returns (best_candidate, similarity_score) or (None, 0.0) if no candidates.
    """
    if not candidates:
        return None, 0.0

    query_fp = text_fingerprint(query)
    best_candidate = None
    best_score = -1.0

    for cand in candidates:
        cand_fp = text_fingerprint(cand)
        score = bitvector_similarity(query_fp, cand_fp)
        if score > best_score:
            best_score = score
            best_candidate = cand

    return best_candidate, best_score
