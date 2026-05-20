"use strict";
// eisenstein-embed — 5-layer matching cascade for JavaScript/TypeScript
// FNV-1a 64-bit fingerprints + Hamming distance bitvector matching
// Mirrors the Python implementation exactly for cross-language compatibility.
Object.defineProperty(exports, "__esModule", { value: true });
exports.VERSION = exports.EisensteinModel = exports.CascadeMatcher = exports.MatchResult = exports.DeadbandCache = void 0;
exports.stemWord = stemWord;
exports.wordFingerprint = wordFingerprint;
exports.textFingerprint = textFingerprint;
exports.hammingDistance = hammingDistance;
exports.bitvectorSimilarity = bitvectorSimilarity;
// --- FNV-1a 64-bit hash (BigInt arithmetic) ---
const FNV_64_OFFSET = 0xcbf29ce484222325n;
const FNV_64_PRIME = 0x100000001b3n;
const MASK_64 = 0xffffffffffffffffn;
function fnv1a64(data) {
    let h = FNV_64_OFFSET;
    for (let i = 0; i < data.length; i++) {
        h ^= BigInt(data.charCodeAt(i));
        h = (h * FNV_64_PRIME) & MASK_64;
    }
    return h;
}
// --- Stopwords (mirrors Python version) ---
const STOPWORDS = new Set([
    "what", "is", "the", "a", "an", "how", "does", "do", "tell", "me",
    "about", "it", "that", "this", "of", "for", "in", "on", "to", "and", "or",
]);
// --- Morphological suffixes (longest-first, mirrors Python) ---
const STEM_SUFFIXES = [
    "ization", "isation",
    "ation", "ition",
    "ment",
    "ness",
    "able", "ible",
    "ful",
    "less",
    "ous",
    "ive",
    "ing",
    "tion", "sion",
    "ity",
    "ize", "ise",
    "est",
    "ed",
    "er",
    "ly",
    "al",
];
const MIN_STEM_LEN = 3;
// --- Public API ---
/** Strip common English suffixes to get a morphological stem. */
function stemWord(word) {
    const w = word.toLowerCase();
    for (const suffix of STEM_SUFFIXES) {
        if (w.endsWith(suffix) && (w.length - suffix.length) >= MIN_STEM_LEN) {
            return w.slice(0, -suffix.length);
        }
    }
    return w;
}
/**
 * Compute a 64-bit fingerprint for a single word.
 *
 * Uses FNV-1a 64-bit over character unigrams and bigrams to set bits in a
 * 64-bit integer.  Deterministic and cross-language compatible (matches
 * Python version exactly).
 */
function wordFingerprint(word) {
    let fp = 0n;
    const w = word.toLowerCase();
    if (!w)
        return fp;
    // Set bits based on character n-grams (unigrams and bigrams)
    for (let i = 0; i < w.length; i++) {
        // Unigram hash
        const h1 = fnv1a64(w[i]);
        const bit1 = Number(h1 % 64n);
        fp |= 1n << BigInt(bit1);
        // Bigram hash
        if (i + 1 < w.length) {
            const bigram = w[i] + w[i + 1];
            const h2 = fnv1a64(bigram);
            const bit2 = Number(h2 % 64n);
            fp |= 1n << BigInt(bit2);
        }
    }
    // Add length-based bit for extra discrimination
    const lengthBit = (w.length * 7) % 64;
    fp |= 1n << BigInt(lengthBit);
    return fp & MASK_64;
}
/** Normalize text: lowercase, strip accents, collapse whitespace. */
function normalizeText(text) {
    return text
        .normalize("NFKD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .trim()
        .replace(/\s+/g, " ");
}
/** Tokenize text into words after normalization. */
function tokenize(text) {
    return normalizeText(text).split(/\s+/).filter(Boolean);
}
/**
 * Compute a combined 64-bit fingerprint for a full text.
 *
 * Aggregates word fingerprints with XOR + left-rotate-by-1 on 64 bits.
 * Stopwords are filtered.  Matches Python implementation exactly.
 */
function textFingerprint(text, useStemming = false) {
    const words = tokenize(text);
    if (words.length === 0)
        return 0n;
    let fp = 0n;
    for (const word of words) {
        const lower = word.toLowerCase();
        if (STOPWORDS.has(lower))
            continue;
        const lookup = useStemming ? stemWord(lower) : lower;
        const wfp = wordFingerprint(lookup);
        // Mix using XOR and rotation to avoid cancellation
        fp ^= wfp;
        // Rotate left by 1 on 64 bits
        fp = ((fp << 1n) | (fp >> 63n)) & MASK_64;
    }
    return fp & MASK_64;
}
/** Count differing bits between two 64-bit fingerprints. */
function hammingDistance(a, b) {
    let x = (a ^ b) & MASK_64;
    let count = 0;
    while (x) {
        x &= x - 1n;
        count++;
    }
    return count;
}
/** Normalized similarity between two 64-bit fingerprints [0, 1]. */
function bitvectorSimilarity(a, b) {
    return 1.0 - hammingDistance(a, b) / 64.0;
}
/** Find the best candidate by bitvector similarity. */
function findBestBitvectorMatch(query, candidates, useStemming = false) {
    if (candidates.length === 0)
        return [null, 0.0];
    const qfp = textFingerprint(query, useStemming);
    let best = null;
    let bestScore = -1.0;
    for (const c of candidates) {
        const cfp = textFingerprint(c, useStemming);
        const score = bitvectorSimilarity(qfp, cfp);
        if (score > bestScore) {
            bestScore = score;
            best = c;
        }
    }
    return [best, bestScore];
}
// --- DeadbandCache ---
class DeadbandCache {
    constructor(threshold = 0.90, maxSize = 1000) {
        this.threshold = threshold;
        this.maxSize = maxSize;
        this.entries = [];
    }
    get(text) {
        const norm = normalizeText(text);
        const fp = textFingerprint(norm);
        let bestResult = null;
        let bestSim = -1.0;
        for (const entry of this.entries) {
            if (norm === entry.text)
                return entry.result;
            const sim = bitvectorSimilarity(fp, entry.fp);
            if (sim > bestSim) {
                bestSim = sim;
                bestResult = entry.result;
            }
        }
        return bestSim >= this.threshold ? bestResult : null;
    }
    set(text, result) {
        const norm = normalizeText(text);
        const fp = textFingerprint(norm);
        for (let i = 0; i < this.entries.length; i++) {
            if (this.entries[i].text === norm) {
                this.entries[i] = { text: norm, fp, result };
                return;
            }
        }
        this.entries.push({ text: norm, fp, result });
        if (this.entries.length > this.maxSize) {
            this.entries.shift();
        }
    }
    clear() {
        this.entries = [];
    }
}
exports.DeadbandCache = DeadbandCache;
// --- MatchResult ---
class MatchResult {
    constructor(bestMatch = null, score = 0.0, method = "none") {
        this.bestMatch = bestMatch;
        this.score = score;
        this.method = method;
    }
    toString() {
        return `MatchResult(bestMatch=${JSON.stringify(this.bestMatch)}, score=${this.score.toFixed(3)}, method=${JSON.stringify(this.method)})`;
    }
}
exports.MatchResult = MatchResult;
// --- CascadeMatcher ---
class CascadeMatcher {
    constructor(bitvectorThreshold = 0.85, deadbandCache) {
        this.bitvectorThreshold = bitvectorThreshold;
        this.deadbandCache = deadbandCache;
    }
    match(query, candidates, useStemming = false) {
        if (candidates.length === 0)
            return new MatchResult(null, 0.0, "none");
        const normQuery = normalizeText(query);
        if (!normQuery)
            return new MatchResult(null, 0.0, "none");
        // 1. EXACT
        for (const c of candidates) {
            if (normalizeText(c) === normQuery) {
                return new MatchResult(c, 1.0, "exact");
            }
        }
        // 2. BITVECTOR
        const [bvCand, bvScore] = findBestBitvectorMatch(query, candidates, useStemming);
        if (bvCand !== null && bvScore >= this.bitvectorThreshold) {
            return new MatchResult(bvCand, bvScore, "bitvector");
        }
        // 3. DEADBAND (cache lookup)
        if (this.deadbandCache) {
            const cached = this.deadbandCache.get(query);
            if (cached)
                return cached;
        }
        // 4. SEMANTIC (stub — no ML model in JS, fall through)
        // 5. DOMAIN (stub — fall through)
        // Fallback to best bitvector match
        if (bvCand !== null) {
            return new MatchResult(bvCand, bvScore, "bitvector");
        }
        return new MatchResult(null, 0.0, "none");
    }
}
exports.CascadeMatcher = CascadeMatcher;
// --- EisensteinModel ---
class EisensteinModel {
    constructor(opts = {}) {
        this.knowledge = new Map();
        this.candidateKeys = [];
        this.useStemming = opts.useStemming ?? false;
        this.deadbandCache = new DeadbandCache(opts.deadbandThreshold ?? 0.90, opts.deadbandMaxSize ?? 1000);
        this.cascade = new CascadeMatcher(opts.bitvectorThreshold ?? 0.85, this.deadbandCache);
    }
    /** Add a knowledge entry (key → value). */
    addKnowledge(key, value) {
        this.knowledge.set(normalizeText(key), value);
        this.candidateKeys = Array.from(this.knowledge.keys());
    }
    /** Match a query against known keys. */
    match(query, candidates) {
        const cands = candidates ?? this.candidateKeys;
        const result = this.cascade.match(query, cands, this.useStemming);
        // If matched against knowledge keys, resolve value
        if (result.bestMatch !== null && candidates === undefined) {
            const value = this.knowledge.get(result.bestMatch);
            if (value !== undefined) {
                return new MatchResult(value, result.score, result.method);
            }
        }
        return result;
    }
}
exports.EisensteinModel = EisensteinModel;
// Version
exports.VERSION = "0.1.0";
