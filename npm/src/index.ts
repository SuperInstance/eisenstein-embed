// eisenstein-embed — 5-layer matching cascade for JavaScript/TypeScript
// FNV-1a 64-bit fingerprints + Hamming distance bitvector matching
// Mirrors the Python implementation exactly for cross-language compatibility.

// --- FNV-1a 64-bit hash (BigInt arithmetic) ---
const FNV_64_OFFSET = 0xCBF29CE484222325n;
const FNV_64_PRIME  = 0x100000001B3n;
const MASK_64       = 0xFFFFFFFFFFFFFFFFn;

function fnv1a64(data: string): bigint {
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
export function stemWord(word: string): string {
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
export function wordFingerprint(word: string): bigint {
  let fp = 0n;
  const w = word.toLowerCase();
  if (!w) return fp;

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
function normalizeText(text: string): string {
  return text
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim()
    .replace(/\s+/g, " ");
}

/** Tokenize text into words after normalization. */
function tokenize(text: string): string[] {
  return normalizeText(text).split(/\s+/).filter(Boolean);
}

/**
 * Compute a combined 64-bit fingerprint for a full text.
 *
 * Aggregates word fingerprints with XOR + left-rotate-by-1 on 64 bits.
 * Stopwords are filtered.  Matches Python implementation exactly.
 */
export function textFingerprint(text: string, useStemming: boolean = false): bigint {
  const words = tokenize(text);
  if (words.length === 0) return 0n;

  let fp = 0n;
  for (const word of words) {
    const lower = word.toLowerCase();
    if (STOPWORDS.has(lower)) continue;
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
export function hammingDistance(a: bigint, b: bigint): number {
  let x = (a ^ b) & MASK_64;
  let count = 0;
  while (x) {
    x &= x - 1n;
    count++;
  }
  return count;
}

/** Normalized similarity between two 64-bit fingerprints [0, 1]. */
export function bitvectorSimilarity(a: bigint, b: bigint): number {
  return 1.0 - hammingDistance(a, b) / 64.0;
}

/** Find the best candidate by bitvector similarity. */
function findBestBitvectorMatch(
  query: string,
  candidates: string[],
  useStemming: boolean = false,
): [string | null, number] {
  if (candidates.length === 0) return [null, 0.0];
  const qfp = textFingerprint(query, useStemming);
  let best: string | null = null;
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

export class DeadbandCache {
  private entries: Array<{ text: string; fp: bigint; result: MatchResult }> = [];

  constructor(
    public threshold: number = 0.90,
    public maxSize: number = 1000,
  ) {}

  get(text: string): MatchResult | null {
    const norm = normalizeText(text);
    const fp = textFingerprint(norm);
    let bestResult: MatchResult | null = null;
    let bestSim = -1.0;
    for (const entry of this.entries) {
      if (norm === entry.text) return entry.result;
      const sim = bitvectorSimilarity(fp, entry.fp);
      if (sim > bestSim) {
        bestSim = sim;
        bestResult = entry.result;
      }
    }
    return bestSim >= this.threshold ? bestResult : null;
  }

  set(text: string, result: MatchResult): void {
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

  clear(): void {
    this.entries = [];
  }
}

// --- MatchResult ---

export class MatchResult {
  constructor(
    public bestMatch: string | null = null,
    public score: number = 0.0,
    public method: string = "none",
  ) {}

  toString(): string {
    return `MatchResult(bestMatch=${JSON.stringify(this.bestMatch)}, score=${this.score.toFixed(3)}, method=${JSON.stringify(this.method)})`;
  }
}

// --- CascadeMatcher ---

export class CascadeMatcher {
  constructor(
    public bitvectorThreshold: number = 0.85,
    public deadbandCache?: DeadbandCache,
  ) {}

  match(
    query: string,
    candidates: string[],
    useStemming: boolean = false,
  ): MatchResult {
    if (candidates.length === 0) return new MatchResult(null, 0.0, "none");

    const normQuery = normalizeText(query);
    if (!normQuery) return new MatchResult(null, 0.0, "none");

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
      if (cached) return cached;
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

// --- EisensteinModel ---

export class EisensteinModel {
  private knowledge: Map<string, string> = new Map();
  private candidateKeys: string[] = [];
  private cascade: CascadeMatcher;
  private deadbandCache: DeadbandCache;
  public useStemming: boolean;

  constructor(
    opts: {
      bitvectorThreshold?: number;
      deadbandThreshold?: number;
      deadbandMaxSize?: number;
      useStemming?: boolean;
    } = {},
  ) {
    this.useStemming = opts.useStemming ?? false;
    this.deadbandCache = new DeadbandCache(
      opts.deadbandThreshold ?? 0.90,
      opts.deadbandMaxSize ?? 1000,
    );
    this.cascade = new CascadeMatcher(
      opts.bitvectorThreshold ?? 0.85,
      this.deadbandCache,
    );
  }

  /** Add a knowledge entry (key → value). */
  addKnowledge(key: string, value: string): void {
    this.knowledge.set(normalizeText(key), value);
    this.candidateKeys = Array.from(this.knowledge.keys());
  }

  /** Match a query against known keys. */
  match(query: string, candidates?: string[]): MatchResult {
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

// Version
export const VERSION = "0.1.0";
