/** Strip common English suffixes to get a morphological stem. */
export declare function stemWord(word: string): string;
/**
 * Compute a 64-bit fingerprint for a single word.
 *
 * Uses FNV-1a 64-bit over character unigrams and bigrams to set bits in a
 * 64-bit integer.  Deterministic and cross-language compatible (matches
 * Python version exactly).
 */
export declare function wordFingerprint(word: string): bigint;
/**
 * Compute a combined 64-bit fingerprint for a full text.
 *
 * Aggregates word fingerprints with XOR + left-rotate-by-1 on 64 bits.
 * Stopwords are filtered.  Matches Python implementation exactly.
 */
export declare function textFingerprint(text: string, useStemming?: boolean): bigint;
/** Count differing bits between two 64-bit fingerprints. */
export declare function hammingDistance(a: bigint, b: bigint): number;
/** Normalized similarity between two 64-bit fingerprints [0, 1]. */
export declare function bitvectorSimilarity(a: bigint, b: bigint): number;
export declare class DeadbandCache {
    threshold: number;
    maxSize: number;
    private entries;
    constructor(threshold?: number, maxSize?: number);
    get(text: string): MatchResult | null;
    set(text: string, result: MatchResult): void;
    clear(): void;
}
export declare class MatchResult {
    bestMatch: string | null;
    score: number;
    method: string;
    constructor(bestMatch?: string | null, score?: number, method?: string);
    toString(): string;
}
export declare class CascadeMatcher {
    bitvectorThreshold: number;
    deadbandCache?: DeadbandCache | undefined;
    constructor(bitvectorThreshold?: number, deadbandCache?: DeadbandCache | undefined);
    match(query: string, candidates: string[], useStemming?: boolean): MatchResult;
}
export declare class EisensteinModel {
    private knowledge;
    private candidateKeys;
    private cascade;
    private deadbandCache;
    useStemming: boolean;
    constructor(opts?: {
        bitvectorThreshold?: number;
        deadbandThreshold?: number;
        deadbandMaxSize?: number;
        useStemming?: boolean;
    });
    /** Add a knowledge entry (key → value). */
    addKnowledge(key: string, value: string): void;
    /** Match a query against known keys. */
    match(query: string, candidates?: string[]): MatchResult;
}
export declare const VERSION = "0.1.0";
