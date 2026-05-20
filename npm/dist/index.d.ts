/** Strip common English suffixes to get a morphological stem. */
export declare function stemWord(word: string): string;
/** Compute a 32-bit FNV-1a fingerprint for a single word. */
export declare function wordFingerprint(word: string): number;
/** Compute a combined 32-bit fingerprint for a full text. */
export declare function textFingerprint(text: string, useStemming?: boolean): number;
/** Count differing bits between two 32-bit fingerprints. */
export declare function hammingDistance(a: number, b: number): number;
/** Normalized similarity between two 32-bit fingerprints [0, 1]. */
export declare function bitvectorSimilarity(a: number, b: number): number;
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
