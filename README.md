# eisenstein-embed

Eisenstein integer embeddings with a 5-layer matching cascade — word fingerprints, bitvector search, deadband caching, and domain-SIF for text similarity.

## What This Gives You

- **5-layer cascade matcher** — bitvector → deadband cache → Eisenstein quantization → domain SIF → BMA monitor
- **Word fingerprints** — hash words to Eisenstein lattice coordinates for fast similarity
- **Bitvector search** — Hamming distance similarity with inverted index acceleration
- **Deadband cache** — memoize lookups within deadband tolerance (lattice-aware caching)
- **Domain SIF** — Smooth Inverse Frequency embeddings adapted for Eisenstein space
- **BMA monitor** — Best Match Average tracking for quality control
- **8 tests** — verified cascade layers and end-to-end matching

## Quick Start

```python
from eisenstein_embed import CascadeMatcher, text_fingerprint

# Build a cascade matcher with documents
matcher = CascadeMatcher()
matcher.add("constraint lattice quantization")
matcher.add("harmonic ring pitch class")
matcher.add("distributed consensus protocol")

# Query
results = matcher.query("lattice snap constraint")
for r in results:
    print(f"[{r.score:.3f}] {r.text}")
```

### Individual Layers

```python
from eisenstein_embed import word_fingerprint, hamming_distance, DeadbandCache

# Word → Eisenstein fingerprint
fp = word_fingerprint("lattice")
print(f"Fingerprint: {fp}")

# Hamming similarity
fp2 = word_fingerprint("grid")
dist = hamming_distance(fp, fp2)
print(f"Distance: {dist}")

# Deadband cache
cache = DeadbandCache(tolerance=0.05)
cache.put("lattice", 0.73)
result = cache.get("grid", 0.71)  # within deadband → cache hit
```

## API Reference

| Layer | Types | Description |
|---|---|---|
| 1. Bitvector | `word_fingerprint`, `text_fingerprint`, `hamming_distance` | Fast approximate matching |
| 2. Deadband Cache | `DeadbandCache` | Lattice-aware memoization |
| 3. Quantization | `SplineLinearQuantizer` | Eisenstein-space quantization |
| 4. Domain SIF | `DomainSIF` | Smooth Inverse Frequency embeddings |
| 5. BMA Monitor | `BMAMonitor` | Quality tracking |
| Cascade | `CascadeMatcher` | All 5 layers in sequence |

## How It Fits

The **semantic search layer** using Eisenstein lattice geometry:

- [eisenstein-triples](https://github.com/SuperInstance/eisenstein-triples) — Eisenstein integer theory
- [eisenstein-vs-z2-rs](https://github.com/SuperInstance/eisenstein-vs-z2-rs) — lattice comparison benchmarks
- [flux-index](https://github.com/SuperInstance/flux-index) — code search using these embeddings
- [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) — lattice snap operations

## Testing

```bash
pip install -e ".[dev]"
pytest -v  # 8 test files
```

## Installation

```bash
pip install eisenstein-embed
```

Requires Python ≥ 3.10.

## License

MIT
