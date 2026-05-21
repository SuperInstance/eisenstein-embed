# eisenstein-embed

## Why hexagonal?

If you need to place points evenly across a 2D surface, you have two choices: a rectangular grid or a hexagonal grid. The hexagonal grid wins. Here's why.

On a rectangular grid, each interior point has 4 equidistant neighbors at distance 1. But the diagonal neighbors are at distance √2 ≈ 1.414. There's a gap — the diagonal is 41% farther than the axial distance. You're wasting space.

On a hexagonal grid, every interior point has 6 equidistant neighbors, all at the same distance. No gaps. No wasted space. This isn't an aesthetic preference — Thue proved in 1910 that the hexagonal arrangement is the densest possible packing of equal circles in 2D. No arrangement, no matter how clever, can do better.

The hexagonal grid is the Eisenstein lattice. Points live at positions `a + bω` where `ω = e^(2πi/3)` and `a, b` are integers.

## The Eisenstein norm

For a point `z = a + bω` on the Eisenstein lattice, the squared distance from the origin is:

```
|z|² = a² + ab + b²
```

Let's work through an example by hand. Take the point `z = 3 + ω` (so `a = 3, b = 1`):

```
|z|² = 3² + 3×1 + 1² = 9 + 3 + 1 = 13
```

Distance from origin: √13 ≈ 3.606.

Now take `z = 2 + 2ω`:

```
|z|² = 2² + 2×2 + 2² = 4 + 4 + 4 = 12
```

Distance from origin: √12 ≈ 3.464.

Every Eisenstein integer has a unique factorization into primes — same as the regular integers, but in a hexagonal world. The norm is always a non-negative integer, which means all the arithmetic stays exact. No floating point.

## What's actually happening?

`eisenstein-embed` uses the Eisenstein lattice as the backbone for static text embeddings. Words get mapped to lattice positions, and the hexagonal structure gives you two things:

1. **Dense packing** — more unique embeddings per unit of storage than a rectangular grid
2. **Exact arithmetic** — the Eisenstein norm is always an integer, so distance computations never drift

The library then layers a 5-level cascade matcher on top: bitvector fingerprints first (fastest), then deadband caching, then full semantic matching. Each level short-circuits if it finds a confident match, so you only pay for the compute you need.

## Install and use

```bash
pip install eisenstein-embed
```

```python
from eisenstein_embed import EisensteinModel, word_fingerprint, hamming_distance

# Create a model (works without a semantic model — bitvector-only mode)
model = EisensteinModel()

# Bitvector fingerprints for fast matching
fp_hello = word_fingerprint("hello")
fp_world = word_fingerprint("world")
print(hamming_distance(fp_hello, fp_world))  # integer distance

# Full cascade matching
model.encode(["hello", "world", "greetings"])
match = model.match("hi")
print(match.best_match, match.score, match.method)
```

## The cascade

Level 1: **Bitvector fingerprints** — hash each word to a compact bit pattern. Hamming distance between patterns gives a fast similarity score. Cost: O(1) per comparison.

Level 2: **Deadband cache** — if you've seen a nearly-identical query before (within a deadband threshold), return the cached result. Cost: O(1) on cache hit.

Level 3: **Domain SIF** — sentence-level encoding tuned to the current domain. Cost: O(n) where n = words in query.

Level 4: **Full semantic model** (optional) — if a Model2Vec model is loaded, fall back to dense embeddings. Cost: O(n × d) where d = embedding dimension.

Level 5: **BMA monitoring** — track match quality over time, alert if accuracy degrades.

## SplineLinear compression

The `SplineLinearQuantizer` compresses embedding vectors by dividing each vector into segments, fitting a linear function (slope + intercept) per segment, and quantizing to 8-bit integers. Typical compression: ~20× versus raw float32 at modest quality loss.

```python
from eisenstein_embed import SplineLinearQuantizer
import numpy as np

q = SplineLinearQuantizer(n_segments=4, bits_per_value=8)
vectors = np.random.randn(1000, 256).astype(np.float32)
compressed = q.compress(vectors)
restored = q.decompress(compressed)
print(q.compression_ratio(vectors, compressed))  # ~20×
```

## Key modules

| Module | What it does |
|--------|-------------|
| `static_model` | Drop-in `Model2Vec` replacement with cascade matching |
| `bitvector` | Word/text fingerprinting and Hamming distance |
| `deadband_cache` | Near-duplicate query caching |
| `domain_sif` | Domain-tuned sentence encoding |
| `cascade` | 5-level matching cascade |
| `eisenstein_quantize` | Spline-linear embedding compression |
| `bma_monitor` | Batch match accuracy tracking |

## Why does this work?

The Eisenstein lattice gives you the densest possible packing in 2D (Thue, 1910). Densest packing means more distinct positions per unit area, which means more unique embeddings per byte. The integer norm means exact arithmetic — no floating-point drift in distance computations. The cascade matcher exploits a simple fact: most matches are obvious (level 1 catches them), so you only run the expensive models on the hard cases.

## License

MIT
