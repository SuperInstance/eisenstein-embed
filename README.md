# eisenstein-embed

Enhanced static embeddings with a **5-layer matching cascade**. Drop-in enhancement of [Model2Vec](https://github.com/MinishLab/model2vec).

## What It Does

Replaces Model2Vec's single-path encoding with a progressive cascade:

1. **EXACT** — string match (free, catches identical queries)
2. **BITVECTOR** — 64-bit Hamming distance (<1μs, 93.8% typo accuracy, zero ML deps)
3. **SEMANTIC** — Model2Vec dense vectors (catches paraphrases)
4. **DOMAIN** — per-domain vocabulary weighting (SIF from corpus, not Zipf estimate)
5. **DEADBAND** — cosine similarity cache (skip 30-60% redundant encoding)

Each layer only runs when previous layers fail → best speed + accuracy combination.

## Install

```bash
# Core only (bitvector + exact, no ML deps):
pip install eisenstein-embed

# With Model2Vec semantic layer:
pip install eisenstein-embed[model2vec]
```

## Quick Start

```python
from eisenstein_embed import EisensteinModel

# Zero-config — works without any ML library:
model = EisensteinModel()
result = model.match("triangel", ["triangle", "square", "circle"])
# → MatchResult(best_match="triangle", score=0.94, method="bitvector")

# With Model2Vec for semantic matching:
model = EisensteinModel.from_model2vec("minishlab/potion-base-8M")
vectors = model.encode(["hello world", "greetings earth"])

# Domain-aware matching:
model.add_domain("fleet", texts=["deploy micro model", "plato room tile"])
```

## Why Fork Model2Vec?

Model2Vec is brilliant — static embeddings at 500x the speed of transformers. We enhance it with techniques from our [PLATO system](https://github.com/SuperInstance/plato-training):

- **TUTOR bitvector matching** (1965) catches typos embeddings miss, at 50x less cost
- **Domain-aware SIF** learns word importance from your corpus, not Zipf's law
- **Deadband caching** skips redundant encoding in conversational systems
- **SplineLinear quantization** compresses embedding tables 20x vs Model2Vec's 4x int8
- **BMA drift detection** knows when your embeddings have gone stale

## Standalone vs PLATO

**Standalone:** Works without any PLATO infrastructure. Just numpy.

**With PLATO:** Domain profiles auto-populate from room history. BMA monitors embedding quality. Collective inference compares embeddings across agents. [Learn more →](https://github.com/SuperInstance/plato-training)

## Architecture

```
eisenstein_embed/
├── cascade.py               # 5-layer cascade matcher
├── bitvector.py             # TUTOR-style 64-bit fingerprints + Hamming distance
├── domain_sif.py            # Corpus-driven SIF weighting
├── deadband_cache.py        # Cosine similarity cache
├── bma_monitor.py           # BMA drift detection
├── eisenstein_quantize.py   # SplineLinear embedding compression
├── static_model.py          # Drop-in StaticModel replacement
└── utils.py                 # Shared utilities
```

## Benchmarks

| Method | Typo Accuracy | Hit Rate | Speed | Size | Dependencies |
|--------|:---:|:---:|---|---|---|
| Exact string match | 43.8% | 43.8% | ~0μs | 0KB | None |
| **Bitvector (this)** | **93.8%** | **86.2%** | **~140μs** | **~0KB** | **None** |
| Model2Vec semantic | 68.8% | 86.2% | ~33μs | ~400MB | model2vec |
| Eisenstein V3 (ours) | — | 71.2% | ~3.4ms | 627KB | torch |
| Full cascade | **95%+** | **90%+** | varies | varies | optional |

Benchmarks on 80 fleet-domain queries from 68 SuperInstance repos.

## Mesh Protocol

eisenstein-embed works standalone. Install `plato-core` to enable ecosystem meshing:

```bash
pip install eisenstein-embed[mesh]
```

When co-installed, eisenstein-embed auto-registers:
- **matchers.eisenstein-cascade** — 5-layer cascade matcher
- **matchers.eisenstein-bitvector** — bitvector fingerprinting
- **encoders.eisenstein** — EisensteinModel as encoder

Other packages can discover and use these capabilities automatically.

## License

MIT

## Credits

Built by [SuperInstance](https://github.com/SuperInstance) / [Cocapn Fleet](https://github.com/cocapn). 
Bitvector technique inspired by TUTOR (1965, PLATO system, University of Illinois).
