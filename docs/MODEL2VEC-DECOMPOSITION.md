# Model2Vec Source Code Decomposition → PLATO Architecture Mapping

**Date:** 2026-05-20  
**Version:** model2vec 0.8.1  
**Author:** Forgemaster ⚒️ (subagent research)

---

## Table of Contents

1. [Source Code Architecture Overview](#1-source-code-architecture-overview)
2. [Annotated Source Code Listing](#2-annotated-source-code-listing)
3. [Extracted Primitives](#3-extracted-primitives)
4. [PLATO Architecture Mapping](#4-plato-architecture-mapping)
5. [Interface Contracts Between Primitives](#5-interface-contracts-between-primitives)
6. [Improvement Opportunities with PLATO](#6-improvement-opportunities-with-plato)
7. [Implementation Priority](#7-implementation-priority)

---

## 1. Source Code Architecture Overview

Model2Vec is organized into 6 modules:

```
model2vec/
├── model.py                    # StaticModel: core inference class
├── quantization.py             # Embedding dtype quantization (float16/int8)
├── vocabulary_quantization.py  # KMeans-based vocabulary clustering
├── utils.py                    # ProgressParallel, logging utilities
├── version.py                  # Version (0.8.1)
│
├── distill/                    # Teacher → Student distillation
│   ├── distillation.py         # Main distill() orchestrator
│   ├── inference.py            # Forward pass, PCA, SIF weighting, pooling modes
│   └── utils.py                # Device selection (cuda/mps/cpu)
│
├── tokenizer/                  # Vocabulary management
│   └── tokenizer.py            # Vocabulary cleaning, token-to-id mapping
│
├── persistence/                # Save/Load
│   ├── persistence.py          # safetensors-based save/load with layout detection
│   ├── datamodels.py           # Layout dataclasses for model2vec, sentence-transformers, nested
│   └── hf.py                   # HuggingFace Hub push/cache
│
├── inference/                  # Classifier pipeline
│   └── model.py                # StaticModelPipeline (sklearn head)
│
├── train/                      # Fine-tuning
│   ├── base.py                 # FinetunableStaticModel (PyTorch, Lightning)
│   ├── classifier.py           # StaticModelForClassification (full training loop)
│   └── utils.py                # PAD token detection
│
└── modelcards/                 # Model card generation
    └── modelcards.py           # HF ModelCard creation/metadata
```

### Data Flow (Distillation Pipeline)

```
Sentence Transformer (Teacher)
    │
    ├── AutoModel.from_pretrained()
    ├── AutoTokenizer.from_pretrained()
    │
    ▼
[Vocabulary Extraction]
    │  tokenizer/ tokenizer.py
    │  - Clean tokens (remove duplicates, regex patterns)
    │  - Map tokens → IDs using teacher tokenizer
    ▼
[Forward Pass] 
    │  distill/inference.py
    │  - Batch tokens through teacher model
    │  - Apply pooling (mean/last/first/pooler)
    │  - Output: (V, D) matrix where V=vocab_size, D=hidden_dim
    ▼
[Post-Processing]
    │  distill/inference.py: post_process_embeddings()
    │  - PCA dimensionality reduction (D → pca_dims)
    │  - SIF/Zipf weighting (frequency-based token weights)
    ▼
[Optional Vocabulary Quantization]
    │  vocabulary_quantization.py
    │  - KMeans clustering of embeddings
    │  - Tokens map → cluster centers
    ▼
[Dtype Quantization]
    │  quantization.py
    │  - Float32 → Float16 or Int8
    ▼
StaticModel (Student)
    │  model.py
    │  - embedding: (V, D) numpy array
    │  - tokenizer: HuggingFace Tokenizer
    │  - weights: (V,) SIF weights
    │  - token_mapping: optional KMeans mapping
    │
    ▼ encode()
    ├── tokenize() → list[list[int]]
    ├── embedding lookup → (seq_len, D)
    ├── weighted mean → (D,)
    └── L2 normalize → (D,)
```

### Data Flow (Inference)

```
Input: "hello world"
    │
    ▼ tokenize()
    Token IDs: [4521, 8892]    (UNK tokens removed, max_length truncated)
    │
    ▼ _encode_helper()
    Remap via token_mapping (if vocab quantized): [4521, 8892] → [23, 156]
    Embedding lookup: embedding[ids] → (2, D)
    Apply weights: emb * weights[ids][:, None] → (2, D)
    │
    ▼ _encode_batch()
    Mean over tokens: (2, D) → (D,)
    L2 normalize (if self.normalize): v / (||v|| + 1e-32)
    │
    ▼
    Output: (D,) numpy array
```

---

## 2. Annotated Source Code Listing

### 2.1 `model.py` — StaticModel (Core)

**Key Classes/Functions:**
- `StaticModel` — The main model. Holds embedding matrix, tokenizer, weights, token_mapping.
- `encode()` — Tokenize → embed → weighted mean → L2 normalize.
- `encode_as_sequence()` — Tokenize → embed → return per-token vectors (no mean).
- `quantize_model()` — Post-hoc quantization/vocab reduction on a loaded model.

**Critical Details:**

```python
# The encode pipeline (simplified):
def _encode_batch(self, sentences, max_length):
    ids = self.tokenize(sentences, max_length)  # [[tok_ids], ...]
    out = []
    for id_list in ids:
        if id_list:
            emb = self._encode_helper(id_list)  # weighted lookup
            out.append(emb.mean(axis=0))         # MEAN AGGREGATION
        else:
            out.append(np.zeros(self.dim))       # zero vector for empty
    out_array = np.stack(out)
    if self.normalize:
        norm = np.linalg.norm(out_array, axis=1, keepdims=True) + 1e-32
        out_array = out_array / norm             # L2 NORMALIZE
    return out_array

def _encode_helper(self, id_list):
    if self.token_mapping is not None:
        id_list_remapped = self.token_mapping[id_list]  # KMeans remap
    emb = self.embedding[id_list_remapped]               # lookup
    if self.weights is not None:
        emb = emb * self.weights[id_list][:, None]       # SIF weighting
    return emb
```

**Key Properties:**
- `dim` → embedding.shape[1]
- `embedding_dtype` → dtype name (float16/float32/int8)
- `vocabulary_quantization` → number of clusters if vocab-quantized

### 2.2 `distill/inference.py` — Embedding Creation & Post-Processing

**Key Functions:**

#### `create_embeddings(model, tokenized, device, pad_token_id, pooling)`
- Batches tokens through the teacher model
- **4 pooling modes**: MEAN, LAST, FIRST, POOLER
- Sorts by length for efficient batching
- Returns `(V, D)` numpy array

```python
# Mean pooling:
mask = attention_mask.float()
lengths = mask.sum(1, keepdim=True).clamp_min_(1.0)
mask = mask / lengths
return torch.bmm(mask[:, None, :], hidden).squeeze(1)

# Last token pooling:
last_idx = (mask.sum(dim=1) - 1).clamp_min(0)
return hidden[batch_indices, last_idx, :]

# First token (CLS) pooling:
return hidden[:, 0, :]

# Pooler output:
return outputs.pooler_output
```

#### `post_process_embeddings(embeddings, pca_dims, sif_coefficient)`

**PCA:**
```python
p = PCA(n_components=pca_dims, svd_solver="full")
embeddings = p.fit_transform(embeddings)
# Reduces (V, 768) → (V, 256) typically
# Reports explained variance ratio
```

**SIF Weighting (Zipf-based):**
```python
# Estimate word frequencies using Zipf's law
inv_rank = 1 / (np.arange(2, embeddings.shape[0] + 2))
proba = inv_rank / np.sum(inv_rank)
weight = sif_coefficient / (sif_coefficient + proba)
# Default sif_coefficient = 1e-4
# Higher rank (rarer words) → higher weight
# This is NOT learned from data — it's estimated from rank ordering
```

### 2.3 `quantization.py` — Dtype Quantization

```python
# Float types: simple astype()
embeddings.astype(np.float16)

# Int8: scale + clip
scale = np.max(np.abs(embeddings)) / 127.0
buf = embeddings.astype(np.float16, copy=True)
np.divide(buf, scale, out=buf)
np.rint(buf, out=buf)
np.clip(buf, -127, 127, out=buf)
quantized = buf.astype(np.int8)

# Dimensionality reduction (truncation, not PCA):
embeddings = embeddings[:, :dimensionality]
# NOTE: This assumes the embeddings were trained with MRL or PCA
# so the first N dimensions are the most important
```

### 2.4 `vocabulary_quantization.py` — KMeans Clustering

```python
def quantize_vocabulary(n_clusters, weights, embeddings):
    # If no weights, use L2 norm as implicit weight
    if weights is None:
        weights = np.linalg.norm(embeddings, axis=1) + 1e-32
        embeddings = embeddings / weights[:, None]  # normalize before clustering
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="random")
    kmeans.fit(embeddings.astype(np.float32))
    token_mapping = kmeans.predict(embeddings)  # maps each token → cluster
    embeddings = kmeans.cluster_centers_  # new embeddings = cluster centers
    
    return embeddings, token_mapping, weights
    # (n_clusters, D), (V,), (V,)
```

### 2.5 `tokenizer/tokenizer.py` — Vocabulary Management

```python
def clean_and_create_vocabulary(model, vocabulary_to_add, token_remove_regex):
    # 1. Remove tokens matching regex (e.g., [unused\d+])
    # 2. Skip duplicates, empty tokens, multi-word tokens
    # 3. Add remaining tokens to vocabulary
    # Uses skeletoken library for tokenizer manipulation

def turn_tokens_into_ids(tokens, model):
    # For each token:
    #   - If in vocabulary → direct ID lookup
    #   - Otherwise → encode through tokenizer
    # Wraps with BOS/EOS if present
```

### 2.6 `distill/distillation.py` — Main Orchestrator

```python
def distill(model_name, vocabulary, device, pca_dims, sif_coefficient, 
            token_remove_pattern, quantize_to, vocabulary_quantization, pooling):
    # 1. Load teacher model + tokenizer from HF
    # 2. Clean vocabulary, create tokenizer
    # 3. Map tokens → IDs
    # 4. Forward pass through teacher → (V, D) embeddings
    # 5. Optionally: vocabulary quantization (KMeans)
    # 6. Post-process: PCA + SIF
    # 7. Quantize dtype
    # 8. Return StaticModel
```

### 2.7 `train/base.py` — Fine-tunable Model

```python
class FinetunableStaticModel(nn.Module):
    # Trainable version of StaticModel
    # - Embeddings can be frozen or fine-tuned
    # - Learnable per-token weights (sigmoid-gated)
    # - Classification head (Linear → ReLU → ... → Linear)
    # 
    # _encode(input_ids):
    #   w = sigmoid(self.w[input_ids])       # learnable weights
    #   embedded = embeddings(token_mapping[input_ids])
    #   weighted = w[:, None, :] @ embedded  # weighted sum
    #   normalized = L2_normalize(weighted)
    #
    # to_static_model():
    #   Extracts embeddings, sigmoid(weights), token_mapping → StaticModel
```

### 2.8 `persistence/persistence.py` — Save/Load

```
Files on disk:
  model.safetensors  → {embeddings, weights, mapping}
  tokenizer.json     → HuggingFace tokenizer
  config.json        → {model_type, apply_pca, sif_coefficient, hidden_dim, normalize, ...}
  modules.json       → sentence-transformers compatibility
  README.md          → model card
```

**3 layout patterns supported:**
1. model2vec flat: `model.safetensors`, `config.json`, `tokenizer.json`
2. sentence-transformers: `config_sentence_transformers.json`
3. nested: `0_StaticEmbedding/` subfolder

---

## 3. Extracted Primitives

### Primitive 1: Token Embedding Table

| Property | Detail |
|---|---|
| **What** | Static lookup: token_id → vector |
| **Format** | `np.ndarray` of shape `(V, D)`, dtype float16/float32/int8 |
| **Creation** | Forward pass of teacher model on vocabulary tokens |
| **Typical sizes** | V=30,000-50,000 tokens, D=256 (after PCA) |
| **Memory** | 30K × 256 × 2 bytes (float16) ≈ 15 MB |
| **Dependencies** | numpy only (no torch needed at inference) |

### Primitive 2: PCA Transform

| Property | Detail |
|---|---|
| **What** | Dimensionality reduction: D_teacher → D_student |
| **Algorithm** | `sklearn.decomposition.PCA(n_components, svd_solver="full")` |
| **Input** | (V, D_teacher) float array |
| **Output** | (V, D_student) float array |
| **Side product** | Explained variance ratio, component matrix |
| **Default** | 768 → 256 (67% reduction) |
| **Dependencies** | sklearn (for fitting), numpy only (for apply) |

### Primitive 3: SIF/Zipf Weighting

| Property | Detail |
|---|---|
| **What** | Frequency-based token weighting for aggregation |
| **Formula** | `weight[i] = a / (a + proba[i])` where `proba[i] = 1/rank[i] / Σ(1/rank)` |
| **Default `a`** | 1e-4 |
| **Assumption** | Token order in vocabulary ≈ frequency rank (Zipf's law) |
| **NOT learned** | Estimated from rank ordering, no corpus needed |
| **Shape** | (V,) numpy array |
| **Dependencies** | numpy only |

### Primitive 4: Token Aggregation (Mean Pooling)

| Property | Detail |
|---|---|
| **What** | Combines token embeddings into sentence vector |
| **Algorithm** | Weighted arithmetic mean: `mean(weights * embeddings, axis=0)` |
| **Alternatives** | Sum, max, attention-weighted |
| **Dependencies** | numpy only |

### Primitive 5: L2 Normalization

| Property | Detail |
|---|---|
| **What** | Post-processing: unit-length vectors |
| **Algorithm** | `v / (||v|| + ε)` where ε=1e-32 |
| **Purpose** | Cosine similarity via dot product |
| **Dependencies** | numpy only |

### Primitive 6: Vocabulary Tokenizer

| Property | Detail |
|---|---|
| **What** | Text → token IDs |
| **Implementation** | HuggingFace `tokenizers.Tokenizer` (Rust-based, fast) |
| **OOV handling** | UNK token removed from output |
| **Preprocessing** | Custom vocabulary, regex-based token removal, dedup |
| **Dependencies** | tokenizers (HuggingFace) |

### Primitive 7: Vocabulary Quantization (KMeans)

| Property | Detail |
|---|---|
| **What** | Cluster similar tokens → shared embedding |
| **Algorithm** | KMeans(n_clusters) on L2-normalized embeddings |
| **Input** | (V, D) embeddings |
| **Output** | (K, D) cluster centers + (V,) token→cluster mapping + (V,) weights |
| **Effect** | V=30K tokens → K=8K clusters (4× compression) |
| **Dependencies** | sklearn (KMeans), numpy |

### Primitive 8: Dtype Quantization

| Property | Detail |
|---|---|
| **What** | Reduce numerical precision |
| **Options** | float32 → float16 (2×), float32 → int8 (4×) |
| **Int8 method** | Scale by max(|emb|)/127, round, clip |
| **Dependencies** | numpy only |

### Primitive 9: Teacher Forward Pass (Distillation Source)

| Property | Detail |
|---|---|
| **What** | Extract per-token embeddings from transformer |
| **Pooling modes** | MEAN (default), LAST, FIRST, POOLER |
| **Batching** | Sorted by length, padded, batch_size=256 |
| **Dependencies** | torch, transformers |

### Primitive 10: Learnable Weights (Fine-tuning)

| Property | Detail |
|---|---|
| **What** | Trainable per-token importance weights |
| **Parametrization** | `sigmoid(w[token_id])` — initialized to 0 (sigmoid(0)=0.5) |
| **Training** | PyTorch Lightning, Adam optimizer, early stopping |
| **Dependencies** | torch, lightning |

### Primitive 11: Classification Head

| Property | Detail |
|---|---|
| **What** | Linear layers on top of sentence embedding |
| **Architecture** | `Linear(D, hidden) → ReLU → ... → Linear(hidden, n_classes)` |
| **Export** | Can be converted to sklearn MLPClassifier |
| **Dependencies** | torch (training), sklearn (inference) |

---

## 4. PLATO Architecture Mapping

### Mapping Table

| Model2Vec Primitive | PLATO Component | Rationale | Location |
|---|---|---|---|
| **Token Embedding Table** | **PLATO Tile** (content-addressed) | Portable, versioned, shareable between agents. Each domain gets its own tile. | `plato-training/store.py` → LocalTileStore |
| **PCA Transform Matrix** | **PLATO Tile** (small, portable) | The PCA components matrix is tiny (D×D'). Any room can apply it. | New: `pca_tile.py` |
| **SIF Weights** | **PLATO Room State** | Domain-specific. Accumulated as rooms process queries. Exportable as tile. | Room-level state |
| **Zipf Estimator** | **PLATO Room Operation** | Pure function: rank → weights. Can be overridden with learned frequencies. | Room method |
| **Token Aggregation (Mean)** | **PLATO Model Component** | Could be a SplineLinear layer (learnable aggregation). Different per room. | `plato-training/rooms/` |
| **L2 Normalization** | **PLATO Store Operation** | Applied before FAISS indexing. Pure function, no state. | Store method |
| **Vocabulary Tokenizer** | **PLATO Tile** | Domain-specific vocabularies as content-addressed tiles. | New: `vocab_tile.py` |
| **KMeans Vocabulary Quantization** | **PLATO Room Operation** | Clustering as a room capability. Cluster assignments as tiles. | Room method |
| **Dtype Quantization** | **PLATO Store Operation** | Applied during save/deploy. Reduces tile size. | Store method |
| **Teacher Forward Pass** | **PLATO Room Operation** (one-time distillation) | Distillation room: takes teacher model, produces student tile. | New: `distillation_room.py` |
| **Learnable Weights** | **PLATO Model Component** | SplineLinear or LoRA over embedding table. | `plato-training/adapters/` |
| **Classification Head** | **PLATO Room** (task-specific) | Each room has its own head. Trained on room-specific data. | Room-level |

### Detailed PLATO Integration Design

#### 4.1 Embedding Tile (Core Artifact)

```python
# A PLATO tile containing a static embedding space
@dataclass
class EmbeddingTile:
    """Content-addressed embedding table."""
    embedding: np.ndarray      # (V, D) — the core lookup
    vocabulary: dict[str, int] # token → index
    weights: np.ndarray        # (V,) — SIF or learned weights
    metadata: dict             # source, dims, dtype, language, domain
    
    @property
    def content_hash(self) -> str:
        """Content-addressed hash for dedup and versioning."""
        # Hash embedding + vocab + weights
        ...
    
    def lookup(self, tokens: list[str]) -> np.ndarray:
        """Look up embeddings for tokens."""
        ids = [self.vocabulary.get(t) for t in tokens if t in self.vocabulary]
        if not ids:
            return np.zeros(self.embedding.shape[1])
        emb = self.embedding[ids]
        if self.weights is not None:
            emb = emb * self.weights[ids][:, None]
        return emb
```

**Per-domain embedding tiles:**
- `fleet-embeddings-v1.tile` — PLATO rooms, fleet ops, agent names
- `constraint-theory-v1.tile` — drift, tiles, rooms, Eisenstein terminology
- `general-english-v1.tile` — distilled from sentence-transformers (baseline)
- `code-v1.tile` — distilled from CodeBERT or similar

#### 4.2 PCA as a Room Operation

```python
class DimensionalityReductionRoom(Room):
    """Applies PCA (or SplineLinear) to reduce embedding dimensions."""
    
    def apply(self, embedding_tile: EmbeddingTile, target_dim: int) -> EmbeddingTile:
        """Reduce dimensions and return new tile."""
        # Option A: PCA (standard)
        pca = PCA(n_components=target_dim)
        reduced = pca.fit_transform(embedding_tile.embedding)
        
        # Option B: SplineLinear (PLATO native — see improvement opportunities)
        # ...
        
        return EmbeddingTile(
            embedding=reduced,
            vocabulary=embedding_tile.vocabulary,
            weights=embedding_tile.weights,
            metadata={**embedding_tile.metadata, "pca_dim": target_dim}
        )
```

#### 4.3 SIF Weighting as Room State

```python
class SIFWeightingRoom(Room):
    """Tracks word frequencies and computes SIF weights."""
    
    def __init__(self):
        self.word_counts: Counter = Counter()
        self.total_count: int = 0
    
    def observe(self, tokens: list[str]):
        """Update frequency counts from observed text."""
        self.word_counts.update(tokens)
        self.total_count += len(tokens)
    
    def compute_weights(self, coefficient: float = 1e-4) -> np.ndarray:
        """Compute SIF weights from observed frequencies."""
        # Sort by frequency (rank)
        sorted_tokens = sorted(self.word_counts.items(), key=lambda x: -x[1])
        weights = np.zeros(len(sorted_tokens))
        for rank, (token, count) in enumerate(sorted_tokens):
            proba = count / self.total_count
            weights[rank] = coefficient / (coefficient + proba)
        return weights
    
    def export_as_tile(self) -> SIFWeightsTile:
        """Export weights for other rooms."""
        ...
```

#### 4.4 Token Aggregation as SplineLinear Component

```python
class SplineAggregation:
    """Replace mean pooling with learnable SplineLinear aggregation."""
    
    def __init__(self, input_dim: int, spline_params):
        self.spline = SplineLinear(input_dim, input_dim, **spline_params)
    
    def aggregate(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Learned aggregation instead of simple mean."""
        # Apply spline to each token, then mean
        transformed = self.spline(token_embeddings)
        return transformed.mean(axis=0)
```

---

## 5. Interface Contracts Between Primitives

### 5.1 Data Format Flow

```
Primitive A → Primitive B | Format | Dimensions | Dependencies
--------------------------|--------|------------|-------------
Tokenizer → Embedding     | list[int] (token IDs) | variable length | tokenizers
Embedding → SIF Weight    | np.ndarray (float32) | (seq_len, D) | numpy
SIF Weight → Aggregation  | np.ndarray (float32) | (seq_len, D) | numpy
Aggregation → Normalize   | np.ndarray (float32) | (D,) | numpy
Normalize → Output        | np.ndarray (float32) | (D,) | numpy

PCA: (V, D_teacher) → (V, D_student) | np.ndarray | numpy + sklearn (fit) / numpy (apply)
KMeans: (V, D) → (K, D) + (V,) mapping | np.ndarray | sklearn
Quantize: (V, D) float32 → (V, D) float16/int8 | np.ndarray | numpy
```

### 5.2 Swap Independence Matrix

| Can swap independently? | PCA | SIF | Aggregation | Normalization | Quantization |
|---|---|---|---|---|---|
| **PCA** | — | ✅ | ✅ | ✅ | ✅ |
| **SIF** | ✅ | — | ✅ (SIF feeds into agg) | ✅ | ✅ |
| **Aggregation** | ✅ | ⚠️ (needs weights) | — | ✅ | ✅ |
| **Normalization** | ✅ | ✅ | ✅ | — | ✅ |
| **Quantization** | ✅ | ✅ | ✅ | ✅ | — |

**Key insight:** All primitives are loosely coupled. The only dependency chain is:
`Tokenizer → Embedding → [PCA] → [SIF weights] → Aggregation → [Normalize] → [Quantize]`

Each step operates on numpy arrays and produces numpy arrays. Any step can be swapped independently.

### 5.3 Dependency Summary

| Primitive | numpy | sklearn | torch | tokenizers | Other |
|---|---|---|---|---|---|
| Token Embedding Table | ✅ | | | | |
| PCA | ✅ | fit only | | | |
| SIF/Zipf | ✅ | | | | |
| Mean Aggregation | ✅ | | | | |
| L2 Normalization | ✅ | | | | |
| Vocabulary Tokenizer | | | | ✅ | skeletoken |
| KMeans Vocab Quant | ✅ | ✅ | | | |
| Dtype Quantization | ✅ | | | | |
| Teacher Forward Pass | ✅ | | ✅ | | transformers |
| Learnable Weights | ✅ | | ✅ | | lightning |
| Classification Head | ✅ | ✅ | ✅ | | |

**Inference-only dependency: numpy + tokenizers** (no torch, no sklearn!)

---

## 6. Improvement Opportunities with PLATO

### 6.1 SplineLinear → Replace PCA (HIGH PRIORITY)

**Current:** PCA is a linear projection. It finds orthogonal directions of maximum variance. But it's:
- Data-agnostic after fitting
- Not parameterized efficiently
- No structure beyond orthogonality

**PLATO alternative:** SplineLinear with Eisenstein lattice weights
- Non-linear dimensionality reduction
- Structured parameterization (fewer parameters than dense linear)
- 20× compression demonstrated on drift-detect
- Could learn domain-specific projections that preserve semantic structure

**Experiment:** Train SplineLinear(D_teacher, D_student) on teacher embeddings. Compare:
1. Reconstruction quality (cosine similarity between PCA-reduced and SplineLinear-reduced)
2. Downstream task performance (STS, classification)
3. Parameter count (SplineLinear vs PCA components)

### 6.2 Deadband Throttle → Skip Redundant Embedding Computation (MEDIUM)

**Current:** Every sentence gets fully encoded, even if similar to cached results.

**PLATO alternative:** Deadband throttle checks if input is within epsilon of recent inputs:
- If `cosine(query, cached) > threshold` → return cached embedding
- If not → compute new embedding
- Fleet-aware: respects training throttle across agents

**Savings:** In conversational settings (PLATO rooms), many queries are near-duplicates. Deadband could skip 30-60% of encoding.

### 6.3 BMA (Bayesian Model Averaging) → Detect Embedding Drift (HIGH)

**Current:** Static embeddings never update. If the domain shifts, you need full redistillation.

**PLATO alternative:** 
- Maintain a rolling window of embeddings and their quality scores
- BMA detects when new embeddings diverge from historical distribution
- Trigger re-embedding of drifted tokens automatically
- Room-level: each room monitors its own embedding quality

### 6.4 Tensor-Spline Compression on Embedding Table (HIGH)

**Current:** Int8 quantization is the only compression (4× lossy).

**PLATO alternative:** SplineLinear parameterization of the entire embedding table:
- Instead of storing (V, D) directly, store the spline parameters
- 20× compression demonstrated on classification tasks
- Could achieve higher compression with less quality loss than int8
- Content-addressed: tiles become tiny

**Experiment:** Parameterize each embedding dimension as a spline over the vocabulary index. Compare:
1. Compression ratio (SplineLinear parameters vs raw embedding bytes)
2. Reconstruction quality
3. Downstream task performance

### 6.5 Domain-Specific Vocabularies from Room History (MEDIUM)

**Current:** Vocabulary comes from the teacher model or a user-provided list.

**PLATO alternative:**
- Each room accumulates its own vocabulary from processed queries
- High-frequency terms get dedicated embeddings
- Low-frequency terms share via KMeans clusters
- Room vocabulary tiles can be merged, split, or shared

**This is the key insight:** PLATO rooms naturally produce domain-specific corpora. The vocabulary should emerge from room activity, not be prescribed.

### 6.6 Collective Embedding via I2I (MEDIUM)

**Current:** Each StaticModel is standalone.

**PLATO alternative:**
- Multiple agents encode the same text with different domain embeddings
- Compare results via I2I protocol
- Where they agree: high confidence
- Where they disagree: interesting — potential for ensemble or domain detection
- "The glitches ARE the research agenda" applies to embedding disagreement

### 6.7 LoRA Fine-Tuning of Embedding Table (LOW — already in codebase)

**Current:** Fine-tuning uses full embedding table updates.

**PLATO alternative:** 
- LoRA adapters on the embedding table (already have `adapters/lora.py`)
- Each room/domain gets its own LoRA adapter
- Base embedding table stays frozen, adapters are tiny tiles
- Multiple rooms can share the same base with different adapters

---

## 7. Implementation Priority

### Phase 1: Core Primitives (Week 1)

1. **EmbeddingTile** — Content-addressed embedding table as PLATO tile
   - Subclass `TrainingTile` from plato-types
   - Content hash over embedding + vocabulary + weights
   - Save/load via safetensors
   - Lookup method with SIF weighting
   - **Files:** `plato-training/embedding_tile.py` (~200 lines)

2. **VocabularyTile** — Domain vocabulary as tile
   - Extract from room history
   - Merge, split, share operations
   - **Files:** `plato-training/vocabulary_tile.py` (~150 lines)

3. **EmbeddingRoom** — Room that can encode text using an EmbeddingTile
   - Load tile, encode queries, return vectors
   - Accumulate SIF weights from observed text
   - **Files:** `plato-training/rooms/embedding_room.py` (~300 lines)

### Phase 2: Distillation Pipeline (Week 2)

4. **DistillationRoom** — Distill teacher → student embedding tile
   - Wrap Model2Vec's distill() as a room operation
   - Output: EmbeddingTile
   - Support multiple pooling modes
   - **Files:** `plato-training/rooms/distillation_room.py` (~400 lines)

5. **PCARoom** — Dimensionality reduction as room operation
   - Fit PCA on embedding tile, produce reduced tile
   - PCA matrix stored as its own tile
   - **Files:** `plato-training/rooms/pca_room.py` (~200 lines)

### Phase 3: PLATO-Native Improvements (Week 3)

6. **SplineLinear Aggregation** — Replace mean pooling
   - Trainable aggregation using tensor-spline
   - Per-room learned aggregation strategy
   - **Files:** `plato-training/rooms/spline_aggregation.py` (~250 lines)

7. **Deadband Encoding Cache** — Skip redundant computation
   - LRU cache with cosine similarity threshold
   - Integrated into EmbeddingRoom
   - **Files:** Extend `embedding_room.py` (~100 lines additional)

### Phase 4: Advanced Features (Week 4)

8. **BMA Drift Detection** — Monitor embedding quality
   - Rolling window statistics on embedding quality
   - Trigger re-embedding when drift exceeds threshold
   - **Files:** `plato-training/rooms/drift_detection.py` (~300 lines)

9. **Tensor-Spline Embedding Compression** — Compress entire tables
   - SplineLinear parameterization of embedding table
   - Compare with int8 quantization
   - **Files:** `plato-training/spline_embeddings.py` (~400 lines)

10. **I2I Collective Embedding** — Multi-agent embedding comparison
    - Encode with multiple domain tiles
    - Compare, detect disagreement
    - Share via bottle protocol
    - **Files:** `plato-training/rooms/collective_embedding.py` (~300 lines)

---

## Appendix A: Complete File Inventory

| File | Lines | Purpose |
|---|---|---|
| `model.py` | ~350 | StaticModel: core encode/decode, quantize_model |
| `quantization.py` | ~70 | DType enum, quantize_embeddings, quantize_and_reduce_dim |
| `vocabulary_quantization.py` | ~35 | quantize_vocabulary (KMeans) |
| `utils.py` | ~90 | ProgressParallel, importable, setup_logging |
| `version.py` | ~2 | Version 0.8.1 |
| `distill/distillation.py` | ~220 | distill(), distill_from_model() orchestrator |
| `distill/inference.py` | ~200 | create_embeddings, post_process_embeddings, pooling modes |
| `distill/utils.py` | ~35 | select_optimal_device |
| `tokenizer/tokenizer.py` | ~100 | clean_and_create_vocabulary, turn_tokens_into_ids |
| `persistence/persistence.py` | ~130 | save_pretrained, load_pretrained |
| `persistence/datamodels.py` | ~40 | Layout dataclass, FOLDER_LAYOUTS |
| `persistence/hf.py` | ~50 | push_folder_to_hub, cache lookup |
| `inference/model.py` | ~300 | StaticModelPipeline (sklearn classifier) |
| `train/base.py` | ~200 | FinetunableStaticModel (PyTorch) |
| `train/classifier.py` | ~400 | StaticModelForClassification (Lightning training) |
| `train/utils.py` | ~20 | get_probable_pad_token_id |
| `modelcards/modelcards.py` | ~60 | create_model_card, get_metadata_from_readme |

**Total: ~2,300 lines of Python**

## Appendix B: Key Constants & Defaults

| Parameter | Default | Notes |
|---|---|---|
| PCA dimensions | 256 | Reduces from 768 (typical) |
| SIF coefficient | 1e-4 | `a / (a + proba)` |
| Quantize dtype | float16 | 2× compression |
| Batch size (distill) | 256 | Token batches through teacher |
| Batch size (encode) | 1024 | Sentence batches through student |
| Multiprocessing threshold | 10,000 | sentences before parallelism |
| Max length | 512 tokens | Truncation limit |
| UNK handling | Removed | UNK tokens dropped from output |
| Normalize | True | L2 normalize output vectors |
| KMeans init | "random" | For vocabulary quantization |
| KMeans random_state | 42 | Reproducible |
| Int8 scale | max(|emb|) / 127 | Symmetric quantization |

## Appendix C: Minimal Inference Dependencies

For **inference only** (no distillation, no training), Model2Vec needs:

```
numpy          — array operations
tokenizers     — HuggingFace tokenizer (Rust-backed, fast)
safetensors    — loading model weights
```

That's it. No torch, no sklearn, no transformers. This makes it ideal for edge deployment.

**PLATO integration opportunity:** Our micro-model deployment pipeline (`deploy_micro()`) targets NPU, CPU-tiny, etc. A Model2Vec inference-only embedding tile would deploy to any target with just numpy.
