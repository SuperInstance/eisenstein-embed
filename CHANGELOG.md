# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-20

### Added
- 5-layer matching cascade (EXACT → BITVECTOR → DEADBAND → SEMANTIC → DOMAIN)
- `MatchResult` with type annotations
- `EisensteinModel` with zero-config bitvector matching
- Model2Vec integration for semantic encoding
- Domain-aware SIF re-weighting
- Deadband cosine-similarity cache
- BMA drift detection and adaptive thresholds
- Eisenstein quantization (spline-linear)
- Thread-safe state management via `threading.Lock`
- Full type annotations on public API
- `save()` and `load()` with return type annotations
