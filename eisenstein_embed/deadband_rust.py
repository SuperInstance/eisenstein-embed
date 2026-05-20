"""Rust deadband acceleration via deadband-rs.

Tries to import from a compiled PyO3 extension built from deadband-rs.
Falls back to the pure-Python DeadbandCache when the Rust extension
is unavailable.

Building the Rust extension for production:
    pip install maturin
    cd deadband-rs
    maturin build --release
    pip install target/wheels/deadband_rs-*.whl

The Rust crate (deadband-rs) provides:
  - hexagonal point sampling (hpdf)
  - Eisenstein lattice arithmetic (eisenstein)
  - BMA (Bounded Modular Arithmetic) acceleration (bma)
  - 360-degree division helpers (div360)
  - Fibonacci spline interpolation (fib_spline)

When compiled as a Python extension, these become available as:
  import deadband_rs
  deadband_rs.in_hexagon(x, y, r)
  deadband_rs.sample_hex(rng_seed, radius)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Try importing the compiled Rust extension
try:
    import deadband_rs as _rust  # type: ignore
    HAS_RUST_DEADBAND = True
except ImportError:
    HAS_RUST_DEADBAND = False


def in_hexagon(x: float, y: float, r: float) -> bool:
    """Check if (x, y) is inside a regular hexagon with circumradius r.

    Uses the Rust extension when available, otherwise falls back to pure Python.
    """
    if HAS_RUST_DEADBAND:
        return _rust.in_hexagon(x, y, r)

    # Pure Python fallback
    ax, ay = abs(x), abs(y)
    import math
    return ay <= r and (math.sqrt(3) * ax + ay) * 0.5 <= r


def sample_hex(rng_seed: int = 0, radius: float = 1.0) -> Tuple[float, float]:
    """Sample a random point uniformly from a regular hexagon.

    Uses the Rust extension when available, otherwise pure Python.
    """
    if HAS_RUST_DEADBAND:
        return _rust.sample_hex(rng_seed, radius)

    # Pure Python fallback — rejection sampling
    import random
    import math
    rng = random.Random(rng_seed)
    while True:
        x = rng.uniform(-radius, radius)
        y = rng.uniform(-radius, radius)
        if in_hexagon(x, y, radius):
            return x, y


def rust_available() -> bool:
    """Return True if the Rust deadband extension is compiled and importable."""
    return HAS_RUST_DEADBAND
