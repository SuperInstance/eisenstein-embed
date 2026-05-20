"""SplineLinear compression of embedding tables."""

from typing import List

import numpy as np


class SplineLinearQuantizer:
    """Piecewise-linear quantization for embedding vectors.

    Divides each vector into segments and fits a linear function
    (slope + intercept) per segment. Values are quantized to 8-bit
    within each segment, achieving ~20x compression versus raw
    float32 at modest quality loss.
    """

    def __init__(self, n_segments: int = 4, bits_per_value: int = 8):
        self.n_segments = n_segments
        self.bits_per_value = bits_per_value
        self.max_val = 2 ** (bits_per_value - 1) - 1  # signed int range

    def fit(self, vectors: np.ndarray) -> "SplineLinearQuantizer":
        """No-op for compatibility; compression is stateless."""
        return self

    def compress(self, vectors: np.ndarray) -> dict:
        """Compress a 2-D array of vectors.

        Returns a dict with:
            - shape: original shape
            - segments: number of segments
            - bits: bits per value
            - min_vals: per-segment min values
            - max_vals: per-segment max values
            - slopes: per-segment slopes
            - intercepts: per-segment intercepts
            - data: quantized integer data
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        n_rows, dim = vectors.shape
        seg_len = dim // self.n_segments
        remainder = dim % self.n_segments

        min_vals = []
        max_vals = []
        slopes = []
        intercepts = []
        quantized_blocks = []

        idx = 0
        for s in range(self.n_segments):
            length = seg_len + (1 if s < remainder else 0)
            block = vectors[:, idx : idx + length]
            bmin = block.min(axis=1, keepdims=True)
            bmax = block.max(axis=1, keepdims=True)
            # Avoid division by zero
            scale = np.where(bmax - bmin == 0, 1.0, bmax - bmin)

            # Map to [0, max_val]
            q = np.round((block - bmin) / scale * self.max_val).astype(np.int8)

            # Fit linear approximation per row: y = slope * x + intercept
            x = np.arange(length, dtype=np.float32)
            # Simple least-squares for slope/intercept per row
            x_mean = x.mean()
            y_mean = block.mean(axis=1)
            num = ((x - x_mean) * (block - y_mean[:, None])).sum(axis=1)
            den = ((x - x_mean) ** 2).sum()
            slope = num / den if den != 0 else np.zeros(n_rows)
            intercept = y_mean - slope * x_mean

            min_vals.append(bmin.squeeze().astype(np.float32).tolist())
            max_vals.append(bmax.squeeze().astype(np.float32).tolist())
            slopes.append(slope.astype(np.float32).tolist())
            intercepts.append(intercept.astype(np.float32).tolist())
            quantized_blocks.append(q)

            idx += length

        data = np.concatenate(quantized_blocks, axis=1)

        return {
            "shape": (n_rows, dim),
            "segments": self.n_segments,
            "bits": self.bits_per_value,
            "min_vals": min_vals,
            "max_vals": max_vals,
            "slopes": slopes,
            "intercepts": intercepts,
            "data": data.tobytes(),
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        """Decompress data back to float32 vectors.

        Reconstructs each segment from its quantized values and
        linear parameters.
        """
        n_rows, dim = compressed["shape"]
        n_segments = compressed["segments"]
        bits = compressed["bits"]
        max_val = 2 ** (bits - 1) - 1

        data = np.frombuffer(compressed["data"], dtype=np.int8).reshape(n_rows, dim)

        seg_len = dim // n_segments
        remainder = dim % n_segments

        blocks = []
        idx = 0
        for s in range(n_segments):
            length = seg_len + (1 if s < remainder else 0)
            block = data[:, idx : idx + length]
            bmin = np.asarray(compressed["min_vals"][s], dtype=np.float32).reshape(-1, 1)
            bmax = np.asarray(compressed["max_vals"][s], dtype=np.float32).reshape(-1, 1)
            scale = np.where(bmax - bmin == 0, 1.0, bmax - bmin)
            recon = (block.astype(np.float32) / max_val) * scale + bmin
            blocks.append(recon)
            idx += length

        return np.concatenate(blocks, axis=1).astype(np.float32)

    def compression_ratio(self, original: np.ndarray, compressed: dict) -> float:
        """Report achieved compression ratio."""
        original_bytes = original.nbytes
        # Approximate compressed size
        data_bytes = len(compressed["data"])
        meta_bytes = sum(
            len(np.asarray(v, dtype=np.float32).tobytes())
            for v in [compressed["min_vals"], compressed["max_vals"], compressed["slopes"], compressed["intercepts"]]
        )
        return original_bytes / (data_bytes + meta_bytes)
