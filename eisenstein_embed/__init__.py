"""Eisenstein Embed — Enhanced static embeddings with a 5-layer matching cascade."""

from eisenstein_embed.static_model import EisensteinModel, MatchResult
from eisenstein_embed.cascade import CascadeMatcher
from eisenstein_embed.bitvector import (
    word_fingerprint,
    text_fingerprint,
    hamming_distance,
    bitvector_similarity,
    find_best_bitvector_match,
)
from eisenstein_embed.deadband_cache import DeadbandCache
from eisenstein_embed.domain_sif import DomainSIF
from eisenstein_embed.bma_monitor import BMAMonitor
from eisenstein_embed.eisenstein_quantize import SplineLinearQuantizer

__all__ = [
    "EisensteinModel",
    "MatchResult",
    "CascadeMatcher",
    "word_fingerprint",
    "text_fingerprint",
    "hamming_distance",
    "bitvector_similarity",
    "find_best_bitvector_match",
    "DeadbandCache",
    "DomainSIF",
    "BMAMonitor",
    "SplineLinearQuantizer",
]

__version__ = "0.1.0"
