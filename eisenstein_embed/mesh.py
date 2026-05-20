"""SuperInstance mesh registration for eisenstein-embed."""

try:
    from plato_core.registry import registry
except ImportError:
    registry = None  # Standalone mode — no mesh


def register_eisenstein(registry):
    """Register eisenstein-embed capabilities with the mesh."""
    from eisenstein_embed.static_model import EisensteinModel
    from eisenstein_embed.bitvector import word_fingerprint, text_fingerprint

    registry.register("matchers", "eisenstein-cascade", EisensteinModel)
    registry.register("matchers", "eisenstein-bitvector", lambda: {
        "word_fingerprint": word_fingerprint,
        "text_fingerprint": text_fingerprint,
    })
    registry.register("encoders", "eisenstein", EisensteinModel)
