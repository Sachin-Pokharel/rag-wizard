from .recursive_chunker import RecursiveChunker
# from .semantic_chunker import SemanticChunker  # when ready
from .base import BaseChunker

CHUNKER_REGISTRY = {
    "recursive": RecursiveChunker,
    # "semantic": SemanticChunker,
}

def get_chunker(chunker_type: str, **kwargs) -> BaseChunker:
    chunker_class = CHUNKER_REGISTRY.get(chunker_type.lower())
    if not chunker_class:
        raise ValueError(f"Unsupported chunker type: {chunker_type}")
    return chunker_class(**kwargs)
