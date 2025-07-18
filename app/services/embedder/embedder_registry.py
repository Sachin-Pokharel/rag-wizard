from .openai_embedder import OpenAIEmbedder
# import other embedders here when ready

EMBEDDER_REGISTRY = {
    "openai": OpenAIEmbedder,
    # "huggingface": HuggingFaceEmbedder,
    # etc.
}

def get_embedder(name: str):
    embedder_cls = EMBEDDER_REGISTRY.get(name.lower())
    if embedder_cls is None:
        raise ValueError(f"Embedder '{name}' not found in registry")
    return embedder_cls()
