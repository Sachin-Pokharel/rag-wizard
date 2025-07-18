from .pymupdf_loader import PyMuPDFLoaderWrapper
from .base import BaseLoader

# More loaders can be imported and registered here
LOADER_REGISTRY = {
    "pymupdf": PyMuPDFLoaderWrapper,
    # "pdfminer": PDFMinerLoaderWrapper,
    # "unstructured": UnstructuredPDFLoaderWrapper,
}

def get_loader(loader_type: str) -> BaseLoader:
    loader_class = LOADER_REGISTRY.get(loader_type.lower())
    if not loader_class:
        raise ValueError(f"Unsupported loader type: {loader_type}")
    return loader_class()
