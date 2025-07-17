import pytest
from pathlib import Path
from .pymupdf_loader import PyMuPDFLoaderWrapper

PROJECT_ROOT = Path(__file__).parents[3]

SINGLE_PDF_PATH = PROJECT_ROOT / "data" / "The-Labour-Act-2017.pdf"
PDF_FOLDER_PATH = PROJECT_ROOT / "data" / "reports"

@pytest.fixture
def loader():
    return PyMuPDFLoaderWrapper()

def test_load_single_file(loader):
    docs = loader.load(str(SINGLE_PDF_PATH))
    assert isinstance(docs, list), "Should return a list"
    assert len(docs) > 0, "Documents list should not be empty"
    assert all(hasattr(doc, "page_content") for doc in docs), "Each doc should have page_content"

def test_load_folder(loader):
    docs = loader.load(str(PDF_FOLDER_PATH))
    assert isinstance(docs, list), "Should return a list"
    assert len(docs) > 1, "Should load multiple documents from folder"
    assert all(hasattr(doc, "page_content") for doc in docs), "Each doc should have page_content"

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
