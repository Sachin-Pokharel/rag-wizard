from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from .base import BaseLoader

class PyMuPDFLoaderWrapper(BaseLoader):
    def load(self, path: str) -> List[Document]:
        documents = []

        p = Path(path)
        if p.is_file():
            loader = PyMuPDFLoader(str(p))
            documents.extend(loader.load())
        elif p.is_dir():
            for file_path in p.glob("*.pdf"):
                loader = PyMuPDFLoader(file_path=str(file_path))
                documents.extend(loader.load())
        else:
            raise FileNotFoundError(f"Path '{path}' is neither a file nor a directory.")
        
        return documents



