from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from app.utils.helpers import clean_and_normalize_text
from .base import BaseLoader
import uuid


class PyMuPDFLoaderWrapper(BaseLoader):
    def load(self, path: str) -> List[Document]:
        documents = []

        p = Path(path)
        if p.is_file():
            loader = PyMuPDFLoader(str(p))
            raw_docs = loader.load()
            documents.extend(self._extract_metadata(raw_docs))
        elif p.is_dir():
            for file_path in p.glob("*.pdf"):
                loader = PyMuPDFLoader(file_path=str(file_path))
                raw_docs = loader.load()
                documents.extend(self._extract_metadata(raw_docs))
        else:
            raise FileNotFoundError(f"Path '{path}' is neither a file nor a directory.")

        return documents

    def _extract_metadata(self, raw_docs: List[Document]) -> List[Document]:
        new_docs = []
        for doc in raw_docs:
            metadata = doc.metadata
            page_no = metadata.get("page", 0)

            new_metadata = {
                "document_id": str(uuid.uuid4()),
                "source": metadata.get("source") or "",
                "title": metadata.get("title") or "",
                "chunk_text_preview": clean_and_normalize_text(doc.page_content[:500]),
                "page_no": page_no,
                "created_at": metadata.get("creationDate")
                or metadata.get("creationdate")
                or "",
                "category": "",
                "author": metadata.get("author") or "",
            }

            new_docs.append(
                Document(
                    page_content=clean_and_normalize_text(doc.page_content),
                    metadata=new_metadata,
                )
            )

        return new_docs
