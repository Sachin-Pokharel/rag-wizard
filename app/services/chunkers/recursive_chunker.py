from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .base import BaseChunker
from typing import List

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size=512, chunk_overlap=20):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        splitted_chunks = self.splitter.split_documents(documents)
        for chunk in splitted_chunks:
            chunk.metadata["text"] = chunk.page_content
        return splitted_chunks
