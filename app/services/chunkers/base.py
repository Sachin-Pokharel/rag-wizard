from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        pass


