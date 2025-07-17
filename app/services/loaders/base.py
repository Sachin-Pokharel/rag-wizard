# loaders/base_loader.py
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> List[Document]:
        """Loads one file or multiple documents from a directory and returns a list of document."""
        pass
