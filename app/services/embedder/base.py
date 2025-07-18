from abc import ABC, abstractmethod
from typing import List

class EmbedderBase(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Returns list of embedding vectors for the input texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimension of the embedding vector."""
        pass
