# embedder/openai_embedder.py
from .base import EmbedderBase
from openai import OpenAI
from typing import List
from app.core.config import settings

class OpenAIEmbedder(EmbedderBase):
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model_name = "text-embedding-3-small"
        self._dimension = 1536

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [record.embedding for record in response.data]


    @property
    def dimension(self) -> int:
        return self._dimension
