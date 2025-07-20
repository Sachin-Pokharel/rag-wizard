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

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text]
        )
        return response.data[0].embedding


    @property
    def dimension(self) -> int:
        return self._dimension
