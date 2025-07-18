from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType,
)
from .base import VectorStoreBase
from typing import List, Dict, Any, Union
from app.services.embedder.embedder_registry import get_embedder
from app.services.embedder.base import EmbedderBase
import uuid
import logging


class MilvusVectorStore(VectorStoreBase):
    def __init__(self, collection_name: str, uri: str, token: str, dim: int):
        self.collection_name = collection_name
        self.dim = dim

        try:
            self.client = MilvusClient(uri=uri, token=token)
            # Test connection by listing collections
            collections = self.client.list_collections()
            print(f"Connected successfully. Collections: {collections}")
        except Exception as e:
            logging.error(f"Failed to connect to Milvus: {e}")
            raise ConnectionError(f"Milvus connection failed: {e}")

        # Create or get collection handle
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        collections = self.client.list_collections()
        if self.collection_name in collections:
            # Collection exists
            return

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=64,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                nullable=True,
            ),
        ]

        schema = CollectionSchema(fields, description="Embedding vector collection")
        self.client.create_collection(self.collection_name, schema=schema)

        # Create index on embedding field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self.client.create_index(self.collection_name, "embedding", index_params=index_params)

        self.client.load_collection(self.collection_name)


    def embed_and_store_documents(
        self,
        documents: List[Dict[str, Any]],
        embedder: Union[EmbedderBase, str],
    ):
        if isinstance(embedder, str):
            embedder = get_embedder(embedder)

        if not isinstance(embedder, EmbedderBase):
            raise ValueError(
                "embedder must be an EmbedderBase instance or valid embedder key"
            )

        entities = []
        texts = [doc.page_content for doc in documents]
        embeddings = embedder.embed_texts(texts)

        for i, doc in enumerate(documents):
            entities.append(
                {
                    "embeddings": embeddings[i],
                    "metadata": doc.metadata,
                }
            )
        
        self.client.insert(self.collection_name, entities)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["metadata"],
        )
        hits = results[0]
        return [
            {"score": hit.score, "metadata": hit.entity.get("metadata")}
            for hit in hits
        ]
