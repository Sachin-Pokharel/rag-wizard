from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from app.services.embedder.embedder_registry import get_embedder
from app.services.embedder.base import EmbedderBase
from typing import List, Dict, Any, Union
from .base import VectorStoreBase
import logging


class MilvusVectorStore(VectorStoreBase):
    TEXT_FIELD = "text"
    DENSE_FIELD = "dense_vector"
    PK_FIELD = "pk"

    def __init__(self, collection_name: str, uri: str, token: str, dim: int, overwrite: bool = False):
        self.collection_name = collection_name
        self.dim = dim
        self.overwrite = overwrite
        connections.connect(uri=uri, token=token)
        logging.info(f"Connected to Milvus at {uri}")
        
        if utility.has_collection(self.collection_name) and not overwrite:
            self.collection = Collection(self.collection_name)
            logging.info(f"Using existing collection: {self.collection_name}")
        else:
            self._create_collection()
        
        self.collection.load()

    def _create_collection(self):
        if utility.has_collection(self.collection_name):
            logging.info(f"Dropping existing collection: {self.collection_name}")
            Collection(self.collection_name).drop()

        fields = [
            FieldSchema(
                name=self.PK_FIELD,
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=100
            ),
            FieldSchema(name=self.TEXT_FIELD, dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name=self.DENSE_FIELD, dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]

        schema = CollectionSchema(fields, description="Hybrid vector search demo")
        self.collection = Collection(name=self.collection_name, schema=schema, consistency_level="Strong")

        self.collection.create_index(field_name=self.DENSE_FIELD, index_params={
            "index_type": "AUTOINDEX", "metric_type": "IP"
        })

        logging.info(f"Created new collection: {self.collection_name}")

    def load_existing_collection(self):
        """Loads an existing collection into memory (used after instantiation)."""
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' does not exist.")
        self.collection = Collection(self.collection_name)
        self.collection.load()
        logging.info(f"Loaded existing collection: {self.collection_name}")

    def embed_and_store_documents(
        self,
        documents: List[Any],
        embedder: Union[EmbedderBase, str]
    ):
        """Embeds documents and inserts them into the collection."""
        if isinstance(embedder, str):
            embedder = get_embedder(embedder)
        if not isinstance(embedder, EmbedderBase):
            raise ValueError("Embedder must be an instance of EmbedderBase or string key")

        texts = [doc.page_content for doc in documents]
        dense_embs = embedder.embed_texts(texts)

        data = [
            None,       # For auto_id=True, primary key is None
            texts,
            dense_embs,
        ]

        self.collection.insert(data)
        self.collection.flush()
        logging.info(f"Inserted {len(texts)} documents into '{self.collection_name}'")

    def search_dense(self, query: str, embedder: Union[EmbedderBase, str], top_k: int = 5):
        """Performs dense (vector) search using the query string."""
        if isinstance(embedder, str):
            embedder = get_embedder(embedder)
        if not isinstance(embedder, EmbedderBase):
            raise ValueError("Embedder must be an instance of EmbedderBase or string key")

        dense_vec = embedder.embed_query(query)

        search_params = {
            "metric_type": "IP",
            "params": {}
        }

        results = self.collection.search(
            data=[dense_vec],
            anns_field=self.DENSE_FIELD,
            param=search_params,
            limit=top_k,
            output_fields=[self.TEXT_FIELD]
        )

        logging.info(f"Search completed. Found {len(results[0]) if results else 0} results.")
        return results[0] if results else []

    def drop(self):
        """Drops the collection."""
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            logging.info(f"Dropped collection '{self.collection_name}'")
