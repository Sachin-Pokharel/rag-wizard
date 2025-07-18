from typing import List, Dict, Any, Optional

class VectorStoreBase:
    def create_collection(self, collection_name: str, dim: int) -> None:
        """Create collection/schema with embedding dimension."""
        raise NotImplementedError()

    def insert_embeddings(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]],
        sparse_embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Insert embeddings and metadata into the vector store."""
        raise NotImplementedError()

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings, returns list of dicts with score and metadata."""
        raise NotImplementedError()
