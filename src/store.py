from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.Client()
            try:
                client.delete_collection(name=self._collection_name)
            except Exception:
                pass
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata)
        metadata["doc_id"] = doc.id
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []
        query_vector = self._embedding_fn(query)
        scored_records = []
        for r in records:
            from .chunking import compute_similarity
            score = compute_similarity(query_vector, r["embedding"])
            scored_records.append({
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "score": score
            })
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        records = [self._make_record(doc) for doc in docs]
        if self._use_chroma and self._collection is not None:
            ids = [r["id"] for r in records]
            documents = [r["content"] for r in records]
            embeddings = [r["embedding"] for r in records]
            metadatas = [r["metadata"] for r in records]
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            try:
                query_embedding = self._embedding_fn(query)
                res = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                formatted = []
                if res and "documents" in res and res["documents"]:
                    ids = res["ids"][0]
                    documents = res["documents"][0]
                    metadatas = res["metadatas"][0]
                    distances = res.get("distances", [[]])[0]
                    for i in range(len(ids)):
                        dist = distances[i] if i < len(distances) else 0.0
                        score = 1.0 - dist
                        formatted.append({
                            "id": ids[i],
                            "content": documents[i],
                            "metadata": metadatas[i],
                            "score": score
                        })
                return formatted
            except Exception:
                return self._search_records(query, self._store, top_k)
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma and self._collection is not None:
            try:
                query_embedding = self._embedding_fn(query)
                where_clause = metadata_filter if metadata_filter else None
                res = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_clause
                )
                formatted = []
                if res and "documents" in res and res["documents"]:
                    ids = res["ids"][0]
                    documents = res["documents"][0]
                    metadatas = res["metadatas"][0]
                    distances = res.get("distances", [[]])[0]
                    for i in range(len(ids)):
                        dist = distances[i] if i < len(distances) else 0.0
                        score = 1.0 - dist
                        formatted.append({
                            "id": ids[i],
                            "content": documents[i],
                            "metadata": metadatas[i],
                            "score": score
                        })
                return formatted
            except Exception:
                pass

        # In-memory fallback
        filtered_records = []
        for r in self._store:
            matches = True
            if metadata_filter:
                for k, v in metadata_filter.items():
                    if r["metadata"].get(k) != v:
                        matches = False
                        break
            if matches:
                filtered_records.append(r)
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        removed = False
        initial_size = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        if len(self._store) < initial_size:
            removed = True
        
        if self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(where={"doc_id": doc_id})
            except Exception:
                pass
        return removed
