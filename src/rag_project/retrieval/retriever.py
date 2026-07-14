from pydantic import BaseModel

from rag_project.ingestion.indexer import get_vector_store


class RetrievedChunk(BaseModel):
    content: str
    source_path: str
    score: float


class Retriever:
    """Holds a single Chroma store instance so the embedding model is loaded
    once per process rather than on every query (see get_vector_store)."""

    def __init__(self, persist_dir: str, embedding_model: str) -> None:
        self._store = get_vector_store(persist_dir, embedding_model)

    def similarity_search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        results = self._store.similarity_search_with_score(query, k=k)
        return [
            RetrievedChunk(
                content=doc.page_content,
                source_path=doc.metadata.get("source_path", ""),
                score=score,
            )
            for doc, score in results
        ]
