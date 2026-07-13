import hashlib

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_project.ingestion.chunker import Chunk

COLLECTION_NAME = "fastapi_docs"


def chunk_id(chunk: Chunk) -> str:
    """Deterministic ID from source path + content, so re-indexing the same
    chunk upserts in place instead of creating a duplicate."""
    digest_input = f"{chunk.source_path}:{chunk.content}".encode()
    return hashlib.sha256(digest_input).hexdigest()


def get_vector_store(persist_dir: str, embedding_model: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def index_chunks(chunks: list[Chunk], persist_dir: str, embedding_model: str) -> int:
    """Embed and upsert chunks into the persisted Chroma collection. Chroma
    metadata can't hold None, so a missing heading is stored as ''."""
    if not chunks:
        return 0

    store = get_vector_store(persist_dir, embedding_model)
    ids = [chunk_id(c) for c in chunks]
    texts = [c.content for c in chunks]
    metadatas = [{"source_path": c.source_path, "heading": c.heading or ""} for c in chunks]

    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(chunks)
