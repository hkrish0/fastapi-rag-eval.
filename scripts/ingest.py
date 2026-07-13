from rag_project.config import get_settings
from rag_project.ingestion.chunker import chunk_file
from rag_project.ingestion.fetch_docs import fetch_docs
from rag_project.ingestion.indexer import index_chunks


def main() -> None:
    settings = get_settings()

    raw_root = fetch_docs(dest_dir=settings.raw_docs_dir)
    md_files = sorted(raw_root.rglob("*.md"))

    chunks = [
        chunk
        for md_file in md_files
        for chunk in chunk_file(md_file, raw_root, settings.chunk_size)
    ]

    indexed_count = index_chunks(
        chunks,
        persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
    )

    print(
        f"Fetched {len(md_files)} docs, indexed {indexed_count} chunks "
        f"into {settings.chroma_persist_dir}"
    )


if __name__ == "__main__":
    main()
