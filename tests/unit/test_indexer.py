from pytest_mock import MockerFixture

from rag_project.ingestion.chunker import Chunk
from rag_project.ingestion.indexer import chunk_id, index_chunks


def test_chunk_id_is_deterministic_and_content_sensitive() -> None:
    chunk = Chunk(content="Hello world", source_path="a/b.md", heading="Intro")
    assert chunk_id(chunk) == chunk_id(chunk)

    different_content = Chunk(content="Different", source_path="a/b.md", heading="Intro")
    assert chunk_id(chunk) != chunk_id(different_content)

    different_source = Chunk(content="Hello world", source_path="a/c.md", heading="Intro")
    assert chunk_id(chunk) != chunk_id(different_source)


def test_index_chunks_upserts_with_deterministic_ids(mocker: MockerFixture) -> None:
    fake_store = mocker.Mock()
    mocker.patch("rag_project.ingestion.indexer.get_vector_store", return_value=fake_store)

    chunks = [
        Chunk(content="A", source_path="x.md", heading="H1"),
        Chunk(content="B", source_path="x.md", heading=None),
    ]

    count = index_chunks(chunks, persist_dir="unused", embedding_model="unused")

    assert count == 2
    fake_store.add_texts.assert_called_once()
    _, kwargs = fake_store.add_texts.call_args
    assert kwargs["texts"] == ["A", "B"]
    assert kwargs["metadatas"] == [
        {"source_path": "x.md", "heading": "H1"},
        {"source_path": "x.md", "heading": ""},
    ]
    assert kwargs["ids"] == [chunk_id(c) for c in chunks]


def test_index_chunks_empty_list_is_noop(mocker: MockerFixture) -> None:
    fake_get_store = mocker.patch("rag_project.ingestion.indexer.get_vector_store")

    count = index_chunks([], persist_dir="unused", embedding_model="unused")

    assert count == 0
    fake_get_store.assert_not_called()
