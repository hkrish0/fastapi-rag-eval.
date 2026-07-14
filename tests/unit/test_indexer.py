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
    fake_store.get.return_value = {"ids": []}
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


def test_index_chunks_deletes_stale_ids_not_in_current_chunk_set(mocker: MockerFixture) -> None:
    fake_store = mocker.Mock()
    mocker.patch("rag_project.ingestion.indexer.get_vector_store", return_value=fake_store)

    chunks = [Chunk(content="A", source_path="x.md", heading="H1")]
    current_id = chunk_id(chunks[0])
    stale_id = "stale-id-from-a-previous-run-whose-source-content-changed"
    fake_store.get.return_value = {"ids": [current_id, stale_id]}

    index_chunks(chunks, persist_dir="unused", embedding_model="unused")

    fake_store.delete.assert_called_once_with(ids=[stale_id])


def test_index_chunks_skips_delete_when_no_stale_ids(mocker: MockerFixture) -> None:
    fake_store = mocker.Mock()
    mocker.patch("rag_project.ingestion.indexer.get_vector_store", return_value=fake_store)

    chunks = [Chunk(content="A", source_path="x.md", heading="H1")]
    fake_store.get.return_value = {"ids": [chunk_id(chunks[0])]}

    index_chunks(chunks, persist_dir="unused", embedding_model="unused")

    fake_store.delete.assert_not_called()


def test_index_chunks_empty_list_is_noop(mocker: MockerFixture) -> None:
    fake_get_store = mocker.patch("rag_project.ingestion.indexer.get_vector_store")

    count = index_chunks([], persist_dir="unused", embedding_model="unused")

    assert count == 0
    fake_get_store.assert_not_called()
