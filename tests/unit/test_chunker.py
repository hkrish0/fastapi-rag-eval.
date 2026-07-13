from pathlib import Path

from rag_project.ingestion.chunker import chunk_file, chunk_text


def test_short_doc_produces_a_single_chunk() -> None:
    text = "# Title\n\nJust one short paragraph.\n"
    chunks = chunk_text(text, source_path="short.md", max_chars=800)

    assert len(chunks) == 1
    assert chunks[0].heading == "Title"
    assert "Just one short paragraph." in chunks[0].content


def test_nested_headers_are_tracked_per_chunk() -> None:
    text = (
        "# Top\n\n"
        "Intro text.\n\n"
        "## Sub A\n\n"
        "Content for sub A, long enough to force a new chunk. " + ("x" * 60) + "\n\n"
        "## Sub B\n\n"
        "Content for sub B, also padded out a fair bit. " + ("y" * 60) + "\n"
    )
    chunks = chunk_text(text, source_path="nested.md", max_chars=100)

    assert len(chunks) >= 2
    headings = [c.heading for c in chunks]
    assert "Sub A" in headings
    assert "Sub B" in headings


def test_code_block_is_never_split_across_chunks() -> None:
    code_lines = "\n".join(f"line_{i} = {i}" for i in range(50))
    text = f"# Example\n\nHere is some code:\n\n```python\n{code_lines}\n```\n"
    # max_chars smaller than the code block itself
    chunks = chunk_text(text, source_path="code.md", max_chars=50)

    code_chunks = [c for c in chunks if "```python" in c.content]
    assert len(code_chunks) == 1
    assert code_chunks[0].content.count("```") == 2
    for i in range(50):
        assert f"line_{i} = {i}" in code_chunks[0].content


def test_long_blank_line_free_list_still_gets_split() -> None:
    # A long bullet list with no blank lines between items (like FastAPI's
    # release-notes.md) must not become a single oversized chunk.
    text = "# Release Notes\n\n" + "\n".join(
        f"* Entry number {i} with some descriptive text padding it out." for i in range(200)
    )
    chunks = chunk_text(text, source_path="release-notes.md", max_chars=800)

    assert len(chunks) > 1
    assert all(len(c.content) <= 900 for c in chunks)  # some slack for join separators


def test_chunk_file_sets_source_path_relative_to_raw_root(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    nested = raw_root / "tutorial" / "first-steps.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("# First Steps\n\nSome content.\n")

    chunks = chunk_file(nested, raw_root=raw_root, max_chars=800)

    assert len(chunks) == 1
    assert chunks[0].source_path == "tutorial/first-steps.md"
