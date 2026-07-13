import io
import tarfile
from pathlib import Path

from pytest_mock import MockerFixture

from rag_project.ingestion.fetch_docs import fetch_docs


def _make_fake_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        def add(name: str, content: bytes) -> None:
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

        add("fastapi-master/docs/en/docs/index.md", b"# FastAPI")
        add(
            "fastapi-master/docs/en/docs/tutorial/first-steps.md",
            b"# First Steps",
        )
        add("fastapi-master/docs/de/docs/index.md", b"# Nicht enthalten")
        add("fastapi-master/README.md", b"not docs")

    return buffer.getvalue()


def test_fetch_docs_extracts_only_matching_docs_subpath(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    fake_response = mocker.Mock()
    fake_response.content = _make_fake_archive()
    fake_response.raise_for_status = mocker.Mock()
    mock_get = mocker.patch(
        "rag_project.ingestion.fetch_docs.httpx.get", return_value=fake_response
    )

    dest = tmp_path / "raw"
    result = fetch_docs(dest_dir=dest, ref="master")

    assert result == dest
    assert (dest / "index.md").read_bytes() == b"# FastAPI"
    assert (dest / "tutorial" / "first-steps.md").read_bytes() == b"# First Steps"
    assert not (dest / "de").exists()
    assert not (dest / "README.md").exists()

    called_url = mock_get.call_args.args[0]
    assert "tiangolo/fastapi" in called_url
    assert "master" in called_url


def test_fetch_docs_is_safe_to_rerun(mocker: MockerFixture, tmp_path: Path) -> None:
    fake_response = mocker.Mock()
    fake_response.content = _make_fake_archive()
    fake_response.raise_for_status = mocker.Mock()
    mocker.patch(
        "rag_project.ingestion.fetch_docs.httpx.get", return_value=fake_response
    )

    dest = tmp_path / "raw"
    fetch_docs(dest_dir=dest, ref="master")
    fetch_docs(dest_dir=dest, ref="master")

    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*.md"))
    assert files == ["index.md", "tutorial/first-steps.md"]
