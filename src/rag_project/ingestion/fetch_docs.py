import io
import tarfile
from pathlib import Path

import httpx

REPO_ARCHIVE_URL = "https://codeload.github.com/tiangolo/fastapi/tar.gz/refs/heads/{ref}"
DOCS_SUBPATH = "docs/en/docs/"


def fetch_docs(dest_dir: str | Path = "data/raw", ref: str = "master") -> Path:
    """Download the FastAPI docs/en/docs markdown tree into dest_dir.

    Safe to re-run: files are overwritten in place, so repeated calls update
    rather than duplicate.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    url = REPO_ARCHIVE_URL.format(ref=ref)
    response = httpx.get(url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()

    resolved_dest = dest.resolve()

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            marker_index = member.name.find(DOCS_SUBPATH)
            if marker_index == -1:
                continue
            relative_path = member.name[marker_index + len(DOCS_SUBPATH) :]
            if not relative_path:
                continue

            target = (dest / relative_path).resolve()
            if not target.is_relative_to(resolved_dest):
                continue  # tar-slip guard: skip entries that escape dest

            extracted_file = tar.extractfile(member)
            if extracted_file is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(extracted_file.read())

    return dest
