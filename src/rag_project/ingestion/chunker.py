import re
from pathlib import Path

from pydantic import BaseModel

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
FENCE_RE = re.compile(r"^```")


class Chunk(BaseModel):
    content: str
    source_path: str
    heading: str | None = None


def _split_blocks(text: str) -> list[str]:
    """Split markdown into packing units. A fenced code block is one atomic
    unit (never split internally). Everything else is split per line, so
    long blank-line-free runs (e.g. large bullet/table lists) can still be
    broken up during packing instead of becoming one oversized chunk."""
    blocks: list[str] = []
    code_buffer: list[str] = []
    in_code_block = False

    for line in text.splitlines():
        if FENCE_RE.match(line):
            code_buffer.append(line)
            if in_code_block:
                blocks.append("\n".join(code_buffer))
                code_buffer = []
            in_code_block = not in_code_block
            continue

        if in_code_block:
            code_buffer.append(line)
            continue

        if line.strip() == "":
            continue

        blocks.append(line)

    if code_buffer:
        blocks.append("\n".join(code_buffer))

    return blocks


def _heading_for_block(block: str, current: str | None) -> str | None:
    match = HEADING_RE.match(block.splitlines()[0])
    return match.group(2).strip() if match else current


def chunk_text(text: str, source_path: str, max_chars: int) -> list[Chunk]:
    """Greedily pack blocks into chunks under max_chars. A single block that
    exceeds max_chars on its own (e.g. a large code block) is still kept
    whole as its own chunk rather than split."""
    blocks = _split_blocks(text)

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_len = 0
    running_heading: str | None = None
    chunk_heading: str | None = None

    def flush_chunk() -> None:
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append(
                Chunk(
                    content="\n\n".join(current_parts).strip(),
                    source_path=source_path,
                    heading=chunk_heading,
                )
            )
        current_parts = []
        current_len = 0

    for block in blocks:
        running_heading = _heading_for_block(block, running_heading)

        if current_parts and current_len + len(block) + 2 > max_chars:
            flush_chunk()

        if not current_parts:
            chunk_heading = running_heading

        current_parts.append(block)
        current_len += len(block) + 2

    flush_chunk()
    return chunks


def chunk_file(path: Path, raw_root: Path, max_chars: int) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    source_path = path.relative_to(raw_root).as_posix()
    return chunk_text(text, source_path=source_path, max_chars=max_chars)
