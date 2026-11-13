"""Fixed-size (strategy A) and structure-aware (strategy B/C) chunking.

Both strategies chunk the same normalized page text and every chunk is a
contiguous slice page.text[char_start:char_end], so a chunk can be traced to an
exact position and both strategies can be compared on the same positions.
Chunks never cross a page boundary (keeps `page` metadata exact; see report
for the trade-off).
"""
from __future__ import annotations

import hashlib
import re

from .models import Chunk, DocumentPage, Paragraph

HEADING_ONLY_MAX_CHARS = 120
MIN_CHUNK_CHARS = 150
MERGE_SLACK = 1.2


def _section_at(page: DocumentPage, offset: int) -> Paragraph:
    for para in page.paragraphs:
        if offset < para.end + 2:            # offset falls in paragraph or its trailing separator
            return para
    return page.paragraphs[-1]


def _make_chunk(page: DocumentPage, strategy: str, index: int, start: int, end: int) -> Chunk:
    text = page.text[start:end]
    para = _section_at(page, start)
    return Chunk(
        chunk_id=f"{strategy}:{page.document_id}:p{page.page:04d}:c{index:03d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={
            "chapter": page.metadata["chapter"],
            "chapter_title": page.metadata["chapter_title"],
            "section_id": para.section_id,
            "section_title": para.section_title,
            "printed_page": page.metadata.get("printed_page") or -1,
            "char_start": start,
            "char_end": end,
            "text_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
            "strategy": strategy,
        },
    )


def fixed_size(page: DocumentPage, size: int, overlap: int, strategy: str = "A") -> list[Chunk]:
    """Character windows of `size` with `overlap`; whitespace at window edges is trimmed."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        s, e = start, end
        while s < e and page.text[s].isspace():
            s += 1
        while e > s and page.text[e - 1].isspace():
            e -= 1
        if e > s:
            chunks.append(_make_chunk(page, strategy, len(chunks), s, e))
        if end == len(page.text):
            break
        start += step
    return chunks


def _split_long(page: DocumentPage, para: Paragraph, max_size: int) -> list[tuple[int, int]]:
    """Split an oversized paragraph at sentence boundaries, hard-splitting a giant sentence."""
    text = page.text[para.start:para.end]
    bounds = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", text)] + [len(text)]
    pieces: list[tuple[int, int]] = []
    cur_start = 0
    for i in range(1, len(bounds)):
        if bounds[i] - cur_start > max_size and bounds[i - 1] > cur_start:
            pieces.append((cur_start, bounds[i - 1]))
            cur_start = bounds[i - 1]
    pieces.append((cur_start, len(text)))
    out: list[tuple[int, int]] = []
    for s, e in pieces:
        while e - s > max_size:                      # a single sentence longer than max_size
            out.append((s, s + max_size))
            s += max_size
        out.append((s, e))
    return [(para.start + s, para.start + len(text[:e].rstrip())) for s, e in out if text[s:e].strip()]


def by_paragraph(page: DocumentPage, max_size: int, strategy: str = "B") -> list[Chunk]:
    """Group whole paragraphs up to `max_size` chars; a heading always starts a new chunk.

    A heading stays attached to the paragraphs that follow it. Chunks that would
    contain only a heading (e.g. a heading at the bottom of a page whose body is
    on the next page) are dropped because they carry no answerable content.
    """
    spans: list[tuple[int, int, bool]] = []          # (start, end, heading_only)
    cur: list[Paragraph] = []

    def flush() -> None:
        if cur:
            heading_only = all(p.is_heading for p in cur)
            spans.append((cur[0].start, cur[-1].end, heading_only))
            cur.clear()

    for para in page.paragraphs:
        length = para.end - para.start
        if para.is_heading:
            if cur and not all(p.is_heading for p in cur):
                flush()                               # heading opens a new chunk
            cur.append(para)
            continue
        if length > max_size:
            # flush grouped content but keep a dangling heading with the first piece
            heading_start = cur[0].start if cur and all(p.is_heading for p in cur) else None
            if heading_start is None:
                flush()
            else:
                cur.clear()
            pieces = _split_long(page, para, max_size)
            for i, (s, e) in enumerate(pieces):
                spans.append((heading_start if (i == 0 and heading_start is not None) else s, e, False))
            continue
        if cur and not all(p.is_heading for p in cur) and para.end - cur[0].start > max_size:
            flush()
        cur.append(para)
    flush()

    heading_starts = {p.start for p in page.paragraphs if p.is_heading}
    kept = [[s, e] for s, e, heading_only in spans
            if not (heading_only and e - s <= HEADING_ONLY_MAX_CHARS)]
    kept = _merge_small(kept, heading_starts, max_size)
    return [_make_chunk(page, strategy, i, s, e) for i, (s, e) in enumerate(kept)]


def _merge_small(spans: list[list[int]], heading_starts: set[int], max_size: int) -> list[list[int]]:
    """Merge fragments < MIN_CHUNK_CHARS (page-boundary sentence tails, 'Solution', captions)
    into the following chunk, or the previous one when the next chunk opens a new heading.
    Merged chunks may reach MERGE_SLACK * max_size."""
    limit = int(MERGE_SLACK * max_size)
    i = 0
    while i < len(spans):
        s, e = spans[i]
        if e - s >= MIN_CHUNK_CHARS or len(spans) == 1:
            i += 1
            continue
        nxt = spans[i + 1] if i + 1 < len(spans) else None
        prev = spans[i - 1] if i > 0 else None
        if nxt and nxt[0] not in heading_starts and nxt[1] - s <= limit:
            nxt[0] = s
            del spans[i]
        elif prev and s not in heading_starts and e - prev[0] <= limit:
            prev[1] = e
            del spans[i]
        else:
            i += 1
    return spans
