"""Read and normalize the PDF corpus while keeping page/section traceability.

Extraction uses PyMuPDF text blocks. Running headers/footers are removed by
position, heading lines are detected by font (bold, >= 10.5pt), and every
paragraph keeps its character span in the normalized page text so chunks can
be traced back to an exact position.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from .models import DocumentPage, Paragraph

HEADER_MAX_Y = 40.0     # running header sits at y~24-34 on a 792pt page
FOOTER_MIN_Y = 750.0    # "Access for free at openstax.org" sits at y~758
HEADING_MIN_SIZE = 10.5
MIN_CONTENT_CHARS = 500

_CHAR_MAP = str.maketrans({
    " ": " ",              # no-break space
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "�": "",               # replacement char (never seen in this corpus, but guard)
})


def normalize_text(text: str) -> str:
    """NFKC + straight quotes + collapsed spaces; paragraph breaks (\\n\\n) are kept."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_inline(text: str) -> str:
    """Single-line form used for one paragraph or heading."""
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def _line_text(line: dict) -> str:
    return "".join(span["text"] for span in line["spans"])


def _is_heading_line(line: dict) -> bool:
    spans = [s for s in line["spans"] if s["text"].strip()]
    return bool(spans) and all("Bold" in s["font"] and s["size"] >= HEADING_MIN_SIZE for s in spans)


def _join_lines(lines: list[str]) -> str:
    out = ""
    for raw in lines:
        piece = raw.strip()
        if not piece:
            continue
        if not out:
            out = piece
        elif out.endswith("-"):
            out += piece          # keep the hyphen, drop the line break ("well-\norganized")
        else:
            out += " " + piece
    return out


def _printed_page(header_texts: list[str]) -> int | None:
    for text in header_texts:
        text = normalize_inline(text)
        match = re.match(r"^(\d+)\s", text) or re.search(r"\s(\d+)$", text)
        if match:
            return int(match.group(1))
    return None


def section_starts(doc: fitz.Document, chapter: int, first: int, last: int) -> dict[int, list[tuple[str, str]]]:
    """TOC level-2 entries of one chapter, grouped by start page -> [(section_id, title_to_match)]."""
    starts: dict[int, list[tuple[str, str]]] = {}
    for level, title, page in doc.get_toc():
        if level != 2 or not (first <= page <= last):
            continue
        title = normalize_inline(title)
        match = re.match(rf"^({chapter}\.\d+)\s", title)
        section_id = match.group(1) if match else f"{chapter}.0"   # Chapter Outline / Introduction
        starts.setdefault(page, []).append((section_id, title))
    return starts


def load_pdf(path: Path, document_id: str, source: str, chapter: int, chapter_title: str,
             first_page: int, last_page: int, log: list[dict[str, Any]]) -> list[DocumentPage]:
    """Extract physical pages [first_page, last_page] (1-based) of one chapter.

    Pages with no text or little text are recorded in ``log``; OCR is not needed
    for this born-digital PDF, so it is not attempted.
    """
    doc = fitz.open(path)
    starts = section_starts(doc, chapter, first_page, last_page)
    toc_titles = {sid: title for page_starts in starts.values() for sid, title in page_starts}
    current = (f"{chapter}.0", "Introduction")
    pages: list[DocumentPage] = []

    for page_no in range(first_page, last_page + 1):
        page = doc[page_no - 1]
        blocks = page.get_text("dict")["blocks"]
        header_texts, raw_paragraphs = [], []          # raw_paragraphs: (text, is_heading)
        n_images = 0
        for block in blocks:
            if block["type"] != 0:
                n_images += 1
                continue
            y0, y1 = block["bbox"][1], block["bbox"][3]
            if y1 <= HEADER_MAX_Y or y0 >= FOOTER_MIN_Y:
                header_texts.append(" ".join(_line_text(l) for l in block["lines"]))
                continue
            pending: list[str] = []
            for line in block["lines"]:
                if _is_heading_line(line):
                    if pending:
                        raw_paragraphs.append((_join_lines(pending), False))
                        pending = []
                    raw_paragraphs.append((_line_text(line), True))
                else:
                    pending.append(_line_text(line))
            if pending:
                raw_paragraphs.append((_join_lines(pending), False))

        # merge consecutive heading lines that belong to one wrapped heading
        merged: list[tuple[str, bool]] = []
        for text, is_heading in raw_paragraphs:
            text = normalize_inline(text)
            if not text:
                continue
            if is_heading and merged and merged[-1][1] and merged[-1][0] + " " + text in toc_titles.values():
                merged[-1] = (merged[-1][0] + " " + text, True)
            elif (not is_heading and merged and not merged[-1][1]
                  and not re.search(r"[.!?:;)\"]$", merged[-1][0]) and re.match(r"[a-z(]", text)):
                # numbered-list layout puts the item's first line in its own block:
                # "...subgroups based on" + "specified factors, ..." -> one paragraph
                merged[-1] = (merged[-1][0] + " " + text, False)
            else:
                merged.append((text, is_heading))

        pending_starts = list(starts.get(page_no, []))
        paragraphs: list[Paragraph] = []
        parts: list[str] = []
        offset = 0
        for text, is_heading in merged:
            if is_heading and pending_starts:
                for i, (sid, title) in enumerate(pending_starts):
                    if text == title:
                        current = (sid, "Introduction" if sid.endswith(".0") else title)
                        pending_starts = pending_starts[i + 1:]
                        break
            if parts:
                offset += 2                                  # the "\n\n" separator
            paragraphs.append(Paragraph(offset, offset + len(text), is_heading, current[0], current[1]))
            parts.append(text)
            offset += len(text)

        for sid, title in pending_starts:
            log.append({"page": page_no, "level": "warning",
                        "message": f"TOC section '{title}' not matched to a heading; section metadata may lag"})
            current = (sid, title)

        text = "\n\n".join(parts)
        if not text:
            log.append({"page": page_no, "level": "error", "message": "no extractable text"})
        elif len(text) < MIN_CONTENT_CHARS:
            log.append({"page": page_no, "level": "info",
                        "message": f"low-content page ({len(text)} chars) - not counted as content page"})

        pages.append(DocumentPage(
            document_id=document_id, source=source, page=page_no, text=text,
            paragraphs=paragraphs,
            metadata={
                "chapter": chapter,
                "chapter_title": chapter_title,
                "printed_page": _printed_page(header_texts),
                "n_images": n_images,
                "n_chars": len(text),
                "is_content_page": len(text) >= MIN_CONTENT_CHARS,
            },
        ))
    return pages
