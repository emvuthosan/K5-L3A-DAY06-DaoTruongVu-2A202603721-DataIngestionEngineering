"""Shared data contracts; keep traceability from chunk back to source."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Paragraph:
    """A contiguous slice [start, end) of the page's normalized text."""
    start: int
    end: int
    is_heading: bool
    section_id: str
    section_title: str


@dataclass
class DocumentPage:
    document_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
