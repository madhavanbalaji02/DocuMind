"""Hierarchical parent-child chunker for layout-aware documents.

PARENT chunks (600-1200 tokens): one per section header or page
CHILD chunks (100-300 tokens): overlap 20%, never split mid-sentence
Tables: entire table = one parent, each row = one child (with header)
Figures: one parent + one child per figure
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aws.textract.parser import LayoutDocument

PARENT_MIN_TOKENS = 600
PARENT_MAX_TOKENS = 1200
CHILD_MIN_TOKENS = 100
CHILD_MAX_TOKENS = 300
OVERLAP_RATIO = 0.20  # 20% overlap between children

_WORDS_PER_TOKEN = 0.75  # rough approximation


def _token_count(text: str) -> int:
    return int(len(text.split()) / _WORDS_PER_TOKEN)


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries (period/exclamation/question followed by space)."""
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text.strip())
    return [s for s in sentences if s.strip()]


@dataclass
class ParentChildPair:
    parent_content: str
    parent_section: str
    parent_element_type: str  # text|table|figure|mixed
    parent_page_start: int
    parent_page_end: int
    children: list["ChildChunkData"] = field(default_factory=list)


@dataclass
class ChildChunkData:
    content: str
    element_type: str  # text|table_row|figure_desc
    page_number: int
    chunk_index: int


class HierarchicalChunker:
    """Chunk a LayoutDocument into hierarchical parent-child pairs."""

    def chunk(self, layout_doc: "LayoutDocument") -> list[ParentChildPair]:
        from aws.textract.parser import ElementType

        pairs: list[ParentChildPair] = []
        current_section_title = ""
        current_texts: list[tuple[str, int]] = []  # (text, page)

        def _flush_text_section() -> None:
            if not current_texts:
                return
            full_text = " ".join(t for t, _ in current_texts)
            pages = [p for _, p in current_texts]
            pair = ParentChildPair(
                parent_content=full_text,
                parent_section=current_section_title,
                parent_element_type="text",
                parent_page_start=pages[0] if pages else 1,
                parent_page_end=pages[-1] if pages else 1,
                children=_make_text_children(full_text, pages[0] if pages else 1),
            )
            pairs.append(pair)
            current_texts.clear()

        for element in layout_doc.elements:
            if element.element_type in (ElementType.TITLE, ElementType.SECTION_HEADER):
                _flush_text_section()
                current_section_title = element.content

            elif element.element_type == ElementType.TEXT:
                current_texts.append((element.content, element.page_number))
                # Flush when parent would exceed max tokens
                total = _token_count(" ".join(t for t, _ in current_texts))
                if total >= PARENT_MAX_TOKENS:
                    _flush_text_section()

            elif element.element_type == ElementType.TABLE:
                _flush_text_section()
                pair = _make_table_pair(element, current_section_title)
                pairs.append(pair)

            elif element.element_type == ElementType.FIGURE:
                _flush_text_section()
                pair = _make_figure_pair(element, current_section_title)
                pairs.append(pair)

        _flush_text_section()
        return pairs


def _make_text_children(text: str, page: int) -> list[ChildChunkData]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    children: list[ChildChunkData] = []
    current_tokens = 0
    current_sents: list[str] = []
    chunk_index = 0
    overlap_sents: list[str] = []

    for sent in sentences:
        sent_tokens = _token_count(sent)

        if current_tokens + sent_tokens > CHILD_MAX_TOKENS and current_sents:
            child_text = " ".join(current_sents)
            if _token_count(child_text) >= CHILD_MIN_TOKENS:
                children.append(ChildChunkData(
                    content=child_text,
                    element_type="text",
                    page_number=page,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

            # Overlap: keep last 20% of sentences
            overlap_count = max(1, int(len(current_sents) * OVERLAP_RATIO))
            overlap_sents = current_sents[-overlap_count:]
            current_sents = list(overlap_sents)
            current_tokens = sum(_token_count(s) for s in current_sents)

        current_sents.append(sent)
        current_tokens += sent_tokens

    # Last chunk
    if current_sents:
        child_text = " ".join(current_sents)
        if child_text.strip():
            children.append(ChildChunkData(
                content=child_text,
                element_type="text",
                page_number=page,
                chunk_index=chunk_index,
            ))

    return children


def _make_table_pair(element, section_title: str) -> ParentChildPair:
    """One parent per table, one child per row (with header prepended)."""
    table_md = element.content
    lines = table_md.strip().splitlines()

    header_line = ""
    separator_line = ""
    data_rows: list[str] = []
    for i, line in enumerate(lines):
        if i == 0:
            header_line = line
        elif i == 1 and all(c in "|-: " for c in line):
            separator_line = line
        else:
            if line.strip():
                data_rows.append(line)

    children: list[ChildChunkData] = []
    for idx, row in enumerate(data_rows):
        row_text = f"{header_line}\n{separator_line}\n{row}" if header_line else row
        children.append(ChildChunkData(
            content=row_text,
            element_type="table_row",
            page_number=element.page_number,
            chunk_index=idx,
        ))

    # If no rows, treat whole table as single child
    if not children:
        children.append(ChildChunkData(
            content=table_md,
            element_type="table_row",
            page_number=element.page_number,
            chunk_index=0,
        ))

    return ParentChildPair(
        parent_content=table_md,
        parent_section=section_title,
        parent_element_type="table",
        parent_page_start=element.page_number,
        parent_page_end=element.page_number,
        children=children,
    )


def _make_figure_pair(element, section_title: str) -> ParentChildPair:
    """One parent + one child per figure description."""
    description = element.content or "[Figure without description]"
    return ParentChildPair(
        parent_content=description,
        parent_section=section_title,
        parent_element_type="figure",
        parent_page_start=element.page_number,
        parent_page_end=element.page_number,
        children=[ChildChunkData(
            content=description,
            element_type="figure_desc",
            page_number=element.page_number,
            chunk_index=0,
        )],
    )
