"""Layout-aware PDF parsing via Amazon Textract.

COST GUARD:
  DetectDocumentText: $0.0015/page (plain text PDFs)
  AnalyzeDocument:    $0.015/page  (PDFs with tables/forms)
  Max pages: 50 (configurable via TEXTRACT_MAX_PAGES)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import boto3
from botocore.exceptions import ClientError

from aws.config import REGION, S3_BUCKET, TEXTRACT_MAX_PAGES


class ElementType(str, Enum):
    TITLE = "TITLE"
    SECTION_HEADER = "SECTION_HEADER"
    TEXT = "TEXT"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    KEY_VALUE = "KEY_VALUE"


@dataclass
class BoundingBox:
    left: float
    top: float
    width: float
    height: float


@dataclass
class LayoutElement:
    element_type: ElementType
    content: str
    page_number: int
    bounding_box: BoundingBox
    confidence: float
    reading_order: int
    parent_section: str = ""
    raw_block: dict = field(default_factory=dict)


@dataclass
class LayoutDocument:
    s3_key: str
    page_count: int
    elements: list[LayoutElement]
    textract_job_id: str = ""
    has_tables: bool = False
    has_figures: bool = False
    cost_estimate_usd: float = 0.0


class DocumentTooLargeError(Exception):
    pass


class TextractLayoutParser:
    """Parse documents from S3 using Textract with layout awareness."""

    def __init__(self) -> None:
        self._textract = boto3.client("textract", region_name=REGION)
        self._s3 = boto3.client("s3", region_name=REGION)
        self._max_pages = int(os.environ.get("TEXTRACT_MAX_PAGES", str(TEXTRACT_MAX_PAGES)))
        self._enabled = os.environ.get("TEXTRACT_ENABLED", "true").lower() == "true"
        self._bucket = os.environ.get("S3_BUCKET", S3_BUCKET)

    def _get_page_count(self, s3_key: str) -> int:
        """Estimate page count from S3 object size (fast, no API call)."""
        try:
            head = self._s3.head_object(Bucket=self._bucket, Key=s3_key)
            size_bytes = head["ContentLength"]
            # Rough heuristic: ~50KB per page for a typical PDF
            return max(1, size_bytes // 51200)
        except Exception:
            return 1

    def _estimate_cost(self, page_count: int, has_tables: bool) -> float:
        if has_tables:
            return page_count * 0.015   # AnalyzeDocument
        return page_count * 0.0015      # DetectDocumentText

    def parse_document(self, s3_key: str) -> LayoutDocument:
        """Parse a document stored in S3. Raises DocumentTooLargeError if > max_pages."""
        if not self._enabled:
            return self._fallback_parse(s3_key)

        page_count = self._get_page_count(s3_key)
        if page_count > self._max_pages:
            raise DocumentTooLargeError(
                f"Document has ~{page_count} pages (max {self._max_pages}). "
                f"Split the document or increase TEXTRACT_MAX_PAGES."
            )

        # First pass: quick single-page check for tables
        has_tables = self._quick_table_check(s3_key)
        cost = self._estimate_cost(page_count, has_tables)
        print(
            f"[Textract] {s3_key}: ~{page_count} pages, "
            f"{'AnalyzeDocument' if has_tables else 'DetectDocumentText'}, "
            f"estimated cost: ${cost:.4f}"
        )

        if has_tables:
            return self._analyze_document(s3_key, page_count, cost)
        else:
            return self._detect_text(s3_key, page_count, cost)

    def _quick_table_check(self, s3_key: str) -> bool:
        """Run synchronous single-page DetectDocumentText to check for TABLE blocks."""
        try:
            resp = self._textract.detect_document_text(
                Document={"S3Object": {"Bucket": self._bucket, "Name": s3_key}}
            )
            # If we see TABLE in block types, use AnalyzeDocument
            for block in resp.get("Blocks", []):
                if block.get("BlockType") == "TABLE":
                    return True
            return False
        except Exception:
            return True  # default to full analysis on error

    def _detect_text(self, s3_key: str, page_count: int, cost: float) -> LayoutDocument:
        """Use async StartDocumentTextDetection for plain text PDFs."""
        resp = self._textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": self._bucket, "Name": s3_key}},
        )
        job_id = resp["JobId"]
        blocks = self._poll_job("text_detection", job_id)
        return self._build_layout_doc(s3_key, page_count, blocks, job_id, cost, has_tables=False)

    def _analyze_document(self, s3_key: str, page_count: int, cost: float) -> LayoutDocument:
        """Use async StartDocumentAnalysis for PDFs with tables/forms."""
        resp = self._textract.start_document_analysis(
            DocumentLocation={"S3Object": {"Bucket": self._bucket, "Name": s3_key}},
            FeatureTypes=["TABLES", "FORMS", "LAYOUT"],
        )
        job_id = resp["JobId"]
        blocks = self._poll_job("document_analysis", job_id)
        return self._build_layout_doc(s3_key, page_count, blocks, job_id, cost, has_tables=True)

    def _poll_job(self, job_type: str, job_id: str, timeout: int = 600) -> list[dict]:
        """Poll Textract until job completes. Returns all blocks."""
        get_fn = (
            self._textract.get_document_text_detection
            if job_type == "text_detection"
            else self._textract.get_document_analysis
        )
        start = time.time()
        wait = 5
        all_blocks: list[dict] = []
        next_token = None

        while True:
            kwargs: dict[str, Any] = {"JobId": job_id}
            if next_token:
                kwargs["NextToken"] = next_token

            resp = get_fn(**kwargs)
            status = resp["JobStatus"]

            if status == "FAILED":
                raise RuntimeError(f"Textract job {job_id} failed: {resp.get('StatusMessage')}")

            if status == "SUCCEEDED":
                all_blocks.extend(resp.get("Blocks", []))
                next_token = resp.get("NextToken")
                if not next_token:
                    break
            else:  # IN_PROGRESS
                if time.time() - start > timeout:
                    raise TimeoutError(f"Textract job {job_id} timed out after {timeout}s")
                time.sleep(wait)
                wait = min(wait * 1.5, 30)  # exponential backoff, cap at 30s

        return all_blocks

    def _build_layout_doc(
        self,
        s3_key: str,
        page_count: int,
        blocks: list[dict],
        job_id: str,
        cost: float,
        has_tables: bool,
    ) -> LayoutDocument:
        elements: list[LayoutElement] = []
        current_section = ""
        reading_order = 0
        has_figures = False

        # Build block lookup by ID
        block_map = {b["Id"]: b for b in blocks}

        # Group CELL blocks by TABLE parent
        table_cells: dict[str, list[dict]] = {}
        for block in blocks:
            if block["BlockType"] == "CELL":
                for rel in block.get("Relationships", []):
                    if rel["Type"] == "CHILD":
                        parent_id = block.get("Id", "")
                        # Find which TABLE owns this CELL
                        pass

        for block in blocks:
            btype = block.get("BlockType", "")
            page = block.get("Page", 1)
            bbox_raw = block.get("Geometry", {}).get("BoundingBox", {})
            bbox = BoundingBox(
                left=bbox_raw.get("Left", 0),
                top=bbox_raw.get("Top", 0),
                width=bbox_raw.get("Width", 0),
                height=bbox_raw.get("Height", 0),
            )
            confidence = block.get("Confidence", 0.0)

            if btype == "LAYOUT_TITLE":
                text = self._extract_text(block, block_map)
                el = LayoutElement(ElementType.TITLE, text, page, bbox, confidence, reading_order)
                elements.append(el)
                current_section = text
                reading_order += 1

            elif btype == "LAYOUT_SECTION_HEADER":
                text = self._extract_text(block, block_map)
                el = LayoutElement(ElementType.SECTION_HEADER, text, page, bbox, confidence, reading_order)
                elements.append(el)
                current_section = text
                reading_order += 1

            elif btype in ("LAYOUT_TEXT", "LINE"):
                text = self._extract_text(block, block_map)
                if text.strip():
                    el = LayoutElement(ElementType.TEXT, text, page, bbox, confidence, reading_order)
                    el.parent_section = current_section
                    elements.append(el)
                    reading_order += 1

            elif btype == "TABLE":
                table_text = self._extract_table_markdown(block, block_map)
                el = LayoutElement(ElementType.TABLE, table_text, page, bbox, confidence, reading_order)
                el.parent_section = current_section
                el.raw_block = block
                elements.append(el)
                reading_order += 1

            elif btype == "LAYOUT_FIGURE":
                has_figures = True
                if confidence >= 0.7:
                    el = LayoutElement(
                        ElementType.FIGURE, "[Figure]", page, bbox, confidence, reading_order
                    )
                    el.parent_section = current_section
                    el.raw_block = block
                    elements.append(el)
                    reading_order += 1

        return LayoutDocument(
            s3_key=s3_key,
            page_count=page_count,
            elements=elements,
            textract_job_id=job_id,
            has_tables=has_tables,
            has_figures=has_figures,
            cost_estimate_usd=cost,
        )

    def _extract_text(self, block: dict, block_map: dict[str, dict]) -> str:
        text_parts: list[str] = []
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for child_id in rel["Ids"]:
                    child = block_map.get(child_id, {})
                    if child.get("BlockType") == "WORD":
                        text_parts.append(child.get("Text", ""))
        return " ".join(text_parts)

    def _extract_table_markdown(self, table_block: dict, block_map: dict[str, dict]) -> str:
        from aws.textract.table_extractor import TableToMarkdown
        extractor = TableToMarkdown()
        return extractor.convert(table_block, block_map)

    def _fallback_parse(self, s3_key: str) -> LayoutDocument:
        """Basic fallback when Textract is disabled — returns single TEXT element."""
        return LayoutDocument(
            s3_key=s3_key,
            page_count=1,
            elements=[LayoutElement(
                ElementType.TEXT,
                f"[Textract disabled — raw text from {s3_key}]",
                1,
                BoundingBox(0, 0, 1, 1),
                1.0,
                0,
            )],
        )
