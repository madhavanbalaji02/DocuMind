"""Lambda Stage 1: layout-aware parsing.

Input:  {s3_key, filename, doc_type, bucket, size_bytes}
Output: {s3_key, parse_result_key, page_count, element_counts, ...input}

PDFs go through Textract. HTML and plain text are parsed directly — Textract
only accepts PDF/images, and the /url route labels every download "web", so
the real format is sniffed from the object's first bytes.

Saves parse results to S3: parsed/{s3_key}.json
"""

import boto3
import json
import os
import re
import time
from html.parser import HTMLParser


def lambda_handler(event, context):
    s3_key = event["s3_key"]
    filename = event["filename"]
    bucket = event.get("bucket", os.environ.get("S3_BUCKET", ""))

    s3 = boto3.client("s3")
    head = s3.get_object(Bucket=bucket, Key=s3_key, Range="bytes=0-1023")["Body"].read()
    doc_type = _sniff_doc_type(head, event.get("doc_type", ""))

    print(f"Stage1: parsing {s3_key} ({doc_type})")
    t0 = time.time()

    job_id = None
    try:
        if doc_type == "pdf":
            elements, page_count, job_id = _parse_pdf(bucket, s3_key)
        else:
            raw = s3.get_object(Bucket=bucket, Key=s3_key)["Body"].read()
            text = raw.decode("utf-8", errors="replace")
            elements = _parse_html(text) if doc_type == "web" else _parse_text(text)
            page_count = 1

        if not elements:
            raise ValueError(f"No text extracted from {s3_key}")

        element_counts = {
            "text": sum(1 for e in elements if e["type"] == "TEXT"),
            "tables": sum(1 for e in elements if e["type"] == "TABLE"),
            "figures": sum(1 for e in elements if e["type"] == "FIGURE"),
        }
        print(f"Parse complete: page_count={page_count}, elements={element_counts}")

        # Save parse results to S3
        parse_result = {
            "s3_key": s3_key,
            "filename": filename,
            "doc_type": doc_type,
            "page_count": page_count,
            "textract_job_id": job_id,
            "element_counts": element_counts,
            "elements": elements,
            "parse_time_s": round(time.time() - t0, 1),
        }
        parse_result_key = f"parsed/{s3_key}.json"
        s3.put_object(
            Bucket=bucket,
            Key=parse_result_key,
            Body=json.dumps(parse_result, default=str).encode(),
            ContentType="application/json",
        )
        print(f"Parse results saved: {parse_result_key}")

    except Exception as e:
        print(f"Stage1 error: {e}")
        raise

    return {
        **event,
        "doc_type": doc_type,
        "parse_result_key": parse_result_key,
        "page_count": page_count,
        "element_counts": element_counts,
        "textract_job_id": job_id,
    }


def _sniff_doc_type(head: bytes, declared: str) -> str:
    if head.lstrip().startswith(b"%PDF"):
        return "pdf"
    sample = head.decode("utf-8", errors="ignore").lower()
    if "<html" in sample or "<!doctype html" in sample or declared == "web":
        return "web"
    return "text"


def _parse_pdf(bucket: str, s3_key: str) -> tuple[list[dict], int, str]:
    textract = boto3.client("textract")

    # TEXTRACT COST GUARD
    # AnalyzeDocument ($0.015/page) only for PDFs with tables
    # DetectDocumentText ($0.0015/page) for plain text PDFs
    MAX_PAGES = int(os.environ.get("TEXTRACT_MAX_PAGES", "50"))

    resp = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": s3_key}},
        FeatureTypes=["TABLES", "FORMS", "LAYOUT"],
    )
    job_id = resp["JobId"]
    print(f"Textract job started: {job_id}")

    blocks = _poll_textract(textract, job_id, timeout=540)  # 9 min Lambda timeout buffer
    page_count = max((b.get("Page", 1) for b in blocks), default=1)

    if page_count > MAX_PAGES:
        raise ValueError(
            f"Document too large: {page_count} pages (max {MAX_PAGES}). "
            f"Split the document and re-upload."
        )

    return _extract_elements(blocks), page_count, job_id


class _HTMLTextExtractor(HTMLParser):
    """Collects visible text as TITLE / SECTION_HEADER / TEXT elements."""

    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form"}
    BLOCK = {"p", "div", "li", "td", "th", "pre", "blockquote", "section", "article", "br"}
    HEADINGS = {"h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self.elements: list[dict] = []
        self._buf: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def _flush(self, etype: str = "TEXT") -> None:
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf.clear()
        if text:
            self.elements.append({"type": etype, "content": text, "page": 1, "confidence": 100})

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._flush()
            self._in_title = True
        elif tag in self.HEADINGS or tag in self.BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._flush("TITLE")
            self._in_title = False
        elif tag in self.HEADINGS:
            self._flush("SECTION_HEADER")
        elif tag in self.BLOCK:
            self._flush()

    def handle_data(self, data):
        if not self._skip_depth:
            self._buf.append(data)


def _parse_html(html: str) -> list[dict]:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    parser._flush()
    return parser.elements


def _parse_text(text: str) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [{"type": "TEXT", "content": re.sub(r"\s+", " ", p), "page": 1, "confidence": 100}
            for p in paragraphs]


def _poll_textract(textract, job_id: str, timeout: int = 540) -> list[dict]:
    """Poll GetDocumentAnalysis until SUCCEEDED. Returns all blocks."""
    import time
    start = time.time()
    wait = 5
    all_blocks: list[dict] = []
    next_token = None

    while True:
        kwargs = {"JobId": job_id}
        if next_token:
            kwargs["NextToken"] = next_token

        resp = textract.get_document_analysis(**kwargs)
        status = resp["JobStatus"]

        if status == "FAILED":
            raise RuntimeError(f"Textract job {job_id} FAILED: {resp.get('StatusMessage')}")

        if status == "SUCCEEDED":
            all_blocks.extend(resp.get("Blocks", []))
            next_token = resp.get("NextToken")
            if not next_token:
                break
        else:
            if time.time() - start > timeout:
                raise TimeoutError(f"Textract job timed out after {timeout}s")
            time.sleep(wait)
            wait = min(wait * 1.5, 30)

    return all_blocks


def _extract_elements(blocks: list[dict]) -> list[dict]:
    """Convert raw Textract blocks to simplified element list."""
    block_map = {b["Id"]: b for b in blocks}
    elements: list[dict] = []

    for block in blocks:
        btype = block.get("BlockType", "")
        if btype in ("LAYOUT_TEXT", "LINE"):
            text = _get_text(block, block_map)
            if text.strip():
                elements.append({
                    "type": "TEXT",
                    "content": text,
                    "page": block.get("Page", 1),
                    "confidence": block.get("Confidence", 0),
                    "bbox": block.get("Geometry", {}).get("BoundingBox", {}),
                })
        elif btype == "LAYOUT_TITLE":
            text = _get_text(block, block_map)
            elements.append({
                "type": "TITLE",
                "content": text,
                "page": block.get("Page", 1),
                "confidence": block.get("Confidence", 0),
            })
        elif btype == "LAYOUT_SECTION_HEADER":
            text = _get_text(block, block_map)
            elements.append({
                "type": "SECTION_HEADER",
                "content": text,
                "page": block.get("Page", 1),
                "confidence": block.get("Confidence", 0),
            })
        elif btype == "TABLE":
            table_md = _table_to_markdown(block, block_map)
            elements.append({
                "type": "TABLE",
                "content": table_md,
                "page": block.get("Page", 1),
                "confidence": block.get("Confidence", 0),
            })
        elif btype == "LAYOUT_FIGURE":
            if block.get("Confidence", 0) >= 0.7:
                elements.append({
                    "type": "FIGURE",
                    "content": "[Figure]",
                    "page": block.get("Page", 1),
                    "confidence": block.get("Confidence", 0),
                    "bbox": block.get("Geometry", {}).get("BoundingBox", {}),
                })

    return elements


def _get_text(block: dict, block_map: dict) -> str:
    words = []
    for rel in block.get("Relationships", []):
        if rel["Type"] == "CHILD":
            for child_id in rel["Ids"]:
                child = block_map.get(child_id, {})
                if child.get("BlockType") == "WORD":
                    words.append(child.get("Text", ""))
    return " ".join(words)


def _table_to_markdown(table_block: dict, block_map: dict) -> str:
    cells: list[dict] = []
    for rel in table_block.get("Relationships", []):
        if rel["Type"] == "CHILD":
            for child_id in rel["Ids"]:
                block = block_map.get(child_id, {})
                if block.get("BlockType") == "CELL":
                    cells.append({
                        "row": block.get("RowIndex", 1) - 1,
                        "col": block.get("ColumnIndex", 1) - 1,
                        "text": _get_text(block, block_map),
                        "is_header": "COLUMN_HEADER" in block.get("EntityTypes", []),
                    })
    if not cells:
        return "[Empty Table]"

    max_row = max(c["row"] for c in cells)
    max_col = max(c["col"] for c in cells)
    grid = [[""] * (max_col + 1) for _ in range(max_row + 1)]
    headers = set()
    for cell in cells:
        grid[cell["row"]][cell["col"]] = cell["text"]
        if cell["is_header"]:
            headers.add(cell["row"])

    lines = []
    for i, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0 or i in headers:
            lines.append("| " + " | ".join("-" * max(3, len(c)) for c in row) + " |")
    return "\n".join(lines)
