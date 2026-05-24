"""Extract and describe figures from PDFs using PyMuPDF + Bedrock Claude Haiku vision.

COST: Claude Haiku vision ~$0.001 per image (800px crop).
Only processes figures > 100x100 pixels with confidence >= 0.7.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import boto3

from aws.config import REGION
from aws.textract.parser import LayoutElement, ElementType


@dataclass
class ImageBlock:
    description: str
    image_type: str        # chart|diagram|signature|figure|table
    page_number: int
    s3_key: str            # crop stored to S3
    bedrock_tokens: int
    cost_usd: float


class MultiModalExtractor:
    """Describe PDF figures using Claude Haiku vision via Bedrock."""

    MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    MIN_PIXEL_SIZE = 100   # skip tiny icons/logos

    def __init__(self) -> None:
        self._bedrock = boto3.client("bedrock-runtime", region_name=REGION)
        self._s3 = boto3.client("s3", region_name=REGION)
        self._bucket = os.environ.get("S3_BUCKET", "")
        self._enabled = os.environ.get("TEXTRACT_ENABLED", "true").lower() == "true"

    def extract_figure(
        self,
        figure_element: LayoutElement,
        pdf_bytes: bytes,
        s3_key: str,
    ) -> ImageBlock | None:
        """Crop figure from PDF and describe it with Claude Haiku vision."""
        if not self._enabled or figure_element.element_type != ElementType.FIGURE:
            return None
        if figure_element.confidence < 0.7:
            return None

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(figure_element.page_number - 1)
        page_rect = page.rect

        bbox = figure_element.bounding_box
        rect = fitz.Rect(
            bbox.left * page_rect.width,
            bbox.top * page_rect.height,
            (bbox.left + bbox.width) * page_rect.width,
            (bbox.top + bbox.height) * page_rect.height,
        )
        pixel_w = rect.width
        pixel_h = rect.height
        if pixel_w < self.MIN_PIXEL_SIZE or pixel_h < self.MIN_PIXEL_SIZE:
            return None

        pix = page.get_pixmap(clip=rect, dpi=150)
        image_bytes = pix.tobytes("png")

        # Store crop to S3
        crop_key = f"figures/{s3_key}_p{figure_element.page_number}_r{figure_element.reading_order}.png"
        if self._bucket:
            try:
                self._s3.put_object(Bucket=self._bucket, Key=crop_key, Body=image_bytes,
                                    ContentType="image/png")
            except Exception:
                pass

        description, tokens, cost = self._describe_with_haiku(image_bytes)
        return ImageBlock(
            description=description,
            image_type=self._classify_type(description),
            page_number=figure_element.page_number,
            s3_key=crop_key,
            bedrock_tokens=tokens,
            cost_usd=cost,
        )

    def _describe_with_haiku(self, image_bytes: bytes) -> tuple[str, int, float]:
        import json
        image_b64 = base64.standard_b64encode(image_bytes).decode()
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are analyzing a figure from a business or research document. "
                            "Describe: (1) figure type (chart/diagram/table/signature/logo), "
                            "(2) key information or data points visible, "
                            "(3) any text labels or axes, "
                            "(4) what this figure represents in context. "
                            "Be concise and factual. Max 150 words."
                        ),
                    },
                ],
            }],
        }
        resp = self._bedrock.invoke_model(
            modelId=self.MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        description = result["content"][0]["text"]
        input_tokens = result.get("usage", {}).get("input_tokens", 0)
        output_tokens = result.get("usage", {}).get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        # Haiku: $0.00025/1K input + $0.00125/1K output
        cost = (input_tokens * 0.00025 + output_tokens * 0.00125) / 1000
        return description, total_tokens, cost

    def _classify_type(self, description: str) -> str:
        desc_lower = description.lower()
        if any(w in desc_lower for w in ["chart", "bar", "pie", "line graph", "scatter"]):
            return "chart"
        if any(w in desc_lower for w in ["diagram", "flow", "architecture", "schema"]):
            return "diagram"
        if "signature" in desc_lower:
            return "signature"
        if "table" in desc_lower:
            return "table"
        return "figure"
