"""Loaders for PDF, web pages, CSV files, and plain text."""

from __future__ import annotations

from src.core.logging import get_logger
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup

logger = get_logger(__name__)


class DocumentLoader:
    """Load raw text from various document sources."""

    async def load_pdf(self, path: str) -> str:
        """Extract text from a PDF file using pypdf."""
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages: list[str] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {page_num}]\n{text}")
        full_text = "\n\n".join(pages)
        logger.info("Loaded PDF %s: %d pages, %d chars", path, len(reader.pages), len(full_text))
        return full_text

    async def load_url(self, url: str) -> str:
        """Fetch a web page and extract readable text via BeautifulSoup."""
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "DocuMind/1.0 research-bot"},
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove boilerplate elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Prefer <article> or <main>, fall back to <body>
        container = soup.find("article") or soup.find("main") or soup.body
        text = container.get_text(separator="\n", strip=True) if container else soup.get_text()

        logger.info("Loaded URL %s: %d chars", url, len(text))
        return text

    async def load_csv(self, path: str) -> str:
        """Load a CSV and convert to a markdown table string."""
        df = pd.read_csv(path)
        md = df.to_markdown(index=False)
        logger.info("Loaded CSV %s: %d rows × %d cols", path, len(df), len(df.columns))
        return md

    async def load_text(self, path: str) -> str:
        """Load a plain text or code file."""
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        logger.info("Loaded text file %s: %d chars", path, len(text))
        return text

    async def load(self, source: str, doc_type: str) -> str:
        """Dispatch to the correct loader based on doc_type."""
        loaders = {
            "pdf": self.load_pdf,
            "web": self.load_url,
            "csv": self.load_csv,
            "text": self.load_text,
            "code": self.load_text,
        }
        loader = loaders.get(doc_type)
        if loader is None:
            raise ValueError(f"Unsupported doc_type: {doc_type!r}. Expected one of {list(loaders)}")
        return await loader(source)
