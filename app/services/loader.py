"""
loader.py — Extracts raw text from three source types:

  text  → returned as-is (caller already has the string)
  url   → fetched with httpx, HTML stripped with BeautifulSoup
  pdf   → parsed with PyMuPDF (fitz); handles multi-page, columns, tables
"""

import httpx
from bs4 import BeautifulSoup
import fitz          # PyMuPDF
import io
from typing import Tuple


# ── Text ──────────────────────────────────────────────────────────────────────

def load_text(raw: str) -> Tuple[str, str]:
    """Returns (text, source_label)."""
    return raw.strip(), "plain_text"


# ── URL ───────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; EzeeChatBot/1.0; +https://ezeechatbot.io)"
    )
}

_UNWANTED_TAGS = [
    "script", "style", "noscript", "header", "footer",
    "nav", "aside", "form", "iframe",
]


def load_url(url: str, timeout: int = 20) -> Tuple[str, str]:
    """Fetch a URL and return clean text."""
    try:
        resp = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ValueError(f"HTTP {e.response.status_code} fetching {url}") from e
    except httpx.RequestError as e:
        raise ValueError(f"Could not reach {url}: {e}") from e

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(_UNWANTED_TAGS):
        tag.decompose()

    # Prefer article/main content blocks when available
    content_block = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id="content")
        or soup.find(class_="content")
        or soup.body
        or soup
    )

    text = content_block.get_text(separator="\n", strip=True)
    # Collapse blank lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cleaned = "\n".join(lines)

    return cleaned, url


# ── PDF ───────────────────────────────────────────────────────────────────────

def load_pdf_bytes(data: bytes, filename: str = "upload.pdf") -> Tuple[str, str]:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {e}") from e

    pages_text = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")   # preserves reading order
        if text.strip():
            pages_text.append(f"[Page {page_num}]\n{text.strip()}")

    doc.close()

    if not pages_text:
        raise ValueError("PDF appears to contain no extractable text (scanned image?).")

    return "\n\n".join(pages_text), filename
