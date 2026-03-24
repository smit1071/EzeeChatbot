"""
upload.py — POST /upload

Accepts three intake modes:
  1. JSON body { source_type: "text", content: "..." }
  2. JSON body { source_type: "url",  url: "https://..." }
  3. multipart/form-data with file= (PDF) + optional bot_name field

Pipeline: load → chunk → embed → index → store → return bot_id
"""

import uuid
import time
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

from app.models.schemas import UploadTextRequest, UploadResponse, SourceType
from app.services.loader import load_text, load_url, load_pdf_bytes
from app.services.chunker import chunk_text
from app.services.embedder import embed_texts
from app.services.vector_store import build_index
from app.core.store import bot_store
import numpy as np

router = APIRouter()


def _ingest(raw_text: str, source_label: str, bot_name: str) -> UploadResponse:
    """Shared ingestion pipeline once raw text is available."""
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Extracted content is empty.")

    # 1. Chunk
    chunk_pairs = chunk_text(raw_text, source_label=source_label)
    if not chunk_pairs:
        raise HTTPException(status_code=400, detail="Could not extract any chunks.")

    chunks = [c for c, _ in chunk_pairs]
    metadata = [m for _, m in chunk_pairs]

    # 2. Embed
    embeddings: np.ndarray = embed_texts(chunks)

    # 3. Build FAISS index
    index = build_index(embeddings)

    # 4. Store in BotRecord
    bot_id = str(uuid.uuid4())
    record = bot_store.create(bot_id=bot_id, name=bot_name)
    record.chunks = chunks
    record.chunk_metadata = metadata
    record.faiss_index = index
    record.embeddings_matrix = embeddings
    bot_store.update(record)

    return UploadResponse(
        bot_id=bot_id,
        bot_name=bot_name,
        chunks_indexed=len(chunks),
        message=f"Knowledge base ready. {len(chunks)} chunks indexed.",
    )


# ── JSON endpoint (text / url) ────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_json(request: UploadTextRequest):
    """Upload plain text or a URL via JSON body."""
    bot_name = request.bot_name or "My Bot"

    if request.source_type == SourceType.text:
        raw, label = load_text(request.content)
    elif request.source_type == SourceType.url:
        try:
            raw, label = load_url(request.url)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    else:
        raise HTTPException(
            status_code=400,
            detail="Use POST /upload/pdf for PDF uploads.",
        )

    return _ingest(raw, label, bot_name)


# ── Multipart endpoint (PDF) ──────────────────────────────────────────────────

@router.post("/upload/pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    bot_name: Optional[str] = Form("My Bot"),
):
    """Upload a PDF file via multipart/form-data."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:   # 50 MB guard
        raise HTTPException(status_code=413, detail="PDF exceeds 50 MB limit.")

    try:
        raw, label = load_pdf_bytes(data, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return _ingest(raw, label, bot_name or "My Bot")
