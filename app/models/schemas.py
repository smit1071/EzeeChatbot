from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Optional, Literal
from enum import Enum


# ── Upload ────────────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    text = "text"
    url = "url"
    pdf = "pdf"   # multipart upload handled separately in the router


class UploadTextRequest(BaseModel):
    source_type: SourceType
    content: Optional[str] = None        # raw text when source_type == "text"
    url: Optional[str] = None            # web URL when source_type == "url"
    bot_name: Optional[str] = "My Bot"

    @field_validator("content")
    @classmethod
    def content_required_for_text(cls, v, info):
        data = info.data
        if data.get("source_type") == SourceType.text and not v:
            raise ValueError("content is required when source_type is 'text'")
        return v

    @field_validator("url")
    @classmethod
    def url_required_for_url(cls, v, info):
        data = info.data
        if data.get("source_type") == SourceType.url and not v:
            raise ValueError("url is required when source_type is 'url'")
        return v


class UploadResponse(BaseModel):
    bot_id: str
    bot_name: str
    chunks_indexed: int
    message: str


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    bot_id: str
    user_message: str
    conversation_history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    answer: str
    sources_used: int          # number of retrieved chunks
    grounded: bool             # False when fallback "not found" response issued
    latency_ms: float


# ── Stats ─────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    bot_id: str
    bot_name: str
    total_messages: int
    avg_latency_ms: float
    estimated_cost_usd: float
    unanswered_questions: int
    chunks_in_index: int
