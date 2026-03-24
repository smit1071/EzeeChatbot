"""
bot_store — lightweight registry that maps bot_id → BotRecord.

Each BotRecord holds:
  - the FAISS index (in-memory, isolated per bot)
  - the chunked text corpus
  - runtime stats
  - conversation window memory (last N turns)
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import numpy as np


@dataclass
class ConversationTurn:
    role: str   # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class BotRecord:
    bot_id: str
    name: str
    created_at: float = field(default_factory=time.time)

    # Knowledge base
    chunks: List[str] = field(default_factory=list)
    chunk_metadata: List[dict] = field(default_factory=list)  # source info per chunk
    faiss_index: Optional[object] = None          # faiss.IndexFlatIP after upload
    embeddings_matrix: Optional[np.ndarray] = None

    # Sliding window memory (list of ConversationTurn, capped at 2*WINDOW_MEMORY_SIZE)
    memory: List[ConversationTurn] = field(default_factory=list)

    # Stats
    total_messages: int = 0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    unanswered_count: int = 0      # responses where bot said "not found"


class BotStore:
    def __init__(self):
        self._store: Dict[str, BotRecord] = {}
        self._lock = threading.Lock()

    def create(self, bot_id: str, name: str) -> BotRecord:
        record = BotRecord(bot_id=bot_id, name=name)
        with self._lock:
            self._store[bot_id] = record
        return record

    def get(self, bot_id: str) -> Optional[BotRecord]:
        with self._lock:
            return self._store.get(bot_id)

    def list_bots(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

    def update(self, record: BotRecord) -> None:
        """Persist any in-place mutations back (no-op for in-memory, useful hook for future persistence)."""
        with self._lock:
            self._store[record.bot_id] = record


# Singleton
bot_store = BotStore()
